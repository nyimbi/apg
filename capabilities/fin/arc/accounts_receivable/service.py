"""Accounts Receivable lifecycle service — standalone-capable via adapter protocols.

Architecture
------------
All external dependencies (auth, audit, notifications, workflow, persistence) are
injected via protocol adapters.  When running standalone the Null* adapters and
InMemoryStore are used automatically — no platform installation required.

Usage (standalone)::

    svc = AccountsReceivableService(tenant_id="acme")
    customer = await svc.create_customer("Acme Corp", 50000, "NET30", "USD")

Usage (platform)::

    from apg_common_auth import AuthService
    svc = AccountsReceivableService(
        tenant_id="acme",
        actor_id="user-123",
        auth=AuthService.from_env(),
        db_url="postgresql+asyncpg://...",
    )
"""

from __future__ import annotations

import math
import textwrap
from copy import deepcopy
from datetime import date, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from uuid6 import uuid7
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache


# ─────────────────────────────────────────────────────────────
# UUID7 shim
# ─────────────────────────────────────────────────────────────

def uuid7str() -> str:
	return str(uuid7())


# ─────────────────────────────────────────────────────────────
# Adapter + Store imports — graceful fallback
# ─────────────────────────────────────────────────────────────

try:
	from .domain.adapters import (
		AuthAdapter,
		AuditAdapter,
		NotifyAdapter,
		NullAuthAdapter,
		NullAuditAdapter,
		NullNotifyAdapter,
		NullWorkflowAdapter,
		WorkflowAdapter,
		get_audit_adapter,
		get_auth_adapter,
		get_notify_adapter,
		get_workflow_adapter,
	)
	from .database.store import Store, get_store
except ImportError:  # pragma: no cover — direct file load in tests/CLI
	from domain.adapters import (  # type: ignore[no-redef]
		AuthAdapter,
		AuditAdapter,
		NotifyAdapter,
		NullAuthAdapter,
		NullAuditAdapter,
		NullNotifyAdapter,
		NullWorkflowAdapter,
		WorkflowAdapter,
		get_audit_adapter,
		get_auth_adapter,
		get_notify_adapter,
		get_workflow_adapter,
	)
	from database.store import Store, get_store  # type: ignore[no-redef]


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

_DUNNING_LEVELS: list[dict[str, Any]] = [
	{"level": 1, "dpd": 7,  "label": "Gentle Reminder",    "action": "email"},
	{"level": 2, "dpd": 21, "label": "Firm Notice",        "action": "email_phone"},
	{"level": 3, "dpd": 45, "label": "Final Notice",       "action": "registered_mail"},
	{"level": 4, "dpd": 60, "label": "External Collector", "action": "collector"},
	{"level": 5, "dpd": 90, "label": "Legal Action",       "action": "legal"},
]

# IFRS 9 ECL provision rates by days-past-due bucket
_ECL_RATES: list[tuple[int, Decimal]] = [
	(30,  Decimal("0.01")),
	(60,  Decimal("0.05")),
	(90,  Decimal("0.20")),
	(120, Decimal("0.50")),
	(9999, Decimal("1.00")),
]

# WHT rates by jurisdiction (%)
_WHT_RATES: dict[str, Decimal] = {
	"KE": Decimal("0.05"),
	"NG": Decimal("0.10"),
	"GH": Decimal("0.05"),
	"ZA": Decimal("0.15"),
	"UG": Decimal("0.06"),
	"TZ": Decimal("0.05"),
}

_TWO_PLACES = Decimal("0.01")


# ─────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────

def _d(v: Any) -> Decimal:
	"""Coerce to Decimal."""
	if isinstance(v, Decimal):
		return v
	return Decimal(str(v))


def _today() -> str:
	return date.today().isoformat()


def _now() -> str:
	return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _days_between(d1: str, d2: str) -> int:
	"""Signed days d2 − d1 (ISO date strings)."""
	return (date.fromisoformat(d2) - date.fromisoformat(d1)).days


def _round2(v: Decimal | int | float) -> Decimal:
	return _d(v).quantize(_TWO_PLACES, rounding=ROUND_HALF_UP)


# ─────────────────────────────────────────────────────────────
# Service
# ─────────────────────────────────────────────────────────────

class AccountsReceivableService:
	"""Full accounts-receivable lifecycle service.

	Covers: customer management, invoice lifecycle, payments & allocation,
	dunning & collections, disputes, bad-debt write-offs, multi-currency
	revaluation, withholding tax, revenue recognition, reporting, e-invoicing,
	and GL entry generation.
	"""

	def __init__(
		self,
		tenant_id: str,
		actor_id: str = "system",
		*,
		auth: AuthAdapter | None = None,
		audit: AuditAdapter | None = None,
		notify: NotifyAdapter | None = None,
		workflow: WorkflowAdapter | None = None,
		db_url: str | None = None,
		store: Store | None = None,
	) -> None:
		self.tenant_id = tenant_id
		self.actor_id = actor_id
		self._auth = auth or get_auth_adapter()
		self._audit = audit or get_audit_adapter()
		self._notify = notify or get_notify_adapter()
		self._workflow = workflow or get_workflow_adapter()
		self._store = store or get_store(db_url)

	# ─────────────────────────────────────────────────────────
	# Internal logging helpers
	# ─────────────────────────────────────────────────────────

	def _log_ctx(self, method: str) -> str:
		return f"[ARC:{self.tenant_id}:{method}]"

	def _log_record_path(self, collection: str, record_id: str) -> str:
		return f"{collection}/{self.tenant_id}/{record_id}"

	async def _log_audit(
		self,
		event_type: str,
		resource_id: str,
		details: dict[str, Any],
	) -> None:
		await self._audit.log_event(
			event_type,
			self.actor_id,
			self.tenant_id,
			resource_id,
			details,
		)

	async def _log_notify(
		self,
		recipient: str,
		channel: str,
		subject: str,
		body: str,
		metadata: dict[str, Any] | None = None,
	) -> None:
		await self._notify.send(recipient, channel, subject, body, metadata)

	# ─────────────────────────────────────────────────────────
	# Store helpers
	# ─────────────────────────────────────────────────────────

	async def _put(self, collection: str, record: dict[str, Any]) -> dict[str, Any]:
		record["tenant_id"] = self.tenant_id
		await self._store.put(collection, record)
		return record

	async def _get(self, collection: str, record_id: str) -> dict[str, Any] | None:
		record = await self._store.get(collection, record_id)
		if record and record.get("tenant_id") != self.tenant_id:
			return None
		return record

	async def _require(self, collection: str, record_id: str, label: str) -> dict[str, Any]:
		record = await self._get(collection, record_id)
		if record is None:
			raise ValueError(f"{label} not found: {record_id}")
		return record

	async def _query(
		self,
		collection: str,
		extra: dict[str, Any] | None = None,
	) -> list[dict[str, Any]]:
		filters: dict[str, Any] = {"tenant_id": self.tenant_id}
		if extra:
			filters.update(extra)
		return await self._store.query(collection, filters, limit=10_000)

	# ─────────────────────────────────────────────────────────────────────────
	# ██  CUSTOMER MANAGEMENT
	# ─────────────────────────────────────────────────────────────────────────

	async def create_customer(
		self,
		name: str,
		credit_limit: float | int | Decimal,
		payment_terms: str,
		currency: str,
		**kwargs: Any,
	) -> dict[str, Any]:
		"""Create an AR customer master record."""
		assert name, "name required"
		assert _d(credit_limit) >= 0, "credit_limit must be >= 0"
		assert payment_terms, "payment_terms required"
		assert currency, "currency required"

		record: dict[str, Any] = {
			"id": uuid7str(),
			"type": "ar_customer",
			"name": name,
			"credit_limit": str(_d(credit_limit)),
			"payment_terms": payment_terms,
			"currency": currency,
			"credit_score": 500,
			"credit_hold": False,
			"credit_hold_reason": None,
			"status": "active",
			"created_at": _now(),
			"updated_at": _now(),
			**{k: v for k, v in kwargs.items() if k not in ("id", "tenant_id")},
		}
		await self._put("ar_customers", record)
		await self._log_audit("customer_created", record["id"], {"name": name, "credit_limit": str(credit_limit)})
		return deepcopy(record)

	async def update_customer(self, customer_id: str, **fields: Any) -> dict[str, Any]:
		"""Update mutable fields on a customer record."""
		record = await self._require("ar_customers", customer_id, "customer")
		forbidden = {"id", "tenant_id", "type", "created_at"}
		for k, v in fields.items():
			if k in forbidden:
				continue
			record[k] = str(v) if isinstance(v, Decimal) else v
		record["updated_at"] = _now()
		await self._put("ar_customers", record)
		await self._log_audit("customer_updated", customer_id, {"fields": list(fields)})
		return deepcopy(record)

	async def get_customer(self, customer_id: str) -> dict[str, Any]:
		return deepcopy(await self._require("ar_customers", customer_id, "customer"))

	async def list_customers(self, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
		rows = await self._query("ar_customers", filters)
		return [deepcopy(r) for r in rows]

	async def apply_credit_hold(self, customer_id: str, reason: str) -> dict[str, Any]:
		"""Place a customer on credit hold — blocks invoice issuance."""
		assert reason, "reason required"
		record = await self._require("ar_customers", customer_id, "customer")
		record["credit_hold"] = True
		record["credit_hold_reason"] = reason
		record["credit_hold_at"] = _now()
		record["updated_at"] = _now()
		await self._put("ar_customers", record)
		await self._log_audit("credit_hold_applied", customer_id, {"reason": reason})
		customer_email = record.get("email") or "finance@customer.invalid"
		await self._log_notify(
			customer_email, "email",
			f"Credit Hold Applied — {record['name']}",
			f"Your account has been placed on credit hold. Reason: {reason}",
		)
		return deepcopy(record)

	async def release_credit_hold(
		self, customer_id: str, reason: str, approval_ref: str
	) -> dict[str, Any]:
		"""Release a customer from credit hold with approval reference."""
		assert reason, "reason required"
		assert approval_ref, "approval_ref required"
		record = await self._require("ar_customers", customer_id, "customer")
		record["credit_hold"] = False
		record["credit_hold_reason"] = None
		record["credit_hold_released_at"] = _now()
		record["credit_hold_release_approval"] = approval_ref
		record["updated_at"] = _now()
		await self._put("ar_customers", record)
		await self._log_audit(
			"credit_hold_released", customer_id,
			{"reason": reason, "approval_ref": approval_ref},
		)
		return deepcopy(record)

	async def check_credit_limit(
		self, customer_id: str, new_invoice_amount: float | int | Decimal
	) -> dict[str, Any]:
		"""Real-time credit limit check including open invoices."""
		customer = await self._require("ar_customers", customer_id, "customer")
		open_invoices = await self._query(
			"ar_invoices",
			{"customer_id": customer_id},
		)
		outstanding = sum(
			_d(inv.get("outstanding_amount", 0))
			for inv in open_invoices
			if inv.get("status") not in ("paid", "cancelled", "void")
		)
		credit_limit = _d(customer.get("credit_limit", 0))
		new_amount = _d(new_invoice_amount)
		available = credit_limit - outstanding
		would_exceed = (outstanding + new_amount) > credit_limit
		return {
			"customer_id": customer_id,
			"credit_limit": str(credit_limit),
			"outstanding_balance": str(_round2(outstanding)),
			"available_credit": str(_round2(available)),
			"new_invoice_amount": str(new_amount),
			"would_exceed": would_exceed,
			"utilisation_pct": str(_round2((outstanding / credit_limit * 100) if credit_limit > 0 else Decimal("100"))),
			"checked_at": _now(),
		}

	async def update_credit_score(self, customer_id: str) -> dict[str, Any]:
		"""Recompute credit score 0–1000 from payment history."""
		customer = await self._require("ar_customers", customer_id, "customer")
		all_invoices = await self._query("ar_invoices", {"customer_id": customer_id})
		paid = [
			inv for inv in all_invoices
			if inv.get("status") == "paid" and inv.get("paid_at")
		]
		if not paid:
			score = 500  # neutral — no history
		else:
			on_time = sum(
				1
				for inv in paid
				if inv.get("paid_at", "9999") <= inv.get("due_date", "9999")
			)
			ratio = on_time / len(paid)
			# Weighted: recency bias (last 12 invoices), payment ratio
			recency_bonus = min(len(paid) / 12, 1.0) * 100
			score = int(ratio * 800 + recency_bonus + 100)
			score = max(0, min(1000, score))

		customer["credit_score"] = score
		customer["credit_score_updated_at"] = _now()
		customer["updated_at"] = _now()
		await self._put("ar_customers", customer)
		await self._log_audit("credit_score_updated", customer_id, {"score": score})
		return deepcopy(customer)

	async def request_credit_limit_increase(
		self,
		customer_id: str,
		requested_limit: float | int | Decimal,
		justification: str,
	) -> dict[str, Any]:
		"""Submit a credit limit increase request for human approval."""
		assert justification, "justification required"
		customer = await self._require("ar_customers", customer_id, "customer")
		request_id = uuid7str()
		request: dict[str, Any] = {
			"id": request_id,
			"type": "credit_limit_request",
			"customer_id": customer_id,
			"current_limit": customer.get("credit_limit", "0"),
			"requested_limit": str(_d(requested_limit)),
			"justification": justification,
			"status": "pending",
			"requested_by": self.actor_id,
			"requested_at": _now(),
		}
		await self._put("ar_credit_requests", request)
		await self._log_audit("credit_limit_increase_requested", customer_id, request)
		wf = await self._workflow.start_workflow(
			"credit_limit_review",
			{"request_id": request_id, "customer_id": customer_id},
		)
		request["workflow_instance_id"] = wf.get("instance_id")
		await self._put("ar_credit_requests", request)
		return deepcopy(request)

	# ─────────────────────────────────────────────────────────────────────────
	# ██  INVOICE LIFECYCLE
	# ─────────────────────────────────────────────────────────────────────────

	async def create_invoice(
		self,
		customer_id: str,
		invoice_date: str,
		due_date: str,
		lines: list[dict[str, Any]],
		currency: str,
		payment_terms: str,
		**kwargs: Any,
	) -> dict[str, Any]:
		"""Create a draft invoice with line items."""
		assert customer_id, "customer_id required"
		assert invoice_date, "invoice_date required"
		assert due_date >= invoice_date, "due_date must not precede invoice_date"
		assert lines, "at least one line required"
		assert currency, "currency required"

		await self._require("ar_customers", customer_id, "customer")

		subtotal = Decimal("0")
		tax_total = Decimal("0")
		processed_lines = []
		for i, line in enumerate(lines):
			assert line.get("description"), f"line {i}: description required"
			qty = _d(line.get("quantity", 0))
			price = _d(line.get("unit_price", 0))
			assert qty > 0, f"line {i}: quantity must be positive"
			assert price >= 0, f"line {i}: unit_price must be non-negative"
			line_total = _round2(qty * price)
			tax_rate = _d(line.get("tax_rate", 0))
			tax_amount = _round2(line_total * tax_rate)
			processed_lines.append({
				"line_no": i + 1,
				"description": line["description"],
				"quantity": str(qty),
				"unit_price": str(price),
				"line_total": str(line_total),
				"tax_rate": str(tax_rate),
				"tax_amount": str(tax_amount),
				"revenue_account": line.get("revenue_account", "4000"),
				"cost_centre": line.get("cost_centre"),
			})
			subtotal += line_total
			tax_total += tax_amount

		total_amount = _round2(subtotal + tax_total)
		assert total_amount > 0, "invoice total must be positive"

		record: dict[str, Any] = {
			"id": uuid7str(),
			"type": "ar_invoice",
			"customer_id": customer_id,
			"invoice_date": invoice_date,
			"due_date": due_date,
			"lines": processed_lines,
			"currency": currency,
			"payment_terms": payment_terms,
			"subtotal": str(subtotal),
			"tax_total": str(tax_total),
			"total_amount": str(total_amount),
			"paid_amount": "0.00",
			"outstanding_amount": str(total_amount),
			"status": "draft",
			"created_at": _now(),
			"updated_at": _now(),
			**{k: v for k, v in kwargs.items() if k not in ("id", "tenant_id")},
		}
		await self._put("ar_invoices", record)
		await self._log_audit("invoice_created", record["id"], {
			"customer_id": customer_id, "total": str(total_amount),
		})
		return deepcopy(record)

	async def validate_invoice(self, invoice_id: str) -> dict[str, Any]:
		"""Validate line totals, tax, and credit limit.  Returns validation report."""
		invoice = await self._require("ar_invoices", invoice_id, "invoice")
		issues: list[str] = []

		# Re-sum lines
		computed_sub = Decimal("0")
		computed_tax = Decimal("0")
		for line in invoice.get("lines", []):
			computed_sub += _d(line["line_total"])
			computed_tax += _d(line["tax_amount"])
		if _round2(computed_sub) != _round2(_d(invoice["subtotal"])):
			issues.append("subtotal_mismatch")
		if _round2(computed_tax) != _round2(_d(invoice["tax_total"])):
			issues.append("tax_total_mismatch")
		if _round2(computed_sub + computed_tax) != _round2(_d(invoice["total_amount"])):
			issues.append("total_amount_mismatch")

		# Credit limit
		credit_check = await self.check_credit_limit(
			invoice["customer_id"], invoice["total_amount"]
		)
		if credit_check["would_exceed"]:
			issues.append("credit_limit_exceeded")

		valid = len(issues) == 0
		invoice["validation_status"] = "valid" if valid else "invalid"
		invoice["validation_issues"] = issues
		invoice["validated_at"] = _now()
		invoice["updated_at"] = _now()
		await self._put("ar_invoices", invoice)
		return {"invoice_id": invoice_id, "valid": valid, "issues": issues, "credit_check": credit_check}

	async def submit_invoice(self, invoice_id: str) -> dict[str, Any]:
		"""Transition draft → submitted."""
		invoice = await self._require("ar_invoices", invoice_id, "invoice")
		assert invoice["status"] == "draft", f"expected draft, got {invoice['status']}"
		invoice["status"] = "submitted"
		invoice["submitted_at"] = _now()
		invoice["updated_at"] = _now()
		await self._put("ar_invoices", invoice)
		await self._log_audit("invoice_submitted", invoice_id, {})
		return deepcopy(invoice)

	async def approve_invoice(self, invoice_id: str, approved_by: str) -> dict[str, Any]:
		"""Transition submitted → approved."""
		assert approved_by, "approved_by required"
		invoice = await self._require("ar_invoices", invoice_id, "invoice")
		assert invoice["status"] == "submitted", f"expected submitted, got {invoice['status']}"
		invoice["status"] = "approved"
		invoice["approved_by"] = approved_by
		invoice["approved_at"] = _now()
		invoice["updated_at"] = _now()
		await self._put("ar_invoices", invoice)
		await self._log_audit("invoice_approved", invoice_id, {"approved_by": approved_by})
		return deepcopy(invoice)

	async def post_invoice(self, invoice_id: str) -> dict[str, Any]:
		"""Transition approved → posted; emit GL journal entry."""
		invoice = await self._require("ar_invoices", invoice_id, "invoice")
		assert invoice["status"] == "approved", f"expected approved, got {invoice['status']}"
		invoice["status"] = "posted"
		invoice["posted_at"] = _now()
		invoice["updated_at"] = _now()
		await self._put("ar_invoices", invoice)
		gl_entry = await self.invoice_gl_entry(invoice_id)
		await self._log_audit("invoice_posted", invoice_id, {"gl_entry_id": gl_entry["id"]})
		return {"invoice": deepcopy(invoice), "gl_entry": gl_entry}

	async def cancel_invoice(self, invoice_id: str, reason: str) -> dict[str, Any]:
		"""Cancel an invoice (only when no payments are allocated)."""
		assert reason, "reason required"
		invoice = await self._require("ar_invoices", invoice_id, "invoice")
		assert invoice["status"] not in ("paid",), "cannot cancel a paid invoice; use credit note"
		assert _d(invoice.get("paid_amount", 0)) == 0, "cannot cancel — payment already applied"
		invoice["status"] = "cancelled"
		invoice["cancelled_reason"] = reason
		invoice["cancelled_at"] = _now()
		invoice["updated_at"] = _now()
		await self._put("ar_invoices", invoice)
		await self._log_audit("invoice_cancelled", invoice_id, {"reason": reason})
		return deepcopy(invoice)

	async def create_credit_note(
		self,
		original_invoice_id: str,
		reason: str,
		lines: list[dict[str, Any]],
		credit_date: str,
	) -> dict[str, Any]:
		"""Issue a credit note against an existing invoice."""
		original = await self._require("ar_invoices", original_invoice_id, "invoice")
		assert reason, "reason required"
		assert lines, "lines required"
		credit = await self.create_invoice(
			customer_id=original["customer_id"],
			invoice_date=credit_date,
			due_date=credit_date,
			lines=lines,
			currency=original["currency"],
			payment_terms=original.get("payment_terms", "immediate"),
			original_invoice_id=original_invoice_id,
			credit_note_reason=reason,
			document_type="credit_note",
		)
		credit["status"] = "approved"
		credit["total_amount"] = str(_d(credit["total_amount"]) * -1)
		credit["outstanding_amount"] = credit["total_amount"]
		credit["updated_at"] = _now()
		await self._put("ar_invoices", credit)
		await self._log_audit("credit_note_created", credit["id"], {
			"original_invoice_id": original_invoice_id, "reason": reason,
		})
		return deepcopy(credit)

	async def create_debit_note(
		self,
		original_invoice_id: str,
		reason: str,
		amount: float | int | Decimal,
		debit_date: str,
	) -> dict[str, Any]:
		"""Issue a debit note to collect additional amounts."""
		original = await self._require("ar_invoices", original_invoice_id, "invoice")
		assert reason, "reason required"
		assert _d(amount) > 0, "amount must be positive"
		debit = await self.create_invoice(
			customer_id=original["customer_id"],
			invoice_date=debit_date,
			due_date=debit_date,
			lines=[{
				"description": reason,
				"quantity": "1",
				"unit_price": str(_d(amount)),
				"tax_rate": "0",
				"revenue_account": original["lines"][0].get("revenue_account", "4000") if original.get("lines") else "4000",
			}],
			currency=original["currency"],
			payment_terms=original.get("payment_terms", "immediate"),
			original_invoice_id=original_invoice_id,
			debit_note_reason=reason,
			document_type="debit_note",
		)
		await self._log_audit("debit_note_created", debit["id"], {
			"original_invoice_id": original_invoice_id, "amount": str(amount),
		})
		return deepcopy(debit)

	async def get_invoice(self, invoice_id: str) -> dict[str, Any]:
		return deepcopy(await self._require("ar_invoices", invoice_id, "invoice"))

	async def list_invoices(self, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
		rows = await self._query("ar_invoices", filters)
		return [deepcopy(r) for r in rows]

	async def send_invoice(self, invoice_id: str, channel: str = "email") -> dict[str, Any]:
		"""Send invoice to customer via specified channel."""
		invoice = await self._require("ar_invoices", invoice_id, "invoice")
		assert invoice["status"] in ("approved", "posted"), \
			f"invoice must be approved/posted to send; status={invoice['status']}"
		customer = await self._require("ar_customers", invoice["customer_id"], "customer")
		recipient = customer.get("email") or customer.get("name")
		subject = f"Invoice {invoice_id} — {invoice['total_amount']} {invoice['currency']}"
		body = (
			f"Dear {customer['name']},\n\n"
			f"Please find attached invoice {invoice_id} for "
			f"{invoice['total_amount']} {invoice['currency']} "
			f"due on {invoice['due_date']}.\n\n"
			f"Payment terms: {invoice.get('payment_terms','—')}\n"
		)
		await self._log_notify(recipient, channel, subject, body, {"invoice_id": invoice_id})
		invoice["sent_at"] = _now()
		invoice["sent_channel"] = channel
		invoice["updated_at"] = _now()
		await self._put("ar_invoices", invoice)
		await self._log_audit("invoice_sent", invoice_id, {"channel": channel, "recipient": recipient})
		return {"invoice_id": invoice_id, "sent_to": recipient, "channel": channel, "sent_at": invoice["sent_at"]}

	# ─────────────────────────────────────────────────────────────────────────
	# ██  PAYMENTS & ALLOCATION
	# ─────────────────────────────────────────────────────────────────────────

	async def record_payment(
		self,
		customer_id: str,
		amount: float | int | Decimal,
		currency: str,
		payment_date: str,
		payment_method: str,
		reference: str,
	) -> dict[str, Any]:
		"""Record a customer payment receipt."""
		await self._require("ar_customers", customer_id, "customer")
		assert _d(amount) > 0, "amount must be positive"
		assert reference, "reference required"
		record: dict[str, Any] = {
			"id": uuid7str(),
			"type": "ar_payment",
			"customer_id": customer_id,
			"amount": str(_d(amount)),
			"unapplied_amount": str(_d(amount)),
			"currency": currency,
			"payment_date": payment_date,
			"payment_method": payment_method,
			"reference": reference,
			"status": "unallocated",
			"allocations": [],
			"created_at": _now(),
			"updated_at": _now(),
		}
		await self._put("ar_payments", record)
		await self._log_audit("payment_recorded", record["id"], {
			"customer_id": customer_id, "amount": str(amount), "method": payment_method,
		})
		return deepcopy(record)

	async def apply_payment(
		self,
		payment_id: str,
		allocations: list[dict[str, Any]],
	) -> dict[str, Any]:
		"""Apply a payment to specific invoices: allocations=[{invoice_id, amount}]."""
		payment = await self._require("ar_payments", payment_id, "payment")
		assert payment["status"] != "reversed", "cannot allocate a reversed payment"

		unapplied = _d(payment["unapplied_amount"])
		allocation_records = []

		for alloc in allocations:
			inv_id = alloc["invoice_id"]
			alloc_amt = _d(alloc["amount"])
			assert alloc_amt > 0, f"allocation amount must be positive for {inv_id}"
			assert alloc_amt <= unapplied, \
				f"allocation {alloc_amt} exceeds unapplied balance {unapplied}"

			invoice = await self._require("ar_invoices", inv_id, "invoice")
			assert invoice["status"] not in ("paid", "cancelled", "void"), \
				f"invoice {inv_id} is {invoice['status']}"
			inv_outstanding = _d(invoice["outstanding_amount"])
			applied = min(alloc_amt, inv_outstanding)

			invoice["paid_amount"] = str(_round2(_d(invoice["paid_amount"]) + applied))
			invoice["outstanding_amount"] = str(_round2(inv_outstanding - applied))
			if _d(invoice["outstanding_amount"]) == 0:
				invoice["status"] = "paid"
				invoice["paid_at"] = _now()
			else:
				invoice["status"] = "partially_paid"
			invoice["updated_at"] = _now()
			await self._put("ar_invoices", invoice)

			alloc_rec = {
				"id": uuid7str(),
				"type": "ar_allocation",
				"payment_id": payment_id,
				"invoice_id": inv_id,
				"amount": str(applied),
				"allocated_at": _now(),
				"status": "active",
			}
			await self._put("ar_allocations", alloc_rec)
			allocation_records.append(alloc_rec)
			payment.setdefault("allocations", []).append(alloc_rec["id"])
			unapplied -= applied

		payment["unapplied_amount"] = str(_round2(unapplied))
		payment["status"] = "fully_applied" if unapplied == 0 else "partially_applied"
		payment["updated_at"] = _now()
		await self._put("ar_payments", payment)
		await self._log_audit("payment_applied", payment_id, {
			"allocations": [a["id"] for a in allocation_records],
		})
		return {"payment": deepcopy(payment), "allocations": allocation_records}

	async def auto_allocate_payment(self, payment_id: str) -> dict[str, Any]:
		"""FIFO allocation — apply payment to oldest open invoices first."""
		payment = await self._require("ar_payments", payment_id, "payment")
		customer_id = payment["customer_id"]
		open_invoices = await self._query("ar_invoices", {"customer_id": customer_id})
		open_invoices = [
			inv for inv in open_invoices
			if inv.get("status") in ("posted", "partially_paid", "approved")
			and _d(inv.get("outstanding_amount", 0)) > 0
		]
		open_invoices.sort(key=lambda inv: inv.get("due_date", "9999-99-99"))

		unapplied = _d(payment["unapplied_amount"])
		if unapplied == 0:
			return {"payment_id": payment_id, "allocations": [], "note": "nothing_to_apply"}

		allocations = []
		for inv in open_invoices:
			if unapplied <= 0:
				break
			outstanding = _d(inv["outstanding_amount"])
			apply_amount = min(unapplied, outstanding)
			allocations.append({"invoice_id": inv["id"], "amount": str(apply_amount)})
			unapplied -= apply_amount

		return await self.apply_payment(payment_id, allocations)

	async def smart_match_payment(self, payment_id: str) -> dict[str, Any]:
		"""Match payment by PO reference, exact amount, and payment history."""
		payment = await self._require("ar_payments", payment_id, "payment")
		customer_id = payment["customer_id"]
		amount = _d(payment["unapplied_amount"])
		reference = payment.get("reference", "")

		open_invoices = await self._query("ar_invoices", {"customer_id": customer_id})
		open_invoices = [
			inv for inv in open_invoices
			if inv.get("status") in ("posted", "partially_paid", "approved")
			and _d(inv.get("outstanding_amount", 0)) > 0
		]

		# Scoring: exact amount match = 100 pts, reference match = 50 pts
		def _score(inv: dict[str, Any]) -> int:
			s = 0
			if _round2(_d(inv["outstanding_amount"])) == _round2(amount):
				s += 100
			if reference and (
				reference in inv.get("id", "") or
				reference in str(inv.get("purchase_order", "")) or
				reference in str(inv.get("invoice_number", ""))
			):
				s += 50
			return s

		scored = sorted(open_invoices, key=_score, reverse=True)
		candidates = [inv for inv in scored if _score(inv) > 0]

		if not candidates:
			# Fall back to FIFO
			return await self.auto_allocate_payment(payment_id)

		allocations = []
		remaining = amount
		for inv in candidates:
			if remaining <= 0:
				break
			outstanding = _d(inv["outstanding_amount"])
			apply_amount = min(remaining, outstanding)
			allocations.append({"invoice_id": inv["id"], "amount": str(apply_amount)})
			remaining -= apply_amount

		return await self.apply_payment(payment_id, allocations)

	async def get_payment(self, payment_id: str) -> dict[str, Any]:
		return deepcopy(await self._require("ar_payments", payment_id, "payment"))

	async def list_unallocated_payments(
		self, customer_id: str | None = None
	) -> list[dict[str, Any]]:
		extra = {"status": "unallocated"}
		if customer_id:
			extra["customer_id"] = customer_id
		rows = await self._query("ar_payments", extra)
		# Also include partially_applied
		extra2 = {"status": "partially_applied"}
		if customer_id:
			extra2["customer_id"] = customer_id
		rows2 = await self._query("ar_payments", extra2)
		return [deepcopy(r) for r in rows + rows2]

	async def reverse_allocation(self, allocation_id: str, reason: str) -> dict[str, Any]:
		"""Reverse an applied allocation — reinstates invoice outstanding balance."""
		assert reason, "reason required"
		alloc = await self._require("ar_allocations", allocation_id, "allocation")
		assert alloc["status"] == "active", "allocation already reversed"

		invoice = await self._require("ar_invoices", alloc["invoice_id"], "invoice")
		rev_amount = _d(alloc["amount"])
		invoice["outstanding_amount"] = str(_round2(_d(invoice["outstanding_amount"]) + rev_amount))
		invoice["paid_amount"] = str(_round2(_d(invoice["paid_amount"]) - rev_amount))
		invoice["status"] = "partially_paid" if _d(invoice["paid_amount"]) > 0 else "posted"
		invoice["updated_at"] = _now()
		await self._put("ar_invoices", invoice)

		payment = await self._require("ar_payments", alloc["payment_id"], "payment")
		payment["unapplied_amount"] = str(_round2(_d(payment["unapplied_amount"]) + rev_amount))
		payment["status"] = "partially_applied" if _d(payment["unapplied_amount"]) < _d(payment["amount"]) else "unallocated"
		payment["updated_at"] = _now()
		await self._put("ar_payments", payment)

		alloc["status"] = "reversed"
		alloc["reversal_reason"] = reason
		alloc["reversed_at"] = _now()
		await self._put("ar_allocations", alloc)

		await self._log_audit("allocation_reversed", allocation_id, {"reason": reason})
		return deepcopy(alloc)

	async def process_bounced_payment(
		self, payment_id: str, bounce_reason: str
	) -> dict[str, Any]:
		"""Mark a payment as bounced, reverse all allocations, apply bounce fee."""
		assert bounce_reason, "bounce_reason required"
		payment = await self._require("ar_payments", payment_id, "payment")

		# Reverse all active allocations
		reversed_allocs = []
		for alloc_id in payment.get("allocations", []):
			alloc = await self._store.get("ar_allocations", alloc_id)
			if alloc and alloc.get("status") == "active":
				rev = await self.reverse_allocation(alloc_id, f"bounce: {bounce_reason}")
				reversed_allocs.append(rev["id"])

		payment["status"] = "bounced"
		payment["bounce_reason"] = bounce_reason
		payment["bounced_at"] = _now()
		payment["updated_at"] = _now()
		await self._put("ar_payments", payment)

		customer = await self._require("ar_customers", payment["customer_id"], "customer")
		await self._log_notify(
			customer.get("email") or customer["name"],
			"email",
			f"Payment Dishonoured — Ref {payment.get('reference')}",
			f"Your payment of {payment['amount']} {payment['currency']} has been returned. "
			f"Reason: {bounce_reason}. Please arrange alternative payment immediately.",
		)
		await self._log_audit("payment_bounced", payment_id, {
			"bounce_reason": bounce_reason, "reversed_allocations": reversed_allocs,
		})
		return deepcopy(payment)

	# ─────────────────────────────────────────────────────────────────────────
	# ██  DUNNING & COLLECTIONS
	# ─────────────────────────────────────────────────────────────────────────

	async def calculate_aging(
		self, as_of_date: str | None = None
	) -> dict[str, Any]:
		"""AR aging by customer: current / 30 / 60 / 90 / 120+ DPD buckets."""
		ref = as_of_date or _today()
		invoices = await self._query("ar_invoices")
		invoices = [
			inv for inv in invoices
			if inv.get("status") not in ("cancelled", "void", "paid")
		]

		# Aggregate per customer
		summary: dict[str, dict[str, Decimal]] = {}
		for inv in invoices:
			cust_id = inv["customer_id"]
			due = inv.get("due_date", ref)
			dpd = _days_between(due, ref)
			outstanding = _d(inv.get("outstanding_amount", 0))
			if cust_id not in summary:
				summary[cust_id] = {
					"current": Decimal(0), "1_30": Decimal(0),
					"31_60": Decimal(0), "61_90": Decimal(0),
					"91_120": Decimal(0), "120_plus": Decimal(0),
				}
			if dpd <= 0:
				summary[cust_id]["current"] += outstanding
			elif dpd <= 30:
				summary[cust_id]["1_30"] += outstanding
			elif dpd <= 60:
				summary[cust_id]["31_60"] += outstanding
			elif dpd <= 90:
				summary[cust_id]["61_90"] += outstanding
			elif dpd <= 120:
				summary[cust_id]["91_120"] += outstanding
			else:
				summary[cust_id]["120_plus"] += outstanding

		total: dict[str, Decimal] = {
			"current": Decimal(0), "1_30": Decimal(0),
			"31_60": Decimal(0), "61_90": Decimal(0),
			"91_120": Decimal(0), "120_plus": Decimal(0),
		}
		customers_out: list[dict[str, Any]] = []
		for cust_id, buckets in summary.items():
			row = {k: str(_round2(v)) for k, v in buckets.items()}
			row["customer_id"] = cust_id
			row["total_outstanding"] = str(_round2(sum(buckets.values())))
			customers_out.append(row)
			for k in total:
				total[k] += buckets[k]

		return {
			"as_of_date": ref,
			"tenant_id": self.tenant_id,
			"customers": customers_out,
			"totals": {k: str(_round2(v)) for k, v in total.items()},
			"grand_total": str(_round2(sum(total.values()))),
			"generated_at": _now(),
		}

	async def calculate_dso(self, period_from: str, period_to: str) -> float:
		"""Days Sales Outstanding = (Average AR / Revenue) × Days in period."""
		days = _days_between(period_from, period_to)
		assert days > 0, "period_to must be after period_from"

		all_invoices = await self._query("ar_invoices")
		period_invoices = [
			inv for inv in all_invoices
			if period_from <= inv.get("invoice_date", "") <= period_to
		]
		revenue = sum(_d(inv["total_amount"]) for inv in period_invoices)
		if revenue == 0:
			return 0.0

		# Average AR = mean of opening and closing balance
		open_ar = sum(
			_d(inv.get("outstanding_amount", 0))
			for inv in all_invoices
			if inv.get("status") not in ("cancelled", "void", "paid")
		)
		dso = float(open_ar / revenue * days)
		return round(dso, 2)

	async def run_dunning(
		self,
		as_of_date: str | None = None,
		dunning_group: str | None = None,
	) -> dict[str, Any]:
		"""Evaluate all overdue invoices and send dunning communications by level."""
		ref = as_of_date or _today()
		invoices = await self._query("ar_invoices")
		overdue = [
			inv for inv in invoices
			if inv.get("status") in ("posted", "partially_paid")
			and inv.get("due_date", "9999") < ref
		]

		actions_taken: list[dict[str, Any]] = []
		for inv in overdue:
			dpd = _days_between(inv["due_date"], ref)
			level_cfg = next(
				(l for l in reversed(_DUNNING_LEVELS) if dpd >= l["dpd"]),
				None,
			)
			if not level_cfg:
				continue
			if dunning_group and inv.get("dunning_group") != dunning_group:
				continue
			result = await self.send_dunning_notification(
				inv["customer_id"], level_cfg["level"]
			)
			result["invoice_id"] = inv["id"]
			result["dpd"] = dpd
			actions_taken.append(result)

		await self._log_audit("dunning_run", "batch", {
			"as_of_date": ref, "invoices_actioned": len(actions_taken),
		})
		return {
			"as_of_date": ref,
			"invoices_evaluated": len(overdue),
			"actions_taken": len(actions_taken),
			"detail": actions_taken,
		}

	async def generate_dunning_letter(
		self, customer_id: str, dunning_level: int
	) -> dict[str, Any]:
		"""Generate a dunning letter document for the given customer and level."""
		customer = await self._require("ar_customers", customer_id, "customer")
		level_cfg = next((l for l in _DUNNING_LEVELS if l["level"] == dunning_level), None)
		assert level_cfg, f"invalid dunning_level {dunning_level}"

		open_invoices = await self._query("ar_invoices", {"customer_id": customer_id})
		open_invoices = [
			inv for inv in open_invoices
			if inv.get("status") in ("posted", "partially_paid")
		]
		total_outstanding = sum(_d(inv.get("outstanding_amount", 0)) for inv in open_invoices)

		letter_id = uuid7str()
		letter_body = textwrap.dedent(f"""
			{level_cfg['label'].upper()}

			Date: {_today()}
			To:   {customer['name']}

			RE: OUTSTANDING BALANCE — {total_outstanding} {customer.get('currency', 'USD')}

			This is a {level_cfg['label']} regarding your outstanding balance.

			{'IMMEDIATE PAYMENT REQUIRED.' if dunning_level >= 3 else 'Please arrange payment at your earliest convenience.'}

			Outstanding invoices:
			""").lstrip()
		for inv in open_invoices:
			letter_body += f"  - {inv['id']}  Due: {inv.get('due_date')}  Amount: {inv.get('outstanding_amount')}\n"

		letter: dict[str, Any] = {
			"id": letter_id,
			"type": "dunning_letter",
			"customer_id": customer_id,
			"dunning_level": dunning_level,
			"level_label": level_cfg["label"],
			"body": letter_body,
			"total_outstanding": str(total_outstanding),
			"generated_at": _now(),
		}
		await self._put("ar_dunning_letters", letter)
		return deepcopy(letter)

	async def send_dunning_notification(
		self,
		customer_id: str,
		dunning_level: int,
		channel: str = "email",
	) -> dict[str, Any]:
		"""Send dunning notification to customer."""
		customer = await self._require("ar_customers", customer_id, "customer")
		letter = await self.generate_dunning_letter(customer_id, dunning_level)
		level_cfg = next(l for l in _DUNNING_LEVELS if l["level"] == dunning_level)
		recipient = customer.get("email") or customer["name"]
		await self._log_notify(
			recipient, channel,
			f"[{level_cfg['label']}] Outstanding Balance Due",
			letter["body"],
			{"dunning_level": dunning_level, "letter_id": letter["id"]},
		)
		await self._log_audit("dunning_sent", customer_id, {
			"level": dunning_level, "channel": channel, "letter_id": letter["id"],
		})
		return {
			"customer_id": customer_id,
			"dunning_level": dunning_level,
			"channel": channel,
			"letter_id": letter["id"],
			"sent_at": _now(),
		}

	async def get_collection_queue(
		self, filters: dict[str, Any] | None = None
	) -> list[dict[str, Any]]:
		"""Prioritised collection queue: risk score × outstanding amount."""
		invoices = await self._query("ar_invoices", filters)
		overdue = [
			inv for inv in invoices
			if inv.get("status") in ("posted", "partially_paid")
			and inv.get("due_date", "9999") < _today()
		]
		# Priority = outstanding_amount × (1 + dpd/30)  — higher = more urgent
		today = _today()
		for inv in overdue:
			dpd = _days_between(inv.get("due_date", today), today)
			inv["_priority"] = float(_d(inv.get("outstanding_amount", 0)) * _d(1 + dpd / 30))
			inv["dpd"] = dpd
		overdue.sort(key=lambda x: x["_priority"], reverse=True)
		return [deepcopy(r) for r in overdue]

	async def schedule_collection_call(
		self,
		customer_id: str,
		agent_id: str,
		scheduled_at: str,
	) -> dict[str, Any]:
		"""Schedule a collections call."""
		await self._require("ar_customers", customer_id, "customer")
		assert agent_id, "agent_id required"
		assert scheduled_at, "scheduled_at required"
		activity: dict[str, Any] = {
			"id": uuid7str(),
			"type": "collection_call_scheduled",
			"customer_id": customer_id,
			"agent_id": agent_id,
			"scheduled_at": scheduled_at,
			"status": "scheduled",
			"created_at": _now(),
		}
		await self._put("ar_collection_activities", activity)
		await self._log_audit("collection_call_scheduled", activity["id"], {
			"customer_id": customer_id, "agent_id": agent_id, "scheduled_at": scheduled_at,
		})
		return deepcopy(activity)

	async def record_collection_outcome(
		self,
		activity_id: str,
		outcome: str,
		notes: str,
		next_action: str,
	) -> dict[str, Any]:
		"""Record the outcome of a collection call/activity."""
		assert outcome, "outcome required"
		activity = await self._require("ar_collection_activities", activity_id, "activity")
		activity["outcome"] = outcome
		activity["notes"] = notes
		activity["next_action"] = next_action
		activity["completed_at"] = _now()
		activity["status"] = "completed"
		activity["updated_at"] = _now()
		await self._put("ar_collection_activities", activity)
		await self._log_audit("collection_outcome_recorded", activity_id, {
			"outcome": outcome, "next_action": next_action,
		})
		return deepcopy(activity)

	async def escalate_to_external_collector(
		self, customer_id: str, collector_name: str
	) -> dict[str, Any]:
		"""Escalate outstanding debt to an external collection agency."""
		assert collector_name, "collector_name required"
		customer = await self._require("ar_customers", customer_id, "customer")
		open_invoices = await self._query("ar_invoices", {"customer_id": customer_id})
		open_invoices = [
			inv for inv in open_invoices
			if inv.get("status") in ("posted", "partially_paid")
		]
		total = sum(_d(inv.get("outstanding_amount", 0)) for inv in open_invoices)

		escalation: dict[str, Any] = {
			"id": uuid7str(),
			"type": "external_collection_escalation",
			"customer_id": customer_id,
			"collector_name": collector_name,
			"invoice_ids": [inv["id"] for inv in open_invoices],
			"total_outstanding": str(total),
			"escalated_by": self.actor_id,
			"escalated_at": _now(),
			"status": "escalated",
		}
		await self._put("ar_escalations", escalation)

		# Update customer record
		customer["collection_status"] = "external_collector"
		customer["collector_name"] = collector_name
		customer["updated_at"] = _now()
		await self._put("ar_customers", customer)

		await self._log_audit("escalated_to_collector", customer_id, {
			"collector": collector_name, "total": str(total),
		})
		return deepcopy(escalation)

	async def customer_statement(
		self, customer_id: str, period_from: str, period_to: str
	) -> dict[str, Any]:
		"""Generate a customer account statement for the given period."""
		customer = await self._require("ar_customers", customer_id, "customer")
		invoices = await self._query("ar_invoices", {"customer_id": customer_id})
		payments = await self._query("ar_payments", {"customer_id": customer_id})

		period_invoices = [
			inv for inv in invoices
			if period_from <= inv.get("invoice_date", "") <= period_to
		]
		period_payments = [
			pay for pay in payments
			if period_from <= pay.get("payment_date", "") <= period_to
		]

		total_invoiced = sum(_d(inv["total_amount"]) for inv in period_invoices)
		total_paid = sum(_d(pay["amount"]) for pay in period_payments)
		closing_balance = sum(
			_d(inv.get("outstanding_amount", 0))
			for inv in invoices
			if inv.get("status") not in ("cancelled", "void", "paid")
		)

		return {
			"customer_id": customer_id,
			"customer_name": customer["name"],
			"period_from": period_from,
			"period_to": period_to,
			"invoices": [deepcopy(i) for i in period_invoices],
			"payments": [deepcopy(p) for p in period_payments],
			"total_invoiced": str(_round2(total_invoiced)),
			"total_paid": str(_round2(total_paid)),
			"closing_balance": str(_round2(closing_balance)),
			"generated_at": _now(),
		}

	# ─────────────────────────────────────────────────────────────────────────
	# ██  DISPUTES
	# ─────────────────────────────────────────────────────────────────────────

	_DISPUTE_TYPES = frozenset({
		"pricing", "quantity", "quality", "delivery",
		"duplicate", "already_paid", "other",
	})
	_RESOLUTION_TYPES = frozenset({
		"credit_note", "payment", "write_off",
		"no_action", "partial_credit",
	})

	async def open_dispute(
		self,
		invoice_id: str,
		dispute_type: str,
		dispute_amount: float | int | Decimal,
		description: str,
		owner: str,
	) -> dict[str, Any]:
		"""Open a dispute on an invoice."""
		assert dispute_type in self._DISPUTE_TYPES, \
			f"dispute_type must be one of {self._DISPUTE_TYPES}"
		assert _d(dispute_amount) > 0, "dispute_amount must be positive"
		assert description, "description required"
		assert owner, "owner required"

		invoice = await self._require("ar_invoices", invoice_id, "invoice")
		record: dict[str, Any] = {
			"id": uuid7str(),
			"type": "ar_dispute",
			"invoice_id": invoice_id,
			"customer_id": invoice["customer_id"],
			"dispute_type": dispute_type,
			"dispute_amount": str(_d(dispute_amount)),
			"description": description,
			"owner": owner,
			"status": "open",
			"investigation_notes": None,
			"resolution_type": None,
			"resolution_notes": None,
			"reviewed_by": None,
			"opened_at": _now(),
			"updated_at": _now(),
		}
		await self._put("ar_disputes", record)
		invoice["status"] = "disputed"
		invoice["updated_at"] = _now()
		await self._put("ar_invoices", invoice)
		await self._log_audit("dispute_opened", record["id"], {
			"invoice_id": invoice_id, "type": dispute_type,
		})
		return deepcopy(record)

	async def investigate_dispute(
		self, dispute_id: str, investigation_notes: str
	) -> dict[str, Any]:
		"""Record investigation notes on an open dispute."""
		assert investigation_notes, "investigation_notes required"
		record = await self._require("ar_disputes", dispute_id, "dispute")
		assert record["status"] == "open", f"dispute not open: {record['status']}"
		record["investigation_notes"] = investigation_notes
		record["status"] = "under_investigation"
		record["investigated_at"] = _now()
		record["updated_at"] = _now()
		await self._put("ar_disputes", record)
		await self._log_audit("dispute_investigated", dispute_id, {})
		return deepcopy(record)

	async def resolve_dispute(
		self,
		dispute_id: str,
		resolution_type: str,
		resolution_notes: str,
		reviewed_by: str,
	) -> dict[str, Any]:
		"""Resolve a dispute with specified resolution type."""
		assert resolution_type in self._RESOLUTION_TYPES, \
			f"resolution_type must be one of {self._RESOLUTION_TYPES}"
		assert resolution_notes, "resolution_notes required"
		assert reviewed_by, "reviewed_by required"

		record = await self._require("ar_disputes", dispute_id, "dispute")
		assert record["status"] in ("open", "under_investigation"), \
			f"cannot resolve dispute in status {record['status']}"

		record["resolution_type"] = resolution_type
		record["resolution_notes"] = resolution_notes
		record["reviewed_by"] = reviewed_by
		record["status"] = "resolved"
		record["resolved_at"] = _now()
		record["updated_at"] = _now()
		await self._put("ar_disputes", record)

		# Reopen invoice (unless write_off — that is handled separately)
		invoice = await self._store.get("ar_invoices", record["invoice_id"])
		if invoice and invoice.get("status") == "disputed":
			invoice["status"] = (
				"paid" if _d(invoice.get("outstanding_amount", 0)) == 0
				else "posted"
			)
			invoice["updated_at"] = _now()
			await self._put("ar_invoices", invoice)

		await self._log_audit("dispute_resolved", dispute_id, {
			"resolution_type": resolution_type, "reviewed_by": reviewed_by,
		})
		return deepcopy(record)

	async def reject_dispute(
		self,
		dispute_id: str,
		reason: str,
		rejected_by: str,
	) -> dict[str, Any]:
		"""Reject a dispute as invalid."""
		assert reason, "reason required"
		assert rejected_by, "rejected_by required"
		record = await self._require("ar_disputes", dispute_id, "dispute")
		assert record["status"] in ("open", "under_investigation")
		record["status"] = "rejected"
		record["rejection_reason"] = reason
		record["rejected_by"] = rejected_by
		record["rejected_at"] = _now()
		record["updated_at"] = _now()
		await self._put("ar_disputes", record)

		invoice = await self._store.get("ar_invoices", record["invoice_id"])
		if invoice and invoice.get("status") == "disputed":
			invoice["status"] = "posted" if _d(invoice.get("outstanding_amount", 0)) > 0 else "paid"
			invoice["updated_at"] = _now()
			await self._put("ar_invoices", invoice)

		await self._log_audit("dispute_rejected", dispute_id, {
			"reason": reason, "rejected_by": rejected_by,
		})
		return deepcopy(record)

	async def list_disputes(self, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
		rows = await self._query("ar_disputes", filters)
		return [deepcopy(r) for r in rows]

	# ─────────────────────────────────────────────────────────────────────────
	# ██  BAD DEBT & WRITE-OFFS
	# ─────────────────────────────────────────────────────────────────────────

	async def calculate_bad_debt_provision(
		self, method: str = "ecl"
	) -> dict[str, Any]:
		"""Calculate IFRS 9 ECL provision by aging bucket.

		Rates: 0–30d: 1%, 31–60d: 5%, 61–90d: 20%, 91–120d: 50%, 120+: 100%
		"""
		assert method in ("ecl", "flat"), f"unsupported method {method}"
		aging = await self.calculate_aging()
		today = aging["as_of_date"]
		totals = aging["totals"]

		buckets = {
			"current": (_d(totals.get("current", 0)), Decimal("0.01")),
			"1_30":    (_d(totals.get("1_30", 0)),    Decimal("0.01")),
			"31_60":   (_d(totals.get("31_60", 0)),   Decimal("0.05")),
			"61_90":   (_d(totals.get("61_90", 0)),   Decimal("0.20")),
			"91_120":  (_d(totals.get("91_120", 0)),  Decimal("0.50")),
			"120_plus":(_d(totals.get("120_plus", 0)),Decimal("1.00")),
		}
		provision_detail: list[dict[str, Any]] = []
		total_provision = Decimal(0)
		for bucket, (balance, rate) in buckets.items():
			provision = _round2(balance * rate)
			provision_detail.append({
				"bucket": bucket,
				"balance": str(balance),
				"ecl_rate": str(rate),
				"provision": str(provision),
			})
			total_provision += provision

		result: dict[str, Any] = {
			"method": method,
			"as_of_date": today,
			"buckets": provision_detail,
			"total_provision": str(_round2(total_provision)),
			"calculated_at": _now(),
		}
		await self._log_audit("bad_debt_provision_calculated", "batch", {
			"method": method, "total_provision": str(total_provision),
		})
		return result

	async def propose_write_off(
		self,
		invoice_ids: list[str],
		reason: str,
		proposed_by: str,
	) -> dict[str, Any]:
		"""Propose a write-off for a list of invoices — requires approval."""
		assert invoice_ids, "invoice_ids required"
		assert reason, "reason required"
		assert proposed_by, "proposed_by required"

		invoices_detail = []
		total_amount = Decimal(0)
		for inv_id in invoice_ids:
			inv = await self._require("ar_invoices", inv_id, "invoice")
			assert inv.get("status") not in ("paid", "cancelled"), \
				f"invoice {inv_id} is already {inv['status']}"
			invoices_detail.append({"id": inv_id, "outstanding": inv.get("outstanding_amount")})
			total_amount += _d(inv.get("outstanding_amount", 0))

		record: dict[str, Any] = {
			"id": uuid7str(),
			"type": "ar_write_off_proposal",
			"invoice_ids": invoice_ids,
			"invoices_detail": invoices_detail,
			"reason": reason,
			"proposed_by": proposed_by,
			"total_amount": str(_round2(total_amount)),
			"status": "pending_approval",
			"proposed_at": _now(),
			"updated_at": _now(),
		}
		await self._put("ar_write_offs", record)
		wf = await self._workflow.start_workflow(
			"write_off_approval",
			{"write_off_id": record["id"], "total": str(total_amount)},
		)
		record["workflow_instance_id"] = wf.get("instance_id")
		await self._put("ar_write_offs", record)
		await self._log_audit("write_off_proposed", record["id"], {
			"invoice_count": len(invoice_ids), "total": str(total_amount),
		})
		return deepcopy(record)

	async def approve_write_off(
		self,
		write_off_id: str,
		approved_by: str,
		gl_account: str,
	) -> dict[str, Any]:
		"""Approve and execute an invoice write-off."""
		assert approved_by, "approved_by required"
		assert gl_account, "gl_account required"
		record = await self._require("ar_write_offs", write_off_id, "write_off")
		assert record["status"] == "pending_approval", \
			f"write-off not pending: {record['status']}"

		# Mark invoices as written off
		for inv_id in record["invoice_ids"]:
			inv = await self._store.get("ar_invoices", inv_id)
			if inv and inv.get("tenant_id") == self.tenant_id:
				inv["status"] = "written_off"
				inv["written_off_at"] = _now()
				inv["write_off_id"] = write_off_id
				inv["updated_at"] = _now()
				await self._put("ar_invoices", inv)

		record["status"] = "approved"
		record["approved_by"] = approved_by
		record["gl_account"] = gl_account
		record["approved_at"] = _now()
		record["updated_at"] = _now()
		await self._put("ar_write_offs", record)

		gl_entry = await self.bad_debt_gl_entry(write_off_id)
		await self._log_audit("write_off_approved", write_off_id, {
			"approved_by": approved_by, "gl_account": gl_account,
		})
		return {"write_off": deepcopy(record), "gl_entry": gl_entry}

	async def reverse_write_off(
		self,
		write_off_id: str,
		reason: str,
		reversed_by: str,
	) -> dict[str, Any]:
		"""Reverse an approved write-off — reinstates invoices to outstanding."""
		assert reason, "reason required"
		assert reversed_by, "reversed_by required"
		record = await self._require("ar_write_offs", write_off_id, "write_off")
		assert record["status"] == "approved", f"write-off not approved: {record['status']}"

		for inv_id in record["invoice_ids"]:
			inv = await self._store.get("ar_invoices", inv_id)
			if inv and inv.get("tenant_id") == self.tenant_id:
				inv["status"] = "posted"
				inv["write_off_reversed_at"] = _now()
				inv["updated_at"] = _now()
				await self._put("ar_invoices", inv)

		record["status"] = "reversed"
		record["reversal_reason"] = reason
		record["reversed_by"] = reversed_by
		record["reversed_at"] = _now()
		record["updated_at"] = _now()
		await self._put("ar_write_offs", record)
		await self._log_audit("write_off_reversed", write_off_id, {
			"reason": reason, "reversed_by": reversed_by,
		})
		return deepcopy(record)

	# ─────────────────────────────────────────────────────────────────────────
	# ██  MULTI-CURRENCY & WHT
	# ─────────────────────────────────────────────────────────────────────────

	async def foreign_currency_revaluation(
		self,
		period: str,
		fx_rates: dict[str, float | Decimal],
	) -> dict[str, Any]:
		"""Unrealised FX gain/loss on open foreign-currency invoices.

		fx_rates: {currency_code: rate_to_functional_currency}
		"""
		assert fx_rates, "fx_rates required"
		functional_ccy = "USD"  # default; override via tenant config
		all_invoices = await self._query("ar_invoices")
		open_foreign = [
			inv for inv in all_invoices
			if inv.get("status") not in ("cancelled", "void", "paid")
			and inv.get("currency") != functional_ccy
			and inv.get("currency") in fx_rates
		]

		revaluation_lines: list[dict[str, Any]] = []
		total_gain_loss = Decimal(0)

		for inv in open_foreign:
			ccy = inv["currency"]
			current_rate = _d(fx_rates[ccy])
			original_rate = _d(inv.get("fx_rate_at_invoice", fx_rates[ccy]))
			outstanding_fc = _d(inv.get("outstanding_amount", 0))
			original_lc = _round2(outstanding_fc * original_rate)
			current_lc = _round2(outstanding_fc * current_rate)
			gain_loss = _round2(current_lc - original_lc)
			total_gain_loss += gain_loss

			revaluation_lines.append({
				"invoice_id": inv["id"],
				"currency": ccy,
				"outstanding_fc": str(outstanding_fc),
				"original_rate": str(original_rate),
				"current_rate": str(current_rate),
				"original_lc": str(original_lc),
				"current_lc": str(current_lc),
				"unrealised_gain_loss": str(gain_loss),
			})
			# Update invoice with current revalued LC amount
			inv["lc_outstanding"] = str(current_lc)
			inv["fx_revalued_at"] = _now()
			inv["updated_at"] = _now()
			await self._put("ar_invoices", inv)

		result: dict[str, Any] = {
			"period": period,
			"functional_currency": functional_ccy,
			"invoices_revalued": len(revaluation_lines),
			"total_unrealised_gain_loss": str(_round2(total_gain_loss)),
			"gain_loss_type": "gain" if total_gain_loss > 0 else "loss",
			"lines": revaluation_lines,
			"revalued_at": _now(),
			"gl_accounts": {
				"unrealised_gain": "7010",
				"unrealised_loss": "8010",
			},
		}
		await self._log_audit("fx_revaluation_run", "batch", {
			"period": period, "total_gain_loss": str(total_gain_loss),
		})
		return result

	async def record_fx_rate(
		self,
		from_currency: str,
		to_currency: str,
		rate: float | Decimal,
		rate_date: str,
	) -> dict[str, Any]:
		"""Record a spot FX rate."""
		assert from_currency and to_currency, "currencies required"
		assert _d(rate) > 0, "rate must be positive"
		record: dict[str, Any] = {
			"id": uuid7str(),
			"type": "ar_fx_rate",
			"from_currency": from_currency,
			"to_currency": to_currency,
			"rate": str(_d(rate)),
			"rate_date": rate_date,
			"recorded_by": self.actor_id,
			"created_at": _now(),
		}
		await self._put("ar_fx_rates", record)
		await self._log_audit("fx_rate_recorded", record["id"], {
			"pair": f"{from_currency}/{to_currency}", "rate": str(rate),
		})
		return deepcopy(record)

	async def apply_withholding_tax(
		self,
		payment_id: str,
		wht_amount: float | Decimal,
		wht_rate: float | Decimal,
		jurisdiction: str,
	) -> dict[str, Any]:
		"""Apply withholding tax to a payment receipt.

		Creates a WHT certificate and adjusts payment unapplied amount.
		"""
		assert _d(wht_amount) > 0, "wht_amount must be positive"
		assert jurisdiction, "jurisdiction required"
		payment = await self._require("ar_payments", payment_id, "payment")

		expected_rate = _WHT_RATES.get(jurisdiction.upper())
		rate = _d(wht_rate)
		if expected_rate and abs(rate - expected_rate) > Decimal("0.01"):
			raise ValueError(
				f"WHT rate {rate} deviates from statutory rate {expected_rate} for {jurisdiction}"
			)

		wht_rec: dict[str, Any] = {
			"id": uuid7str(),
			"type": "ar_wht_record",
			"payment_id": payment_id,
			"customer_id": payment["customer_id"],
			"wht_amount": str(_d(wht_amount)),
			"wht_rate": str(rate),
			"wht_jurisdiction": jurisdiction.upper(),
			"status": "recorded",
			"created_at": _now(),
		}
		await self._put("ar_wht_records", wht_rec)

		# Gross up the payment — WHT was deducted at source
		payment["gross_amount"] = str(_round2(_d(payment["amount"]) + _d(wht_amount)))
		payment["wht_amount"] = str(_d(wht_amount))
		payment["wht_jurisdiction"] = jurisdiction.upper()
		payment["updated_at"] = _now()
		await self._put("ar_payments", payment)

		await self._log_audit("wht_applied", payment_id, {
			"wht_amount": str(wht_amount), "jurisdiction": jurisdiction,
		})
		return deepcopy(wht_rec)

	async def withholding_tax_certificate(
		self, customer_id: str, period: str
	) -> dict[str, Any]:
		"""Generate WHT certificate for a customer for the given period (YYYY-MM)."""
		customer = await self._require("ar_customers", customer_id, "customer")
		wht_records = await self._query("ar_wht_records", {"customer_id": customer_id})
		period_records = [
			r for r in wht_records
			if r.get("created_at", "").startswith(period)
		]
		total_wht = sum(_d(r.get("wht_amount", 0)) for r in period_records)
		cert: dict[str, Any] = {
			"id": uuid7str(),
			"type": "wht_certificate",
			"customer_id": customer_id,
			"customer_name": customer["name"],
			"period": period,
			"wht_records": [r["id"] for r in period_records],
			"total_withheld": str(_round2(total_wht)),
			"issued_by": self.actor_id,
			"issued_at": _now(),
		}
		await self._put("ar_wht_certificates", cert)
		return deepcopy(cert)

	# ─────────────────────────────────────────────────────────────────────────
	# ██  REVENUE RECOGNITION
	# ─────────────────────────────────────────────────────────────────────────

	_RECOG_METHODS = frozenset({"point_in_time", "over_time", "milestone"})

	async def create_performance_obligation(
		self,
		invoice_id: str,
		description: str,
		transaction_price: float | Decimal,
		method: str,
	) -> dict[str, Any]:
		"""Create an IFRS 15 performance obligation linked to an invoice."""
		assert description, "description required"
		assert method in self._RECOG_METHODS, f"method must be one of {self._RECOG_METHODS}"
		assert _d(transaction_price) > 0, "transaction_price must be positive"
		await self._require("ar_invoices", invoice_id, "invoice")

		record: dict[str, Any] = {
			"id": uuid7str(),
			"type": "performance_obligation",
			"invoice_id": invoice_id,
			"description": description,
			"transaction_price": str(_d(transaction_price)),
			"recognised_amount": "0.00",
			"unrecognised_amount": str(_d(transaction_price)),
			"recognition_method": method,
			"status": "pending",
			"created_at": _now(),
			"updated_at": _now(),
		}
		await self._put("ar_performance_obligations", record)
		await self._log_audit("performance_obligation_created", record["id"], {
			"invoice_id": invoice_id, "method": method,
		})
		return deepcopy(record)

	async def recognise_revenue(
		self,
		obligation_id: str,
		recognised_amount: float | Decimal,
		recognition_date: str,
	) -> dict[str, Any]:
		"""Recognise revenue against a performance obligation."""
		assert _d(recognised_amount) > 0, "recognised_amount must be positive"
		obligation = await self._require(
			"ar_performance_obligations", obligation_id, "obligation"
		)
		unrecognised = _d(obligation["unrecognised_amount"])
		amount = _d(recognised_amount)
		assert amount <= unrecognised, \
			f"recognised {amount} exceeds unrecognised balance {unrecognised}"

		obligation["recognised_amount"] = str(
			_round2(_d(obligation["recognised_amount"]) + amount)
		)
		obligation["unrecognised_amount"] = str(_round2(unrecognised - amount))
		obligation["status"] = (
			"fully_recognised"
			if _round2(unrecognised - amount) == 0
			else "partially_recognised"
		)
		obligation["last_recognised_at"] = recognition_date
		obligation["updated_at"] = _now()
		await self._put("ar_performance_obligations", obligation)

		entry: dict[str, Any] = {
			"id": uuid7str(),
			"type": "revenue_recognition_entry",
			"obligation_id": obligation_id,
			"amount": str(amount),
			"recognition_date": recognition_date,
			"created_at": _now(),
		}
		await self._put("ar_revenue_entries", entry)
		await self._log_audit("revenue_recognised", obligation_id, {
			"amount": str(amount), "date": recognition_date,
		})
		return {"obligation": deepcopy(obligation), "entry": entry}

	async def revenue_recognition_report(
		self, period_from: str, period_to: str
	) -> dict[str, Any]:
		"""IFRS 15 revenue recognition report for the given period."""
		entries = await self._query("ar_revenue_entries")
		period_entries = [
			e for e in entries
			if period_from <= e.get("recognition_date", "") <= period_to
		]
		total_recognised = sum(_d(e.get("amount", 0)) for e in period_entries)

		obligations = await self._query("ar_performance_obligations")
		deferred = sum(
			_d(ob.get("unrecognised_amount", 0))
			for ob in obligations
			if ob.get("status") not in ("fully_recognised",)
		)
		return {
			"period_from": period_from,
			"period_to": period_to,
			"entries": [deepcopy(e) for e in period_entries],
			"total_recognised": str(_round2(total_recognised)),
			"total_deferred": str(_round2(deferred)),
			"generated_at": _now(),
		}

	# ─────────────────────────────────────────────────────────────────────────
	# ██  REPORTING
	# ─────────────────────────────────────────────────────────────────────────

	async def ar_aging_report(self, as_of_date: str | None = None) -> dict[str, Any]:
		"""Full AR aging report — delegates to calculate_aging and enriches with names."""
		aging = await self.calculate_aging(as_of_date)
		for row in aging["customers"]:
			cust = await self._store.get("ar_customers", row["customer_id"])
			row["customer_name"] = cust["name"] if cust else "—"
		return aging

	async def dso_trend(self, periods: int = 12) -> list[dict[str, Any]]:
		"""Compute DSO for each of the last N calendar months."""
		assert periods > 0, "periods must be positive"
		today = date.today()
		result = []
		for i in range(periods - 1, -1, -1):
			first_day = (today.replace(day=1) - timedelta(days=i * 30)).replace(day=1)
			last_day = (first_day.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
			p_from = first_day.isoformat()
			p_to = last_day.isoformat()
			dso = await self.calculate_dso(p_from, p_to)
			result.append({
				"period_from": p_from,
				"period_to": p_to,
				"dso_days": dso,
			})
		return result

	async def collection_performance_report(
		self, period_from: str, period_to: str
	) -> dict[str, Any]:
		"""Collections effectiveness: calls made, outcomes, promise-to-pay rate."""
		activities = await self._query("ar_collection_activities")
		period_acts = [
			a for a in activities
			if period_from <= a.get("created_at", "")[:10] <= period_to
		]
		completed = [a for a in period_acts if a.get("status") == "completed"]
		promises = [
			a for a in completed
			if "promise" in str(a.get("outcome", "")).lower()
		]
		return {
			"period_from": period_from,
			"period_to": period_to,
			"total_activities": len(period_acts),
			"completed": len(completed),
			"promises_to_pay": len(promises),
			"promise_rate": (
				f"{len(promises)/len(completed)*100:.1f}%"
				if completed else "0%"
			),
			"generated_at": _now(),
		}

	async def cash_collection_forecast(self, days: int = 90) -> dict[str, Any]:
		"""Predict cash collections over next N days using payment pattern analysis."""
		assert days > 0, "days must be positive"
		today = date.today()
		customers = await self._query("ar_customers")
		forecast_lines: list[dict[str, Any]] = []
		total_forecast = Decimal(0)

		for customer in customers:
			cust_id = customer["id"]
			all_pays = await self._query("ar_payments", {"customer_id": cust_id})
			paid_pays = [p for p in all_pays if p.get("status") in ("fully_applied", "partially_applied")]

			# Average days to pay from invoice due date
			avg_dtp = 5  # default
			if len(paid_pays) >= 3:
				all_inv = await self._query("ar_invoices", {"customer_id": cust_id})
				inv_by_id = {inv["id"]: inv for inv in all_inv}
				dtps = []
				for pay in paid_pays[-12:]:
					for alloc_id in pay.get("allocations", []):
						alloc = await self._store.get("ar_allocations", alloc_id)
						if alloc:
							inv = inv_by_id.get(alloc.get("invoice_id", ""))
							if inv and inv.get("due_date") and pay.get("payment_date"):
								dtps.append(_days_between(inv["due_date"], pay["payment_date"]))
				if dtps:
					avg_dtp = int(sum(dtps) / len(dtps))

			# Open invoices for this customer
			open_invs = await self._query("ar_invoices", {"customer_id": cust_id})
			for inv in open_invs:
				if inv.get("status") not in ("posted", "partially_paid"):
					continue
				predicted_date = (
					date.fromisoformat(inv["due_date"]) + timedelta(days=max(0, avg_dtp))
				)
				if (predicted_date - today).days <= days:
					amt = _d(inv.get("outstanding_amount", 0))
					forecast_lines.append({
						"customer_id": cust_id,
						"customer_name": customer["name"],
						"invoice_id": inv["id"],
						"due_date": inv["due_date"],
						"predicted_payment_date": predicted_date.isoformat(),
						"predicted_amount": str(amt),
						"avg_days_to_pay": avg_dtp,
					})
					total_forecast += amt

		forecast_lines.sort(key=lambda x: x["predicted_payment_date"])
		return {
			"forecast_days": days,
			"total_forecast": str(_round2(total_forecast)),
			"lines": forecast_lines,
			"generated_at": _now(),
		}

	async def ar_kpi_dashboard(self) -> dict[str, Any]:
		"""DSO, collection rate, dispute rate, write-off rate, aging totals."""
		today = _today()
		month_ago = (date.today() - timedelta(days=30)).isoformat()

		dso = await self.calculate_dso(month_ago, today)
		aging = await self.calculate_aging(today)
		grand_total = _d(aging["totals"].get("current", 0)) + \
		              _d(aging["totals"].get("1_30", 0)) + \
		              _d(aging["totals"].get("31_60", 0)) + \
		              _d(aging["totals"].get("61_90", 0)) + \
		              _d(aging["totals"].get("91_120", 0)) + \
		              _d(aging["totals"].get("120_plus", 0))

		# Collection rate: payments in last 30 days / invoiced last 30 days
		invoices_30 = [
			inv for inv in await self._query("ar_invoices")
			if inv.get("invoice_date", "") >= month_ago
		]
		payments_30 = [
			p for p in await self._query("ar_payments")
			if p.get("payment_date", "") >= month_ago
		]
		invoiced_30 = sum(_d(inv["total_amount"]) for inv in invoices_30)
		collected_30 = sum(_d(p["amount"]) for p in payments_30)
		collection_rate = (
			_round2(collected_30 / invoiced_30 * 100) if invoiced_30 > 0 else Decimal(0)
		)

		# Dispute rate
		all_inv = await self._query("ar_invoices")
		disputes = await self._query("ar_disputes")
		open_disputes = [d for d in disputes if d.get("status") in ("open", "under_investigation")]
		dispute_rate = (
			_round2(Decimal(len(open_disputes)) / Decimal(len(all_inv)) * 100)
			if all_inv else Decimal(0)
		)

		# Write-off rate
		write_offs = await self._query("ar_write_offs", {"status": "approved"})
		total_written_off = sum(_d(wo.get("total_amount", 0)) for wo in write_offs)
		write_off_rate = (
			_round2(total_written_off / invoiced_30 * 100) if invoiced_30 > 0 else Decimal(0)
		)

		return {
			"tenant_id": self.tenant_id,
			"as_of": today,
			"dso_days": dso,
			"collection_rate_pct": str(collection_rate),
			"dispute_rate_pct": str(dispute_rate),
			"write_off_rate_pct": str(write_off_rate),
			"open_ar_total": str(_round2(grand_total)),
			"aging_buckets": aging["totals"],
			"generated_at": _now(),
		}

	async def intercompany_ar_report(
		self, counterpart_entity: str
	) -> dict[str, Any]:
		"""AR balances owed by a specific intercompany counterpart."""
		assert counterpart_entity, "counterpart_entity required"
		customers = await self._query("ar_customers", {"customer_type": "intercompany"})
		ic_customers = [
			c for c in customers
			if c.get("entity_code") == counterpart_entity
			or c.get("name") == counterpart_entity
		]
		cust_ids = {c["id"] for c in ic_customers}
		invoices = await self._query("ar_invoices")
		ic_invoices = [
			inv for inv in invoices
			if inv.get("customer_id") in cust_ids
			and inv.get("status") not in ("cancelled", "void", "paid")
		]
		total_outstanding = sum(_d(inv.get("outstanding_amount", 0)) for inv in ic_invoices)
		return {
			"counterpart_entity": counterpart_entity,
			"customers": [deepcopy(c) for c in ic_customers],
			"open_invoices": [deepcopy(inv) for inv in ic_invoices],
			"total_outstanding": str(_round2(total_outstanding)),
			"as_of": _today(),
			"generated_at": _now(),
		}

	# ─────────────────────────────────────────────────────────────────────────
	# ██  ELECTRONIC INVOICING
	# ─────────────────────────────────────────────────────────────────────────

	async def generate_ubl_invoice(self, invoice_id: str) -> str:
		"""Generate a UBL 2.1 compliant XML invoice document."""
		invoice = await self._require("ar_invoices", invoice_id, "invoice")
		customer = await self._require("ar_customers", invoice["customer_id"], "customer")

		lines_xml = ""
		for line in invoice.get("lines", []):
			lines_xml += textwrap.dedent(f"""
			<cac:InvoiceLine>
				<cbc:ID>{line['line_no']}</cbc:ID>
				<cbc:InvoicedQuantity>{line['quantity']}</cbc:InvoicedQuantity>
				<cbc:LineExtensionAmount currencyID="{invoice['currency']}">{line['line_total']}</cbc:LineExtensionAmount>
				<cac:Item>
					<cbc:Description>{line['description']}</cbc:Description>
				</cac:Item>
				<cac:Price>
					<cbc:PriceAmount currencyID="{invoice['currency']}">{line['unit_price']}</cbc:PriceAmount>
				</cac:Price>
			</cac:InvoiceLine>""")

		xml = textwrap.dedent(f"""<?xml version="1.0" encoding="UTF-8"?>
		<Invoice xmlns="urn:oasis:names:specification:ubl:schema:xsd:Invoice-2"
		         xmlns:cac="urn:oasis:names:specification:ubl:schema:xsd:CommonAggregateComponents-2"
		         xmlns:cbc="urn:oasis:names:specification:ubl:schema:xsd:CommonBasicComponents-2">
			<cbc:UBLVersionID>2.1</cbc:UBLVersionID>
			<cbc:ID>{invoice_id}</cbc:ID>
			<cbc:IssueDate>{invoice.get('invoice_date', _today())}</cbc:IssueDate>
			<cbc:DueDate>{invoice.get('due_date', _today())}</cbc:DueDate>
			<cbc:DocumentCurrencyCode>{invoice['currency']}</cbc:DocumentCurrencyCode>
			<cac:AccountingCustomerParty>
				<cac:Party>
					<cac:PartyName><cbc:Name>{customer['name']}</cbc:Name></cac:PartyName>
				</cac:Party>
			</cac:AccountingCustomerParty>
			<cac:LegalMonetaryTotal>
				<cbc:LineExtensionAmount currencyID="{invoice['currency']}">{invoice.get('subtotal', '0')}</cbc:LineExtensionAmount>
				<cbc:TaxExclusiveAmount currencyID="{invoice['currency']}">{invoice.get('subtotal', '0')}</cbc:TaxExclusiveAmount>
				<cbc:TaxInclusiveAmount currencyID="{invoice['currency']}">{invoice['total_amount']}</cbc:TaxInclusiveAmount>
				<cbc:PayableAmount currencyID="{invoice['currency']}">{invoice.get('outstanding_amount', invoice['total_amount'])}</cbc:PayableAmount>
			</cac:LegalMonetaryTotal>
			{lines_xml}
		</Invoice>""").strip()

		await self._log_audit("ubl_invoice_generated", invoice_id, {})
		return xml

	async def send_invoice_email(
		self, invoice_id: str, recipient_email: str
	) -> dict[str, Any]:
		"""Send invoice to an explicit recipient email address."""
		assert recipient_email, "recipient_email required"
		invoice = await self._require("ar_invoices", invoice_id, "invoice")
		customer = await self._require("ar_customers", invoice["customer_id"], "customer")
		subject = (
			f"Invoice {invoice_id} — {invoice['total_amount']} {invoice['currency']} "
			f"due {invoice.get('due_date', '')}"
		)
		body = (
			f"Dear {customer['name']},\n\n"
			f"Please find your invoice details below:\n\n"
			f"  Invoice ID : {invoice_id}\n"
			f"  Amount     : {invoice['total_amount']} {invoice['currency']}\n"
			f"  Due Date   : {invoice.get('due_date', '—')}\n"
			f"  Terms      : {invoice.get('payment_terms', '—')}\n\n"
			f"UBL XML is available on request.\n"
		)
		await self._log_notify(
			recipient_email, "email", subject, body,
			{"invoice_id": invoice_id},
		)
		await self._log_audit("invoice_emailed", invoice_id, {"recipient": recipient_email})
		return {
			"invoice_id": invoice_id,
			"sent_to": recipient_email,
			"subject": subject,
			"sent_at": _now(),
		}

	# ─────────────────────────────────────────────────────────────────────────
	# ██  GL INTEGRATION
	# ─────────────────────────────────────────────────────────────────────────

	async def invoice_gl_entry(self, invoice_id: str) -> dict[str, Any]:
		"""Generate GL journal: AR Dr / Revenue Cr / VAT Payable Cr."""
		invoice = await self._require("ar_invoices", invoice_id, "invoice")
		subtotal = _d(invoice.get("subtotal", invoice["total_amount"]))
		tax = _d(invoice.get("tax_total", 0))
		total = _d(invoice["total_amount"])

		entry: dict[str, Any] = {
			"id": uuid7str(),
			"type": "gl_journal_entry",
			"source": "ar_invoice",
			"source_id": invoice_id,
			"date": invoice.get("invoice_date", _today()),
			"description": f"AR Invoice {invoice_id}",
			"lines": [
				{
					"account": "1200",
					"account_name": "Accounts Receivable",
					"debit": str(total),
					"credit": "0.00",
					"currency": invoice["currency"],
				},
				{
					"account": "4000",
					"account_name": "Revenue",
					"debit": "0.00",
					"credit": str(subtotal),
					"currency": invoice["currency"],
				},
			],
			"status": "posted",
			"created_at": _now(),
		}
		if tax > 0:
			entry["lines"].append({
				"account": "2200",
				"account_name": "VAT Payable",
				"debit": "0.00",
				"credit": str(tax),
				"currency": invoice["currency"],
			})
		await self._put("ar_gl_entries", entry)
		return deepcopy(entry)

	async def payment_gl_entry(self, payment_id: str) -> dict[str, Any]:
		"""Generate GL journal: Cash Dr / AR Cr."""
		payment = await self._require("ar_payments", payment_id, "payment")
		amount = _d(payment["amount"])
		entry: dict[str, Any] = {
			"id": uuid7str(),
			"type": "gl_journal_entry",
			"source": "ar_payment",
			"source_id": payment_id,
			"date": payment.get("payment_date", _today()),
			"description": f"AR Payment {payment_id} — Ref {payment.get('reference','')}",
			"lines": [
				{
					"account": "1000",
					"account_name": "Cash / Bank",
					"debit": str(amount),
					"credit": "0.00",
					"currency": payment["currency"],
				},
				{
					"account": "1200",
					"account_name": "Accounts Receivable",
					"debit": "0.00",
					"credit": str(amount),
					"currency": payment["currency"],
				},
			],
			"status": "posted",
			"created_at": _now(),
		}
		await self._put("ar_gl_entries", entry)
		return deepcopy(entry)

	async def bad_debt_gl_entry(self, write_off_id: str) -> dict[str, Any]:
		"""Generate GL journal: Bad Debt Expense Dr / AR Cr."""
		write_off = await self._require("ar_write_offs", write_off_id, "write_off")
		amount = _d(write_off["total_amount"])
		gl_account = write_off.get("gl_account", "8500")
		entry: dict[str, Any] = {
			"id": uuid7str(),
			"type": "gl_journal_entry",
			"source": "ar_write_off",
			"source_id": write_off_id,
			"date": _today(),
			"description": f"Bad Debt Write-off {write_off_id}",
			"lines": [
				{
					"account": gl_account,
					"account_name": "Bad Debt Expense",
					"debit": str(amount),
					"credit": "0.00",
					"currency": "USD",
				},
				{
					"account": "1200",
					"account_name": "Accounts Receivable",
					"debit": "0.00",
					"credit": str(amount),
					"currency": "USD",
				},
			],
			"status": "posted",
			"created_at": _now(),
		}
		await self._put("ar_gl_entries", entry)
		return deepcopy(entry)

	# ─────────────────────────────────────────────────────────────────────────
	# ██  RECURRING BILLING
	# ─────────────────────────────────────────────────────────────────────────

	async def create_recurring_schedule(
		self,
		customer_id: str,
		template_lines: list[dict[str, Any]],
		frequency: str,
		start_date: str,
		end_date: str | None = None,
		payment_terms: str = "NET30",
	) -> dict[str, Any]:
		"""Create a recurring invoice schedule.

		frequency: daily | weekly | monthly | quarterly | annually
		"""
		_VALID_FREQUENCIES = {"daily", "weekly", "monthly", "quarterly", "annually"}
		assert frequency in _VALID_FREQUENCIES, f"frequency must be one of {_VALID_FREQUENCIES}"
		assert template_lines, "template_lines required"
		assert start_date, "start_date required"
		await self._require("ar_customers", customer_id, "customer")

		record: dict[str, Any] = {
			"id": uuid7str(),
			"type": "ar_recurring_schedule",
			"customer_id": customer_id,
			"template_lines": template_lines,
			"frequency": frequency,
			"start_date": start_date,
			"end_date": end_date,
			"payment_terms": payment_terms,
			"next_run_date": start_date,
			"run_count": 0,
			"status": "active",
			"created_by": self.actor_id,
			"created_at": _now(),
			"updated_at": _now(),
		}
		await self._put("ar_recurring_schedules", record)
		await self._log_audit("recurring_schedule_created", record["id"], {
			"customer_id": customer_id, "frequency": frequency, "start_date": start_date,
		})
		return deepcopy(record)

	async def run_recurring_invoicing(self, as_of_date: str | None = None) -> dict[str, Any]:
		"""Generate invoices for all due recurring schedules as of the given date."""
		_FREQ_DAYS: dict[str, int] = {
			"daily": 1, "weekly": 7, "monthly": 30, "quarterly": 91, "annually": 365,
		}
		ref = as_of_date or _today()
		schedules = await self._query("ar_recurring_schedules", {"status": "active"})
		due = [s for s in schedules if s.get("next_run_date", "9999") <= ref]

		created_invoices: list[dict[str, Any]] = []
		for sched in due:
			terms = sched.get("payment_terms", "NET30")
			net_days = int("".join(filter(str.isdigit, terms)) or "30")
			inv_date = sched["next_run_date"]
			due_date_obj = date.fromisoformat(inv_date) + timedelta(days=net_days)
			customer = await self._require("ar_customers", sched["customer_id"], "customer")
			inv = await self.create_invoice(
				customer_id=sched["customer_id"],
				invoice_date=inv_date,
				due_date=due_date_obj.isoformat(),
				lines=sched["template_lines"],
				currency=customer.get("currency", "USD"),
				payment_terms=sched.get("payment_terms", "NET30"),
				recurring_schedule_id=sched["id"],
			)
			await self.submit_invoice(inv["id"])
			created_invoices.append(inv)

			step_days = _FREQ_DAYS.get(sched["frequency"], 30)
			next_date = (date.fromisoformat(inv_date) + timedelta(days=step_days)).isoformat()
			sched["next_run_date"] = next_date
			sched["run_count"] = sched.get("run_count", 0) + 1
			if sched.get("end_date") and next_date > sched["end_date"]:
				sched["status"] = "completed"
			sched["updated_at"] = _now()
			await self._put("ar_recurring_schedules", sched)

		await self._log_audit("recurring_invoicing_run", "batch", {
			"as_of_date": ref, "schedules_processed": len(due), "invoices_created": len(created_invoices),
		})
		return {
			"as_of_date": ref,
			"schedules_processed": len(due),
			"invoices_created": len(created_invoices),
			"invoice_ids": [inv["id"] for inv in created_invoices],
		}

	# ─────────────────────────────────────────────────────────────────────────
	# ██  DYNAMIC EARLY-PAYMENT DISCOUNTS
	# ─────────────────────────────────────────────────────────────────────────

	async def calculate_dynamic_discount(
		self,
		invoice_id: str,
		cost_of_capital_pct: float | Decimal = Decimal("10.0"),
	) -> dict[str, Any]:
		"""Sliding-scale early-payment discount tiers for an invoice.

		Break-even daily rate = cost_of_capital / 365.
		Tiers offered at 2, 5, 10, 15 days before original due date.
		"""
		invoice = await self._require("ar_invoices", invoice_id, "invoice")
		assert invoice["status"] not in ("paid", "cancelled", "void"), \
			f"invoice is already {invoice['status']}"

		outstanding = _d(invoice["outstanding_amount"])
		due = date.fromisoformat(invoice["due_date"])
		today_date = date.today()
		days_to_due = (due - today_date).days
		assert days_to_due > 0, "invoice is past due — discount not applicable"

		annual_rate = _d(cost_of_capital_pct) / 100
		daily_rate = annual_rate / 365

		tiers: list[dict[str, Any]] = []
		for pay_in_days in (2, 5, 10, 15):
			if pay_in_days >= days_to_due:
				continue
			days_saved = days_to_due - pay_in_days
			discount_pct = _round2(daily_rate * days_saved * 100)
			discount_amount = _round2(outstanding * daily_rate * days_saved)
			discounted_amount = _round2(outstanding - discount_amount)
			tiers.append({
				"pay_by": (today_date + timedelta(days=pay_in_days)).isoformat(),
				"days_accelerated": days_saved,
				"discount_pct": str(discount_pct),
				"discount_amount": str(discount_amount),
				"amount_payable": str(discounted_amount),
			})

		offer: dict[str, Any] = {
			"id": uuid7str(),
			"type": "ar_discount_offer",
			"invoice_id": invoice_id,
			"customer_id": invoice["customer_id"],
			"original_amount": str(outstanding),
			"original_due_date": invoice["due_date"],
			"cost_of_capital_pct": str(cost_of_capital_pct),
			"tiers": tiers,
			"status": "open",
			"generated_at": _now(),
		}
		await self._put("ar_discount_offers", offer)
		await self._log_audit("discount_offer_generated", invoice_id, {
			"tiers": len(tiers), "cost_of_capital_pct": str(cost_of_capital_pct),
		})
		return deepcopy(offer)

	async def accept_early_payment_discount(
		self,
		offer_id: str,
		tier_index: int,
	) -> dict[str, Any]:
		"""Accept a specific discount tier; adjusts outstanding invoice amount."""
		offer = await self._require("ar_discount_offers", offer_id, "offer")
		assert offer["status"] == "open", f"offer is {offer['status']}"
		tiers = offer.get("tiers", [])
		assert 0 <= tier_index < len(tiers), f"tier_index {tier_index} out of range"

		tier = tiers[tier_index]
		invoice = await self._require("ar_invoices", offer["invoice_id"], "invoice")
		invoice["outstanding_amount"] = tier["amount_payable"]
		invoice["discount_applied_pct"] = tier["discount_pct"]
		invoice["discount_applied_amount"] = tier["discount_amount"]
		invoice["discount_due_by"] = tier["pay_by"]
		invoice["updated_at"] = _now()
		await self._put("ar_invoices", invoice)

		offer["status"] = "accepted"
		offer["accepted_tier"] = tier_index
		offer["accepted_at"] = _now()
		offer["updated_at"] = _now()
		await self._put("ar_discount_offers", offer)

		await self._log_audit("discount_accepted", offer_id, {
			"invoice_id": offer["invoice_id"], "tier": tier_index, "discount_pct": tier["discount_pct"],
		})
		return {"offer": deepcopy(offer), "invoice": deepcopy(invoice)}

	# ─────────────────────────────────────────────────────────────────────────
	# ██  INSTALMENT PLANS
	# ─────────────────────────────────────────────────────────────────────────

	async def create_instalment_plan(
		self,
		invoice_id: str,
		num_instalments: int,
		frequency: str,
		first_due_date: str,
	) -> dict[str, Any]:
		"""Split an outstanding invoice into a structured instalment plan.

		frequency: weekly | monthly | quarterly
		"""
		_FREQ_DAYS: dict[str, int] = {"weekly": 7, "monthly": 30, "quarterly": 91}
		assert frequency in _FREQ_DAYS, f"frequency must be one of {set(_FREQ_DAYS)}"
		assert num_instalments >= 2, "num_instalments must be >= 2"

		invoice = await self._require("ar_invoices", invoice_id, "invoice")
		assert invoice["status"] not in ("paid", "cancelled", "void"), \
			f"invoice is {invoice['status']}"

		outstanding = _d(invoice["outstanding_amount"])
		instalment_amount = _round2(outstanding / num_instalments)
		last_amount = _round2(outstanding - instalment_amount * (num_instalments - 1))

		step = _FREQ_DAYS[frequency]
		plan_id = uuid7str()
		instalments: list[dict[str, Any]] = []
		base = date.fromisoformat(first_due_date)
		for i in range(num_instalments):
			due = (base + timedelta(days=step * i)).isoformat()
			amt = last_amount if i == num_instalments - 1 else instalment_amount
			inst: dict[str, Any] = {
				"id": uuid7str(),
				"type": "ar_instalment",
				"plan_id": plan_id,
				"invoice_id": invoice_id,
				"customer_id": invoice["customer_id"],
				"instalment_no": i + 1,
				"amount": str(amt),
				"due_date": due,
				"status": "pending",
				"created_at": _now(),
			}
			await self._put("ar_instalments", inst)
			instalments.append(inst)

		plan: dict[str, Any] = {
			"id": plan_id,
			"type": "ar_instalment_plan",
			"invoice_id": invoice_id,
			"customer_id": invoice["customer_id"],
			"total_amount": str(outstanding),
			"num_instalments": num_instalments,
			"frequency": frequency,
			"first_due_date": first_due_date,
			"instalment_ids": [inst["id"] for inst in instalments],
			"status": "active",
			"created_by": self.actor_id,
			"created_at": _now(),
			"updated_at": _now(),
		}
		await self._put("ar_instalment_plans", plan)

		invoice["instalment_plan_id"] = plan_id
		invoice["status"] = "partially_paid"
		invoice["updated_at"] = _now()
		await self._put("ar_invoices", invoice)

		await self._log_audit("instalment_plan_created", plan_id, {
			"invoice_id": invoice_id, "num_instalments": num_instalments, "frequency": frequency,
		})
		return {"plan": deepcopy(plan), "instalments": [deepcopy(i) for i in instalments]}

	async def process_instalment_payment(
		self,
		instalment_id: str,
		payment_id: str,
	) -> dict[str, Any]:
		"""Mark a specific instalment as paid from the given payment record."""
		instalment = await self._require("ar_instalments", instalment_id, "instalment")
		assert instalment["status"] == "pending", f"instalment is {instalment['status']}"

		result = await self.apply_payment(
			payment_id,
			[{"invoice_id": instalment["invoice_id"], "amount": instalment["amount"]}],
		)

		instalment["status"] = "paid"
		instalment["payment_id"] = payment_id
		instalment["paid_at"] = _now()
		instalment["updated_at"] = _now()
		await self._put("ar_instalments", instalment)

		plan = await self._store.get("ar_instalment_plans", instalment["plan_id"])
		if plan and plan.get("tenant_id") == self.tenant_id:
			all_insts = await self._query("ar_instalments", {"plan_id": plan["id"]})
			if all(inst.get("status") == "paid" for inst in all_insts):
				plan["status"] = "completed"
				plan["completed_at"] = _now()
				plan["updated_at"] = _now()
				await self._put("ar_instalment_plans", plan)

		await self._log_audit("instalment_paid", instalment_id, {
			"payment_id": payment_id, "plan_id": instalment["plan_id"],
		})
		return {"instalment": deepcopy(instalment), "payment_result": result}

	# ─────────────────────────────────────────────────────────────────────────
	# ██  PERIOD-CLOSE CHECKLIST
	# ─────────────────────────────────────────────────────────────────────────

	async def run_period_close_checklist(self, period: str) -> dict[str, Any]:
		"""Execute the AR period-close checklist for the given period (YYYY-MM).

		Steps:
		  1. Validate no unposted approved invoices remain
		  2. Foreign-currency revaluation (last recorded rates)
		  3. Bad-debt ECL provision calculation
		  4. AR sub-ledger to GL reconciliation
		  5. Aging report snapshot
		  6. Dunning letter archive count
		"""
		assert period and len(period) == 7, "period must be YYYY-MM"

		period_from = f"{period}-01"
		year, month = int(period[:4]), int(period[5:7])
		next_month_first = date(year + (month // 12), (month % 12) + 1, 1)
		period_to = (next_month_first - timedelta(days=1)).isoformat()

		checklist_id = uuid7str()
		steps: list[dict[str, Any]] = []

		def _step(name: str, status: str, detail: Any) -> dict[str, Any]:
			return {"step": name, "status": status, "detail": detail, "checked_at": _now()}

		unposted = await self._query("ar_invoices", {"status": "approved"})
		period_unposted = [inv for inv in unposted if inv.get("invoice_date", "").startswith(period)]
		steps.append(_step(
			"no_unposted_invoices",
			"pass" if not period_unposted else "fail",
			{"unposted_count": len(period_unposted), "invoice_ids": [i["id"] for i in period_unposted]},
		))

		fx_rates_raw = await self._query("ar_fx_rates")
		latest_rates: dict[str, Decimal] = {}
		for r in sorted(fx_rates_raw, key=lambda x: x.get("rate_date", "")):
			if r.get("to_currency") == "USD":
				latest_rates[r["from_currency"]] = _d(r["rate"])
		if latest_rates:
			reval = await self.foreign_currency_revaluation(period, latest_rates)
			steps.append(_step(
				"fx_revaluation", "pass",
				{"invoices_revalued": reval["invoices_revalued"], "gain_loss": reval["total_unrealised_gain_loss"]},
			))
		else:
			steps.append(_step("fx_revaluation", "skipped", {"reason": "no_fx_rates_recorded"}))

		provision = await self.calculate_bad_debt_provision()
		steps.append(_step("bad_debt_provision", "pass", {"total_provision": provision["total_provision"]}))

		all_open_inv = await self._query("ar_invoices")
		ar_balance = sum(
			_d(inv.get("outstanding_amount", 0))
			for inv in all_open_inv
			if inv.get("status") not in ("cancelled", "void", "paid", "written_off")
		)
		gl_entries = await self._query("ar_gl_entries")
		gl_ar_balance = Decimal(0)
		for entry in gl_entries:
			for line in entry.get("lines", []):
				if line.get("account") == "1200":
					gl_ar_balance += _d(line.get("debit", 0)) - _d(line.get("credit", 0))
		reconciled = abs(ar_balance - gl_ar_balance) < Decimal("0.01")
		steps.append(_step(
			"ar_gl_reconciliation", "pass" if reconciled else "warn",
			{
				"ar_subledger_balance": str(_round2(ar_balance)),
				"gl_ar_balance": str(_round2(gl_ar_balance)),
				"difference": str(_round2(abs(ar_balance - gl_ar_balance))),
			},
		))

		aging = await self.calculate_aging(period_to)
		steps.append(_step(
			"aging_snapshot", "pass",
			{"grand_total": aging["grand_total"], "as_of_date": aging["as_of_date"]},
		))

		dunning_letters = await self._query("ar_dunning_letters")
		period_letters = [dl for dl in dunning_letters if dl.get("generated_at", "").startswith(period)]
		steps.append(_step("dunning_letter_archive", "pass", {"letters_archived": len(period_letters)}))

		overall_status = "pass" if all(s["status"] in ("pass", "skipped") for s in steps) else "fail"
		checklist: dict[str, Any] = {
			"id": checklist_id,
			"type": "ar_period_close",
			"period": period,
			"period_from": period_from,
			"period_to": period_to,
			"steps": steps,
			"overall_status": overall_status,
			"executed_by": self.actor_id,
			"executed_at": _now(),
		}
		await self._put("ar_period_close_checklists", checklist)
		await self._log_audit("period_close_checklist_run", checklist_id, {
			"period": period, "overall_status": overall_status,
		})
		return deepcopy(checklist)

	# ─────────────────────────────────────────────────────────────────────────
	# ██  CUSTOMER CHURN RISK SCORING
	# ─────────────────────────────────────────────────────────────────────────

	async def calculate_churn_risk_score(self, customer_id: str) -> dict[str, Any]:
		"""Compute a 0.0–1.0 churn risk score from AR payment behaviour signals.

		Weights: 35% payment delay trend, 25% dispute frequency,
		         20% credit-hold history, 20% outstanding/credit-limit ratio.
		"""
		customer = await self._require("ar_customers", customer_id, "customer")
		all_invoices = await self._query("ar_invoices", {"customer_id": customer_id})
		all_disputes = await self._query("ar_disputes", {"customer_id": customer_id})

		paid_invoices = [
			inv for inv in all_invoices
			if inv.get("status") == "paid" and inv.get("paid_at") and inv.get("due_date")
		]
		paid_invoices.sort(key=lambda x: x.get("paid_at", ""))
		dtps: list[float] = [
			_days_between(inv["due_date"], inv["paid_at"][:10])
			for inv in paid_invoices[-12:]
		]

		delay_slope = 0.0
		if len(dtps) >= 2:
			n = len(dtps)
			xs = list(range(n))
			x_mean = sum(xs) / n
			y_mean = sum(dtps) / n
			num = sum((xs[i] - x_mean) * (dtps[i] - y_mean) for i in range(n))
			den = sum((xs[i] - x_mean) ** 2 for i in range(n))
			delay_slope = num / den if den != 0 else 0.0

		total_invoices = max(len(all_invoices), 1)
		dispute_freq = len(all_disputes) / total_invoices
		credit_hold_flag = 1.0 if customer.get("credit_hold") else 0.0
		credit_limit = float(_d(customer.get("credit_limit", 1) or 1))
		outstanding = sum(
			float(_d(inv.get("outstanding_amount", 0)))
			for inv in all_invoices
			if inv.get("status") not in ("cancelled", "void", "paid")
		)
		outstanding_ratio = min(outstanding / credit_limit, 1.0) if credit_limit > 0 else 1.0

		delay_score = min(max(delay_slope / 30, 0.0), 1.0)
		raw = (
			0.35 * delay_score +
			0.25 * min(dispute_freq, 1.0) +
			0.20 * credit_hold_flag +
			0.20 * outstanding_ratio
		)
		score = round(min(max(raw, 0.0), 1.0), 4)

		churn_record: dict[str, Any] = {
			"id": uuid7str(),
			"type": "ar_churn_score",
			"customer_id": customer_id,
			"churn_risk_score": score,
			"risk_band": "high" if score >= 0.7 else ("medium" if score >= 0.4 else "low"),
			"signals": {
				"payment_delay_slope_days": round(delay_slope, 3),
				"dispute_frequency": round(dispute_freq, 4),
				"credit_hold": bool(credit_hold_flag),
				"outstanding_ratio": round(outstanding_ratio, 4),
			},
			"scored_at": _now(),
		}
		await self._put("ar_churn_scores", churn_record)
		await self._log_audit("churn_risk_scored", customer_id, {
			"score": score, "risk_band": churn_record["risk_band"],
		})
		return deepcopy(churn_record)

	async def run_churn_scoring(self) -> dict[str, Any]:
		"""Score all active customers for churn risk; internally alert on score >= 0.7."""
		customers = await self._query("ar_customers", {"status": "active"})
		results: list[dict[str, Any]] = []
		elevated: list[str] = []

		for customer in customers:
			score_rec = await self.calculate_churn_risk_score(customer["id"])
			results.append(score_rec)
			if score_rec["churn_risk_score"] >= 0.7:
				elevated.append(customer["id"])
				await self._log_notify(
					self.actor_id, "internal",
					f"Churn Risk Elevated — {customer['name']}",
					f"Customer {customer['name']} churn score {score_rec['churn_risk_score']:.2f}. "
					f"Signals: {score_rec['signals']}",
					{"customer_id": customer["id"], "score": score_rec["churn_risk_score"]},
				)

		await self._log_audit("churn_scoring_batch_run", "batch", {
			"customers_scored": len(results), "elevated_count": len(elevated),
		})
		return {
			"customers_scored": len(results),
			"elevated_risk_customers": elevated,
			"scores": results,
			"run_at": _now(),
		}

	# ─────────────────────────────────────────────────────────────────────────
	# ██  IFRS 9 SCENARIO-BASED ECL
	# ─────────────────────────────────────────────────────────────────────────

	async def calculate_ecl_scenarios(
		self,
		scenarios: dict[str, dict[str, float | Decimal]] | None = None,
	) -> dict[str, Any]:
		"""IFRS 9 multi-scenario ECL: base / adverse / severe with probability weighting.

		Default scenarios are conservative. Caller may supply custom per-bucket loss rates.
		Probability weights: base 60%, adverse 30%, severe 10%.
		"""
		_DEFAULT_SCENARIOS: dict[str, dict[str, Decimal]] = {
			"base": {
				"current": Decimal("0.005"), "1_30": Decimal("0.010"),
				"31_60": Decimal("0.050"),  "61_90": Decimal("0.150"),
				"91_120": Decimal("0.400"), "120_plus": Decimal("0.850"),
			},
			"adverse": {
				"current": Decimal("0.015"), "1_30": Decimal("0.030"),
				"31_60": Decimal("0.100"),  "61_90": Decimal("0.300"),
				"91_120": Decimal("0.650"), "120_plus": Decimal("1.000"),
			},
			"severe": {
				"current": Decimal("0.030"), "1_30": Decimal("0.060"),
				"31_60": Decimal("0.200"),  "61_90": Decimal("0.500"),
				"91_120": Decimal("0.800"), "120_plus": Decimal("1.000"),
			},
		}
		scenario_rates = scenarios or _DEFAULT_SCENARIOS
		aging = await self.calculate_aging()
		totals = aging["totals"]

		scenario_results: dict[str, Any] = {}
		for scenario_name, rates in scenario_rates.items():
			provision_total = Decimal(0)
			bucket_detail: list[dict[str, Any]] = []
			for bucket in ("current", "1_30", "31_60", "61_90", "91_120", "120_plus"):
				balance = _d(totals.get(bucket, 0))
				rate = _d(rates.get(bucket, 0))
				provision = _round2(balance * rate)
				provision_total += provision
				bucket_detail.append({
					"bucket": bucket, "balance": str(balance),
					"loss_rate": str(rate), "provision": str(provision),
				})
			scenario_results[scenario_name] = {
				"buckets": bucket_detail,
				"total_provision": str(_round2(provision_total)),
			}

		_DEFAULT_WEIGHTS = {"base": Decimal("0.60"), "adverse": Decimal("0.30"), "severe": Decimal("0.10")}
		pwecl = sum(
			_d(scenario_results[scen]["total_provision"]) * weight
			for scen, weight in _DEFAULT_WEIGHTS.items()
			if scen in scenario_results
		)

		result: dict[str, Any] = {
			"as_of_date": aging["as_of_date"],
			"scenarios": scenario_results,
			"probability_weighted_ecl": str(_round2(pwecl)),
			"weights_used": {k: str(v) for k, v in _DEFAULT_WEIGHTS.items()},
			"calculated_at": _now(),
		}
		await self._log_audit("ecl_scenarios_calculated", "batch", {
			"scenario_count": len(scenario_results), "pwecl": str(_round2(pwecl)),
		})
		return result

	# ─────────────────────────────────────────────────────────────────────────
	# ██  CUSTOMER PORTAL TOKENS
	# ─────────────────────────────────────────────────────────────────────────

	async def generate_customer_portal_token(
		self,
		customer_id: str,
		expires_in_hours: int = 72,
	) -> dict[str, Any]:
		"""Issue a scoped access token for the customer self-service portal.

		Returns the raw token once — the hash is stored. Caller delivers token to customer.
		"""
		import hashlib
		import secrets
		assert expires_in_hours > 0, "expires_in_hours must be positive"
		await self._require("ar_customers", customer_id, "customer")

		raw_token = secrets.token_hex(32)
		token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
		expires_at = (
			datetime.utcnow() + timedelta(hours=expires_in_hours)
		).isoformat(timespec="seconds") + "Z"

		record: dict[str, Any] = {
			"id": uuid7str(),
			"type": "ar_portal_token",
			"customer_id": customer_id,
			"token_hash": token_hash,
			"expires_at": expires_at,
			"status": "active",
			"created_at": _now(),
		}
		await self._put("ar_portal_tokens", record)
		await self._log_audit("portal_token_issued", customer_id, {"expires_at": expires_at})
		return {
			"token": raw_token,
			"token_id": record["id"],
			"customer_id": customer_id,
			"expires_at": expires_at,
		}

	async def portal_get_open_invoices(self, token: str, customer_id: str) -> list[dict[str, Any]]:
		"""Return open invoices for a customer after validating their portal token."""
		import hashlib
		token_hash = hashlib.sha256(token.encode()).hexdigest()
		tokens = await self._query("ar_portal_tokens", {"customer_id": customer_id, "status": "active"})
		valid = next((t for t in tokens if t.get("token_hash") == token_hash), None)
		if not valid:
			raise PermissionError("invalid or expired portal token")
		if valid["expires_at"] < _now():
			raise PermissionError("portal token has expired")

		invoices = await self._query("ar_invoices", {"customer_id": customer_id})
		open_invoices = [
			inv for inv in invoices
			if inv.get("status") not in ("cancelled", "void", "paid", "draft")
		]
		_STRIP = {"tenant_id", "gl_entry_id", "workflow_instance_id"}
		return [{k: v for k, v in deepcopy(inv).items() if k not in _STRIP} for inv in open_invoices]

	# ─────────────────────────────────────────────────────────────────────────
	# ██  BANK STATEMENT INGESTION
	# ─────────────────────────────────────────────────────────────────────────

	async def ingest_bank_statement(
		self,
		statement_lines: list[dict[str, Any]],
		statement_date: str,
		bank_account: str,
	) -> dict[str, Any]:
		"""Parse and auto-reconcile a bank statement against open AR.

		Each line: {amount, currency, reference, customer_id (optional), value_date}.
		Lines with identifiable customers are auto-matched via smart_match_payment.
		Unmatched lines are included in the result for manual review.
		"""
		assert statement_lines, "statement_lines required"
		assert statement_date, "statement_date required"
		assert bank_account, "bank_account required"

		matched: list[dict[str, Any]] = []
		unmatched: list[dict[str, Any]] = []

		for line in statement_lines:
			amount = _d(line.get("amount", 0))
			if amount <= 0:
				unmatched.append({**line, "reason": "non_positive_amount"})
				continue

			customer_id = line.get("customer_id")
			if not customer_id:
				all_customers = await self._query("ar_customers")
				matched_cust = next(
					(c for c in all_customers if line.get("reference", "") in c.get("name", "")),
					None,
				)
				customer_id = matched_cust["id"] if matched_cust else None

			if not customer_id:
				unmatched.append({**line, "reason": "customer_not_identified"})
				continue

			payment = await self.record_payment(
				customer_id=customer_id,
				amount=amount,
				currency=line.get("currency", "USD"),
				payment_date=line.get("value_date", statement_date),
				payment_method="bank_transfer",
				reference=line.get("reference", "BANK-STMT"),
			)
			match_result = await self.smart_match_payment(payment["id"])
			matched.append({
				"statement_line": line,
				"payment_id": payment["id"],
				"match_result": match_result,
			})

		statement_record: dict[str, Any] = {
			"id": uuid7str(),
			"type": "ar_bank_statement",
			"bank_account": bank_account,
			"statement_date": statement_date,
			"lines_total": len(statement_lines),
			"matched_count": len(matched),
			"unmatched_count": len(unmatched),
			"matched": matched,
			"unmatched": unmatched,
			"reconciled_at": _now(),
		}
		await self._put("ar_bank_statements", statement_record)
		await self._log_audit("bank_statement_ingested", statement_record["id"], {
			"bank_account": bank_account, "matched": len(matched), "unmatched": len(unmatched),
		})
		return deepcopy(statement_record)



# ─────────────────────────────────────────────────────────────
# Backwards-compat alias
# ─────────────────────────────────────────────────────────────

	async def ml_collection_risk(self, *args, **kwargs):
		"""AI-powered accounts receivable collection risk and DSO prediction. Requires OLLAMA_BASE_URL."""
		import os
		if not os.environ.get("OLLAMA_BASE_URL"):
			return {"ml_enhanced": False}
		try:
			from capabilities.common.mlx import MLCapability
			ml = MLCapability()
			result = await ml.score(kwargs, task="accounts_receivable_collection_risk")
			return {"collection_risk": round(result.score,3), "risk_factors": result.factors, "ml_enhanced": True}
		except Exception:
			return {"ml_enhanced": False}

ARCService = AccountsReceivableService
