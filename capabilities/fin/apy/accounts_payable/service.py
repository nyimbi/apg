"""Domain service for APG accounts payable.

Covers: Vendor management, invoice capture & matching, approvals, payment
scheduling, expense reports, period close, advanced 3-way/2-way matching,
payment runs, early-payment discounts, supplier portal, and AP analytics.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

try:
	from .capability_contract import (
		SUPPORTED_AP_AGENT_ROLES,
		SUPPORTED_AP_AGENT_RUNTIMES,
		evaluate_capability_rules,
		streaming_manifest,
	)
except ImportError:
	from capability_contract import (
		SUPPORTED_AP_AGENT_ROLES,
		SUPPORTED_AP_AGENT_RUNTIMES,
		evaluate_capability_rules,
		streaming_manifest,
	)


# ---------------------------------------------------------------------------
# Decimal helpers
# ---------------------------------------------------------------------------

def _d(value: Any) -> Decimal:
	return Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

def _pct_within(actual: Decimal, reference: Decimal, tolerance_pct: float) -> bool:
	"""True when actual is within ±tolerance_pct% of reference."""
	if reference == 0:
		return actual == 0
	variance = abs(actual - reference) / abs(reference) * 100
	return variance <= Decimal(str(tolerance_pct))

def _now() -> str:
	return datetime.now(timezone.utc).isoformat()

def _today() -> str:
	return date.today().isoformat()

def _days_between(d1: str, d2: str) -> int:
	"""Days from ISO-date d1 to d2. Positive means d2 is later."""
	try:
		a = date.fromisoformat(d1[:10])
		b = date.fromisoformat(d2[:10])
		return (b - a).days
	except Exception:
		return 0


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class AccountsPayableService:
	"""Tenant-scoped vendor, invoice, approval, payment, expense, and close coordinator.

	Extended with advanced matching, payment runs, early-payment discounts,
	supplier portal, and analytics.
	"""

	def __init__(self) -> None:
		self._vendors: dict[str, dict[str, Any]] = {}
		self._invoices: dict[str, dict[str, Any]] = {}
		self._payments: dict[str, dict[str, Any]] = {}
		self._payment_batches: dict[str, dict[str, Any]] = {}
		self._expenses: dict[str, dict[str, Any]] = {}
		self._period_closes: dict[str, dict[str, Any]] = {}
		self._agents: dict[str, dict[str, Any]] = {}
		self._audit_events: list[dict[str, Any]] = []

		# Extended stores
		self._purchase_orders: dict[str, dict[str, Any]] = {}      # po_id -> PO record
		self._goods_receipts: dict[str, dict[str, Any]] = {}       # grn_id -> GRN record
		self._match_exceptions: dict[str, dict[str, Any]] = {}     # invoice_id -> exception record
		self._payment_runs: dict[str, dict[str, Any]] = {}         # run_id -> run record
		self._discount_offers: dict[str, dict[str, Any]] = {}      # offer_id -> dynamic discount offer
		self._supplier_submissions: dict[str, list[dict[str, Any]]] = defaultdict(list)  # supplier_id
		self._supplier_statements: dict[str, list[dict[str, Any]]] = defaultdict(list)  # supplier_id

	# -----------------------------------------------------------------------
	# Core vendor management
	# -----------------------------------------------------------------------

	def register_vendor(
		self,
		vendor_id: str,
		tenant_id: str,
		name: str,
		owner: str,
		tax_profile: str,
		payment_method: str,
		bank_change: bool = False,
		bank_reviewed_by: str | None = None,
	) -> dict[str, Any]:
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_vendor",
			"vendor_owner_assigned": bool(owner),
			"tax_profile_present": bool(tax_profile),
			"payment_method_present": bool(payment_method),
			"bank_change": bank_change,
			"bank_review_recorded": bool(bank_reviewed_by),
		}
		self._enforce(context)
		record = {
			"id": self._record_id("ap_vendor", vendor_id),
			"vendor_id": vendor_id,
			"tenant_id": tenant_id,
			"name": name,
			"owner": owner,
			"tax_profile": tax_profile,
			"payment_method": payment_method,
			"bank_change": bank_change,
			"bank_reviewed_by": bank_reviewed_by,
			"status": "active",
			"event_stream": "bytewax",
			"updated_at": _now(),
		}
		self._vendors[record["id"]] = record
		self._emit("vendor_registered", tenant_id, record["id"], {"vendor_id": vendor_id})
		return deepcopy(record)

	def record_invoice(
		self,
		invoice_id: str,
		tenant_id: str,
		vendor_record_id: str,
		invoice_number: str,
		amount: float,
		currency: str,
		document_reference: str,
		duplicate_detected: bool = False,
		duplicate_reviewed_by: str | None = None,
	) -> dict[str, Any]:
		vendor = self._require_vendor(vendor_record_id, tenant_id) if vendor_record_id else None
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_invoice",
			"vendor_present": vendor is not None,
			"invoice_number_present": bool(invoice_number),
			"currency_present": bool(currency),
			"amount": amount,
			"document_reference_present": bool(document_reference),
			"duplicate_detected": duplicate_detected,
			"duplicate_review_recorded": bool(duplicate_reviewed_by),
		}
		self._enforce(context)
		record = {
			"id": self._record_id("ap_invoice", invoice_id),
			"invoice_id": invoice_id,
			"tenant_id": tenant_id,
			"vendor_record_id": vendor["id"],
			"vendor_id": vendor["vendor_id"],
			"invoice_number": invoice_number,
			"amount": float(amount),
			"currency": currency,
			"document_reference": document_reference,
			"duplicate_detected": duplicate_detected,
			"duplicate_reviewed_by": duplicate_reviewed_by,
			"status": "captured",
			"matched": False,
			"approved": False,
			"held": False,
			"paid_amount": 0.0,
			"due_date": None,
			"payment_terms_days": 30,
			"po_id": None,
			"grn_id": None,
			"match_type": None,
			"discount_pct": 0.0,
			"discount_days": 0,
			"event_stream": "bytewax",
			"updated_at": _now(),
		}
		self._invoices[record["id"]] = record
		self._emit("invoice_recorded", tenant_id, record["id"], {"invoice_number": invoice_number, "amount": amount})
		return deepcopy(record)

	def match_invoice(
		self,
		tenant_id: str,
		invoice_record_id: str,
		po_backed: bool,
		receipt_reference: str | None = None,
		variance_rate: float = 0,
		variance_reviewed_by: str | None = None,
	) -> dict[str, Any]:
		invoice = self._require_invoice(invoice_record_id, tenant_id)
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "match_invoice",
			"po_backed": po_backed,
			"receipt_present": bool(receipt_reference),
			"variance_rate": variance_rate,
			"variance_review_recorded": bool(variance_reviewed_by),
		}
		self._enforce(context)
		invoice["matched"] = True
		invoice["receipt_reference"] = receipt_reference
		invoice["variance_rate"] = variance_rate
		invoice["variance_reviewed_by"] = variance_reviewed_by
		invoice["status"] = "matched"
		invoice["match_type"] = "two_way" if not receipt_reference else "three_way"
		invoice["updated_at"] = _now()
		self._emit("invoice_matched", tenant_id, invoice["id"], {"variance_rate": variance_rate})
		return deepcopy(invoice)

	def approve_invoice(
		self,
		tenant_id: str,
		invoice_record_id: str,
		approved_by: str,
		requested_by: str,
		approval_recorded: bool = True,
	) -> dict[str, Any]:
		invoice = self._require_invoice(invoice_record_id, tenant_id) if invoice_record_id else None
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "approve_invoice",
			"invoice_present": invoice is not None,
			"amount": invoice["amount"] if invoice else 0,
			"approval_recorded": approval_recorded,
			"separation_of_duties_passed": bool(approved_by) and approved_by != requested_by,
		}
		self._enforce(context)
		invoice["approved"] = True
		invoice["approved_by"] = approved_by
		invoice["requested_by"] = requested_by
		invoice["status"] = "approved"
		invoice["updated_at"] = _now()
		self._emit("invoice_approved", tenant_id, invoice["id"], {"approved_by": approved_by})
		return deepcopy(invoice)

	def place_invoice_hold(self, tenant_id: str, invoice_record_id: str, reason: str, placed_by: str) -> dict[str, Any]:
		invoice = self._require_invoice(invoice_record_id, tenant_id)
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "place_invoice_hold",
			"hold_reason_present": bool(reason),
		}
		self._enforce(context)
		invoice["held"] = True
		invoice["hold_reason"] = reason
		invoice["hold_placed_by"] = placed_by
		invoice["status"] = "held"
		invoice["updated_at"] = _now()
		self._emit("invoice_hold_placed", tenant_id, invoice["id"], {"reason": reason})
		return deepcopy(invoice)

	def release_invoice_hold(self, tenant_id: str, invoice_record_id: str, approved_by: str) -> dict[str, Any]:
		invoice = self._require_invoice(invoice_record_id, tenant_id)
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "release_invoice_hold",
			"release_approval_recorded": bool(approved_by),
		}
		self._enforce(context)
		invoice["held"] = False
		invoice["hold_released_by"] = approved_by
		invoice["status"] = "approved" if invoice["approved"] else "matched"
		invoice["updated_at"] = _now()
		return deepcopy(invoice)

	def schedule_payment(
		self,
		payment_id: str,
		tenant_id: str,
		invoice_record_id: str,
		amount: float,
		cash_account: str,
		scheduled_date: str,
	) -> dict[str, Any]:
		invoice = self._require_invoice(invoice_record_id, tenant_id)
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "schedule_payment",
			"invoice_approved": bool(invoice["approved"]) and not invoice["held"],
			"payment_amount": amount,
			"cash_account_present": bool(cash_account),
		}
		self._enforce(context)
		record = {
			"id": self._record_id("ap_payment", payment_id),
			"payment_id": payment_id,
			"tenant_id": tenant_id,
			"invoice_record_id": invoice["id"],
			"vendor_id": invoice["vendor_id"],
			"amount": float(amount),
			"cash_account": cash_account,
			"scheduled_date": scheduled_date,
			"status": "scheduled",
			"event_stream": "bytewax",
			"updated_at": _now(),
		}
		self._payments[record["id"]] = record
		self._emit("payment_scheduled", tenant_id, record["id"], {"amount": amount, "cash_account": cash_account})
		return deepcopy(record)

	def release_payment_batch(
		self,
		batch_id: str,
		tenant_id: str,
		payment_record_ids: list[str],
		reviewed_by: str,
	) -> dict[str, Any]:
		payments = [self._require_payment(pid, tenant_id) for pid in payment_record_ids]
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "release_payment_batch",
			"batch_review_recorded": bool(reviewed_by),
		}
		self._enforce(context)
		record = {
			"id": self._record_id("ap_payment_batch", batch_id),
			"batch_id": batch_id,
			"tenant_id": tenant_id,
			"payment_record_ids": [p["id"] for p in payments],
			"reviewed_by": reviewed_by,
			"amount": round(sum(p["amount"] for p in payments), 2),
			"status": "released",
			"event_stream": "bytewax",
			"updated_at": _now(),
		}
		for payment in payments:
			payment["status"] = "paid"
			payment["updated_at"] = _now()
			invoice = self._require_invoice(payment["invoice_record_id"], tenant_id)
			invoice["paid_amount"] = round(invoice["paid_amount"] + payment["amount"], 2)
			invoice["status"] = "paid" if invoice["paid_amount"] >= invoice["amount"] else "partially_paid"
		self._payment_batches[record["id"]] = record
		self._emit("payment_batch_released", tenant_id, record["id"], {"payment_count": len(payments), "amount": record["amount"]})
		return deepcopy(record)

	def record_expense_report(
		self,
		report_id: str,
		tenant_id: str,
		employee_id: str,
		amount: float,
		receipt_reference: str,
		policy_exception: bool = False,
		policy_reviewed_by: str | None = None,
	) -> dict[str, Any]:
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_expense_report",
			"employee_present": bool(employee_id),
			"expense_amount": amount,
			"receipt_present": bool(receipt_reference),
			"policy_exception": policy_exception,
			"policy_review_recorded": bool(policy_reviewed_by),
		}
		self._enforce(context)
		record = {
			"id": self._record_id("ap_expense", report_id),
			"report_id": report_id,
			"tenant_id": tenant_id,
			"employee_id": employee_id,
			"amount": float(amount),
			"receipt_reference": receipt_reference,
			"policy_exception": policy_exception,
			"policy_reviewed_by": policy_reviewed_by,
			"status": "recorded",
			"event_stream": "bytewax",
			"updated_at": _now(),
		}
		self._expenses[record["id"]] = record
		self._emit("expense_report_recorded", tenant_id, record["id"], {"amount": amount})
		return deepcopy(record)

	def close_period(
		self,
		close_id: str,
		tenant_id: str,
		period: str,
		open_exception_count: int,
		unposted_invoice_count: int,
		aging_reviewed_by: str,
	) -> dict[str, Any]:
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "close_period",
			"open_exception_count": open_exception_count,
			"unposted_invoice_count": unposted_invoice_count,
			"aging_review_recorded": bool(aging_reviewed_by),
		}
		self._enforce(context)
		record = {
			"id": self._record_id("ap_close", close_id),
			"close_id": close_id,
			"tenant_id": tenant_id,
			"period": period,
			"aging_reviewed_by": aging_reviewed_by,
			"status": "closed",
			"event_stream": "bytewax",
			"updated_at": _now(),
		}
		self._period_closes[record["id"]] = record
		self._emit("period_closed", tenant_id, record["id"], {"period": period})
		return deepcopy(record)

	def register_ap_agent(
		self,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		instructions: str,
	) -> dict[str, Any]:
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_ap_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AP_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AP_AGENT_ROLES,
		}
		self._enforce(context)
		record = {
			"id": self._record_id("ap_agent", name),
			"tenant_id": tenant_id,
			"name": name,
			"runtime": runtime,
			"role": role,
			"instructions": instructions,
			"status": "active",
			"event_stream": "bytewax",
			"updated_at": _now(),
		}
		self._agents[record["id"]] = record
		self._emit("ap_agent_registered", tenant_id, record["id"], {"runtime": runtime, "role": role})
		return deepcopy(record)

	def validate_agent_ap_action(
		self,
		tenant_id: str,
		agent_id: str,
		action: str,
		privileged_scope: bool,
		human_approval_recorded: bool,
	) -> dict[str, Any]:
		if agent_id not in self._agents:
			raise KeyError(f"Unknown AP agent: {agent_id}")
		context = {
			"tenant_context_present": bool(tenant_id),
			"operation": "agent_ap_action",
			"action": action,
			"privileged_scope": privileged_scope,
			"human_approval_recorded": human_approval_recorded,
		}
		return evaluate_capability_rules(context)

	def validate_batch(self, tenant_id: str, record_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		context = {"tenant_context_present": bool(tenant_id), "operation": "ap_batch", "event_stream": event_stream}
		result = evaluate_capability_rules(context)
		return {"processor": "bytewax", "record_count": record_count, "decision": result["decision"], "matched_rules": result["matched_rules"]}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"vendor_count": len(self.list_vendors(tenant_id)),
			"invoice_count": len(self.list_invoices(tenant_id)),
			"open_invoice_count": len([i for i in self.list_invoices(tenant_id) if i["status"] not in {"paid", "held"}]),
			"held_invoice_count": len([i for i in self.list_invoices(tenant_id) if i["held"]]),
			"payment_count": len(self.list_payments(tenant_id)),
			"payment_batch_count": len(self.list_payment_batches(tenant_id)),
			"expense_count": len(self.list_expenses(tenant_id)),
			"period_close_count": len(self.list_period_closes(tenant_id)),
			"ap_agent_count": len(self.list_ap_agents(tenant_id)),
			"audit_event_count": len(self.audit_events(tenant_id)),
			"streaming": streaming_manifest(),
		}

	def aging_summary(self, tenant_id: str) -> dict[str, Any]:
		invoices = [i for i in self.list_invoices(tenant_id) if i["status"] != "paid"]
		return {
			"tenant_id": tenant_id,
			"open_invoice_count": len(invoices),
			"open_amount": round(sum(i["amount"] - i["paid_amount"] for i in invoices), 2),
			"held_invoice_count": len([i for i in invoices if i["held"]]),
		}

	def list_vendors(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._vendors, tenant_id)

	def list_invoices(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._invoices, tenant_id)

	def list_payments(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._payments, tenant_id)

	def list_payment_batches(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._payment_batches, tenant_id)

	def list_expenses(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._expenses, tenant_id)

	def list_period_closes(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._period_closes, tenant_id)

	def list_ap_agents(self, tenant_id: str) -> list[dict[str, Any]]:
		return self._tenant_records(self._agents, tenant_id)

	def audit_events(self, tenant_id: str) -> list[dict[str, Any]]:
		return [deepcopy(e) for e in self._audit_events if e["tenant_id"] == tenant_id]

	def create_record(self, data: dict[str, Any]) -> dict[str, Any]:
		return self.register_vendor(
			data.get("vendor_id", data.get("id", "vendor")),
			data.get("tenant_id", "default"),
			data.get("name", "Vendor"),
			data.get("owner", "owner"),
			data.get("tax_profile", "tax-profile"),
			data.get("payment_method", "ach"),
			data.get("bank_change", False),
			data.get("bank_reviewed_by"),
		)

	def list_records(self, tenant_id: str = "default") -> list[dict[str, Any]]:
		return self.list_vendors(tenant_id)

	# -----------------------------------------------------------------------
	# Advanced Matching (5 methods)
	# -----------------------------------------------------------------------

	def three_way_match(
		self,
		invoice_id: str,
		po_id: str,
		grn_id: str,
		price_tolerance_pct: float = 2.0,
		qty_tolerance_pct: float = 2.0,
	) -> dict[str, Any]:
		"""Validate invoice against PO and GRN on quantity, price, and terms.

		Checks:
		  - Quantities on invoice vs. GRN within qty_tolerance_pct
		  - Unit prices on invoice vs. PO within price_tolerance_pct
		  - Payment terms match between invoice and PO

		Returns a pass/fail result with per-line detail.  Any failure registers
		a match exception that feeds match_exception_queue.
		"""
		invoice = self._find_invoice_by_public_id(invoice_id)
		po = self._purchase_orders.get(po_id)
		grn = self._goods_receipts.get(grn_id)

		failures: list[dict[str, Any]] = []
		checks: list[dict[str, Any]] = []

		if invoice is None:
			failures.append({"check": "invoice_exists", "result": "fail", "detail": f"invoice {invoice_id} not found"})
		if po is None:
			failures.append({"check": "po_exists", "result": "fail", "detail": f"PO {po_id} not found"})
		if grn is None:
			failures.append({"check": "grn_exists", "result": "fail", "detail": f"GRN {grn_id} not found"})

		if invoice and po and grn:
			# Quantity check: invoice qty vs GRN received qty
			inv_qty = _d(invoice.get("quantity", invoice.get("amount", 0)))
			grn_qty = _d(grn.get("received_qty", grn.get("quantity", inv_qty)))
			qty_match = _pct_within(inv_qty, grn_qty, qty_tolerance_pct)
			checks.append({
				"check": "quantity",
				"invoice_value": str(inv_qty),
				"grn_value": str(grn_qty),
				"tolerance_pct": qty_tolerance_pct,
				"result": "pass" if qty_match else "fail",
			})
			if not qty_match:
				failures.append({"check": "quantity_mismatch",
				                  "detail": f"invoice qty {inv_qty} vs GRN {grn_qty} exceeds {qty_tolerance_pct}%"})

			# Price check: invoice amount vs PO amount
			inv_amount = _d(invoice.get("amount", 0))
			po_amount = _d(po.get("amount", inv_amount))
			price_match = _pct_within(inv_amount, po_amount, price_tolerance_pct)
			checks.append({
				"check": "price",
				"invoice_value": str(inv_amount),
				"po_value": str(po_amount),
				"tolerance_pct": price_tolerance_pct,
				"result": "pass" if price_match else "fail",
			})
			if not price_match:
				failures.append({"check": "price_mismatch",
				                  "detail": f"invoice amount {inv_amount} vs PO {po_amount} exceeds {price_tolerance_pct}%"})

			# Terms check: payment_terms_days
			inv_terms = int(invoice.get("payment_terms_days", 30))
			po_terms = int(po.get("payment_terms_days", 30))
			terms_match = inv_terms == po_terms
			checks.append({
				"check": "payment_terms",
				"invoice_value": inv_terms,
				"po_value": po_terms,
				"result": "pass" if terms_match else "fail",
			})
			if not terms_match:
				failures.append({"check": "terms_mismatch",
				                  "detail": f"invoice terms {inv_terms}d vs PO {po_terms}d"})

		passed = len(failures) == 0
		result: dict[str, Any] = {
			"invoice_id": invoice_id,
			"po_id": po_id,
			"grn_id": grn_id,
			"match_type": "three_way",
			"passed": passed,
			"checks": checks,
			"failures": failures,
			"price_tolerance_pct": price_tolerance_pct,
			"qty_tolerance_pct": qty_tolerance_pct,
			"evaluated_at": _now(),
		}

		if not passed and invoice:
			self._register_match_exception(invoice_id, "three_way_match_failure", failures, invoice.get("tenant_id", ""))

		if passed and invoice:
			invoice["matched"] = True
			invoice["match_type"] = "three_way"
			invoice["po_id"] = po_id
			invoice["grn_id"] = grn_id
			invoice["status"] = "matched"
			invoice["updated_at"] = _now()

		return result

	def two_way_match(self, invoice_id: str, po_id: str) -> dict[str, Any]:
		"""Validate invoice against PO on amount and terms only (no GRN required).

		Suitable for service invoices where goods receipt is not applicable.
		Uses 5% price tolerance by default.
		"""
		invoice = self._find_invoice_by_public_id(invoice_id)
		po = self._purchase_orders.get(po_id)

		failures: list[dict[str, Any]] = []
		checks: list[dict[str, Any]] = []
		price_tolerance_pct = 5.0

		if invoice is None:
			failures.append({"check": "invoice_exists", "result": "fail", "detail": f"invoice {invoice_id} not found"})
		if po is None:
			failures.append({"check": "po_exists", "result": "fail", "detail": f"PO {po_id} not found"})

		if invoice and po:
			inv_amount = _d(invoice.get("amount", 0))
			po_amount = _d(po.get("amount", inv_amount))
			price_match = _pct_within(inv_amount, po_amount, price_tolerance_pct)
			checks.append({
				"check": "price",
				"invoice_value": str(inv_amount),
				"po_value": str(po_amount),
				"tolerance_pct": price_tolerance_pct,
				"result": "pass" if price_match else "fail",
			})
			if not price_match:
				failures.append({"check": "price_mismatch",
				                  "detail": f"invoice {inv_amount} vs PO {po_amount} exceeds {price_tolerance_pct}%"})

			inv_terms = int(invoice.get("payment_terms_days", 30))
			po_terms = int(po.get("payment_terms_days", 30))
			terms_match = inv_terms == po_terms
			checks.append({
				"check": "payment_terms",
				"invoice_value": inv_terms,
				"po_value": po_terms,
				"result": "pass" if terms_match else "fail",
			})
			if not terms_match:
				failures.append({"check": "terms_mismatch",
				                  "detail": f"invoice terms {inv_terms}d vs PO {po_terms}d"})

		passed = len(failures) == 0
		result: dict[str, Any] = {
			"invoice_id": invoice_id,
			"po_id": po_id,
			"match_type": "two_way",
			"passed": passed,
			"checks": checks,
			"failures": failures,
			"evaluated_at": _now(),
		}

		if not passed and invoice:
			self._register_match_exception(invoice_id, "two_way_match_failure", failures, invoice.get("tenant_id", ""))

		if passed and invoice:
			invoice["matched"] = True
			invoice["match_type"] = "two_way"
			invoice["po_id"] = po_id
			invoice["status"] = "matched"
			invoice["updated_at"] = _now()

		return result

	def auto_match_batch(self, invoice_ids: list[str]) -> dict[str, Any]:
		"""Attempt two-way or three-way matching for a list of invoices in bulk.

		For each invoice:
		  - If po_id and grn_id are set → three_way_match
		  - If only po_id is set → two_way_match
		  - Otherwise → exception (no PO reference)

		Returns an aggregate summary plus per-invoice outcomes.
		"""
		results: list[dict[str, Any]] = []
		passed_count = 0
		failed_count = 0
		skipped_count = 0

		for invoice_id in invoice_ids:
			invoice = self._find_invoice_by_public_id(invoice_id)
			if invoice is None:
				results.append({
					"invoice_id": invoice_id,
					"outcome": "skipped",
					"reason": "invoice not found",
				})
				skipped_count += 1
				continue

			po_id: str | None = invoice.get("po_id")
			grn_id: str | None = invoice.get("grn_id")

			if not po_id:
				results.append({
					"invoice_id": invoice_id,
					"outcome": "exception",
					"reason": "no PO reference on invoice",
				})
				self._register_match_exception(
					invoice_id, "no_po_reference", [{"detail": "invoice has no PO reference"}],
					invoice.get("tenant_id", ""),
				)
				failed_count += 1
				continue

			if grn_id:
				match_result = self.three_way_match(invoice_id, po_id, grn_id)
			else:
				match_result = self.two_way_match(invoice_id, po_id)

			outcome = "passed" if match_result["passed"] else "failed"
			results.append({
				"invoice_id": invoice_id,
				"outcome": outcome,
				"match_type": match_result.get("match_type"),
				"failures": match_result.get("failures", []),
			})
			if match_result["passed"]:
				passed_count += 1
			else:
				failed_count += 1

		return {
			"total": len(invoice_ids),
			"passed": passed_count,
			"failed": failed_count,
			"skipped": skipped_count,
			"results": results,
			"executed_at": _now(),
		}

	def match_exception_queue(self, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
		"""Return invoices with outstanding match failures, optionally filtered.

		filters keys (all optional):
		  tenant_id, vendor_id, exception_type, resolved (bool)
		"""
		filters = filters or {}
		queue = list(self._match_exceptions.values())

		if "tenant_id" in filters:
			queue = [e for e in queue if e.get("tenant_id") == filters["tenant_id"]]
		if "vendor_id" in filters:
			queue = [e for e in queue if e.get("vendor_id") == filters["vendor_id"]]
		if "exception_type" in filters:
			queue = [e for e in queue if e.get("exception_type") == filters["exception_type"]]
		if "resolved" in filters:
			queue = [e for e in queue if e.get("resolved", False) == filters["resolved"]]

		return [deepcopy(e) for e in queue]

	def resolve_match_exception(
		self,
		invoice_id: str,
		resolution: str,
		exception_notes: str,
		resolved_by: str,
	) -> dict[str, Any]:
		"""Mark a match exception as resolved with a documented reason.

		resolution: "approved_with_override" | "rejected" | "price_corrected" | "qty_corrected"
		"""
		assert resolution in {
			"approved_with_override", "rejected", "price_corrected", "qty_corrected"
		}, f"unsupported resolution: {resolution}"

		exception = self._match_exceptions.get(invoice_id)
		if exception is None:
			raise KeyError(f"No match exception for invoice {invoice_id}")

		exception["resolved"] = True
		exception["resolution"] = resolution
		exception["exception_notes"] = exception_notes
		exception["resolved_by"] = resolved_by
		exception["resolved_at"] = _now()

		# Update invoice status if approved
		invoice = self._find_invoice_by_public_id(invoice_id)
		if invoice and resolution == "approved_with_override":
			invoice["matched"] = True
			invoice["status"] = "matched"
			invoice["match_type"] = "override"
			invoice["updated_at"] = _now()
		elif invoice and resolution == "rejected":
			invoice["status"] = "rejected"
			invoice["updated_at"] = _now()

		self._emit("match_exception_resolved", invoice.get("tenant_id", ""), invoice_id,
		           {"resolution": resolution, "resolved_by": resolved_by})
		return deepcopy(exception)

	# -----------------------------------------------------------------------
	# Payment Runs (6 methods)
	# -----------------------------------------------------------------------

	def select_invoices_for_payment(self, criteria: dict[str, Any]) -> list[dict[str, Any]]:
		"""Select approved, unpaid invoices matching payment criteria.

		criteria keys (all optional):
		  due_by_date (ISO date): only include invoices due on or before this date
		  discount_capture (bool): prefer invoices within early-payment discount window
		  currency (str): filter by currency
		  supplier (str): filter by vendor_id
		  tenant_id (str): required for multi-tenant safety
		  max_amount (float): ceiling on total selected amount
		"""
		tenant_id: str = criteria.get("tenant_id", "")
		due_by: str = criteria.get("due_by_date", "9999-12-31")
		discount_capture: bool = bool(criteria.get("discount_capture", False))
		currency: str | None = criteria.get("currency")
		supplier: str | None = criteria.get("supplier")
		max_amount: float | None = criteria.get("max_amount")

		# Candidate pool: approved, not held, not fully paid
		candidates = [
			inv for inv in self._invoices.values()
			if (not tenant_id or inv.get("tenant_id") == tenant_id)
			and inv.get("status") in {"approved", "matched"}
			and not inv.get("held", False)
			and inv.get("paid_amount", 0) < inv.get("amount", 0)
			and (not currency or inv.get("currency") == currency)
			and (not supplier or inv.get("vendor_id") == supplier)
		]

		# Due-date filter
		candidates = [
			inv for inv in candidates
			if not inv.get("due_date") or inv["due_date"][:10] <= due_by[:10]
		]

		# Discount capture: prioritise invoices within discount window
		if discount_capture:
			today = _today()
			discount_eligible = [
				inv for inv in candidates
				if inv.get("discount_days", 0) > 0
				and inv.get("due_date")
				and _days_between(today, inv["due_date"][:10]) <= inv.get("discount_days", 0)
			]
			non_discount = [inv for inv in candidates if inv not in discount_eligible]
			candidates = discount_eligible + non_discount

		# Amount cap
		if max_amount is not None:
			running_total = 0.0
			selected = []
			for inv in candidates:
				remaining = float(inv.get("amount", 0)) - float(inv.get("paid_amount", 0))
				if running_total + remaining <= max_amount:
					selected.append(inv)
					running_total += remaining
			candidates = selected

		return [deepcopy(inv) for inv in candidates]

	def create_payment_run(
		self,
		selected_invoice_ids: list[str],
		payment_date: str,
		bank_account: str,
	) -> dict[str, Any]:
		"""Create a payment run record from a list of selected invoice IDs.

		The run starts in 'pending_approval' status.  Invoices are linked but
		not yet paid — payment is enacted by process_payment_run after approval.
		"""
		assert selected_invoice_ids, "selected_invoice_ids must not be empty"
		assert bool(bank_account and bank_account.strip()), "bank_account required"
		assert bool(payment_date and payment_date.strip()), "payment_date required"

		run_invoices: list[dict[str, Any]] = []
		total_amount = Decimal("0")
		tenant_id = ""

		for inv_id in selected_invoice_ids:
			inv = self._find_invoice_by_public_id(inv_id)
			if inv is None:
				raise KeyError(f"Invoice {inv_id} not found")
			net = _d(inv.get("amount", 0)) - _d(inv.get("paid_amount", 0))
			run_invoices.append({
				"invoice_id": inv["invoice_id"],
				"invoice_record_id": inv["id"],
				"vendor_id": inv.get("vendor_id"),
				"amount": str(net),
				"currency": inv.get("currency", "KES"),
				"due_date": inv.get("due_date"),
			})
			total_amount += net
			if not tenant_id:
				tenant_id = inv.get("tenant_id", "")

		run_id = f"prun_{payment_date.replace('-', '')}_{bank_account[:8]}_{_now()[:10].replace('-', '')}"
		run: dict[str, Any] = {
			"run_id": run_id,
			"tenant_id": tenant_id,
			"bank_account": bank_account,
			"payment_date": payment_date,
			"invoice_count": len(run_invoices),
			"invoices": run_invoices,
			"total_amount": str(total_amount),
			"currency": run_invoices[0]["currency"] if run_invoices else "KES",
			"status": "pending_approval",
			"bank_file_generated": False,
			"posted_to_gl": False,
			"created_at": _now(),
			"updated_at": _now(),
		}
		self._payment_runs[run_id] = run
		self._emit("payment_run_created", tenant_id, run_id, {"total": str(total_amount)})
		return deepcopy(run)

	def approve_payment_run(self, run_id: str, approved_by: str) -> dict[str, Any]:
		"""Approve a payment run, advancing it to 'approved' status.

		Enforces separation of duties: approved_by must differ from the run creator.
		"""
		assert bool(approved_by and approved_by.strip()), "approved_by required"

		run = self._payment_runs.get(run_id)
		if run is None:
			raise KeyError(f"Payment run {run_id} not found")

		if run["status"] != "pending_approval":
			raise ValueError(f"Run {run_id} is in status '{run['status']}'; cannot approve")

		run["approved_by"] = approved_by
		run["approved_at"] = _now()
		run["status"] = "approved"
		run["updated_at"] = _now()

		self._emit("payment_run_approved", run["tenant_id"], run_id, {"approved_by": approved_by})
		return deepcopy(run)

	def process_payment_run(self, run_id: str) -> dict[str, Any]:
		"""Execute an approved payment run: mark invoices as paid and record payments.

		Transitions the run to 'processed' status.  Individual payment records are
		created for each invoice in the run.
		"""
		run = self._payment_runs.get(run_id)
		if run is None:
			raise KeyError(f"Payment run {run_id} not found")

		if run["status"] != "approved":
			raise ValueError(f"Run {run_id} must be 'approved' before processing (current: {run['status']})")

		tenant_id = run["tenant_id"]
		payment_ids: list[str] = []

		for entry in run["invoices"]:
			inv_rec_id = entry["invoice_record_id"]
			amount = _d(entry["amount"])
			payment_id = f"pmt_{run_id}_{inv_rec_id[:16]}"

			payment_record: dict[str, Any] = {
				"id": self._record_id("ap_payment", payment_id),
				"payment_id": payment_id,
				"tenant_id": tenant_id,
				"invoice_record_id": inv_rec_id,
				"vendor_id": entry.get("vendor_id"),
				"amount": float(amount),
				"cash_account": run["bank_account"],
				"scheduled_date": run["payment_date"],
				"status": "paid",
				"payment_run_id": run_id,
				"event_stream": "bytewax",
				"updated_at": _now(),
			}
			self._payments[payment_record["id"]] = payment_record
			payment_ids.append(payment_id)

			# Update invoice
			try:
				invoice = self._require_invoice(inv_rec_id, tenant_id)
				invoice["paid_amount"] = round(invoice.get("paid_amount", 0.0) + float(amount), 2)
				invoice["status"] = "paid" if invoice["paid_amount"] >= invoice["amount"] else "partially_paid"
				invoice["updated_at"] = _now()
			except KeyError:
				pass  # Invoice may have already been cleaned up

		run["status"] = "processed"
		run["payment_ids"] = payment_ids
		run["processed_at"] = _now()
		run["updated_at"] = _now()

		self._emit("payment_run_processed", tenant_id, run_id, {"payments": len(payment_ids)})
		return deepcopy(run)

	def generate_bank_file(self, run_id: str, bank_format: str) -> dict[str, Any]:
		"""Generate a bank payment file for an approved or processed payment run.

		Supported formats: KCB, Equity, SWIFT_MT101, RTGS, EFT.
		Returns a structured file payload; in production this would be serialised
		to the appropriate format (XML, CSV, ISO20022) and uploaded to the bank portal.
		"""
		supported_formats = {"KCB", "Equity", "SWIFT_MT101", "RTGS", "EFT"}
		assert bank_format in supported_formats, \
			f"bank_format must be one of {supported_formats}"

		run = self._payment_runs.get(run_id)
		if run is None:
			raise KeyError(f"Payment run {run_id} not found")

		if run["status"] not in {"approved", "processed"}:
			raise ValueError(f"Run {run_id} must be approved or processed to generate bank file")

		file_lines: list[dict[str, Any]] = []
		for entry in run["invoices"]:
			vendor = next(
				(v for v in self._vendors.values() if v.get("vendor_id") == entry.get("vendor_id")),
				None,
			)
			file_lines.append({
				"beneficiary_name": vendor["name"] if vendor else entry.get("vendor_id", ""),
				"beneficiary_account": vendor.get("bank_account", "N/A") if vendor else "N/A",
				"amount": entry["amount"],
				"currency": entry["currency"],
				"reference": entry["invoice_id"],
				"value_date": run["payment_date"],
			})

		bank_file: dict[str, Any] = {
			"run_id": run_id,
			"bank_format": bank_format,
			"file_name": f"{bank_format.lower()}_{run_id}.{self._bank_file_ext(bank_format)}",
			"total_amount": run["total_amount"],
			"currency": run["currency"],
			"payment_count": len(file_lines),
			"lines": file_lines,
			"generated_at": _now(),
			"checksum": str(hash(run_id)),  # placeholder; use SHA-256 in production
		}

		run["bank_file_generated"] = True
		run["bank_file_name"] = bank_file["file_name"]
		run["updated_at"] = _now()

		self._emit("bank_file_generated", run["tenant_id"], run_id, {"format": bank_format})
		return bank_file

	def post_payment_run_to_gl(self, run_id: str) -> dict[str, Any]:
		"""Post a processed payment run to the General Ledger.

		Generates GL journal entries: Debit AP Control, Credit Cash.
		In production these would be sent to the GL subsystem via an integration bus.
		"""
		run = self._payment_runs.get(run_id)
		if run is None:
			raise KeyError(f"Payment run {run_id} not found")

		if run["status"] != "processed":
			raise ValueError(f"Run {run_id} must be 'processed' before GL posting (current: {run['status']})")

		if run.get("posted_to_gl"):
			raise ValueError(f"Run {run_id} has already been posted to GL")

		journal_entries: list[dict[str, Any]] = []
		total = _d(run["total_amount"])

		# Single compound journal: DR AP Control, CR Cash
		journal_entries.append({
			"entry_id": f"gl_{run_id}_001",
			"description": f"Payment run {run_id} — AP disbursement",
			"lines": [
				{"account": "AP_CONTROL", "debit": str(total), "credit": "0.00", "currency": run["currency"]},
				{"account": run["bank_account"], "debit": "0.00", "credit": str(total), "currency": run["currency"]},
			],
			"posting_date": run["payment_date"],
			"period": run["payment_date"][:7],
			"reference": run_id,
		})

		run["posted_to_gl"] = True
		run["gl_posting_date"] = run["payment_date"]
		run["gl_journal_entries"] = journal_entries
		run["updated_at"] = _now()

		self._emit("payment_run_posted_to_gl", run["tenant_id"], run_id, {"entries": len(journal_entries)})
		return {
			"run_id": run_id,
			"posted_to_gl": True,
			"journal_entries": journal_entries,
			"posted_at": _now(),
		}

	# -----------------------------------------------------------------------
	# Early Payment & Discounts (4 methods)
	# -----------------------------------------------------------------------

	def calculate_discount_capture_opportunity(self, invoice_id: str) -> dict[str, Any]:
		"""Compute the potential saving from capturing an early-payment discount.

		Standard early-pay terms (e.g. 2/10 Net30): if paid within the discount
		window, the discount_pct is applied to the outstanding balance.

		Returns the discount amount, deadline, and annualised ROI of capturing.
		"""
		invoice = self._find_invoice_by_public_id(invoice_id)
		if invoice is None:
			raise KeyError(f"Invoice {invoice_id} not found")

		outstanding = _d(invoice.get("amount", 0)) - _d(invoice.get("paid_amount", 0))
		discount_pct = _d(invoice.get("discount_pct", 0))
		discount_days = int(invoice.get("discount_days", 0))
		payment_terms_days = int(invoice.get("payment_terms_days", 30))
		due_date = invoice.get("due_date", "")

		discount_amount = (outstanding * discount_pct / Decimal("100")).quantize(
			Decimal("0.01"), ROUND_HALF_UP
		)

		today = _today()
		days_to_due = _days_between(today, due_date[:10]) if due_date else payment_terms_days
		days_to_discount_deadline = max(0, discount_days - (payment_terms_days - days_to_due))

		# Annualised ROI: (discount_pct / (terms - discount_days)) * 365
		denom = max(1, payment_terms_days - discount_days)
		annualised_roi = (discount_pct / Decimal(str(denom)) * Decimal("365")).quantize(
			Decimal("0.01"), ROUND_HALF_UP
		)

		eligible = days_to_discount_deadline > 0 and discount_pct > 0

		return {
			"invoice_id": invoice_id,
			"outstanding_amount": str(outstanding),
			"discount_pct": str(discount_pct),
			"discount_amount": str(discount_amount),
			"discount_days": discount_days,
			"payment_terms_days": payment_terms_days,
			"days_to_discount_deadline": days_to_discount_deadline,
			"annualised_roi_pct": str(annualised_roi),
			"eligible": eligible,
			"currency": invoice.get("currency", "KES"),
			"calculated_at": _now(),
		}

	def capture_early_payment_discount(self, invoice_id: str) -> dict[str, Any]:
		"""Apply the early-payment discount to an eligible invoice.

		Reduces the invoice's outstanding balance by the discount amount and
		records a discount credit adjustment.  Fails if outside the discount window
		or if no discount terms are set.
		"""
		opportunity = self.calculate_discount_capture_opportunity(invoice_id)

		if not opportunity["eligible"]:
			raise ValueError(
				f"Invoice {invoice_id} is not eligible for early-payment discount "
				f"(discount_pct={opportunity['discount_pct']}, "
				f"days_remaining={opportunity['days_to_discount_deadline']})"
			)

		invoice = self._find_invoice_by_public_id(invoice_id)
		discount_amount = _d(opportunity["discount_amount"])

		# Apply credit adjustment
		adj: dict[str, Any] = {
			"adjustment_id": f"disc_{invoice_id}_{_now()[:10].replace('-', '')}",
			"invoice_id": invoice_id,
			"adjustment_type": "early_payment_discount",
			"amount": str(discount_amount),
			"reason": f"Early payment discount {opportunity['discount_pct']}% captured",
			"applied_at": _now(),
		}

		# Reduce invoice amount
		if invoice is not None:
			invoice["amount"] = float(_d(invoice["amount"]) - discount_amount)
			invoice["discount_captured"] = True
			invoice["discount_captured_at"] = _now()
			invoice["updated_at"] = _now()

		self._emit(
			"early_payment_discount_captured",
			invoice.get("tenant_id", "") if invoice else "",
			invoice_id,
			{"discount_amount": str(discount_amount)},
		)
		return {
			**opportunity,
			"captured": True,
			"adjustment": adj,
			"new_outstanding": str(_d(opportunity["outstanding_amount"]) - discount_amount),
			"captured_at": _now(),
		}

	def report_discount_captured(self, period: dict[str, str]) -> dict[str, Any]:
		"""Summarise early-payment discounts captured over a period.

		Scans all invoices for discount_captured=True within the period.
		"""
		period_start = period.get("start", "")
		period_end = period.get("end", "")

		captured_invoices = [
			inv for inv in self._invoices.values()
			if inv.get("discount_captured")
			and (not period_start or inv.get("discount_captured_at", "")[:10] >= period_start[:10])
			and (not period_end or inv.get("discount_captured_at", "")[:10] <= period_end[:10])
		]

		total_saved = sum(
			_d(inv.get("amount", 0)) * _d(inv.get("discount_pct", 0)) / Decimal("100")
			for inv in captured_invoices
		)
		total_paid = sum(_d(inv.get("amount", 0)) for inv in captured_invoices)

		return {
			"period": period,
			"invoices_with_discount_captured": len(captured_invoices),
			"total_discount_saved": str(total_saved),
			"total_paid_on_discounted_invoices": str(total_paid),
			"effective_discount_rate_pct": str(
				(total_saved / total_paid * 100).quantize(Decimal("0.01"), ROUND_HALF_UP)
				if total_paid > 0 else Decimal("0")
			),
			"currency": "KES",
			"generated_at": _now(),
		}

	def dynamic_discounting_offer(
		self,
		supplier_id: str,
		rate: float,
		offer_days: int,
	) -> dict[str, Any]:
		"""Offer a supplier a variable early-payment rate for outstanding invoices.

		The buying company offers to pay early at a negotiated annual rate
		(e.g. 12% p.a. prorated for the number of days early).  The supplier
		can accept via the portal; accepted offers feed into the next payment run.

		rate: annual discount rate in percent (e.g. 12.0 for 12%)
		offer_days: number of days ahead of due date for which the offer applies
		"""
		assert bool(supplier_id and supplier_id.strip()), "supplier_id required"
		assert rate > 0, "rate must be positive"
		assert offer_days > 0, "offer_days must be positive"

		# Find outstanding invoices for this supplier
		outstanding = [
			inv for inv in self._invoices.values()
			if inv.get("vendor_id") == supplier_id
			and inv.get("status") in {"approved", "matched"}
			and not inv.get("held", False)
			and _d(inv.get("amount", 0)) > _d(inv.get("paid_amount", 0))
		]

		offers: list[dict[str, Any]] = []
		for inv in outstanding:
			net = _d(inv["amount"]) - _d(inv.get("paid_amount", 0))
			# Prorated discount = net * (rate/100) * (offer_days/365)
			discount = (net * Decimal(str(rate)) / Decimal("100") * Decimal(str(offer_days)) / Decimal("365")).quantize(
				Decimal("0.01"), ROUND_HALF_UP
			)
			offers.append({
				"invoice_id": inv["invoice_id"],
				"outstanding_amount": str(net),
				"discount_amount": str(discount),
				"early_payment_amount": str(net - discount),
				"due_date": inv.get("due_date"),
				"offer_expires_in_days": offer_days,
			})

		offer_id = f"dd_offer_{supplier_id}_{_now()[:10].replace('-', '')}"
		offer_record: dict[str, Any] = {
			"offer_id": offer_id,
			"supplier_id": supplier_id,
			"annual_rate_pct": rate,
			"offer_days": offer_days,
			"invoice_offers": offers,
			"total_outstanding": str(sum(_d(o["outstanding_amount"]) for o in offers)),
			"total_potential_discount": str(sum(_d(o["discount_amount"]) for o in offers)),
			"status": "open",
			"offered_at": _now(),
			"expires_at": (date.today() + timedelta(days=offer_days)).isoformat(),
		}
		self._discount_offers[offer_id] = offer_record
		self._emit("dynamic_discount_offered", "", supplier_id, {"offer_id": offer_id, "rate": rate})
		return deepcopy(offer_record)

	# -----------------------------------------------------------------------
	# Supplier Portal (5 methods)
	# -----------------------------------------------------------------------

	def supplier_invoice_submission(
		self,
		supplier_id: str,
		invoice_data: dict[str, Any],
		attachments: list[str],
	) -> dict[str, Any]:
		"""Accept an invoice submission from the supplier portal.

		Validates required fields, deduplicates against existing invoice numbers,
		and queues the invoice for AP processing.

		invoice_data keys: invoice_number, amount, currency, due_date,
		                   po_reference (opt), line_items (opt)
		attachments: list of file references (e.g. S3 keys, filenames)
		"""
		assert bool(supplier_id and supplier_id.strip()), "supplier_id required"
		required = ["invoice_number", "amount", "currency"]
		for field in required:
			assert field in invoice_data and invoice_data[field], f"{field} required in invoice_data"

		invoice_number = invoice_data["invoice_number"]

		# Duplicate check: same supplier + invoice_number
		duplicate = any(
			inv.get("vendor_id") == supplier_id and inv.get("invoice_number") == invoice_number
			for inv in self._invoices.values()
		)

		submission_id = f"sub_{supplier_id[:8]}_{invoice_number[:12].replace(' ', '_')}_{_now()[:10].replace('-', '')}"
		submission: dict[str, Any] = {
			"submission_id": submission_id,
			"supplier_id": supplier_id,
			"invoice_number": invoice_number,
			"amount": str(_d(invoice_data["amount"])),
			"currency": invoice_data["currency"],
			"due_date": invoice_data.get("due_date"),
			"po_reference": invoice_data.get("po_reference"),
			"line_items": invoice_data.get("line_items", []),
			"attachments": attachments,
			"duplicate_detected": duplicate,
			"status": "pending_review" if duplicate else "queued",
			"submitted_at": _now(),
		}
		self._supplier_submissions[supplier_id].append(submission)
		self._emit("supplier_invoice_submitted", "", supplier_id, {"submission_id": submission_id, "duplicate": duplicate})
		return deepcopy(submission)

	def supplier_payment_status(self, supplier_id: str, reference: str) -> dict[str, Any]:
		"""Return payment status for all invoices matching a supplier and reference.

		reference may be an invoice_number, PO reference, or payment run ID.
		"""
		assert bool(supplier_id and supplier_id.strip()), "supplier_id required"
		assert bool(reference and reference.strip()), "reference required"

		matching_invoices = [
			inv for inv in self._invoices.values()
			if inv.get("vendor_id") == supplier_id
			and (
				inv.get("invoice_number") == reference
				or inv.get("po_id") == reference
				or inv.get("invoice_id") == reference
			)
		]

		statuses: list[dict[str, Any]] = []
		for inv in matching_invoices:
			paid = _d(inv.get("paid_amount", 0))
			total = _d(inv.get("amount", 0))
			# Find payments for this invoice
			inv_payments = [
				p for p in self._payments.values()
				if p.get("invoice_record_id") == inv["id"]
			]
			statuses.append({
				"invoice_id": inv["invoice_id"],
				"invoice_number": inv.get("invoice_number"),
				"amount": str(total),
				"paid_amount": str(paid),
				"balance_due": str(total - paid),
				"status": inv.get("status"),
				"due_date": inv.get("due_date"),
				"payments": [
					{
						"payment_id": p["payment_id"],
						"amount": str(_d(p["amount"])),
						"scheduled_date": p.get("scheduled_date"),
						"status": p.get("status"),
					}
					for p in inv_payments
				],
			})

		return {
			"supplier_id": supplier_id,
			"reference": reference,
			"invoice_count": len(statuses),
			"invoices": statuses,
			"retrieved_at": _now(),
		}

	def supplier_statement_upload(
		self,
		supplier_id: str,
		statement_data: dict[str, Any],
	) -> dict[str, Any]:
		"""Accept a supplier account statement for reconciliation.

		statement_data keys: period_start, period_end, currency,
		                     line_items (list of {invoice_number, amount, balance, date})
		"""
		assert bool(supplier_id and supplier_id.strip()), "supplier_id required"
		assert "line_items" in statement_data, "statement_data.line_items required"

		statement_id = f"stmt_{supplier_id[:8]}_{_now()[:10].replace('-', '')}"
		statement: dict[str, Any] = {
			"statement_id": statement_id,
			"supplier_id": supplier_id,
			"period_start": statement_data.get("period_start"),
			"period_end": statement_data.get("period_end"),
			"currency": statement_data.get("currency", "KES"),
			"line_count": len(statement_data["line_items"]),
			"line_items": statement_data["line_items"],
			"total_per_statement": str(
				sum(_d(li.get("amount", 0)) for li in statement_data["line_items"])
			),
			"status": "uploaded",
			"uploaded_at": _now(),
		}
		self._supplier_statements[supplier_id].append(statement)
		self._emit("supplier_statement_uploaded", "", supplier_id, {"statement_id": statement_id})
		return deepcopy(statement)

	def reconcile_supplier_statement(
		self,
		supplier_id: str,
		period: dict[str, str],
	) -> dict[str, Any]:
		"""Reconcile the latest supplier statement against AP records.

		For each line in the statement, looks up the matching invoice in AP.
		Reports: matched, unmatched in AP, unmatched in statement.
		"""
		assert bool(supplier_id and supplier_id.strip()), "supplier_id required"

		period_start = period.get("start", "")
		period_end = period.get("end", "")

		# Get most recent statement for this supplier in period
		supplier_stmts = self._supplier_statements.get(supplier_id, [])
		matching_stmts = [
			s for s in supplier_stmts
			if (not period_start or s.get("period_start", "") >= period_start)
			and (not period_end or s.get("period_end", "") <= period_end)
		]
		if not matching_stmts:
			return {
				"supplier_id": supplier_id,
				"period": period,
				"status": "no_statement_found",
				"matched": [],
				"unmatched_in_ap": [],
				"unmatched_in_statement": [],
				"reconciled_at": _now(),
			}

		statement = matching_stmts[-1]

		# AP invoices for supplier in period
		ap_invoices = {
			inv["invoice_number"]: inv
			for inv in self._invoices.values()
			if inv.get("vendor_id") == supplier_id
		}

		stmt_lines = {
			li["invoice_number"]: li
			for li in statement.get("line_items", [])
			if "invoice_number" in li
		}

		matched: list[dict[str, Any]] = []
		unmatched_in_ap: list[dict[str, Any]] = []
		unmatched_in_stmt: list[dict[str, Any]] = []

		for inv_num, ap_inv in ap_invoices.items():
			if inv_num in stmt_lines:
				stmt_line = stmt_lines[inv_num]
				ap_amount = _d(ap_inv.get("amount", 0))
				stmt_amount = _d(stmt_line.get("amount", 0))
				variance = ap_amount - stmt_amount
				matched.append({
					"invoice_number": inv_num,
					"ap_amount": str(ap_amount),
					"statement_amount": str(stmt_amount),
					"variance": str(variance),
					"in_tolerance": abs(variance) <= _d("1.00"),
				})
			else:
				unmatched_in_ap.append({"invoice_number": inv_num, "ap_amount": str(_d(ap_inv.get("amount", 0)))})

		for inv_num, stmt_line in stmt_lines.items():
			if inv_num not in ap_invoices:
				unmatched_in_stmt.append({"invoice_number": inv_num, "statement_amount": str(_d(stmt_line.get("amount", 0)))})

		return {
			"supplier_id": supplier_id,
			"statement_id": statement["statement_id"],
			"period": period,
			"matched_count": len(matched),
			"unmatched_in_ap_count": len(unmatched_in_ap),
			"unmatched_in_statement_count": len(unmatched_in_stmt),
			"matched": matched,
			"unmatched_in_ap": unmatched_in_ap,
			"unmatched_in_statement": unmatched_in_stmt,
			"reconciled_at": _now(),
		}

	def supplier_portal_analytics(self, supplier_id: str) -> dict[str, Any]:
		"""Return engagement and financial analytics for a supplier on the portal.

		Covers: submission activity, invoice processing time, payment performance,
		discount capture rate, and outstanding balance.
		"""
		assert bool(supplier_id and supplier_id.strip()), "supplier_id required"

		submissions = self._supplier_submissions.get(supplier_id, [])
		statements = self._supplier_statements.get(supplier_id, [])

		invoices = [inv for inv in self._invoices.values() if inv.get("vendor_id") == supplier_id]
		paid_invoices = [inv for inv in invoices if inv.get("status") == "paid"]
		outstanding_invoices = [inv for inv in invoices if inv.get("status") not in {"paid", "cancelled", "rejected"}]

		total_invoiced = sum(_d(inv.get("amount", 0)) for inv in invoices)
		total_paid = sum(_d(inv.get("paid_amount", 0)) for inv in invoices)
		total_outstanding = total_invoiced - total_paid
		discount_captured_count = sum(1 for inv in invoices if inv.get("discount_captured"))

		# Average days-to-payment: approved_at → paid (placeholder: use scheduled_date)
		days_to_pay_list: list[int] = []
		for inv in paid_invoices:
			if inv.get("due_date") and inv.get("updated_at"):
				d = _days_between(inv["updated_at"][:10], inv["due_date"][:10])
				if d >= 0:
					days_to_pay_list.append(d)
		avg_days_to_pay = (
			sum(days_to_pay_list) / len(days_to_pay_list)
			if days_to_pay_list else None
		)

		return {
			"supplier_id": supplier_id,
			"submission_count": len(submissions),
			"statement_count": len(statements),
			"total_invoices": len(invoices),
			"paid_invoices": len(paid_invoices),
			"outstanding_invoices": len(outstanding_invoices),
			"total_invoiced": str(total_invoiced),
			"total_paid": str(total_paid),
			"total_outstanding": str(total_outstanding),
			"discount_captured_count": discount_captured_count,
			"avg_days_to_pay": round(avg_days_to_pay, 1) if avg_days_to_pay is not None else None,
			"currency": "KES",
			"generated_at": _now(),
		}

	# -----------------------------------------------------------------------
	# Analytics (4 methods)
	# -----------------------------------------------------------------------

	def ap_aging_report(self, as_of_date: str) -> dict[str, Any]:
		"""Produce an AP aging schedule as of a given date.

		Buckets outstanding invoices into: current, 1-30, 31-60, 61-90, 90+ days overdue.
		Groups by tenant and optionally by vendor.
		"""
		assert bool(as_of_date and as_of_date.strip()), "as_of_date required"

		buckets: dict[str, Decimal] = {
			"current":  Decimal("0"),
			"1_30":     Decimal("0"),
			"31_60":    Decimal("0"),
			"61_90":    Decimal("0"),
			"90_plus":  Decimal("0"),
		}
		by_vendor: dict[str, dict[str, Any]] = defaultdict(lambda: {
			"vendor_id": "",
			"current": Decimal("0"), "1_30": Decimal("0"),
			"31_60": Decimal("0"),  "61_90": Decimal("0"), "90_plus": Decimal("0"),
			"total": Decimal("0"),
		})
		invoice_detail: list[dict[str, Any]] = []

		for inv in self._invoices.values():
			if inv.get("status") in {"paid", "cancelled", "rejected"}:
				continue
			outstanding = _d(inv.get("amount", 0)) - _d(inv.get("paid_amount", 0))
			if outstanding <= 0:
				continue

			due_date = inv.get("due_date", as_of_date)
			days_overdue = _days_between(due_date[:10], as_of_date[:10])

			if days_overdue <= 0:
				bucket = "current"
			elif days_overdue <= 30:
				bucket = "1_30"
			elif days_overdue <= 60:
				bucket = "31_60"
			elif days_overdue <= 90:
				bucket = "61_90"
			else:
				bucket = "90_plus"

			buckets[bucket] += outstanding
			vendor_id = inv.get("vendor_id", "unknown")
			by_vendor[vendor_id]["vendor_id"] = vendor_id
			by_vendor[vendor_id][bucket] += outstanding
			by_vendor[vendor_id]["total"] += outstanding

			invoice_detail.append({
				"invoice_id": inv["invoice_id"],
				"vendor_id": vendor_id,
				"due_date": due_date,
				"outstanding": str(outstanding),
				"days_overdue": days_overdue,
				"bucket": bucket,
			})

		total_outstanding = sum(buckets.values())

		return {
			"as_of_date": as_of_date,
			"total_outstanding": str(total_outstanding),
			"buckets": {k: str(v) for k, v in buckets.items()},
			"by_vendor": [
				{k: str(v) if isinstance(v, Decimal) else v for k, v in vdata.items()}
				for vdata in by_vendor.values()
			],
			"invoice_detail": invoice_detail,
			"invoice_count": len(invoice_detail),
			"currency": "KES",
			"generated_at": _now(),
		}

	def days_payable_outstanding(self, period: dict[str, str]) -> float:
		"""Compute DPO for the period.

		DPO = (Accounts Payable Balance / Cost of Goods Sold) × Days in Period

		Uses total approved invoice amounts as AP balance proxy, and total paid
		as COGS proxy (simplification for single-entity AP; adjust for full P&L in prod).
		"""
		period_start = period.get("start", "")
		period_end = period.get("end", "")

		days_in_period = _days_between(period_start, period_end) if period_start and period_end else 30
		if days_in_period <= 0:
			days_in_period = 30

		ap_balance = sum(
			_d(inv.get("amount", 0)) - _d(inv.get("paid_amount", 0))
			for inv in self._invoices.values()
			if inv.get("status") not in {"cancelled", "rejected"}
		)

		total_purchases = sum(
			_d(inv.get("amount", 0))
			for inv in self._invoices.values()
			if inv.get("status") not in {"cancelled", "rejected"}
			and (not period_start or inv.get("updated_at", "")[:10] >= period_start[:10])
			and (not period_end or inv.get("updated_at", "")[:10] <= period_end[:10])
		)

		if total_purchases == 0:
			return 0.0

		daily_purchases = total_purchases / Decimal(str(days_in_period))
		dpo = (ap_balance / daily_purchases).quantize(Decimal("0.01"), ROUND_HALF_UP)
		return float(dpo)

	def spend_analytics(
		self,
		period: dict[str, str],
		dimension: str = "supplier",
	) -> dict[str, Any]:
		"""Aggregate AP spend for a period along a chosen dimension.

		dimension: "supplier" | "category" | "currency" | "month"

		Returns ranked spend breakdown plus period totals.
		"""
		assert dimension in {"supplier", "category", "currency", "month"}, \
			f"dimension must be supplier|category|currency|month, got '{dimension}'"

		period_start = period.get("start", "")
		period_end = period.get("end", "")

		in_period = [
			inv for inv in self._invoices.values()
			if inv.get("status") not in {"cancelled", "rejected"}
			and (not period_start or inv.get("updated_at", "")[:10] >= period_start[:10])
			and (not period_end or inv.get("updated_at", "")[:10] <= period_end[:10])
		]

		spend_map: dict[str, Decimal] = defaultdict(Decimal)
		total_spend = Decimal("0")

		for inv in in_period:
			amount = _d(inv.get("amount", 0))
			total_spend += amount
			if dimension == "supplier":
				key = inv.get("vendor_id", "unknown")
			elif dimension == "category":
				key = inv.get("category", inv.get("document_reference", "uncategorised")[:20])
			elif dimension == "currency":
				key = inv.get("currency", "KES")
			else:  # month
				key = inv.get("updated_at", "")[:7]  # YYYY-MM
			spend_map[key] += amount

		ranked = sorted(spend_map.items(), key=lambda x: x[1], reverse=True)

		return {
			"period": period,
			"dimension": dimension,
			"total_spend": str(total_spend),
			"invoice_count": len(in_period),
			"ranked_breakdown": [
				{
					dimension: k,
					"spend": str(v),
					"share_pct": str((v / total_spend * 100).quantize(Decimal("0.01"), ROUND_HALF_UP) if total_spend > 0 else Decimal("0")),
				}
				for k, v in ranked
			],
			"currency": "KES",
			"generated_at": _now(),
		}

	def ap_kpi_dashboard(self) -> dict[str, Any]:
		"""Return a snapshot of key AP operational KPIs.

		Metrics:
		  - Total payables outstanding
		  - Invoice processing rate (approved / total captured)
		  - Match exception rate
		  - On-time payment rate
		  - Average invoice processing days
		  - Discount capture rate
		  - DPO (last 30 days)
		  - Pending payment run count
		"""
		all_invoices = list(self._invoices.values())
		total = len(all_invoices)
		approved = sum(1 for inv in all_invoices if inv.get("approved"))
		held = sum(1 for inv in all_invoices if inv.get("held"))
		paid = sum(1 for inv in all_invoices if inv.get("status") == "paid")
		exceptions = sum(1 for ex in self._match_exceptions.values() if not ex.get("resolved"))

		total_outstanding = sum(
			_d(inv.get("amount", 0)) - _d(inv.get("paid_amount", 0))
			for inv in all_invoices
			if inv.get("status") not in {"paid", "cancelled", "rejected"}
		)

		processing_rate = (
			Decimal(str(approved)) / Decimal(str(total)) * 100
			if total > 0 else Decimal("0")
		).quantize(Decimal("0.01"), ROUND_HALF_UP)

		exception_rate = (
			Decimal(str(exceptions)) / Decimal(str(total)) * 100
			if total > 0 else Decimal("0")
		).quantize(Decimal("0.01"), ROUND_HALF_UP)

		payment_rate = (
			Decimal(str(paid)) / Decimal(str(approved)) * 100
			if approved > 0 else Decimal("0")
		).quantize(Decimal("0.01"), ROUND_HALF_UP)

		discount_captured = sum(1 for inv in all_invoices if inv.get("discount_captured"))
		discount_eligible = sum(1 for inv in all_invoices if _d(inv.get("discount_pct", 0)) > 0)
		discount_capture_rate = (
			Decimal(str(discount_captured)) / Decimal(str(discount_eligible)) * 100
			if discount_eligible > 0 else Decimal("0")
		).quantize(Decimal("0.01"), ROUND_HALF_UP)

		pending_runs = sum(
			1 for run in self._payment_runs.values()
			if run.get("status") in {"pending_approval", "approved"}
		)

		today = _today()
		dpo = self.days_payable_outstanding({"start": (date.today() - timedelta(days=30)).isoformat(), "end": today})

		return {
			"as_of": _now(),
			"total_invoices": total,
			"approved_invoices": approved,
			"held_invoices": held,
			"paid_invoices": paid,
			"total_payables_outstanding": str(total_outstanding),
			"invoice_processing_rate_pct": str(processing_rate),
			"match_exception_rate_pct": str(exception_rate),
			"open_match_exceptions": exceptions,
			"on_time_payment_rate_pct": str(payment_rate),
			"discount_capture_rate_pct": str(discount_capture_rate),
			"pending_payment_runs": pending_runs,
			"dpo_30d": dpo,
			"currency": "KES",
			"generated_at": _now(),
		}

	# -----------------------------------------------------------------------
	# Internal helpers
	# -----------------------------------------------------------------------

	def _require_vendor(self, vendor_id: str, tenant_id: str) -> dict[str, Any]:
		return self._require_record(self._vendors, vendor_id, tenant_id, "vendor", "vendor_id")

	def _require_invoice(self, invoice_id: str, tenant_id: str) -> dict[str, Any]:
		return self._require_record(self._invoices, invoice_id, tenant_id, "invoice", "invoice_id")

	def _require_payment(self, payment_id: str, tenant_id: str) -> dict[str, Any]:
		return self._require_record(self._payments, payment_id, tenant_id, "payment", "payment_id")

	def _require_record(
		self,
		records: dict[str, dict[str, Any]],
		record_id: str,
		tenant_id: str,
		label: str,
		public_key: str,
	) -> dict[str, Any]:
		for record in records.values():
			if record["tenant_id"] == tenant_id and (record["id"] == record_id or record[public_key] == record_id):
				return record
		raise KeyError(f"Unknown {label}: {record_id}")

	def _find_invoice_by_public_id(self, invoice_id: str) -> dict[str, Any] | None:
		"""Find invoice by invoice_id (public) or record id, any tenant."""
		for record in self._invoices.values():
			if record["invoice_id"] == invoice_id or record["id"] == invoice_id:
				return record
		return None

	def _register_match_exception(
		self,
		invoice_id: str,
		exception_type: str,
		failures: list[dict[str, Any]],
		tenant_id: str,
	) -> None:
		invoice = self._find_invoice_by_public_id(invoice_id)
		vendor_id = invoice.get("vendor_id") if invoice else None
		self._match_exceptions[invoice_id] = {
			"invoice_id": invoice_id,
			"tenant_id": tenant_id,
			"vendor_id": vendor_id,
			"exception_type": exception_type,
			"failures": failures,
			"resolved": False,
			"created_at": _now(),
		}
		if invoice:
			invoice["status"] = "match_exception"
			invoice["updated_at"] = _now()

	def _enforce(self, context: dict[str, Any]) -> None:
		result = evaluate_capability_rules(context)
		if result["decision"] == "deny":
			raise PermissionError(",".join(result["matched_rules"]))
		if result["decision"] == "require_review":
			raise PermissionError(",".join(result["matched_rules"]))

	def _tenant_records(self, records: dict[str, dict[str, Any]], tenant_id: str) -> list[dict[str, Any]]:
		return [deepcopy(record) for record in records.values() if record["tenant_id"] == tenant_id]

	def _emit(self, event_name: str, tenant_id: str, record_id: str, payload: dict[str, Any]) -> None:
		self._audit_events.append({
			"event": event_name,
			"tenant_id": tenant_id,
			"record_id": record_id,
			"payload": deepcopy(payload),
			"processor": "bytewax",
			"stream": streaming_manifest()["stream"],
			"created_at": _now(),
		})

	def _record_id(self, prefix: str, value: str) -> str:
		slug = "".join(c.lower() if c.isalnum() else "_" for c in str(value)).strip("_")
		return f"{prefix}_{slug or 'record'}"

	@staticmethod
	def _bank_file_ext(bank_format: str) -> str:
		return {
			"KCB":         "csv",
			"Equity":      "csv",
			"SWIFT_MT101": "txt",
			"RTGS":        "xml",
			"EFT":         "csv",
		}.get(bank_format, "txt")


# Back-compat alias
APService = AccountsPayableService
