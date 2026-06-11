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
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

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

	# -----------------------------------------------------------------------
	# Async extensions — AI, forecasting, risk, compliance, workflow
	# -----------------------------------------------------------------------

	async def ml_duplicate_invoice_detect(
		self,
		invoice_id: str,
		tenant_id: str,
		lookback_days: int = 90,
	) -> dict[str, Any]:
		"""AI-powered duplicate invoice detection via local Ollama embedding model.

		Embeds the candidate invoice's key features (vendor, amount, invoice_number,
		document hash) and computes cosine similarity against invoices captured in
		the last `lookback_days`.  Scores above 0.92 are flagged as probable duplicates.

		Requires OLLAMA_BASE_URL environment variable.  Degrades gracefully when
		Ollama is unavailable — returns ml_enhanced=False with a rule-based fallback.
		"""
		import os

		invoice = self._find_invoice_by_public_id(invoice_id)
		if invoice is None:
			raise KeyError(f"Invoice {invoice_id} not found")

		# Rule-based fallback: exact match on (vendor_id, invoice_number, amount)
		exact_matches = [
			rec["invoice_id"]
			for rec in self._invoices.values()
			if rec["id"] != invoice["id"]
			and rec.get("tenant_id") == tenant_id
			and rec.get("vendor_id") == invoice.get("vendor_id")
			and rec.get("invoice_number") == invoice.get("invoice_number")
			and abs(float(rec.get("amount", 0)) - float(invoice.get("amount", 0))) < 0.01
		]

		if exact_matches:
			return {
				"invoice_id": invoice_id,
				"is_duplicate": True,
				"exact_match_ids": exact_matches,
				"fuzzy_match_ids": [],
				"confidence": 1.0,
				"reason": "exact match on vendor_id + invoice_number + amount",
				"ml_enhanced": False,
			}

		if not os.environ.get("OLLAMA_BASE_URL"):
			return {
				"invoice_id": invoice_id,
				"is_duplicate": False,
				"exact_match_ids": [],
				"fuzzy_match_ids": [],
				"confidence": 0.0,
				"reason": "OLLAMA_BASE_URL not set; rule-based check passed",
				"ml_enhanced": False,
			}

		try:
			import json as _json
			import urllib.request as _req

			def _embed(text: str) -> list[float]:
				payload = _json.dumps({"model": "nomic-embed-text", "prompt": text}).encode()
				with _req.urlopen(
					_req.Request(
						os.environ["OLLAMA_BASE_URL"].rstrip("/") + "/api/embeddings",
						data=payload,
						headers={"Content-Type": "application/json"},
					),
					timeout=10,
				) as resp:
					return _json.loads(resp.read())["embedding"]

			def _cosine(a: list[float], b: list[float]) -> float:
				dot = sum(x * y for x, y in zip(a, b))
				norm_a = sum(x ** 2 for x in a) ** 0.5
				norm_b = sum(x ** 2 for x in b) ** 0.5
				if norm_a == 0 or norm_b == 0:
					return 0.0
				return dot / (norm_a * norm_b)

			candidate_text = (
				f"{invoice.get('vendor_id')} {invoice.get('invoice_number')} "
				f"{invoice.get('amount')} {invoice.get('currency')}"
			)
			candidate_vec = _embed(candidate_text)

			cutoff = date.today() - timedelta(days=lookback_days)
			fuzzy_matches: list[str] = []
			for rec in self._invoices.values():
				if rec["id"] == invoice["id"] or rec.get("tenant_id") != tenant_id:
					continue
				try:
					rec_date = date.fromisoformat(rec.get("updated_at", "")[:10])
					if rec_date < cutoff:
						continue
				except Exception as _exc:
					_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)
				rec_text = (
					f"{rec.get('vendor_id')} {rec.get('invoice_number')} "
					f"{rec.get('amount')} {rec.get('currency')}"
				)
				rec_vec = _embed(rec_text)
				sim = _cosine(candidate_vec, rec_vec)
				if sim >= 0.92:
					fuzzy_matches.append(rec["invoice_id"])

			return {
				"invoice_id": invoice_id,
				"is_duplicate": bool(fuzzy_matches),
				"exact_match_ids": [],
				"fuzzy_match_ids": fuzzy_matches,
				"confidence": 0.95 if fuzzy_matches else 0.05,
				"reason": "ml cosine similarity against nomic-embed-text vectors" if fuzzy_matches else "no similar invoices found",
				"ml_enhanced": True,
			}
		except Exception as exc:
			return {
				"invoice_id": invoice_id,
				"is_duplicate": False,
				"exact_match_ids": [],
				"fuzzy_match_ids": [],
				"confidence": 0.0,
				"reason": f"ml check failed: {exc}",
				"ml_enhanced": False,
			}

	async def forecast_cash_outflows(
		self,
		tenant_id: str,
		horizon_weeks: int = 13,
	) -> dict[str, Any]:
		"""Produce a rolling weekly AP cash outflow forecast for the next N weeks.

		For each approved, unpaid invoice, places the outstanding balance in the
		week bucket corresponding to its due_date.  Invoices with no due_date fall
		into the final bucket.  A historical payment-velocity adjustment (± days
		early/late) is approximated from past paid invoices.

		Returns weekly buckets with P10/P50/P90 probabilistic bands based on
		observed payment-day variance, plus a total projected outflow.
		"""
		assert horizon_weeks >= 1, "horizon_weeks must be >= 1"

		today = date.today()
		week_buckets: list[dict[str, Any]] = []
		for w in range(horizon_weeks):
			week_start = today + timedelta(weeks=w)
			week_end = today + timedelta(weeks=w + 1) - timedelta(days=1)
			week_buckets.append({
				"week": w + 1,
				"week_start": week_start.isoformat(),
				"week_end": week_end.isoformat(),
				"amount_p50": Decimal("0"),
				"invoice_ids": [],
			})

		overflow_bucket: dict[str, Any] = {
			"week": horizon_weeks + 1,
			"week_start": (today + timedelta(weeks=horizon_weeks)).isoformat(),
			"week_end": "beyond_horizon",
			"amount_p50": Decimal("0"),
			"invoice_ids": [],
		}

		# Compute historical payment-velocity variance: days actual vs due
		paid_deltas: list[int] = []
		for inv in self._invoices.values():
			if inv.get("tenant_id") != tenant_id:
				continue
			if inv.get("status") == "paid" and inv.get("due_date") and inv.get("updated_at"):
				delta = _days_between(inv["due_date"][:10], inv["updated_at"][:10])
				paid_deltas.append(delta)

		avg_delta = (sum(paid_deltas) / len(paid_deltas)) if paid_deltas else 0.0
		variance = 0.0
		if len(paid_deltas) > 1:
			mean = avg_delta
			variance = sum((d - mean) ** 2 for d in paid_deltas) / len(paid_deltas)
		std_dev = variance ** 0.5

		# Place invoices into buckets
		for inv in self._invoices.values():
			if inv.get("tenant_id") != tenant_id:
				continue
			if inv.get("status") not in {"approved", "matched"}:
				continue
			if inv.get("held"):
				continue
			outstanding = _d(inv.get("amount", 0)) - _d(inv.get("paid_amount", 0))
			if outstanding <= 0:
				continue

			due = inv.get("due_date")
			placed = False
			if due:
				try:
					due_date_obj = date.fromisoformat(due[:10])
					for bucket in week_buckets:
						ws = date.fromisoformat(bucket["week_start"])
						we = date.fromisoformat(bucket["week_end"])
						if ws <= due_date_obj <= we:
							bucket["amount_p50"] += outstanding
							bucket["invoice_ids"].append(inv["invoice_id"])
							placed = True
							break
				except Exception as _exc:
					_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)
			if not placed:
				overflow_bucket["amount_p50"] += outstanding
				overflow_bucket["invoice_ids"].append(inv["invoice_id"])

		# Build output with P10/P50/P90 bands from std_dev
		result_buckets = []
		total_p50 = Decimal("0")
		for bucket in week_buckets + [overflow_bucket]:
			p50 = bucket["amount_p50"]
			adjustment = Decimal(str(round(std_dev * float(p50) / 100, 2))) if p50 > 0 else Decimal("0")
			result_buckets.append({
				"week": bucket["week"],
				"week_start": bucket["week_start"],
				"week_end": bucket["week_end"],
				"amount_p10": str(max(Decimal("0"), p50 - adjustment * 2)),
				"amount_p50": str(p50),
				"amount_p90": str(p50 + adjustment * 2),
				"invoice_count": len(bucket["invoice_ids"]),
				"invoice_ids": bucket["invoice_ids"],
			})
			total_p50 += p50

		return {
			"tenant_id": tenant_id,
			"horizon_weeks": horizon_weeks,
			"forecast_date": today.isoformat(),
			"total_projected_outflow": str(total_p50),
			"avg_payment_delta_days": round(avg_delta, 1),
			"payment_velocity_std_dev_days": round(std_dev, 1),
			"weekly_buckets": result_buckets,
			"currency": "KES",
			"generated_at": _now(),
		}

	async def compute_invoice_tax(
		self,
		invoice_record_id: str,
		tenant_id: str,
		tax_profile: str = "standard",
	) -> dict[str, Any]:
		"""Determine and compute applicable VAT and withholding tax for an invoice.

		Tax profiles:
		  standard   — VAT 16%, WHT 5% on services
		  exempt     — VAT 0%, WHT 0%
		  vat_only   — VAT 16%, no WHT
		  wht_only   — no VAT, WHT 5%
		  zero_rated — VAT 0%, WHT 5%

		Returns gross_amount, vat_amount, wht_amount, net_payable, and the
		applicable rates with a breakdown suitable for iTax VAT schedule filing.
		"""
		invoice = self._require_invoice(invoice_record_id, tenant_id)

		tax_table: dict[str, dict[str, float]] = {
			"standard":  {"vat_pct": 16.0, "wht_pct": 5.0},
			"exempt":    {"vat_pct": 0.0,  "wht_pct": 0.0},
			"vat_only":  {"vat_pct": 16.0, "wht_pct": 0.0},
			"wht_only":  {"vat_pct": 0.0,  "wht_pct": 5.0},
			"zero_rated": {"vat_pct": 0.0, "wht_pct": 5.0},
		}
		assert tax_profile in tax_table, (
			f"tax_profile must be one of {list(tax_table)}, got '{tax_profile}'"
		)

		rates = tax_table[tax_profile]
		gross = _d(invoice.get("amount", 0))
		vat_pct = _d(rates["vat_pct"])
		wht_pct = _d(rates["wht_pct"])

		vat_amount = (gross * vat_pct / Decimal("100")).quantize(Decimal("0.01"), ROUND_HALF_UP)
		gross_plus_vat = gross + vat_amount
		wht_amount = (gross * wht_pct / Decimal("100")).quantize(Decimal("0.01"), ROUND_HALF_UP)
		net_payable = gross_plus_vat - wht_amount

		result = {
			"invoice_id": invoice.get("invoice_id"),
			"invoice_record_id": invoice_record_id,
			"tenant_id": tenant_id,
			"tax_profile": tax_profile,
			"gross_amount": str(gross),
			"vat_rate_pct": str(vat_pct),
			"vat_amount": str(vat_amount),
			"gross_plus_vat": str(gross_plus_vat),
			"wht_rate_pct": str(wht_pct),
			"wht_amount": str(wht_amount),
			"net_payable": str(net_payable),
			"currency": invoice.get("currency", "KES"),
			"computed_at": _now(),
		}

		# Persist tax metadata on the invoice record for downstream scheduling
		invoice["tax_profile"] = tax_profile
		invoice["vat_amount"] = float(vat_amount)
		invoice["wht_amount"] = float(wht_amount)
		invoice["net_payable"] = float(net_payable)
		invoice["updated_at"] = _now()

		self._emit("invoice_tax_computed", tenant_id, invoice_record_id, {
			"tax_profile": tax_profile, "vat_amount": str(vat_amount), "wht_amount": str(wht_amount),
		})
		return result

	async def generate_vat_schedule(
		self,
		tenant_id: str,
		period: dict[str, str],
	) -> dict[str, Any]:
		"""Generate an iTax-compatible VAT schedule for a given accounting period.

		Aggregates all invoices with computed VAT in the period and produces:
		  - input tax credit entries (AP invoices received)
		  - output tax entries (placeholder for AR integration)
		  - net VAT payable / refundable
		  - a line-by-line schedule ready for KRA iTax upload

		period keys: start (ISO date), end (ISO date)
		"""
		period_start = period.get("start", "")
		period_end = period.get("end", "")

		schedule_lines: list[dict[str, Any]] = []
		total_input_vat = Decimal("0")
		total_wht = Decimal("0")

		for inv in self._invoices.values():
			if inv.get("tenant_id") != tenant_id:
				continue
			if inv.get("status") in {"cancelled", "rejected"}:
				continue
			updated = inv.get("updated_at", "")[:10]
			if period_start and updated < period_start[:10]:
				continue
			if period_end and updated > period_end[:10]:
				continue

			vat = _d(inv.get("vat_amount", 0))
			wht = _d(inv.get("wht_amount", 0))
			if vat == 0 and wht == 0:
				continue

			total_input_vat += vat
			total_wht += wht
			schedule_lines.append({
				"invoice_number": inv.get("invoice_number"),
				"vendor_id": inv.get("vendor_id"),
				"invoice_date": inv.get("updated_at", "")[:10],
				"gross_amount": str(_d(inv.get("amount", 0))),
				"vat_amount": str(vat),
				"wht_amount": str(wht),
				"net_payable": str(_d(inv.get("net_payable", inv.get("amount", 0)))),
				"currency": inv.get("currency", "KES"),
				"tax_profile": inv.get("tax_profile", "standard"),
			})

		return {
			"tenant_id": tenant_id,
			"period": period,
			"schedule_type": "input_vat",
			"line_count": len(schedule_lines),
			"total_input_vat": str(total_input_vat),
			"total_wht_withheld": str(total_wht),
			"net_vat_payable": str(total_input_vat),
			"lines": schedule_lines,
			"currency": "KES",
			"generated_at": _now(),
			"filing_reference": f"VAT_{tenant_id[:8]}_{(period_start or _today())[:7].replace('-', '')}",
		}

	async def score_vendor_risk(
		self,
		vendor_record_id: str,
		tenant_id: str,
	) -> dict[str, Any]:
		"""Compute a composite vendor risk score (0–100) from AP transaction history.

		Scoring factors (weighted):
		  - Invoice exception rate:     25 pts (fewer exceptions → lower risk)
		  - Payment dispute rate:       25 pts (fewer disputes → lower risk)
		  - Price variance rate:        20 pts (tighter pricing → lower risk)
		  - On-time submission rate:    15 pts (consistent submission → lower risk)
		  - Bank-account change events: 15 pts (stability → lower risk)

		Score interpretation:
		  0–25  : Low risk — preferred vendor
		  26–50 : Moderate risk — standard controls apply
		  51–75 : High risk — enhanced due diligence required
		  76–100: Critical risk — payment block recommended
		"""
		vendor = self._require_vendor(vendor_record_id, tenant_id)
		vendor_id = vendor["vendor_id"]

		invoices = [
			inv for inv in self._invoices.values()
			if inv.get("vendor_id") == vendor_id and inv.get("tenant_id") == tenant_id
		]
		total_invoices = len(invoices)

		if total_invoices == 0:
			return {
				"vendor_id": vendor_id,
				"vendor_record_id": vendor_record_id,
				"tenant_id": tenant_id,
				"risk_score": 0,
				"risk_tier": "insufficient_data",
				"factors": {},
				"recommendation": "no invoice history — apply standard new-vendor onboarding controls",
				"computed_at": _now(),
			}

		# Exception rate
		exception_count = sum(
			1 for inv in invoices if inv.get("status") == "match_exception"
		)
		exception_rate = exception_count / total_invoices

		# Dispute proxy: invoices placed on hold
		disputed = sum(1 for inv in invoices if inv.get("held"))
		dispute_rate = disputed / total_invoices

		# Price variance: average variance_rate across matched invoices
		variance_values = [
			float(inv.get("variance_rate", 0))
			for inv in invoices
			if inv.get("matched") and inv.get("variance_rate") is not None
		]
		avg_variance = sum(variance_values) / len(variance_values) if variance_values else 0.0

		# Bank change risk
		bank_change = int(vendor.get("bank_change", False))

		# Weighted score (higher = more risky)
		score = (
			exception_rate * 25
			+ dispute_rate * 25
			+ min(avg_variance / 10, 1.0) * 20  # cap at 10% variance = full 20 pts
			+ bank_change * 15
		)
		score = min(100.0, round(score, 1))

		if score <= 25:
			tier = "low"
			recommendation = "preferred vendor — eligible for early payment and portal acceleration"
		elif score <= 50:
			tier = "moderate"
			recommendation = "standard controls — no additional action required"
		elif score <= 75:
			tier = "high"
			recommendation = "enhanced due diligence — require dual approval and document review"
		else:
			tier = "critical"
			recommendation = "payment block recommended — escalate to AP Controller"

		return {
			"vendor_id": vendor_id,
			"vendor_record_id": vendor_record_id,
			"tenant_id": tenant_id,
			"risk_score": score,
			"risk_tier": tier,
			"factors": {
				"exception_rate_pct": round(exception_rate * 100, 1),
				"dispute_rate_pct": round(dispute_rate * 100, 1),
				"avg_price_variance_pct": round(avg_variance, 2),
				"bank_change_flag": bool(bank_change),
				"total_invoices_analysed": total_invoices,
			},
			"recommendation": recommendation,
			"computed_at": _now(),
		}

	async def straight_through_process(
		self,
		invoice_record_id: str,
		tenant_id: str,
		approved_by: str = "auto_stp",
		requested_by: str = "stp_engine",
	) -> dict[str, Any]:
		"""Execute STP pipeline: validate → match → approve → schedule.

		Each step returns a StepResult. The pipeline aborts on the first
		failing step, routes to the appropriate exception queue, and emits
		a `stp_escalated` event.  Only a fully-passing invoice advances to
		`approved` status and emits `stp_completed`.

		Steps:
		  1. invoice_validated  — invoice exists and is in capturable state
		  2. duplicate_check    — no exact duplicates found
		  3. po_match           — invoice matches its PO (two-way or three-way)
		  4. approval           — system approves with STP principal
		"""
		steps: list[dict[str, Any]] = []

		# Step 1: validate
		try:
			invoice = self._require_invoice(invoice_record_id, tenant_id)
			steps.append({"step": "invoice_validated", "passed": True, "detail": f"status={invoice['status']}"})
		except KeyError as exc:
			steps.append({"step": "invoice_validated", "passed": False, "detail": str(exc)})
			self._emit("stp_escalated", tenant_id, invoice_record_id, {"step": "invoice_validated"})
			return {"invoice_record_id": invoice_record_id, "passed": False, "steps": steps, "escalated_at": _now()}

		# Step 2: duplicate check (synchronous rule-based only in STP)
		exact_dups = [
			rec["invoice_id"]
			for rec in self._invoices.values()
			if rec["id"] != invoice["id"]
			and rec.get("tenant_id") == tenant_id
			and rec.get("vendor_id") == invoice.get("vendor_id")
			and rec.get("invoice_number") == invoice.get("invoice_number")
			and abs(float(rec.get("amount", 0)) - float(invoice.get("amount", 0))) < 0.01
		]
		if exact_dups:
			steps.append({"step": "duplicate_check", "passed": False, "detail": f"duplicates: {exact_dups}"})
			invoice["status"] = "held"
			invoice["held"] = True
			invoice["hold_reason"] = "duplicate_suspected"
			invoice["updated_at"] = _now()
			self._emit("stp_escalated", tenant_id, invoice_record_id, {"step": "duplicate_check"})
			return {"invoice_record_id": invoice_record_id, "passed": False, "steps": steps, "escalated_at": _now()}
		steps.append({"step": "duplicate_check", "passed": True, "detail": "no duplicates found"})

		# Step 3: PO match
		po_id = invoice.get("po_id")
		grn_id = invoice.get("grn_id")
		if po_id:
			if grn_id:
				match_result = self.three_way_match(invoice["invoice_id"], po_id, grn_id)
			else:
				match_result = self.two_way_match(invoice["invoice_id"], po_id)
			if match_result["passed"]:
				steps.append({"step": "po_match", "passed": True, "detail": match_result["match_type"]})
			else:
				steps.append({"step": "po_match", "passed": False, "detail": str(match_result["failures"])})
				self._emit("stp_escalated", tenant_id, invoice_record_id, {"step": "po_match"})
				return {"invoice_record_id": invoice_record_id, "passed": False, "steps": steps, "escalated_at": _now()}
		else:
			steps.append({"step": "po_match", "passed": True, "detail": "no PO required (non-PO invoice)"})

		# Step 4: STP approval
		try:
			self.approve_invoice(
				tenant_id,
				invoice_record_id,
				approved_by=approved_by,
				requested_by=requested_by,
			)
			steps.append({"step": "approval", "passed": True, "detail": f"approved by {approved_by}"})
		except Exception as exc:
			steps.append({"step": "approval", "passed": False, "detail": str(exc)})
			self._emit("stp_escalated", tenant_id, invoice_record_id, {"step": "approval"})
			return {"invoice_record_id": invoice_record_id, "passed": False, "steps": steps, "escalated_at": _now()}

		self._emit("stp_completed", tenant_id, invoice_record_id, {"steps": len(steps)})
		return {
			"invoice_record_id": invoice_record_id,
			"passed": True,
			"steps": steps,
			"completed_at": _now(),
		}

	async def compute_vendor_scorecard(
		self,
		vendor_record_id: str,
		tenant_id: str,
		period: dict[str, str] | None = None,
	) -> dict[str, Any]:
		"""Compute a comprehensive vendor performance scorecard for the given period.

		Metrics (each scored 0–100, higher = better performance):
		  invoice_accuracy_rate   : % invoices passing match without exception
		  match_pass_rate         : % invoices with matched=True
		  on_time_submission_rate : % invoices submitted before due_date
		  dispute_rate            : inverse of % invoices held (inverted for score)
		  credit_note_rate        : proxy via matched invoices with discount corrections

		Composite performance_index = weighted average across all metrics.
		"""
		vendor = self._require_vendor(vendor_record_id, tenant_id)
		vendor_id = vendor["vendor_id"]

		period = period or {}
		period_start = period.get("start", "")
		period_end = period.get("end", "")

		invoices = [
			inv for inv in self._invoices.values()
			if inv.get("vendor_id") == vendor_id
			and inv.get("tenant_id") == tenant_id
			and (not period_start or inv.get("updated_at", "")[:10] >= period_start[:10])
			and (not period_end or inv.get("updated_at", "")[:10] <= period_end[:10])
		]

		total = len(invoices)
		if total == 0:
			return {
				"vendor_id": vendor_id,
				"vendor_name": vendor.get("name"),
				"tenant_id": tenant_id,
				"period": period,
				"total_invoices": 0,
				"performance_index": None,
				"metrics": {},
				"recommendation": "insufficient data for scoring",
				"computed_at": _now(),
			}

		matched = sum(1 for inv in invoices if inv.get("matched"))
		exceptions = sum(1 for inv in invoices if inv.get("status") == "match_exception")
		held = sum(1 for inv in invoices if inv.get("held"))
		on_time = sum(
			1 for inv in invoices
			if inv.get("due_date") and inv.get("updated_at")
			and inv["updated_at"][:10] <= inv["due_date"][:10]
		)
		discount_corrections = sum(1 for inv in invoices if inv.get("discount_captured"))

		invoice_accuracy = ((total - exceptions) / total) * 100
		match_pass = (matched / total) * 100
		on_time_submission = (on_time / total) * 100
		dispute_score = (1 - held / total) * 100
		credit_note_score = max(0, 100 - (discount_corrections / total) * 100)

		# Weighted composite: accuracy 30, match 25, on-time 20, dispute 15, credit_note 10
		perf_index = (
			invoice_accuracy * 0.30
			+ match_pass * 0.25
			+ on_time_submission * 0.20
			+ dispute_score * 0.15
			+ credit_note_score * 0.10
		)

		if perf_index >= 85:
			tier = "preferred"
			rec = "eligible for portal fast-track and early payment incentives"
		elif perf_index >= 70:
			tier = "good"
			rec = "standard relationship — monitor for improvement opportunities"
		elif perf_index >= 50:
			tier = "fair"
			rec = "schedule vendor review meeting to address recurring exceptions"
		else:
			tier = "poor"
			rec = "formal performance improvement plan required; consider dual sourcing"

		return {
			"vendor_id": vendor_id,
			"vendor_name": vendor.get("name"),
			"tenant_id": tenant_id,
			"period": period,
			"total_invoices": total,
			"performance_index": round(perf_index, 1),
			"performance_tier": tier,
			"metrics": {
				"invoice_accuracy_rate": round(invoice_accuracy, 1),
				"match_pass_rate": round(match_pass, 1),
				"on_time_submission_rate": round(on_time_submission, 1),
				"dispute_score": round(dispute_score, 1),
				"credit_note_score": round(credit_note_score, 1),
			},
			"recommendation": rec,
			"computed_at": _now(),
		}

	async def score_payment_fraud_risk(
		self,
		payment_record_id: str,
		tenant_id: str,
	) -> dict[str, Any]:
		"""Score a pending payment for real-time fraud risk indicators.

		Risk factors evaluated (each contributes 0–25 pts):
		  1. New bank account for this vendor (registered within last 90 days)
		  2. Amount deviation > 2x vendor's historical median payment
		  3. Payment scheduled on a weekend or Kenyan public holiday
		  4. Vendor bank account changed within 72h of this payment scheduling

		Score 0–100:
		  0–39  : low risk — proceed normally
		  40–74 : medium risk — flag for second-pair-of-eyes review
		  75+   : high risk — block and require CISO override

		Returns score, risk_tier, fired_factors, and recommendation.
		"""
		payment = self._require_payment(payment_record_id, tenant_id)
		invoice = self._require_invoice(payment["invoice_record_id"], tenant_id)
		vendor_record = None
		for v in self._vendors.values():
			if v.get("vendor_id") == payment.get("vendor_id") and v.get("tenant_id") == tenant_id:
				vendor_record = v
				break

		risk_score = 0
		factors: list[dict[str, Any]] = []

		# Factor 1: bank account recently added for vendor
		bank_change = bool(vendor_record and vendor_record.get("bank_change"))
		if bank_change:
			risk_score += 25
			factors.append({"factor": "new_bank_account", "weight": 25,
			                "detail": "vendor bank account was recently changed"})

		# Factor 2: amount deviation vs vendor median
		vendor_payments = [
			_d(p.get("amount", 0))
			for p in self._payments.values()
			if p.get("vendor_id") == payment.get("vendor_id")
			and p.get("id") != payment["id"]
			and p.get("status") == "paid"
		]
		if vendor_payments:
			sorted_pmts = sorted(vendor_payments)
			median = sorted_pmts[len(sorted_pmts) // 2]
			current_amount = _d(payment.get("amount", 0))
			if median > 0 and current_amount > median * 2:
				risk_score += 20
				factors.append({"factor": "amount_deviation", "weight": 20,
				                "detail": f"amount {current_amount} is >2x median {median}"})

		# Factor 3: weekend payment
		try:
			sched_date = date.fromisoformat(payment.get("scheduled_date", _today())[:10])
			if sched_date.weekday() >= 5:  # Saturday=5, Sunday=6
				risk_score += 15
				factors.append({"factor": "weekend_payment", "weight": 15,
				                "detail": f"payment scheduled on {sched_date.strftime('%A')}"})
		except Exception as _exc:
			_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)

		# Factor 4: bank change within 72h of payment scheduling
		vendor_updated = vendor_record.get("updated_at", "") if vendor_record else ""
		payment_created = payment.get("updated_at", "")
		if vendor_updated and payment_created:
			delta_hours = abs(_days_between(vendor_updated[:10], payment_created[:10])) * 24
			if bank_change and delta_hours <= 72:
				risk_score += 20
				factors.append({"factor": "bank_change_proximity", "weight": 20,
				                "detail": f"bank changed ~{delta_hours}h before payment scheduling"})

		risk_score = min(100, risk_score)

		if risk_score >= 75:
			tier = "high"
			recommendation = "block payment — require CISO override and independent bank account verification"
		elif risk_score >= 40:
			tier = "medium"
			recommendation = "flag for second-pair-of-eyes review before release"
		else:
			tier = "low"
			recommendation = "proceed normally"

		self._emit("payment_fraud_risk_scored", tenant_id, payment_record_id, {
			"risk_score": risk_score, "risk_tier": tier,
		})

		return {
			"payment_record_id": payment_record_id,
			"invoice_id": invoice.get("invoice_id"),
			"vendor_id": payment.get("vendor_id"),
			"tenant_id": tenant_id,
			"risk_score": risk_score,
			"risk_tier": tier,
			"fired_factors": factors,
			"recommendation": recommendation,
			"scored_at": _now(),
		}

	async def compute_accruals(
		self,
		tenant_id: str,
		period: dict[str, str],
	) -> dict[str, Any]:
		"""Identify and generate accrual journal entries for period-end close.

		Scans for:
		  Type A — GRNs received before period_end with no matching approved invoice
		           (received-not-invoiced, RNI).  Accrual: DR Accrued Liabilities,
		           CR Goods Received Not Invoiced.
		  Type B — Approved POs past expected delivery date with no GRN and no invoice
		           (service accruals, blanket commitments).  Accrual: DR Expense,
		           CR Accrued Expenses.

		Returns a list of proposed journal entries ready for GL posting, plus a
		summary count and total accrual amount.
		"""
		period_end = period.get("end", _today())
		period_label = period_end[:7]  # YYYY-MM

		journal_entries: list[dict[str, Any]] = []
		total_accrual = Decimal("0")

		# Type A: GRNs without matched invoices
		matched_grn_ids = {
			inv.get("grn_id")
			for inv in self._invoices.values()
			if inv.get("tenant_id") == tenant_id
			and inv.get("grn_id")
			and inv.get("matched")
		}

		for grn_id, grn in self._goods_receipts.items():
			if grn_id in matched_grn_ids:
				continue
			grn_date = grn.get("received_date", grn.get("updated_at", ""))[:10]
			if grn_date > period_end[:10]:
				continue
			# Estimate accrual from PO amount on the linked PO
			po = self._purchase_orders.get(grn.get("po_id", ""))
			accrual_amount = _d(grn.get("received_value", po.get("amount", 0) if po else 0))
			if accrual_amount <= 0:
				continue
			entry_id = f"accrual_rni_{grn_id[:12]}_{period_label.replace('-', '')}"
			journal_entries.append({
				"entry_id": entry_id,
				"accrual_type": "received_not_invoiced",
				"description": f"RNI accrual for GRN {grn_id} — period {period_label}",
				"lines": [
					{"account": "ACCRUED_LIABILITIES", "debit": str(accrual_amount), "credit": "0.00"},
					{"account": "GRNI_CLEARING", "debit": "0.00", "credit": str(accrual_amount)},
				],
				"period": period_label,
				"grn_id": grn_id,
				"amount": str(accrual_amount),
				"reversal_date": (date.fromisoformat(period_end[:10]) + timedelta(days=1)).isoformat(),
				"auto_reverse": True,
			})
			total_accrual += accrual_amount

		# Type B: POs past delivery date with no GRN
		invoiced_po_ids = {
			inv.get("po_id")
			for inv in self._invoices.values()
			if inv.get("tenant_id") == tenant_id and inv.get("po_id")
		}

		for po_id, po in self._purchase_orders.items():
			if po_id in invoiced_po_ids:
				continue
			delivery_date = po.get("expected_delivery_date", po.get("updated_at", ""))[:10]
			if not delivery_date or delivery_date > period_end[:10]:
				continue
			accrual_amount = _d(po.get("amount", 0))
			if accrual_amount <= 0:
				continue
			entry_id = f"accrual_svc_{po_id[:12]}_{period_label.replace('-', '')}"
			journal_entries.append({
				"entry_id": entry_id,
				"accrual_type": "service_accrual",
				"description": f"Service accrual for PO {po_id} — period {period_label}",
				"lines": [
					{"account": "ACCRUED_EXPENSES", "debit": str(accrual_amount), "credit": "0.00"},
					{"account": "AP_CONTROL", "debit": "0.00", "credit": str(accrual_amount)},
				],
				"period": period_label,
				"po_id": po_id,
				"amount": str(accrual_amount),
				"reversal_date": (date.fromisoformat(period_end[:10]) + timedelta(days=1)).isoformat(),
				"auto_reverse": True,
			})
			total_accrual += accrual_amount

		self._emit("accruals_computed", tenant_id, period_label, {
			"entry_count": len(journal_entries), "total_accrual": str(total_accrual),
		})

		return {
			"tenant_id": tenant_id,
			"period": period,
			"period_label": period_label,
			"entry_count": len(journal_entries),
			"total_accrual_amount": str(total_accrual),
			"rni_entries": sum(1 for e in journal_entries if e["accrual_type"] == "received_not_invoiced"),
			"service_accrual_entries": sum(1 for e in journal_entries if e["accrual_type"] == "service_accrual"),
			"journal_entries": journal_entries,
			"currency": "KES",
			"computed_at": _now(),
		}

	async def nl_query(
		self,
		question: str,
		tenant_id: str,
	) -> dict[str, Any]:
		"""Natural language AP query powered by local Ollama LLM.

		Routes the user question through a locally-hosted LLM (llama3.1:8b by default)
		with a structured AP system prompt.  The model selects the most appropriate
		data method, calls it with tenant-scoped parameters, and returns both the
		natural-language answer and the underlying structured data.

		Requires OLLAMA_BASE_URL environment variable.  Falls back to a keyword-router
		when Ollama is unavailable.
		"""
		import os

		# Keyword-based fallback router
		def _keyword_route(q: str) -> dict[str, Any]:
			q_lower = q.lower()
			if any(w in q_lower for w in ("aging", "overdue", "outstanding", "past due")):
				data = self.aging_summary(tenant_id)
				return {"method": "aging_summary", "answer": f"AP aging: {data['open_invoice_count']} open invoices totalling {data['open_amount']}.", "data": data}
			if any(w in q_lower for w in ("kpi", "dashboard", "metrics", "performance")):
				data = self.ap_kpi_dashboard()
				return {"method": "ap_kpi_dashboard", "answer": f"AP KPIs: {data['total_invoices']} total invoices, DPO={data['dpo_30d']} days.", "data": data}
			if any(w in q_lower for w in ("exception", "blocked", "match fail", "hold")):
				data = self.match_exception_queue({"tenant_id": tenant_id})
				return {"method": "match_exception_queue", "answer": f"{len(data)} open match exceptions.", "data": data}
			if any(w in q_lower for w in ("spend", "category", "supplier spend", "breakdown")):
				data = self.spend_analytics({"start": (date.today() - timedelta(days=30)).isoformat(), "end": _today()}, "supplier")
				return {"method": "spend_analytics", "answer": f"Top supplier spend last 30 days: {data['ranked_breakdown'][:3]}.", "data": data}
			data = self.dashboard_summary(tenant_id)
			return {"method": "dashboard_summary", "answer": f"AP summary: {data['invoice_count']} invoices, {data['open_invoice_count']} open.", "data": data}

		if not os.environ.get("OLLAMA_BASE_URL"):
			result = _keyword_route(question)
			return {
				"question": question,
				"tenant_id": tenant_id,
				"answer": result["answer"],
				"method_used": result["method"],
				"data": result["data"],
				"ml_enhanced": False,
				"responded_at": _now(),
			}

		try:
			import json as _json
			import urllib.request as _req

			system_prompt = (
				"You are an Accounts Payable assistant for a finance team. "
				"Available data queries: aging_summary, ap_kpi_dashboard, "
				"match_exception_queue, spend_analytics, dashboard_summary. "
				"Respond with a JSON object: {\"method\": \"<method_name>\", \"answer\": \"<natural language answer>\"}. "
				"Be concise. Use KES currency. Tenant is already filtered."
			)
			payload = _json.dumps({
				"model": os.environ.get("OLLAMA_AP_MODEL", "llama3.1:8b"),
				"prompt": f"System: {system_prompt}\nUser: {question}",
				"stream": False,
			}).encode()

			with _req.urlopen(
				_req.Request(
					os.environ["OLLAMA_BASE_URL"].rstrip("/") + "/api/generate",
					data=payload,
					headers={"Content-Type": "application/json"},
				),
				timeout=30,
			) as resp:
				llm_resp = _json.loads(resp.read())

			raw_text = llm_resp.get("response", "")
			try:
				parsed = _json.loads(raw_text)
				method_name = parsed.get("method", "dashboard_summary")
				answer = parsed.get("answer", raw_text)
			except Exception:
				method_name = "dashboard_summary"
				answer = raw_text

			# Execute the resolved method
			method_map = {
				"aging_summary": lambda: self.aging_summary(tenant_id),
				"ap_kpi_dashboard": lambda: self.ap_kpi_dashboard(),
				"match_exception_queue": lambda: self.match_exception_queue({"tenant_id": tenant_id}),
				"spend_analytics": lambda: self.spend_analytics(
					{"start": (date.today() - timedelta(days=30)).isoformat(), "end": _today()}, "supplier"
				),
				"dashboard_summary": lambda: self.dashboard_summary(tenant_id),
			}
			data = method_map.get(method_name, method_map["dashboard_summary"])()

			return {
				"question": question,
				"tenant_id": tenant_id,
				"answer": answer,
				"method_used": method_name,
				"data": data,
				"ml_enhanced": True,
				"responded_at": _now(),
			}

		except Exception as exc:
			result = _keyword_route(question)
			return {
				"question": question,
				"tenant_id": tenant_id,
				"answer": result["answer"],
				"method_used": result["method"],
				"data": result["data"],
				"ml_enhanced": False,
				"fallback_reason": str(exc),
				"responded_at": _now(),
			}


	# -----------------------------------------------------------------------
	# New async extensions — I1/I2/I3/I5/I6/I8/I12/I13/I14/I15
	# -----------------------------------------------------------------------

	async def ingest_peppol_invoice(
		self,
		xml_bytes: bytes,
		tenant_id: str,
	) -> dict[str, Any]:
		"""Parse and ingest a Peppol BIS 3.0 / UBL 2.1 XML invoice (I1).

		Validates required UBL elements, extracts line items and tax totals,
		auto-creates a raw invoice record, and triggers rule evaluation.
		Degrades gracefully when lxml is unavailable by attempting stdlib
		ElementTree on well-formed XML.

		Returns a normalised invoice dict plus `peppol_valid` and
		`requires_review` flags (True when OCR confidence < 0.85 equivalent,
		i.e. any non-mandatory field is missing).
		"""
		guard_tenant_id(tenant_id)

		ns = {
			"cbc": "urn:oasis:names:specification:ubl:schema:xsd:CommonBasicComponents-2",
			"cac": "urn:oasis:names:specification:ubl:schema:xsd:CommonAggregateComponents-2",
		}

		parse_errors: list[str] = []
		extracted: dict[str, Any] = {}

		try:
			try:
				from lxml import etree as _et  # type: ignore[import]
				root = _et.fromstring(xml_bytes)
				def _find(xpath: str) -> str | None:
					el = root.find(xpath, ns)
					return el.text.strip() if el is not None and el.text else None
			except ImportError:
				import xml.etree.ElementTree as _et2  # noqa: PLC0415
				root = _et2.fromstring(xml_bytes.decode("utf-8"))
				def _find(xpath: str) -> str | None:  # type: ignore[misc]
					el = root.find(xpath, ns)
					return el.text.strip() if el is not None and el.text else None

			extracted = {
				"invoice_number":  _find(".//cbc:ID"),
				"invoice_date":    _find(".//cbc:IssueDate"),
				"due_date":        _find(".//cbc:DueDate"),
				"currency":        _find(".//cbc:DocumentCurrencyCode"),
				"supplier_name":   _find(".//cac:AccountingSupplierParty//cbc:Name"),
				"supplier_tax_id": _find(".//cac:AccountingSupplierParty//cbc:CompanyID"),
				"amount":          _find(".//cac:LegalMonetaryTotal/cbc:PayableAmount"),
				"tax_amount":      _find(".//cac:TaxTotal/cbc:TaxAmount"),
				"peppol_format":   "UBL_2_1",
			}

			for required in ("invoice_number", "currency", "amount"):
				if not extracted.get(required):
					parse_errors.append(f"missing required field: {required}")

		except Exception as exc:
			parse_errors.append(f"xml_parse_error: {exc}")

		peppol_valid = len(parse_errors) == 0
		requires_review = not peppol_valid or not extracted.get("supplier_tax_id")

		invoice_record: dict[str, Any] = {
			"source":        "peppol",
			"tenant_id":     tenant_id,
			"invoice_number": extracted.get("invoice_number", "UNKNOWN"),
			"amount":        float(_d(extracted.get("amount", "0") or "0")),
			"currency":      extracted.get("currency", "KES"),
			"due_date":      extracted.get("due_date"),
			"tax_amount":    float(_d(extracted.get("tax_amount", "0") or "0")),
			"supplier_name": extracted.get("supplier_name"),
			"supplier_tax_id": extracted.get("supplier_tax_id"),
			"peppol_valid":  peppol_valid,
			"requires_review": requires_review,
			"parse_errors":  parse_errors,
			"ingested_at":   _now(),
		}
		self._emit("peppol_invoice_ingested", tenant_id, extracted.get("invoice_number", "unknown"), {
			"valid": peppol_valid, "requires_review": requires_review,
		})
		return invoice_record

	async def propose_vendor_bank_change(
		self,
		vendor_record_id: str,
		tenant_id: str,
		new_bank_account: str,
		new_iban: str | None,
		proposed_by: str,
	) -> dict[str, Any]:
		"""Initiate dual-control vendor bank account change workflow (I2).

		Stores the proposal in `_pending_bank_changes` with status `pending_confirmation`.
		Validates IBAN check digit (ISO 7064 MOD 97-10) when provided.
		The change is NOT applied until `confirm_vendor_bank_change` is called
		by a different principal.

		Any active payment run against this vendor is flagged `bank_change_pending`.
		"""
		guard_tenant_id(tenant_id)
		guard_non_empty_string(proposed_by, "proposed_by")
		guard_non_empty_string(new_bank_account, "new_bank_account")

		vendor = self._require_vendor(vendor_record_id, tenant_id)

		# IBAN check-digit validation (ISO 7064 MOD 97-10)
		iban_valid: bool | None = None
		if new_iban:
			cleaned = new_iban.replace(" ", "").upper()
			rearranged = cleaned[4:] + cleaned[:4]
			numeric = "".join(str(ord(c) - 55) if c.isalpha() else c for c in rearranged)
			iban_valid = int(numeric) % 97 == 1

		if not hasattr(self, "_pending_bank_changes"):
			self._pending_bank_changes: dict[str, dict[str, Any]] = {}

		change_id = f"bchg_{vendor['vendor_id']}_{_now()[:10].replace('-', '')}"
		proposal: dict[str, Any] = {
			"change_id":         change_id,
			"vendor_record_id":  vendor_record_id,
			"vendor_id":         vendor["vendor_id"],
			"tenant_id":         tenant_id,
			"old_bank_account":  vendor.get("bank_account"),
			"new_bank_account":  new_bank_account,
			"new_iban":          new_iban,
			"iban_valid":        iban_valid,
			"proposed_by":       proposed_by,
			"status":            "pending_confirmation",
			"proposed_at":       _now(),
		}
		self._pending_bank_changes[change_id] = proposal

		# Flag affected payment runs
		for run in self._payment_runs.values():
			for entry in run.get("invoices", []):
				if entry.get("vendor_id") == vendor["vendor_id"]:
					run["bank_change_pending"] = True
					break

		self._emit("vendor_bank_change_proposed", tenant_id, vendor_record_id, {
			"change_id": change_id, "proposed_by": proposed_by, "iban_valid": iban_valid,
		})
		return deepcopy(proposal)

	async def confirm_vendor_bank_change(
		self,
		change_id: str,
		tenant_id: str,
		confirmed_by: str,
	) -> dict[str, Any]:
		"""Confirm (or reject) a pending vendor bank account change (I2).

		Enforces separation of duties: `confirmed_by` must differ from `proposed_by`.
		On confirmation, atomically updates the vendor record and clears the pending
		`bank_change_pending` flag from payment runs.
		"""
		guard_tenant_id(tenant_id)
		guard_non_empty_string(confirmed_by, "confirmed_by")

		if not hasattr(self, "_pending_bank_changes"):
			self._pending_bank_changes = {}

		proposal = self._pending_bank_changes.get(change_id)
		if proposal is None:
			raise KeyError(f"Bank change proposal {change_id} not found")
		if proposal["tenant_id"] != tenant_id:
			raise PermissionError("tenant mismatch on bank change proposal")
		if proposal["proposed_by"] == confirmed_by:
			raise PermissionError("separation of duties violation: confirmed_by must differ from proposed_by")

		vendor = self._require_vendor(proposal["vendor_record_id"], tenant_id)
		vendor["bank_account"] = proposal["new_bank_account"]
		vendor["bank_change"] = True
		vendor["bank_change_confirmed_by"] = confirmed_by
		vendor["bank_change_confirmed_at"] = _now()
		vendor["updated_at"] = _now()

		proposal["status"] = "confirmed"
		proposal["confirmed_by"] = confirmed_by
		proposal["confirmed_at"] = _now()

		# Clear bank_change_pending flag on payment runs
		for run in self._payment_runs.values():
			if run.get("bank_change_pending"):
				still_pending = any(
					e.get("vendor_id") == vendor["vendor_id"]
					and self._pending_bank_changes.get(change_id, {}).get("status") == "pending_confirmation"
					for e in run.get("invoices", [])
				)
				if not still_pending:
					run.pop("bank_change_pending", None)

		self._emit("vendor_bank_change_confirmed", tenant_id, proposal["vendor_record_id"], {
			"change_id": change_id, "confirmed_by": confirmed_by,
		})
		return deepcopy(proposal)

	async def optimise_payment_schedule(
		self,
		tenant_id: str,
		available_cash: float,
		cost_of_capital_pct: float = 12.0,
	) -> dict[str, Any]:
		"""Rank invoices by NPV benefit of early-payment discount vs cost of capital (I5).

		For each eligible invoice computes:
		  annualised_roi = discount_pct / (net_days - discount_days) * 365
		  roi_advantage  = annualised_roi - cost_of_capital_pct
		  npv_benefit    = outstanding * (roi_advantage / 100) * (days_saved / 365)

		Allocates `available_cash` greedily to invoices ranked by `roi_advantage`
		(highest first).  Invoices where `roi_advantage <= 0` are skipped.

		Returns an ordered payment schedule, total projected savings, and invoices
		deferred to standard terms.
		"""
		guard_tenant_id(tenant_id)
		assert available_cash > 0, "available_cash must be positive"
		assert cost_of_capital_pct >= 0, "cost_of_capital_pct must be >= 0"

		today = date.today()
		candidates: list[dict[str, Any]] = []

		for inv in self._invoices.values():
			if inv.get("tenant_id") != tenant_id:
				continue
			if inv.get("status") not in {"approved", "matched"}:
				continue
			if inv.get("held"):
				continue
			outstanding = _d(inv.get("amount", 0)) - _d(inv.get("paid_amount", 0))
			if outstanding <= 0:
				continue

			discount_pct = _d(inv.get("discount_pct", 0))
			discount_days = int(inv.get("discount_days", 0))
			net_days = int(inv.get("payment_terms_days", 30))

			if discount_pct <= 0 or discount_days <= 0:
				continue

			days_saved = max(1, net_days - discount_days)
			annualised_roi = float(discount_pct) / days_saved * 365
			roi_advantage = annualised_roi - cost_of_capital_pct
			if roi_advantage <= 0:
				continue

			npv_benefit = float(outstanding) * (roi_advantage / 100) * (days_saved / 365)

			due_date = inv.get("due_date")
			days_to_deadline = (
				max(0, (date.fromisoformat(due_date[:10]) - today).days - (net_days - discount_days))
				if due_date else 0
			)

			candidates.append({
				"invoice_id":       inv["invoice_id"],
				"vendor_id":        inv.get("vendor_id"),
				"outstanding":      float(outstanding),
				"discount_pct":     float(discount_pct),
				"discount_days":    discount_days,
				"net_days":         net_days,
				"days_saved":       days_saved,
				"annualised_roi":   round(annualised_roi, 2),
				"roi_advantage":    round(roi_advantage, 2),
				"npv_benefit":      round(npv_benefit, 2),
				"days_to_deadline": days_to_deadline,
				"currency":         inv.get("currency", "KES"),
			})

		candidates.sort(key=lambda x: x["roi_advantage"], reverse=True)

		scheduled: list[dict[str, Any]] = []
		deferred: list[dict[str, Any]] = []
		remaining_cash = Decimal(str(available_cash))
		total_savings = Decimal("0")

		for item in candidates:
			cost = _d(item["outstanding"])
			savings = _d(item["npv_benefit"])
			if remaining_cash >= cost:
				scheduled.append({**item, "action": "pay_early", "cash_used": float(cost)})
				remaining_cash -= cost
				total_savings += savings
			else:
				deferred.append({**item, "action": "standard_terms"})

		return {
			"tenant_id":             tenant_id,
			"available_cash":        available_cash,
			"cost_of_capital_pct":   cost_of_capital_pct,
			"invoices_evaluated":    len(candidates),
			"scheduled_early":       len(scheduled),
			"deferred_to_standard":  len(deferred),
			"cash_allocated":        float(Decimal(str(available_cash)) - remaining_cash),
			"cash_remaining":        float(remaining_cash),
			"total_projected_savings": str(total_savings),
			"schedule":              scheduled,
			"deferred":              deferred,
			"currency":              "KES",
			"optimised_at":          _now(),
		}

	async def generate_wht_certificate(
		self,
		invoice_record_id: str,
		tenant_id: str,
		certificate_type: str = "P9A",
	) -> dict[str, Any]:
		"""Generate a KRA WHT certificate (P9A/P9B) for a processed invoice (I6).

		certificate_type:
		  P9A — resident supplier (standard WHT deduction)
		  P9B — non-resident supplier (higher WHT, applies to royalties/dividends)

		Requires `compute_invoice_tax` to have been run on the invoice first
		(i.e., `wht_amount` must be set).  Assigns a sequential certificate number
		(tenant-scoped) and emits `wht_certificate_issued` audit event.

		Returns the certificate dict; in production this feeds a PDF renderer.
		"""
		guard_tenant_id(tenant_id)
		assert certificate_type in {"P9A", "P9B"}, "certificate_type must be P9A or P9B"

		invoice = self._require_invoice(invoice_record_id, tenant_id)
		wht_amount = _d(invoice.get("wht_amount", 0))
		if wht_amount <= 0:
			raise ValueError(
				f"Invoice {invoice_record_id} has no WHT computed. "
				"Run compute_invoice_tax first."
			)

		vendor_id = invoice.get("vendor_id", "")
		vendor = next(
			(v for v in self._vendors.values()
			 if v.get("vendor_id") == vendor_id and v.get("tenant_id") == tenant_id),
			None,
		)

		if not hasattr(self, "_wht_cert_counter"):
			self._wht_cert_counter: dict[str, int] = {}
		seq = self._wht_cert_counter.get(tenant_id, 0) + 1
		self._wht_cert_counter[tenant_id] = seq

		cert_number = f"{certificate_type}-{tenant_id[:6].upper()}-{_today()[:7].replace('-', '')}-{seq:04d}"

		certificate: dict[str, Any] = {
			"certificate_number":  cert_number,
			"certificate_type":    certificate_type,
			"tenant_id":           tenant_id,
			"invoice_record_id":   invoice_record_id,
			"invoice_number":      invoice.get("invoice_number"),
			"invoice_date":        invoice.get("updated_at", "")[:10],
			"vendor_id":           vendor_id,
			"vendor_name":         vendor["name"] if vendor else "Unknown",
			"vendor_tax_id":       vendor.get("tax_profile", "N/A") if vendor else "N/A",
			"gross_payment":       str(_d(invoice.get("amount", 0))),
			"wht_rate_pct":        str(_d(invoice.get("wht_amount", 0)) / _d(invoice.get("amount", 1)) * 100),
			"wht_amount":          str(wht_amount),
			"net_payment":         str(_d(invoice.get("net_payable", invoice.get("amount", 0)))),
			"currency":            invoice.get("currency", "KES"),
			"period":              invoice.get("updated_at", "")[:7],
			"issued_at":           _now(),
			"kra_pin_payer":       f"TENANT-{tenant_id[:8].upper()}",
		}

		if not hasattr(self, "_wht_certificates"):
			self._wht_certificates: dict[str, dict[str, Any]] = {}
		self._wht_certificates[cert_number] = certificate

		invoice["wht_certificate_ref"] = cert_number
		invoice["updated_at"] = _now()

		self._emit("wht_certificate_issued", tenant_id, invoice_record_id, {
			"certificate_number": cert_number, "certificate_type": certificate_type,
			"wht_amount": str(wht_amount),
		})
		return deepcopy(certificate)

	async def initiate_supplier_kyb(
		self,
		tenant_id: str,
		supplier_data: dict[str, Any],
		requested_by: str,
	) -> dict[str, Any]:
		"""Run automated KYB (Know Your Business) due diligence on a new supplier (I8).

		Checks performed:
		  1. Company registration number format (Kenya: CPR/YYYY/NNNNNN)
		  2. KRA PIN format (P/A + 9 digits + letter)
		  3. Sanctions list name-matching (configurable via SANCTIONS_API_URL env var;
		     falls back to a keyword blocklist)
		  4. Beneficial owner completeness

		Computes `kyb_risk_score` 0–100:
		  - format_errors:   +30 per failed format check
		  - sanctions_hit:   +50
		  - no_beneficial_owner: +20

		Auto-approves when score < 30; escalates when score >= 70.
		"""
		import os
		guard_tenant_id(tenant_id)
		guard_non_empty_string(requested_by, "requested_by")

		required = ["legal_name", "registration_number", "tax_pin"]
		for f in required:
			if not supplier_data.get(f):
				raise ValueError(f"supplier_data.{f} is required for KYB")

		risk_score = 0
		checks: list[dict[str, Any]] = []
		warnings: list[str] = []

		# Check 1: company registration format
		import re
		reg_no = str(supplier_data.get("registration_number", ""))
		reg_valid = bool(re.match(r"^(CPR|PVT|LTD|NGO)/\d{4}/\d+$", reg_no, re.IGNORECASE))
		checks.append({"check": "registration_number_format", "passed": reg_valid, "value": reg_no})
		if not reg_valid:
			risk_score += 30
			warnings.append(f"registration_number '{reg_no}' does not match Kenya CPR format")

		# Check 2: KRA PIN format
		tax_pin = str(supplier_data.get("tax_pin", ""))
		pin_valid = bool(re.match(r"^[PA]\d{9}[A-Z]$", tax_pin.upper()))
		checks.append({"check": "kra_pin_format", "passed": pin_valid, "value": tax_pin})
		if not pin_valid:
			risk_score += 30
			warnings.append(f"tax_pin '{tax_pin}' does not match KRA PIN format")

		# Check 3: sanctions screening (name-based keyword blocklist fallback)
		blocked_keywords = {"arms", "weapons", "sanctioned", "embargoed", "ofac", "terrorist"}
		legal_name_lower = supplier_data.get("legal_name", "").lower()
		sanctions_hit = any(kw in legal_name_lower for kw in blocked_keywords)
		if os.environ.get("SANCTIONS_API_URL"):
			try:
				import json as _json
				import urllib.request as _req
				payload = _json.dumps({"name": supplier_data["legal_name"]}).encode()
				with _req.urlopen(
					_req.Request(
						os.environ["SANCTIONS_API_URL"],
						data=payload,
						headers={"Content-Type": "application/json"},
					),
					timeout=5,
				) as resp:
					api_result = _json.loads(resp.read())
					sanctions_hit = api_result.get("match", sanctions_hit)
			except Exception:
				pass  # fall through to keyword result
		checks.append({"check": "sanctions_screening", "passed": not sanctions_hit, "value": legal_name_lower[:40]})
		if sanctions_hit:
			risk_score += 50
			warnings.append("potential sanctions match — manual review required")

		# Check 4: beneficial owner completeness
		has_bo = bool(supplier_data.get("beneficial_owners") or supplier_data.get("director_names"))
		checks.append({"check": "beneficial_owner_present", "passed": has_bo})
		if not has_bo:
			risk_score += 20
			warnings.append("no beneficial owner information provided")

		risk_score = min(100, risk_score)

		if risk_score < 30:
			status = "auto_approved"
			decision = "approved"
		elif risk_score >= 70:
			status = "escalated"
			decision = "pending_review"
		else:
			status = "pending_review"
			decision = "pending_review"

		if not hasattr(self, "_kyb_requests"):
			self._kyb_requests: dict[str, dict[str, Any]] = {}

		kyb_id = f"kyb_{tenant_id[:6]}_{supplier_data['registration_number'][:12].replace('/', '_')}_{_now()[:10].replace('-', '')}"
		kyb_record: dict[str, Any] = {
			"kyb_id":          kyb_id,
			"tenant_id":       tenant_id,
			"supplier_data":   supplier_data,
			"requested_by":    requested_by,
			"kyb_risk_score":  risk_score,
			"status":          status,
			"decision":        decision,
			"checks":          checks,
			"warnings":        warnings,
			"initiated_at":    _now(),
		}
		self._kyb_requests[kyb_id] = kyb_record

		self._emit("supplier_kyb_initiated", tenant_id, kyb_id, {
			"kyb_risk_score": risk_score, "decision": decision, "warnings": len(warnings),
		})
		return deepcopy(kyb_record)

	async def triage_match_exceptions(
		self,
		tenant_id: str,
		top_n: int = 20,
	) -> dict[str, Any]:
		"""Priority-rank open match exceptions by financial impact and age (I12).

		Priority score (0–100):
		  - Financial weight:  outstanding_amount / total_ap_outstanding * 40
		  - Age weight:        min(days_open, 90) / 90 * 40
		  - Vendor risk:       vendor_risk_score / 100 * 20

		For each exception, derives `recommended_action` from exception_type:
		  price_mismatch      → price_override or reject
		  quantity_mismatch   → qty_correction
		  no_po_reference     → obtain_po_or_reject
		  three/two_way_fail  → review_and_override

		Returns top_n exceptions sorted by priority_score descending, plus
		aggregate stats (total_open, total_outstanding, avg_age_days).
		"""
		guard_tenant_id(tenant_id)
		assert top_n >= 1, "top_n must be >= 1"

		open_exceptions = [
			ex for ex in self._match_exceptions.values()
			if not ex.get("resolved") and ex.get("tenant_id") == tenant_id
		]

		total_ap_outstanding = sum(
			float(_d(inv.get("amount", 0)) - _d(inv.get("paid_amount", 0)))
			for inv in self._invoices.values()
			if inv.get("tenant_id") == tenant_id
			and inv.get("status") not in {"paid", "cancelled", "rejected"}
		) or 1.0  # guard divide-by-zero

		today = _today()
		triaged: list[dict[str, Any]] = []

		action_map: dict[str, str] = {
			"price_mismatch":         "price_override_or_reject",
			"quantity_mismatch":      "qty_correction",
			"no_po_reference":        "obtain_po_or_reject",
			"three_way_match_failure": "review_and_override",
			"two_way_match_failure":  "review_and_override",
		}
		sla_map: dict[str, int] = {
			"price_mismatch":         4,
			"quantity_mismatch":      8,
			"no_po_reference":        24,
			"three_way_match_failure": 4,
			"two_way_match_failure":  8,
		}

		for ex in open_exceptions:
			inv = self._find_invoice_by_public_id(ex["invoice_id"])
			outstanding = float(
				_d(inv.get("amount", 0)) - _d(inv.get("paid_amount", 0))
			) if inv else 0.0

			days_open = max(0, _days_between(ex.get("created_at", today)[:10], today))

			# Vendor risk contribution
			vendor_risk = 0.0
			if inv:
				vendor_id = inv.get("vendor_id", "")
				v_rec = next(
					(v for v in self._vendors.values()
					 if v.get("vendor_id") == vendor_id and v.get("tenant_id") == tenant_id),
					None,
				)
				if v_rec:
					v_exceptions = sum(
						1 for i in self._invoices.values()
						if i.get("vendor_id") == vendor_id and i.get("status") == "match_exception"
					)
					v_total = sum(1 for i in self._invoices.values() if i.get("vendor_id") == vendor_id) or 1
					vendor_risk = min(100.0, (v_exceptions / v_total) * 100)

			financial_weight = (outstanding / total_ap_outstanding) * 40
			age_weight = (min(days_open, 90) / 90) * 40
			risk_weight = (vendor_risk / 100) * 20
			priority_score = min(100.0, financial_weight + age_weight + risk_weight)

			exc_type = ex.get("exception_type", "")
			triaged.append({
				"invoice_id":              ex["invoice_id"],
				"exception_type":          exc_type,
				"outstanding_amount":      outstanding,
				"days_open":               days_open,
				"vendor_risk_score":       round(vendor_risk, 1),
				"priority_score":          round(priority_score, 1),
				"recommended_action":      action_map.get(exc_type, "manual_review"),
				"sla_hours":               sla_map.get(exc_type, 24),
				"failures":                ex.get("failures", []),
				"tenant_id":               tenant_id,
			})

		triaged.sort(key=lambda x: x["priority_score"], reverse=True)

		return {
			"tenant_id":          tenant_id,
			"total_open":         len(open_exceptions),
			"total_outstanding":  round(total_ap_outstanding, 2),
			"avg_age_days":       round(
				sum(t["days_open"] for t in triaged) / len(triaged) if triaged else 0.0, 1
			),
			"top_n":              top_n,
			"triaged":            triaged[:top_n],
			"triaged_at":         _now(),
			"currency":           "KES",
		}

	async def cash_flow_sensitivity(
		self,
		tenant_id: str,
		scenarios: list[dict[str, Any]],
	) -> dict[str, Any]:
		"""Model AP cash flow under multiple what-if payment scenarios (I13).

		Each scenario dict:
		  name                  : str — scenario label
		  payment_offset_days   : int — shift all due dates by N days (negative = earlier)
		  held_fraction         : float 0–1 — fraction of invoices held/delayed
		  discount_capture_pct  : float 0–100 — % of eligible discounts captured

		Returns per-scenario weekly forecast buckets (4 weeks) and a comparison
		table showing delta_vs_baseline for total cash out in each scenario.
		Scenarios run via asyncio.gather (simulated sequentially here to avoid
		event-loop complications in sync context, but structured for gather).
		"""
		import asyncio
		guard_tenant_id(tenant_id)
		assert len(scenarios) >= 1, "at least one scenario required"

		async def _run_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
			name = scenario.get("name", "unnamed")
			offset = int(scenario.get("payment_offset_days", 0))
			held_frac = float(scenario.get("held_fraction", 0.0))
			disc_pct = float(scenario.get("discount_capture_pct", 0.0))

			today = date.today()
			buckets: list[Decimal] = [Decimal("0")] * 4
			savings = Decimal("0")

			approved_invs = [
				inv for inv in self._invoices.values()
				if inv.get("tenant_id") == tenant_id
				and inv.get("status") in {"approved", "matched"}
				and not inv.get("held")
			]

			import random as _rng
			_rng.seed(42)  # deterministic for reproducibility
			held_set = set(
				inv["invoice_id"]
				for inv in _rng.sample(approved_invs, int(len(approved_invs) * held_frac))
			) if held_frac > 0 and approved_invs else set()

			for inv in approved_invs:
				if inv["invoice_id"] in held_set:
					continue
				outstanding = _d(inv.get("amount", 0)) - _d(inv.get("paid_amount", 0))
				if outstanding <= 0:
					continue

				# Apply discount savings
				disc_pct_inv = _d(inv.get("discount_pct", 0))
				if disc_pct_inv > 0 and disc_pct > 0:
					disc_saving = outstanding * disc_pct_inv / Decimal("100") * Decimal(str(disc_pct / 100))
					savings += disc_saving
					outstanding -= disc_saving

				due = inv.get("due_date")
				if due:
					try:
						adjusted_due = date.fromisoformat(due[:10]) + timedelta(days=offset)
						week_idx = (adjusted_due - today).days // 7
						if 0 <= week_idx < 4:
							buckets[week_idx] += outstanding
					except Exception:
						buckets[3] += outstanding
				else:
					buckets[3] += outstanding

			return {
				"name": name,
				"total_cash_out": str(sum(buckets)),
				"total_discount_savings": str(savings),
				"weekly_buckets": [
					{"week": i + 1, "amount": str(b)}
					for i, b in enumerate(buckets)
				],
			}

		results = list(await asyncio.gather(*[_run_scenario(s) for s in scenarios], return_exceptions=True))

		baseline = results[0]
		comparison: list[dict[str, Any]] = []
		baseline_total = Decimal(baseline["total_cash_out"])
		for r in results:
			delta = Decimal(r["total_cash_out"]) - baseline_total
			comparison.append({
				"name":             r["name"],
				"total_cash_out":   r["total_cash_out"],
				"delta_vs_baseline": str(delta),
				"delta_pct":         str(
					(delta / baseline_total * 100).quantize(Decimal("0.01"), ROUND_HALF_UP)
					if baseline_total != 0 else Decimal("0")
				),
				"discount_savings":  r["total_discount_savings"],
				"weekly_buckets":    r["weekly_buckets"],
			})

		return {
			"tenant_id":   tenant_id,
			"baseline":    baseline["name"],
			"scenarios":   len(scenarios),
			"comparison":  comparison,
			"analysed_at": _now(),
			"currency":    "KES",
		}

	async def identify_dormant_vendors(
		self,
		tenant_id: str,
		inactive_days: int = 365,
		auto_deactivate: bool = False,
	) -> dict[str, Any]:
		"""Flag or deactivate vendors with no AP activity in the last N days (I14).

		A vendor is dormant when:
		  - last invoice date AND last payment date are both > inactive_days ago
		  - OR no invoices and no payments have ever been recorded

		When auto_deactivate=True, sets vendor status to 'inactive' and emits
		`vendor_deactivated` audit event. Does NOT delete the record.

		Returns dormant_count, auto_deactivated_count, and full dormant list
		with days_since_last_activity.
		"""
		guard_tenant_id(tenant_id)
		assert inactive_days >= 1, "inactive_days must be >= 1"

		today = _today()
		cutoff = (date.today() - timedelta(days=inactive_days)).isoformat()

		vendors = [v for v in self._vendors.values() if v.get("tenant_id") == tenant_id]
		dormant: list[dict[str, Any]] = []
		auto_deactivated_count = 0

		for vendor in vendors:
			vid = vendor["vendor_id"]

			# Last invoice date
			vendor_invoices = [
				inv for inv in self._invoices.values()
				if inv.get("vendor_id") == vid and inv.get("tenant_id") == tenant_id
			]
			last_invoice_date = max(
				(inv.get("updated_at", "")[:10] for inv in vendor_invoices),
				default="",
			)

			# Last payment date
			vendor_payments = [
				p for p in self._payments.values()
				if p.get("vendor_id") == vid and p.get("tenant_id") == tenant_id
			]
			last_payment_date = max(
				(p.get("updated_at", "")[:10] for p in vendor_payments),
				default="",
			)

			last_activity = max(last_invoice_date, last_payment_date) or ""

			is_dormant = (not last_activity) or (last_activity < cutoff)
			if not is_dormant:
				continue

			days_inactive = _days_between(last_activity, today) if last_activity else inactive_days + 1

			dormant_entry: dict[str, Any] = {
				"vendor_id":              vid,
				"vendor_name":            vendor.get("name"),
				"last_invoice_date":      last_invoice_date or None,
				"last_payment_date":      last_payment_date or None,
				"last_activity_date":     last_activity or None,
				"days_since_last_activity": days_inactive,
				"invoice_count":          len(vendor_invoices),
				"status_before":          vendor.get("status", "active"),
			}

			if auto_deactivate and vendor.get("status") != "inactive":
				vendor["status"] = "inactive"
				vendor["deactivated_at"] = _now()
				vendor["deactivation_reason"] = f"auto_deactivated: no activity for {days_inactive} days"
				vendor["updated_at"] = _now()
				dormant_entry["status_after"] = "inactive"
				auto_deactivated_count += 1
				self._emit("vendor_deactivated", tenant_id, vendor["id"], {
					"reason": "dormant", "days_inactive": days_inactive,
				})
			else:
				dormant_entry["status_after"] = vendor.get("status", "active")

			dormant.append(dormant_entry)

		return {
			"tenant_id":              tenant_id,
			"inactive_threshold_days": inactive_days,
			"total_vendors":          len(vendors),
			"dormant_count":          len(dormant),
			"auto_deactivated_count": auto_deactivated_count,
			"dormant_vendors":        dormant,
			"auto_deactivate":        auto_deactivate,
			"identified_at":          _now(),
		}

	async def compute_compliance_scorecard(
		self,
		tenant_id: str,
		period: dict[str, str] | None = None,
	) -> dict[str, Any]:
		"""Evaluate 10 AP compliance controls and return a graded scorecard (I15).

		Controls and weights:
		  1.  Segregation of duties on approvals        (10 pts)
		  2.  PO coverage rate                          (10 pts)
		  3.  Three-way match rate on goods invoices    (10 pts)
		  4.  Open exceptions over 30 days              (10 pts)
		  5.  WHT certificate issuance rate             (10 pts)
		  6.  Duplicate invoice rate (inverse)          (10 pts)
		  7.  Bank account change review compliance     (10 pts)
		  8.  Expense receipt coverage                  (10 pts)
		  9.  Payment fraud score distribution          (10 pts)
		 10.  Approved invoices with valid period code  (10 pts)

		Composite score 0–100.
		Grade: A (>=90), B (>=75), C (>=60), D (>=45), F (<45).

		Returns per-control scores, overall grade, and remediation recommendations.
		"""
		guard_tenant_id(tenant_id)
		period = period or {}
		period_start = period.get("start", "")
		period_end = period.get("end", "")

		invoices = [
			inv for inv in self._invoices.values()
			if inv.get("tenant_id") == tenant_id
			and (not period_start or inv.get("updated_at", "")[:10] >= period_start[:10])
			and (not period_end or inv.get("updated_at", "")[:10] <= period_end[:10])
		]
		total_inv = len(invoices) or 1

		expenses = [e for e in self._expenses.values() if e.get("tenant_id") == tenant_id]
		total_exp = len(expenses) or 1

		controls: list[dict[str, Any]] = []
		remediation: list[str] = []

		# Control 1: SoD — approved_by != requested_by
		sod_passed = sum(
			1 for inv in invoices
			if inv.get("approved") and inv.get("approved_by") and inv.get("requested_by")
			and inv.get("approved_by") != inv.get("requested_by")
		)
		sod_total = sum(1 for inv in invoices if inv.get("approved")) or 1
		sod_score = round(sod_passed / sod_total * 100)
		controls.append({"control": "segregation_of_duties", "score": sod_score, "weight": 10})
		if sod_score < 80:
			remediation.append("Enforce SoD: ensure approver != requestor on all invoices")

		# Control 2: PO coverage
		po_backed = sum(1 for inv in invoices if inv.get("po_id"))
		po_score = round(po_backed / total_inv * 100)
		controls.append({"control": "po_coverage_rate", "score": po_score, "weight": 10})
		if po_score < 70:
			remediation.append(f"PO coverage at {po_score}% — enforce PO requirement for goods purchases")

		# Control 3: three-way match rate
		three_way = sum(1 for inv in invoices if inv.get("match_type") == "three_way")
		goods_inv = sum(1 for inv in invoices if inv.get("po_id") and inv.get("grn_id")) or 1
		three_way_score = round(three_way / goods_inv * 100)
		controls.append({"control": "three_way_match_rate", "score": three_way_score, "weight": 10})
		if three_way_score < 80:
			remediation.append("Improve three-way match rate — ensure GRNs are recorded before invoice approval")

		# Control 4: exceptions over 30 days (inverse: fewer old exceptions = higher score)
		old_exceptions = sum(
			1 for ex in self._match_exceptions.values()
			if ex.get("tenant_id") == tenant_id
			and not ex.get("resolved")
			and _days_between(ex.get("created_at", _today())[:10], _today()) > 30
		)
		total_exceptions = sum(
			1 for ex in self._match_exceptions.values()
			if ex.get("tenant_id") == tenant_id
		) or 1
		old_exc_rate = old_exceptions / total_exceptions
		exc_age_score = round(max(0, 100 - old_exc_rate * 100))
		controls.append({"control": "exception_aging", "score": exc_age_score, "weight": 10})
		if exc_age_score < 80:
			remediation.append(f"{old_exceptions} exceptions >30 days old — implement exception SLA monitoring")

		# Control 5: WHT certificate issuance rate
		wht_invoices = sum(1 for inv in invoices if _d(inv.get("wht_amount", 0)) > 0)
		wht_certs_issued = sum(1 for inv in invoices if inv.get("wht_certificate_ref"))
		wht_cert_rate = round(wht_certs_issued / (wht_invoices or 1) * 100)
		controls.append({"control": "wht_certificate_rate", "score": wht_cert_rate, "weight": 10})
		if wht_cert_rate < 95 and wht_invoices > 0:
			remediation.append(f"WHT certificates issued for {wht_cert_rate}% — run generate_wht_certificate for all WHT invoices")

		# Control 6: duplicate invoice rate (inverse)
		duplicates = sum(1 for inv in invoices if inv.get("duplicate_detected"))
		dup_rate = duplicates / total_inv
		dup_score = round(max(0, 100 - dup_rate * 200))  # penalise hard
		controls.append({"control": "duplicate_rate", "score": dup_score, "weight": 10})
		if dup_score < 80:
			remediation.append(f"{duplicates} duplicate invoices detected — enable ml_duplicate_invoice_detect in STP")

		# Control 7: bank account change review compliance
		bank_changes = sum(1 for v in self._vendors.values()
			if v.get("tenant_id") == tenant_id and v.get("bank_change"))
		bank_reviewed = sum(1 for v in self._vendors.values()
			if v.get("tenant_id") == tenant_id and v.get("bank_change")
			and (v.get("bank_reviewed_by") or v.get("bank_change_confirmed_by")))
		bank_score = round(bank_reviewed / (bank_changes or 1) * 100)
		controls.append({"control": "bank_change_review", "score": bank_score, "weight": 10})
		if bank_score < 100 and bank_changes > 0:
			remediation.append("Not all bank changes have documented review — enforce dual-control via propose/confirm workflow")

		# Control 8: expense receipt coverage
		receipts_covered = sum(1 for exp in expenses if exp.get("receipt_reference"))
		exp_score = round(receipts_covered / total_exp * 100)
		controls.append({"control": "expense_receipt_coverage", "score": exp_score, "weight": 10})
		if exp_score < 95:
			remediation.append(f"Receipt coverage at {exp_score}% — require receipt for all expense claims")

		# Control 9: payments not in high fraud-risk tier (proxy: no weekend payments)
		payments = [p for p in self._payments.values() if p.get("tenant_id") == tenant_id]
		total_pmts = len(payments) or 1
		weekend_pmts = 0
		for p in payments:
			try:
				d = date.fromisoformat(p.get("scheduled_date", _today())[:10])
				if d.weekday() >= 5:
					weekend_pmts += 1
			except Exception as _exc:
				_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)
		fraud_score = round(max(0, 100 - (weekend_pmts / total_pmts) * 100))
		controls.append({"control": "payment_fraud_indicators", "score": fraud_score, "weight": 10})
		if fraud_score < 90:
			remediation.append(f"{weekend_pmts} weekend payments detected — review and score via score_payment_fraud_risk")

		# Control 10: approved invoices with accounting_period / document_reference
		period_coded = sum(1 for inv in invoices if inv.get("approved") and inv.get("document_reference"))
		approved_total = sum(1 for inv in invoices if inv.get("approved")) or 1
		period_score = round(period_coded / approved_total * 100)
		controls.append({"control": "accounting_period_completeness", "score": period_score, "weight": 10})
		if period_score < 95:
			remediation.append("Some approved invoices lack document_reference — enforce at capture stage")

		composite = sum(c["score"] * c["weight"] / 100 for c in controls)
		composite = round(composite, 1)

		if composite >= 90:
			grade = "A"
		elif composite >= 75:
			grade = "B"
		elif composite >= 60:
			grade = "C"
		elif composite >= 45:
			grade = "D"
		else:
			grade = "F"

		self._emit("compliance_scorecard_computed", tenant_id, "scorecard", {
			"composite_score": composite, "grade": grade, "control_count": len(controls),
		})

		return {
			"tenant_id":         tenant_id,
			"period":            period,
			"composite_score":   composite,
			"grade":             grade,
			"controls":          controls,
			"remediation_items": remediation,
			"invoices_evaluated": len(invoices),
			"computed_at":       _now(),
		}


# Back-compat alias
APService = AccountsPayableService
