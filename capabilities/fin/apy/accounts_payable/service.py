"""Domain service for APG accounts payable."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
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


class AccountsPayableService:
	"""Tenant-scoped vendor, invoice, approval, payment, expense, and close coordinator."""

	def __init__(self) -> None:
		self._vendors: dict[str, dict[str, Any]] = {}
		self._invoices: dict[str, dict[str, Any]] = {}
		self._payments: dict[str, dict[str, Any]] = {}
		self._payment_batches: dict[str, dict[str, Any]] = {}
		self._expenses: dict[str, dict[str, Any]] = {}
		self._period_closes: dict[str, dict[str, Any]] = {}
		self._agents: dict[str, dict[str, Any]] = {}
		self._audit_events: list[dict[str, Any]] = []

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
			"updated_at": self._now(),
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
			"event_stream": "bytewax",
			"updated_at": self._now(),
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
		invoice["updated_at"] = self._now()
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
		invoice["updated_at"] = self._now()
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
		invoice["updated_at"] = self._now()
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
		invoice["updated_at"] = self._now()
		return deepcopy(invoice)

	def schedule_payment(self, payment_id: str, tenant_id: str, invoice_record_id: str, amount: float, cash_account: str, scheduled_date: str) -> dict[str, Any]:
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
			"updated_at": self._now(),
		}
		self._payments[record["id"]] = record
		self._emit("payment_scheduled", tenant_id, record["id"], {"amount": amount, "cash_account": cash_account})
		return deepcopy(record)

	def release_payment_batch(self, batch_id: str, tenant_id: str, payment_record_ids: list[str], reviewed_by: str) -> dict[str, Any]:
		payments = [self._require_payment(payment_id, tenant_id) for payment_id in payment_record_ids]
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
			"payment_record_ids": [payment["id"] for payment in payments],
			"reviewed_by": reviewed_by,
			"amount": round(sum(payment["amount"] for payment in payments), 2),
			"status": "released",
			"event_stream": "bytewax",
			"updated_at": self._now(),
		}
		for payment in payments:
			payment["status"] = "paid"
			payment["updated_at"] = self._now()
			invoice = self._require_invoice(payment["invoice_record_id"], tenant_id)
			invoice["paid_amount"] = round(invoice["paid_amount"] + payment["amount"], 2)
			invoice["status"] = "paid" if invoice["paid_amount"] >= invoice["amount"] else "partially_paid"
		self._payment_batches[record["id"]] = record
		self._emit("payment_batch_released", tenant_id, record["id"], {"payment_count": len(payments), "amount": record["amount"]})
		return deepcopy(record)

	def record_expense_report(self, report_id: str, tenant_id: str, employee_id: str, amount: float, receipt_reference: str, policy_exception: bool = False, policy_reviewed_by: str | None = None) -> dict[str, Any]:
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
			"updated_at": self._now(),
		}
		self._expenses[record["id"]] = record
		self._emit("expense_report_recorded", tenant_id, record["id"], {"amount": amount})
		return deepcopy(record)

	def close_period(self, close_id: str, tenant_id: str, period: str, open_exception_count: int, unposted_invoice_count: int, aging_reviewed_by: str) -> dict[str, Any]:
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
			"updated_at": self._now(),
		}
		self._period_closes[record["id"]] = record
		self._emit("period_closed", tenant_id, record["id"], {"period": period})
		return deepcopy(record)

	def register_ap_agent(self, tenant_id: str, name: str, runtime: str, role: str, instructions: str) -> dict[str, Any]:
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
			"updated_at": self._now(),
		}
		self._agents[record["id"]] = record
		self._emit("ap_agent_registered", tenant_id, record["id"], {"runtime": runtime, "role": role})
		return deepcopy(record)

	def validate_agent_ap_action(self, tenant_id: str, agent_id: str, action: str, privileged_scope: bool, human_approval_recorded: bool) -> dict[str, Any]:
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
			"open_invoice_count": len([item for item in self.list_invoices(tenant_id) if item["status"] not in {"paid", "held"}]),
			"held_invoice_count": len([item for item in self.list_invoices(tenant_id) if item["held"]]),
			"payment_count": len(self.list_payments(tenant_id)),
			"payment_batch_count": len(self.list_payment_batches(tenant_id)),
			"expense_count": len(self.list_expenses(tenant_id)),
			"period_close_count": len(self.list_period_closes(tenant_id)),
			"ap_agent_count": len(self.list_ap_agents(tenant_id)),
			"audit_event_count": len(self.audit_events(tenant_id)),
			"streaming": streaming_manifest(),
		}

	def aging_summary(self, tenant_id: str) -> dict[str, Any]:
		invoices = [invoice for invoice in self.list_invoices(tenant_id) if invoice["status"] != "paid"]
		return {
			"tenant_id": tenant_id,
			"open_invoice_count": len(invoices),
			"open_amount": round(sum(invoice["amount"] - invoice["paid_amount"] for invoice in invoices), 2),
			"held_invoice_count": len([invoice for invoice in invoices if invoice["held"]]),
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
		return [deepcopy(event) for event in self._audit_events if event["tenant_id"] == tenant_id]

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

	def _require_vendor(self, vendor_id: str, tenant_id: str) -> dict[str, Any]:
		return self._require_record(self._vendors, vendor_id, tenant_id, "vendor", "vendor_id")

	def _require_invoice(self, invoice_id: str, tenant_id: str) -> dict[str, Any]:
		return self._require_record(self._invoices, invoice_id, tenant_id, "invoice", "invoice_id")

	def _require_payment(self, payment_id: str, tenant_id: str) -> dict[str, Any]:
		return self._require_record(self._payments, payment_id, tenant_id, "payment", "payment_id")

	def _require_record(self, records: dict[str, dict[str, Any]], record_id: str, tenant_id: str, label: str, public_key: str) -> dict[str, Any]:
		for record in records.values():
			if record["tenant_id"] == tenant_id and (record["id"] == record_id or record[public_key] == record_id):
				return record
		raise KeyError(f"Unknown {label}: {record_id}")

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
			"created_at": self._now(),
		})

	def _record_id(self, prefix: str, value: str) -> str:
		slug = "".join(character.lower() if character.isalnum() else "_" for character in str(value)).strip("_")
		return f"{prefix}_{slug or 'record'}"

	def _now(self) -> str:
		return datetime.now(timezone.utc).isoformat()


APService = AccountsPayableService
