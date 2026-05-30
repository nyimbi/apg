"""Dependency-light Accounts Receivable lifecycle service."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import uuid4

try:
	from .capability_contract import (
		ARC_EVENT_STREAM,
		STREAMING,
		SUPPORTED_ARC_AGENT_ROLES,
		SUPPORTED_ARC_AGENT_RUNTIMES,
		SUPPORTED_CUSTOMER_TYPES,
		SUPPORTED_DISPUTE_REASONS,
		SUPPORTED_PAYMENT_METHODS,
		evaluate_capability_rules,
	)
except ImportError:  # pragma: no cover - supports direct file loading in tests
	from capability_contract import (  # type: ignore
		ARC_EVENT_STREAM,
		STREAMING,
		SUPPORTED_ARC_AGENT_ROLES,
		SUPPORTED_ARC_AGENT_RUNTIMES,
		SUPPORTED_CUSTOMER_TYPES,
		SUPPORTED_DISPUTE_REASONS,
		SUPPORTED_PAYMENT_METHODS,
		evaluate_capability_rules,
	)


class AccountsReceivableService:
	"""In-memory executable service for the ARC lifecycle packet."""

	def __init__(self, tenant_id: str | None = None, user_id: str | None = None) -> None:
		self.tenant_id = tenant_id
		self.user_id = user_id
		self.customers: dict[str, dict[str, Any]] = {}
		self.credit_assessments: dict[str, dict[str, Any]] = {}
		self.invoices: dict[str, dict[str, Any]] = {}
		self.payments: dict[str, dict[str, Any]] = {}
		self.cash_applications: dict[str, dict[str, Any]] = {}
		self.collection_activities: dict[str, dict[str, Any]] = {}
		self.disputes: dict[str, dict[str, Any]] = {}
		self.agents: dict[str, dict[str, Any]] = {}
		self._audit_events: list[dict[str, Any]] = []

	def _tenant(self, tenant_id: str | None = None) -> str:
		value = tenant_id or self.tenant_id
		if not value:
			raise PermissionError("tenant_context_required")
		return value

	def _record_id(self, prefix: str, explicit: str | None = None) -> str:
		return explicit or f"{prefix}-{uuid4().hex[:12]}"

	def _now(self) -> str:
		return datetime.utcnow().isoformat(timespec="seconds") + "Z"

	def _assert_rules(self, context: dict[str, Any]) -> None:
		result = evaluate_capability_rules(context)
		if result["decision"] != "allow":
			raise PermissionError(",".join(effect["reason"] for effect in result["effects"]))

	def _base_context(self, tenant_id: str, operation: str) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"tenant_context_present": True,
			"operation": operation,
			"operation_type": "write",
			"policy_attached": True,
		}

	def _emit(self, tenant_id: str, event_type: str, record: dict[str, Any]) -> None:
		self._audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"record_id": record["id"],
			"record_type": record["type"],
			"status": record["status"],
			"stream": ARC_EVENT_STREAM,
			"processor": "bytewax",
			"emitted_at": self._now(),
		})

	def create_customer(
		self,
		customer_id: str,
		tenant_id: str,
		customer_code: str,
		legal_name: str,
		customer_type: str,
		currency: str = "USD",
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		context = self._base_context(tenant, "create_customer")
		context.update({
			"customer_code_present": bool(customer_code),
			"legal_name_present": bool(legal_name),
			"customer_type_supported": customer_type in SUPPORTED_CUSTOMER_TYPES,
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("cust", customer_id),
			"type": "ar_customer",
			"tenant_id": tenant,
			"customer_code": customer_code,
			"legal_name": legal_name,
			"customer_type": customer_type,
			"currency": currency,
			"credit_limit": Decimal("0"),
			"credit_score": None,
			"credit_hold": False,
			"status": "active",
			"created_at": self._now(),
		}
		self.customers[record["id"]] = record
		self._emit(tenant, "customer_created", record)
		return deepcopy(record)

	def assess_credit(
		self,
		assessment_id: str,
		tenant_id: str,
		customer_id: str,
		credit_limit: float | int | Decimal,
		credit_score: float,
		reviewed_by: str | None = None,
		credit_hold: bool = False,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		customer = self.customers.get(customer_id)
		context = self._base_context(tenant, "assess_credit")
		context.update({
			"customer_present": bool(customer and customer["tenant_id"] == tenant),
			"credit_limit_present": credit_limit is not None,
			"credit_score": credit_score,
			"credit_review_recorded": bool(reviewed_by),
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("credit", assessment_id),
			"type": "credit_assessment",
			"tenant_id": tenant,
			"customer_id": customer_id,
			"credit_limit": Decimal(str(credit_limit)),
			"credit_score": credit_score,
			"credit_hold": credit_hold,
			"reviewed_by": reviewed_by,
			"status": "reviewed" if reviewed_by else "assessed",
			"created_at": self._now(),
		}
		self.credit_assessments[record["id"]] = record
		customer["credit_limit"] = Decimal(str(credit_limit))
		customer["credit_score"] = credit_score
		customer["credit_hold"] = credit_hold
		self._emit(tenant, "credit_assessed", record)
		return deepcopy(record)

	def create_invoice(
		self,
		invoice_id: str,
		tenant_id: str,
		customer_id: str,
		invoice_number: str,
		invoice_date: str,
		due_date: str,
		lines: list[dict[str, Any]],
		currency: str = "USD",
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		customer = self.customers.get(customer_id)
		total = sum(Decimal(str(line.get("quantity", 0))) * Decimal(str(line.get("unit_price", 0))) for line in lines)
		context = self._base_context(tenant, "create_invoice")
		context.update({
			"customer_present": bool(customer and customer["tenant_id"] == tenant),
			"invoice_number_present": bool(invoice_number),
			"invoice_dates_present": bool(invoice_date and due_date),
			"due_date_valid": bool(invoice_date and due_date and due_date >= invoice_date),
			"invoice_line_count": len(lines),
			"invoice_total": total,
		})
		self._assert_rules(context)
		for line in lines:
			if (
				not line.get("description")
				or Decimal(str(line.get("quantity", 0))) <= 0
				or Decimal(str(line.get("unit_price", -1))) < 0
				or not line.get("revenue_account")
			):
				raise PermissionError("invoice_line_invalid")
		record = {
			"id": self._record_id("inv", invoice_id),
			"type": "ar_invoice",
			"tenant_id": tenant,
			"customer_id": customer_id,
			"invoice_number": invoice_number,
			"invoice_date": invoice_date,
			"due_date": due_date,
			"currency": currency,
			"lines": deepcopy(lines),
			"total_amount": total,
			"paid_amount": Decimal("0"),
			"outstanding_amount": total,
			"approved_by": None,
			"status": "draft",
			"created_at": self._now(),
		}
		self.invoices[record["id"]] = record
		self._emit(tenant, "invoice_created", record)
		return deepcopy(record)

	def issue_invoice(self, invoice_id: str, tenant_id: str, approved_by: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		invoice = self.invoices.get(invoice_id)
		customer = self.customers.get(invoice["customer_id"]) if invoice else None
		context = self._base_context(tenant, "issue_invoice")
		context.update({
			"credit_hold": bool(customer and customer.get("credit_hold")),
			"approval_recorded": bool(approved_by),
		})
		self._assert_rules(context)
		if not invoice or invoice["tenant_id"] != tenant:
			raise PermissionError("invoice_required")
		invoice["approved_by"] = approved_by
		invoice["status"] = "issued"
		invoice["issued_at"] = self._now()
		self._emit(tenant, "invoice_issued", invoice)
		return deepcopy(invoice)

	def record_payment(
		self,
		payment_id: str,
		tenant_id: str,
		customer_id: str,
		payment_reference: str,
		payment_date: str,
		amount: float | int | Decimal,
		method: str,
		cash_account_id: str,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		customer = self.customers.get(customer_id)
		context = self._base_context(tenant, "record_payment")
		context.update({
			"customer_present": bool(customer and customer["tenant_id"] == tenant),
			"payment_reference_present": bool(payment_reference),
			"payment_date_present": bool(payment_date),
			"payment_amount": amount,
			"payment_method_supported": method in SUPPORTED_PAYMENT_METHODS,
			"cash_account_present": bool(cash_account_id),
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("pay", payment_id),
			"type": "ar_payment",
			"tenant_id": tenant,
			"customer_id": customer_id,
			"payment_reference": payment_reference,
			"payment_date": payment_date,
			"amount": Decimal(str(amount)),
			"unapplied_amount": Decimal(str(amount)),
			"method": method,
			"cash_account_id": cash_account_id,
			"status": "recorded",
			"created_at": self._now(),
		}
		self.payments[record["id"]] = record
		self._emit(tenant, "payment_recorded", record)
		return deepcopy(record)

	def apply_cash(
		self,
		application_id: str,
		tenant_id: str,
		payment_id: str,
		invoice_id: str,
		allocation_amount: float | int | Decimal,
		reviewed_by: str | None = None,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		payment = self.payments.get(payment_id)
		invoice = self.invoices.get(invoice_id)
		allocation = Decimal(str(allocation_amount))
		overapplication = bool(invoice and allocation > invoice["outstanding_amount"])
		unapplied_after = (payment["unapplied_amount"] - allocation) if payment else Decimal("0")
		context = self._base_context(tenant, "apply_cash")
		context.update({
			"payment_present": bool(payment and payment["tenant_id"] == tenant),
			"invoice_present": bool(invoice and invoice["tenant_id"] == tenant),
			"allocation_amount": allocation,
			"overapplication": overapplication,
			"unapplied_amount": unapplied_after,
			"cash_application_review_recorded": bool(reviewed_by),
		})
		self._assert_rules(context)
		if allocation > payment["unapplied_amount"]:
			raise PermissionError("payment_unapplied_amount_exceeded")
		invoice["paid_amount"] += allocation
		invoice["outstanding_amount"] -= allocation
		invoice["status"] = "paid" if invoice["outstanding_amount"] == 0 else "partially_paid"
		payment["unapplied_amount"] = unapplied_after
		payment["status"] = "applied" if unapplied_after == 0 else "partially_applied"
		record = {
			"id": self._record_id("apply", application_id),
			"type": "cash_application",
			"tenant_id": tenant,
			"payment_id": payment_id,
			"invoice_id": invoice_id,
			"allocation_amount": allocation,
			"reviewed_by": reviewed_by,
			"status": "applied",
			"created_at": self._now(),
		}
		self.cash_applications[record["id"]] = record
		self._emit(tenant, "cash_applied", record)
		return deepcopy(record)

	def record_collection_activity(
		self,
		activity_id: str,
		tenant_id: str,
		invoice_id: str,
		contact_method: str,
		priority: str,
		outcome: str,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		invoice = self.invoices.get(invoice_id)
		overdue = bool(invoice and invoice["tenant_id"] == tenant and invoice["status"] in {"issued", "partially_paid", "disputed"})
		context = self._base_context(tenant, "record_collection_activity")
		context.update({
			"overdue_invoice_present": overdue,
			"contact_method_present": bool(contact_method),
			"priority_present": bool(priority),
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("collect", activity_id),
			"type": "collection_activity",
			"tenant_id": tenant,
			"invoice_id": invoice_id,
			"contact_method": contact_method,
			"priority": priority,
			"outcome": outcome,
			"status": "recorded",
			"created_at": self._now(),
		}
		self.collection_activities[record["id"]] = record
		self._emit(tenant, "collection_activity_recorded", record)
		return deepcopy(record)

	def open_dispute(self, dispute_id: str, tenant_id: str, invoice_id: str, reason: str, owner: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		invoice = self.invoices.get(invoice_id)
		context = self._base_context(tenant, "open_dispute")
		context.update({
			"invoice_present": bool(invoice and invoice["tenant_id"] == tenant),
			"dispute_reason_supported": reason in SUPPORTED_DISPUTE_REASONS,
			"owner_present": bool(owner),
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("dispute", dispute_id),
			"type": "ar_dispute",
			"tenant_id": tenant,
			"invoice_id": invoice_id,
			"reason": reason,
			"owner": owner,
			"status": "open",
			"created_at": self._now(),
		}
		self.disputes[record["id"]] = record
		invoice["status"] = "disputed"
		self._emit(tenant, "dispute_opened", record)
		return deepcopy(record)

	def resolve_dispute(self, dispute_id: str, tenant_id: str, resolution: str, reviewed_by: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		dispute = self.disputes.get(dispute_id)
		context = self._base_context(tenant, "resolve_dispute")
		context.update({"resolution_review_recorded": bool(reviewed_by)})
		self._assert_rules(context)
		if not dispute or dispute["tenant_id"] != tenant:
			raise PermissionError("dispute_required")
		dispute["resolution"] = resolution
		dispute["reviewed_by"] = reviewed_by
		dispute["status"] = "resolved"
		invoice = self.invoices.get(dispute["invoice_id"])
		if invoice and invoice["tenant_id"] == tenant and invoice["status"] == "disputed":
			invoice["status"] = "issued" if invoice["outstanding_amount"] > 0 else "paid"
		self._emit(tenant, "dispute_resolved", dispute)
		return deepcopy(dispute)

	def register_arc_agent(self, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		context = self._base_context(tenant, "register_arc_agent")
		context.update({
			"agent_runtime_supported": runtime in SUPPORTED_ARC_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_ARC_AGENT_ROLES,
		})
		self._assert_rules(context)
		record = {
			"id": self._record_id("agent"),
			"type": "arc_agent",
			"tenant_id": tenant,
			"name": name,
			"runtime": runtime,
			"role": role,
			"scope": scope,
			"status": "active",
			"created_at": self._now(),
		}
		self.agents[record["id"]] = record
		self._emit(tenant, "arc_agent_registered", record)
		return deepcopy(record)

	def validate_agent_arc_action(
		self,
		tenant_id: str,
		agent_id: str,
		action: str,
		privileged_scope: bool,
		human_approval_recorded: bool,
	) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		agent = self.agents.get(agent_id)
		if not agent or agent["tenant_id"] != tenant:
			raise PermissionError("arc_agent_required")
		result = evaluate_capability_rules({
			"tenant_id": tenant,
			"tenant_context_present": True,
			"operation": "agent_arc_action",
			"action": action,
			"privileged_scope": privileged_scope,
			"human_approval_recorded": human_approval_recorded,
		})
		if result["decision"] != "allow":
			raise PermissionError(",".join(effect["reason"] for effect in result["effects"]))
		return result

	def validate_batch(self, tenant_id: str, event_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		self._assert_rules({
			"tenant_id": tenant,
			"tenant_context_present": True,
			"operation": "arc_batch",
			"event_stream": event_stream,
		})
		return {
			"tenant_id": tenant,
			"event_count": event_count,
			"processor": "bytewax",
			"stream": ARC_EVENT_STREAM,
		}

	def aging_summary(self, tenant_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		buckets = {"current": Decimal("0"), "overdue": Decimal("0"), "disputed": Decimal("0"), "paid": Decimal("0")}
		for invoice in self.invoices.values():
			if invoice["tenant_id"] != tenant:
				continue
			if invoice["status"] == "disputed":
				buckets["disputed"] += invoice["outstanding_amount"]
			elif invoice["status"] in {"issued", "partially_paid"}:
				buckets["overdue"] += invoice["outstanding_amount"]
			elif invoice["status"] == "paid":
				buckets["paid"] += invoice["total_amount"]
			else:
				buckets["current"] += invoice["outstanding_amount"]
		return {key: str(value) for key, value in buckets.items()}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		return {
			"tenant_id": tenant,
			"customer_count": len([record for record in self.customers.values() if record["tenant_id"] == tenant]),
			"credit_assessment_count": len([record for record in self.credit_assessments.values() if record["tenant_id"] == tenant]),
			"invoice_count": len([record for record in self.invoices.values() if record["tenant_id"] == tenant]),
			"payment_count": len([record for record in self.payments.values() if record["tenant_id"] == tenant]),
			"cash_application_count": len([record for record in self.cash_applications.values() if record["tenant_id"] == tenant]),
			"collection_activity_count": len([record for record in self.collection_activities.values() if record["tenant_id"] == tenant]),
			"dispute_count": len([record for record in self.disputes.values() if record["tenant_id"] == tenant]),
			"arc_agent_count": len([record for record in self.agents.values() if record["tenant_id"] == tenant]),
			"audit_event_count": len([event for event in self._audit_events if event["tenant_id"] == tenant]),
			"aging": self.aging_summary(tenant),
			"streaming": deepcopy(STREAMING),
		}

	def audit_events(self, tenant_id: str) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [deepcopy(event) for event in self._audit_events if event["tenant_id"] == tenant]

	def list_records(self, collection: str, tenant_id: str) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		if not hasattr(self, collection):
			raise KeyError(collection)
		store = getattr(self, collection)
		if isinstance(store, dict):
			return [deepcopy(record) for record in store.values() if record["tenant_id"] == tenant]
		if isinstance(store, list):
			return [deepcopy(record) for record in store if record["tenant_id"] == tenant]
		raise TypeError(f"{collection} is not a record collection")


ARCService = AccountsReceivableService
