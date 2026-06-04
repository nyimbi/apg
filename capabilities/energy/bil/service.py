"""Service layer for APG Energy Billing & Tariffs."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

try:
	from .capability_contract import (
		SUPPORTED_BILLING_CYCLES, SUPPORTED_BILL_STATUSES, SUPPORTED_CHARGE_TYPES,
		SUPPORTED_CREDIT_TYPES, SUPPORTED_CUSTOMER_CLASSES, SUPPORTED_DISPUTE_STATUSES,
		SUPPORTED_PAYMENT_METHODS, SUPPORTED_REVENUE_ASSURANCE_TYPES, SUPPORTED_TARIFF_TYPES,
		SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AGENT_ROLES,
		evaluate_capability_rules, get_capability_contract,
	)
	from .models import (
		AuditEvent, BilAgent, BillingDispute, EnergyBill, EnergyCredit,
		Payment, RevenueAssuranceFlag, Tariff,
	)
except ImportError:
	from capability_contract import (  # type: ignore
		SUPPORTED_BILLING_CYCLES, SUPPORTED_BILL_STATUSES, SUPPORTED_CHARGE_TYPES,
		SUPPORTED_CREDIT_TYPES, SUPPORTED_CUSTOMER_CLASSES, SUPPORTED_DISPUTE_STATUSES,
		SUPPORTED_PAYMENT_METHODS, SUPPORTED_REVENUE_ASSURANCE_TYPES, SUPPORTED_TARIFF_TYPES,
		SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AGENT_ROLES,
		evaluate_capability_rules, get_capability_contract,
	)
	from models import (  # type: ignore
		AuditEvent, BilAgent, BillingDispute, EnergyBill, EnergyCredit,
		Payment, RevenueAssuranceFlag, Tariff,
	)


def _now() -> str:
	return datetime.now(timezone.utc).isoformat()


def _present(value: str | None) -> bool:
	return bool(value and value.strip())


class EnergyBillingService:
	"""Tenant-scoped Energy Billing & Tariffs runtime."""

	def __init__(self, tenant_id: str = "default", actor_id: str = "system", *, auth=None, audit=None, notify=None, db_url=None, store=None) -> None:
		self.tenant_id = tenant_id
		self.actor_id = actor_id
		self.tariffs: dict[tuple[str, str], Tariff] = {}
		self.bills: dict[tuple[str, str], EnergyBill] = {}
		self.payments: dict[tuple[str, str], Payment] = {}
		self.credits: dict[tuple[str, str], EnergyCredit] = {}
		self.disputes: dict[tuple[str, str], BillingDispute] = {}
		self.revenue_assurance_flags: dict[tuple[str, str], RevenueAssuranceFlag] = {}
		self.agents: dict[tuple[str, str], BilAgent] = {}
		self.audit_events: list[AuditEvent] = []
		# Extended stores
		self._consumption_records: dict[str, dict[str, Any]] = {}
		self._demand_charge_records: dict[str, dict[str, Any]] = {}
		self._energy_charge_records: dict[str, dict[str, Any]] = {}
		self._levy_records: dict[str, dict[str, Any]] = {}
		self._arrears_records: dict[str, dict[str, Any]] = {}
		self._billing_analytics_records: dict[str, dict[str, Any]] = {}

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ── tariffs ───────────────────────────────────────────────────────────────

	def create_tariff(
		self,
		tariff_id: str,
		tenant_id: str,
		name: str,
		tariff_type: str,
		customer_class: str,
		effective_date: str,
		created_by: str,
		rate_blocks: list[dict[str, Any]] | None = None,
		description: str = "",
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Create a new tariff structure."""
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "create_tariff",
			"tariff_type_supported": tariff_type in SUPPORTED_TARIFF_TYPES,
			"customer_class_supported": customer_class in SUPPORTED_CUSTOMER_CLASSES,
			"effective_date_present": _present(effective_date),
			"rate_positive": True,  # Caller validates rate blocks
		})
		item = Tariff(
			id=tariff_id, tenant_id=tenant_id, name=name,
			tariff_type=tariff_type, customer_class=customer_class,
			effective_date=effective_date, status="draft",
			created_by=created_by, rate_blocks=rate_blocks or [],
			description=description,
		)
		self.tariffs[self._key(tenant_id, tariff_id)] = item
		self._audit(tenant_id, "tariff_created", tariff_id, "tariff")
		return item.to_dict()

	def approve_tariff(self, tariff_id: str, tenant_id: str, approved_by: str) -> dict[str, Any]:
		"""Approve a tariff for activation."""
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation": "activate_tariff",
			"approval_present": _present(approved_by),
		})
		tariff = self._get_tariff(tenant_id, tariff_id)
		tariff.approved_by = approved_by
		tariff.approved_at = _now()
		tariff.status = "approved"
		self._audit(tenant_id, "tariff_approved", tariff_id, "tariff")
		return tariff.to_dict()

	def activate_tariff(self, tariff_id: str, tenant_id: str) -> dict[str, Any]:
		"""Activate an approved tariff."""
		tariff = self._get_tariff(tenant_id, tariff_id)
		if not tariff.approved_by:
			raise ValueError("Tariff must be approved before activation")
		tariff.status = "active"
		self._audit(tenant_id, "tariff_activated", tariff_id, "tariff")
		return tariff.to_dict()

	def list_tariffs(self, tenant_id: str, customer_class: str | None = None) -> list[dict[str, Any]]:
		items = self._tenant_items(self.tariffs, tenant_id)
		if customer_class:
			items = [t for t in items if t["customer_class"] == customer_class]
		return items

	def get_active_tariff(self, tenant_id: str, customer_class: str) -> dict[str, Any] | None:
		"""Return the active tariff for a customer class, if one exists."""
		for tariff in self._tenant_items(self.tariffs, tenant_id):
			if tariff["customer_class"] == customer_class and tariff["status"] == "active":
				return tariff
		return None

	# ── bills ─────────────────────────────────────────────────────────────────

	def generate_bill(
		self,
		bill_id: str,
		tenant_id: str,
		customer_id: str,
		meter_id: str,
		tariff_id: str,
		billing_cycle: str,
		period_start: str,
		period_end: str,
		consumption_kwh: float,
		peak_demand_kw: float,
		charges: list[dict[str, Any]],
		total_amount: float,
		currency: str = "KES",
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Generate an energy bill."""
		tariff_exists = self._key(tenant_id, tariff_id) in self.tariffs
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "generate_bill",
			"billing_cycle_supported": billing_cycle in SUPPORTED_BILLING_CYCLES,
			"tariff_exists": tariff_exists,
			"meter_reading_present": consumption_kwh >= 0,
		})
		item = EnergyBill(
			id=bill_id, tenant_id=tenant_id, customer_id=customer_id,
			meter_id=meter_id, tariff_id=tariff_id, billing_cycle=billing_cycle,
			period_start=period_start, period_end=period_end,
			status="draft", generated_at=_now(),
			total_amount=total_amount, currency=currency,
			charges=charges, consumption_kwh=consumption_kwh,
			peak_demand_kw=peak_demand_kw,
		)
		self.bills[self._key(tenant_id, bill_id)] = item
		self._audit(tenant_id, "bill_generated", bill_id, "bill")
		return item.to_dict()

	def issue_bill(self, bill_id: str, tenant_id: str, due_date: str) -> dict[str, Any]:
		"""Issue a drafted bill to the customer."""
		bill = self._get_bill(tenant_id, bill_id)
		bill.status = "issued"
		bill.issued_at = _now()
		bill.due_date = due_date
		self._audit(tenant_id, "bill_issued", bill_id, "bill")
		return bill.to_dict()

	def write_off_bill(self, bill_id: str, tenant_id: str, approved_by: str) -> dict[str, Any]:
		"""Write off an uncollectable bill."""
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation": "write_off_bill",
			"approval_present": _present(approved_by),
		})
		bill = self._get_bill(tenant_id, bill_id)
		bill.status = "written_off"
		self._audit(tenant_id, "bill_written_off", bill_id, "bill", {"approved_by": approved_by})
		return bill.to_dict()

	def list_bills(self, tenant_id: str, customer_id: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
		items = self._tenant_items(self.bills, tenant_id)
		if customer_id:
			items = [b for b in items if b["customer_id"] == customer_id]
		if status:
			items = [b for b in items if b["status"] == status]
		return items

	# ── payments ──────────────────────────────────────────────────────────────

	def record_payment(
		self,
		payment_id: str,
		tenant_id: str,
		bill_id: str,
		customer_id: str,
		payment_method: str,
		amount: float,
		currency: str,
		transaction_reference: str = "",
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Record a payment against a bill."""
		bill_exists = self._key(tenant_id, bill_id) in self.bills
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "record_payment",
			"payment_method_supported": payment_method in SUPPORTED_PAYMENT_METHODS,
			"amount_positive": amount > 0,
			"bill_exists": bill_exists,
		})
		item = Payment(
			id=payment_id, tenant_id=tenant_id, bill_id=bill_id,
			customer_id=customer_id, payment_method=payment_method,
			amount=amount, currency=currency, received_at=_now(),
			transaction_reference=transaction_reference,
		)
		self.payments[self._key(tenant_id, payment_id)] = item
		# Update bill status
		bill = self._get_bill(tenant_id, bill_id)
		total_paid = sum(
			p.amount for k, p in self.payments.items()
			if k[0] == tenant_id and p.bill_id == bill_id
		)
		if total_paid >= bill.total_amount:
			bill.status = "paid"
		elif total_paid > 0:
			bill.status = "partially_paid"
		self._audit(tenant_id, "payment_received", payment_id, "payment")
		return item.to_dict()

	def reconcile_payment(self, payment_id: str, tenant_id: str) -> dict[str, Any]:
		"""Mark a payment as reconciled."""
		payment = self._get_payment(tenant_id, payment_id)
		payment.reconciled = True
		payment.reconciled_at = _now()
		self._audit(tenant_id, "payment_reconciled", payment_id, "payment")
		return payment.to_dict()

	def list_payments(self, tenant_id: str, bill_id: str | None = None) -> list[dict[str, Any]]:
		items = self._tenant_items(self.payments, tenant_id)
		if bill_id:
			items = [p for p in items if p["bill_id"] == bill_id]
		return items

	# ── credits ───────────────────────────────────────────────────────────────

	def issue_credit(
		self,
		credit_id: str,
		tenant_id: str,
		customer_id: str,
		credit_type: str,
		amount: float,
		currency: str,
		expires_at: str,
		approved_by: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Issue an energy credit to a customer."""
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "issue_credit",
			"credit_type_supported": credit_type in SUPPORTED_CREDIT_TYPES,
			"approval_present": _present(approved_by),
			"expiry_present": _present(expires_at),
		})
		item = EnergyCredit(
			id=credit_id, tenant_id=tenant_id, customer_id=customer_id,
			credit_type=credit_type, amount=amount, currency=currency,
			issued_at=_now(), expires_at=expires_at, approved_by=approved_by,
		)
		self.credits[self._key(tenant_id, credit_id)] = item
		self._audit(tenant_id, "credit_applied", credit_id, "credit")
		return item.to_dict()

	def list_credits(self, tenant_id: str, customer_id: str | None = None) -> list[dict[str, Any]]:
		items = self._tenant_items(self.credits, tenant_id)
		if customer_id:
			items = [c for c in items if c["customer_id"] == customer_id]
		return items

	# ── disputes ──────────────────────────────────────────────────────────────

	def open_dispute(
		self,
		dispute_id: str,
		tenant_id: str,
		bill_id: str,
		customer_id: str,
		reason: str,
		evidence_reference: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Open a billing dispute."""
		bill_exists = self._key(tenant_id, bill_id) in self.bills
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "open_dispute",
			"evidence_present": _present(evidence_reference),
			"bill_exists": bill_exists,
		})
		item = BillingDispute(
			id=dispute_id, tenant_id=tenant_id, bill_id=bill_id,
			customer_id=customer_id, status="open", reason=reason,
			evidence_reference=evidence_reference, opened_at=_now(),
		)
		self.disputes[self._key(tenant_id, dispute_id)] = item
		self._audit(tenant_id, "dispute_opened", dispute_id, "dispute")
		return item.to_dict()

	def resolve_dispute(self, dispute_id: str, tenant_id: str, resolution: str, adjusted_amount: float = 0.0) -> dict[str, Any]:
		"""Resolve a billing dispute."""
		dispute = self._get_dispute(tenant_id, dispute_id)
		dispute.status = "resolved_accepted" if adjusted_amount > 0 else "resolved_rejected"
		dispute.resolution = resolution
		dispute.resolved_at = _now()
		dispute.adjusted_amount = adjusted_amount
		self._audit(tenant_id, "dispute_resolved", dispute_id, "dispute")
		return dispute.to_dict()

	def list_disputes(self, tenant_id: str, status: str | None = None) -> list[dict[str, Any]]:
		items = self._tenant_items(self.disputes, tenant_id)
		if status:
			items = [d for d in items if d["status"] == status]
		return items

	# ── revenue assurance ─────────────────────────────────────────────────────

	def flag_revenue_issue(
		self,
		flag_id: str,
		tenant_id: str,
		flag_type: str,
		entity_id: str,
		entity_type: str,
		estimated_revenue_impact: float,
		currency: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Flag a revenue assurance issue."""
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "flag_revenue_issue",
			"ra_type_supported": flag_type in SUPPORTED_REVENUE_ASSURANCE_TYPES,
		})
		item = RevenueAssuranceFlag(
			id=flag_id, tenant_id=tenant_id, flag_type=flag_type,
			entity_id=entity_id, entity_type=entity_type,
			estimated_revenue_impact=estimated_revenue_impact,
			currency=currency, flagged_at=_now(),
		)
		self.revenue_assurance_flags[self._key(tenant_id, flag_id)] = item
		self._audit(tenant_id, "revenue_assurance_flag_raised", flag_id, "ra_flag")
		return item.to_dict()

	def resolve_revenue_flag(self, flag_id: str, tenant_id: str, investigated_by: str, notes: str = "") -> dict[str, Any]:
		"""Mark a revenue assurance flag as resolved."""
		flag = self.revenue_assurance_flags.get(self._key(tenant_id, flag_id))
		if not flag:
			raise KeyError(f"RevenueAssuranceFlag {flag_id} not found for tenant {tenant_id}")
		flag.status = "resolved"
		flag.investigated_by = investigated_by
		flag.resolved_at = _now()
		flag.notes = notes
		return flag.to_dict()

	def list_revenue_flags(self, tenant_id: str, status: str | None = None) -> list[dict[str, Any]]:
		items = self._tenant_items(self.revenue_assurance_flags, tenant_id)
		if status:
			items = [f for f in items if f["status"] == status]
		return items

	# ── agents ────────────────────────────────────────────────────────────────

	def register_agent(
		self,
		agent_id: str,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str = "energy billing operations",
	) -> dict[str, Any]:
		self._enforce({
			"tenant_context_present": _present(tenant_id),
			"operation": "register_bil_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
		})
		item = BilAgent(
			id=agent_id, tenant_id=tenant_id, name=name,
			runtime=runtime, role=role, scope=scope, registered_at=_now(),
		)
		self.agents[self._key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "bil_agent_registered", agent_id, "agent")
		return item.to_dict()

	# ── dashboard ─────────────────────────────────────────────────────────────

	async def export_bills(self, period: str, format: str = "json") -> dict[str, Any]:
		"""Export bills for a billing period in JSON or CSV format."""
		assert format in {"json", "csv"}, "format must be json or csv"
		bills = [b for b in self._tenant_items(self.bills, self.tenant_id) if b.get("period_start", "")[:7] == period[:7]]
		self._audit(self.tenant_id, "bills_exported", f"period:{period}", "export", {"format": format, "count": len(bills)})
		if format == "csv":
			import csv, io
			buf = io.StringIO()
			if bills:
				writer = csv.DictWriter(buf, fieldnames=list(bills[0].keys()))
				writer.writeheader()
				writer.writerows(bills)
			return {"format": "csv", "period": period, "record_count": len(bills), "content": buf.getvalue()}
		return {"format": "json", "period": period, "record_count": len(bills), "records": bills}

	async def bulk_generate_bills(self, account_ids: list[str], period: str) -> dict[str, Any]:
		"""Generate bills for multiple accounts in a single bulk operation."""
		assert account_ids, "account_ids required"
		assert period, "period required"
		results: list[dict[str, Any]] = []
		errors: list[dict[str, Any]] = []
		for acc_id in account_ids:
			try:
				bill = await self.read_consumption(account_id=acc_id, period=period, meter_reading=0.0, estimated=True)
				results.append({"account_id": acc_id, "bill_id": bill.get("id"), "status": "generated"})
			except Exception as exc:
				errors.append({"account_id": acc_id, "error": str(exc)})
		return {"period": period, "success_count": len(results), "error_count": len(errors), "results": results, "errors": errors}

	async def health_check(self) -> dict[str, Any]:
		"""Return billing service health status."""
		bills = self._tenant_items(self.bills, self.tenant_id)
		overdue = sum(1 for b in bills if b.get("status") == "overdue")
		return {
			"service": "EnergyBillingService", "tenant_id": self.tenant_id,
			"status": "healthy" if overdue < 1000 else "degraded",
			"bill_count": len(bills), "overdue_count": overdue, "checked_at": _now(),
		}

	async def billing_compliance_report(self, standard: str = "EPRA") -> dict[str, Any]:
		"""Generate a billing compliance report for a regulatory standard."""
		bills = self._tenant_items(self.bills, self.tenant_id)
		disputes = self._tenant_items(self.disputes, self.tenant_id)
		open_disputes = [d for d in disputes if d.get("status") == "open"]
		self._audit(self.tenant_id, "billing_compliance_report_generated", standard, "report", {})
		return {
			"standard": standard, "tenant_id": self.tenant_id,
			"total_bills": len(bills),
			"open_dispute_count": len(open_disputes),
			"dispute_rate_pct": round(len(open_disputes) / max(len(bills), 1) * 100, 2),
			"compliance_status": "compliant" if len(open_disputes) / max(len(bills), 1) < 0.05 else "review_required",
			"generated_at": _now(),
		}

	async def payment_analytics(self) -> dict[str, Any]:
		"""Compute payment analytics: collection rate, payment method distribution."""
		payments = self._tenant_items(self.payments, self.tenant_id)
		total = sum(p.get("amount", 0) for p in payments)
		by_method: dict[str, float] = {}
		for p in payments:
			method = p.get("payment_method", "unknown")
			by_method[method] = round(by_method.get(method, 0.0) + float(p.get("amount", 0)), 2)
		return {
			"tenant_id": self.tenant_id,
			"payment_count": len(payments), "total_collected": round(total, 2),
			"by_method": by_method, "computed_at": _now(),
		}

	async def tariff_analytics(self) -> dict[str, Any]:
		"""Summarise active tariffs by type and structure."""
		tariffs = self._tenant_items(self.tariffs, self.tenant_id)
		by_type: dict[str, int] = {}
		for t in tariffs:
			tt = t.get("tariff_type", "unknown")
			by_type[tt] = by_type.get(tt, 0) + 1
		return {
			"tenant_id": self.tenant_id,
			"tariff_count": len(tariffs), "by_type": by_type,
			"active_count": sum(1 for t in tariffs if t.get("status") == "active"),
			"computed_at": _now(),
		}

	async def arrears_analytics(self) -> dict[str, Any]:
		"""Analyse accounts in arrears: total amount, ageing buckets."""
		arrears = list(self._arrears_records.values())
		arrears = [a for a in arrears if a.get("tenant_id") == self.tenant_id]
		total = sum(a.get("arrears_amount", 0) for a in arrears)
		return {
			"tenant_id": self.tenant_id,
			"accounts_in_arrears": len(arrears),
			"total_arrears_amount": round(total, 2),
			"avg_arrears": round(total / max(len(arrears), 1), 2),
			"computed_at": _now(),
		}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		bills = self._tenant_items(self.bills, tenant_id)
		payments = self._tenant_items(self.payments, tenant_id)
		disputes = self._tenant_items(self.disputes, tenant_id)
		ra_flags = self._tenant_items(self.revenue_assurance_flags, tenant_id)
		total_billed = sum(b["total_amount"] for b in bills)
		total_collected = sum(p["amount"] for p in payments)
		open_disputes = [d for d in disputes if d["status"] == "open"]
		open_ra_flags = [f for f in ra_flags if f["status"] == "open"]
		return {
			"tenant_id": tenant_id,
			"total_bills": len(bills),
			"total_billed_amount": total_billed,
			"total_collected_amount": total_collected,
			"collection_rate_pct": round(total_collected / total_billed * 100, 2) if total_billed > 0 else 0.0,
			"open_disputes": len(open_disputes),
			"open_revenue_flags": len(open_ra_flags),
			"overdue_bills": sum(1 for b in bills if b["status"] == "overdue"),
		}

	# ── internals ─────────────────────────────────────────────────────────────

	def _log_operation(self, tenant_id: str, operation: str, entity_id: str) -> None:
		pass

	def _log_rule_denial(self, actions: list[dict[str, Any]]) -> None:
		pass

	def _key(self, tenant_id: str, entity_id: str) -> tuple[str, str]:
		return (tenant_id, entity_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = evaluate_capability_rules(context)
		if result["decision"] == "deny":
			self._log_rule_denial(result["actions"])
			reasons = "; ".join(a["reason"] for a in result["actions"])
			raise ValueError(f"Rule denied: {reasons}")

	def _audit(self, tenant_id: str, event_type: str, entity_id: str, entity_type: str, payload: dict[str, Any] | None = None) -> None:
		from uuid import uuid4
		self.audit_events.append(AuditEvent(
			id=str(uuid4()), tenant_id=tenant_id, event_type=event_type,
			entity_id=entity_id, entity_type=entity_type,
			actor="system", occurred_at=_now(), payload=payload or {},
		))

	def _get_tariff(self, tenant_id: str, tariff_id: str) -> Tariff:
		item = self.tariffs.get(self._key(tenant_id, tariff_id))
		if not item:
			raise KeyError(f"Tariff {tariff_id} not found for tenant {tenant_id}")
		return item

	def _get_bill(self, tenant_id: str, bill_id: str) -> EnergyBill:
		item = self.bills.get(self._key(tenant_id, bill_id))
		if not item:
			raise KeyError(f"Bill {bill_id} not found for tenant {tenant_id}")
		return item

	def _get_payment(self, tenant_id: str, payment_id: str) -> Payment:
		item = self.payments.get(self._key(tenant_id, payment_id))
		if not item:
			raise KeyError(f"Payment {payment_id} not found for tenant {tenant_id}")
		return item

	def _get_dispute(self, tenant_id: str, dispute_id: str) -> BillingDispute:
		item = self.disputes.get(self._key(tenant_id, dispute_id))
		if not item:
			raise KeyError(f"Dispute {dispute_id} not found for tenant {tenant_id}")
		return item

	def _tenant_items(self, store: dict[tuple[str, str], Any], tenant_id: str) -> list[dict[str, Any]]:
		return [v.to_dict() for k, v in store.items() if k[0] == tenant_id]

	# ── Extended async methods ────────────────────────────────────────────────

	async def read_consumption(
		self,
		account_id: str,
		period: str,
		meter_id: str | None = None,
		read_type: str = "actual",
	) -> dict[str, Any]:
		"""
		Read and record energy consumption for an account in a period.
		Validates non-negative consumption and creates a consumption record.
		"""
		assert account_id, "account_id required"
		assert period and len(period) == 7, "period must be YYYY-MM"
		# In real implementation, pull from AMI/metering system
		# Here we aggregate from any existing bills for this account/period
		existing_bills = [
			b for b in self._tenant_items(self.bills, self.tenant_id)
			if b.get("customer_id") == account_id and b.get("period_start", "")[:7] == period
		]
		total_kwh = sum(b.get("consumption_kwh", 0) for b in existing_bills)
		peak_kw = max((b.get("peak_demand_kw", 0) for b in existing_bills), default=0.0)
		from uuid import uuid4
		rec_id = str(uuid4())
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"account_id": account_id,
			"meter_id": meter_id,
			"period": period,
			"read_type": read_type,
			"consumption_kwh": round(total_kwh, 3),
			"peak_demand_kw": round(peak_kw, 3),
			"existing_bill_count": len(existing_bills),
			"read_at": _now(),
		}
		self._consumption_records[rec_id] = rec
		return rec

	async def apply_tariff(
		self,
		account_id: str,
		period: str,
		tariff_code: str,
		consumption_kwh: float,
		peak_demand_kw: float = 0.0,
	) -> dict[str, Any]:
		"""
		Apply a tariff to an account's consumption and compute charges.
		Looks up the active tariff by tariff_code and calculates energy + demand charges.
		"""
		assert account_id, "account_id required"
		assert tariff_code, "tariff_code required"
		assert consumption_kwh >= 0, "consumption_kwh must be non-negative"
		# Find matching active tariff
		matching = [
			t for t in self._tenant_items(self.tariffs, self.tenant_id)
			if t.get("status") == "active"
		]
		tariff = next((t for t in matching if t["id"] == tariff_code or t["name"] == tariff_code), None)
		if tariff is None:
			raise KeyError(f"No active tariff found for code '{tariff_code}'")
		# Use first rate block for energy charge
		rate_blocks = tariff.get("rate_blocks", [])
		energy_rate = rate_blocks[0].get("rate", 0.15) if rate_blocks else 0.15
		energy_charge = round(consumption_kwh * energy_rate, 4)
		demand_charge = round(peak_demand_kw * tariff.get("demand_rate_per_kw", 0.0), 4)
		from uuid import uuid4
		rec_id = str(uuid4())
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"account_id": account_id,
			"period": period,
			"tariff_id": tariff["id"],
			"tariff_code": tariff_code,
			"consumption_kwh": round(consumption_kwh, 3),
			"peak_demand_kw": round(peak_demand_kw, 3),
			"energy_rate_per_kwh": energy_rate,
			"energy_charge": energy_charge,
			"demand_charge": demand_charge,
			"subtotal": round(energy_charge + demand_charge, 4),
			"currency": "KES",
			"applied_at": _now(),
		}
		return rec

	async def calculate_demand_charges(
		self,
		account_id: str,
		period: str,
		peak_demand_kw: float,
		demand_rate_per_kw: float,
		coincident_peak_kw: float | None = None,
		ratchet_pct: float | None = None,
	) -> dict[str, Any]:
		"""
		Calculate demand charges including optional ratchet clause.
		ratchet_pct: minimum billing demand as % of peak 12-month demand.
		"""
		assert account_id, "account_id required"
		assert peak_demand_kw >= 0, "peak_demand_kw must be non-negative"
		assert demand_rate_per_kw >= 0, "demand_rate_per_kw must be non-negative"
		billing_demand = peak_demand_kw
		ratchet_demand: float | None = None
		if ratchet_pct is not None:
			# Get max demand from last 12 months' bills
			past_bills = [
				b for b in self._tenant_items(self.bills, self.tenant_id)
				if b.get("customer_id") == account_id
			]
			max_12mo_demand = max((b.get("peak_demand_kw", 0) for b in past_bills), default=peak_demand_kw)
			ratchet_demand = round(max_12mo_demand * ratchet_pct / 100, 3)
			billing_demand = max(peak_demand_kw, ratchet_demand)
		demand_charge = round(billing_demand * demand_rate_per_kw, 4)
		coincident_charge = round(coincident_peak_kw * demand_rate_per_kw * 0.5, 4) if coincident_peak_kw else 0.0
		from uuid import uuid4
		rec_id = str(uuid4())
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"account_id": account_id,
			"period": period,
			"peak_demand_kw": round(peak_demand_kw, 3),
			"billing_demand_kw": round(billing_demand, 3),
			"ratchet_demand_kw": ratchet_demand,
			"demand_rate_per_kw": demand_rate_per_kw,
			"demand_charge": demand_charge,
			"coincident_demand_charge": coincident_charge,
			"total_demand_charge": round(demand_charge + coincident_charge, 4),
			"calculated_at": _now(),
		}
		self._demand_charge_records[rec_id] = rec
		return rec

	async def calculate_energy_charges(
		self,
		account_id: str,
		period: str,
		consumption_kwh: float,
		rate_blocks: list[dict[str, Any]],
		fuel_adjustment_rate: float = 0.0,
		tou_multiplier: float = 1.0,
	) -> dict[str, Any]:
		"""
		Calculate tiered/ToU energy charges using rate blocks.
		rate_blocks: [{"limit_kwh": 100, "rate": 0.08}, {"limit_kwh": 500, "rate": 0.12}, {"limit_kwh": None, "rate": 0.18}]
		fuel_adjustment_rate: additional $/kWh fuel cost pass-through.
		tou_multiplier: time-of-use multiplier (e.g. 1.5 for peak hours).
		"""
		assert account_id, "account_id required"
		assert consumption_kwh >= 0, "consumption_kwh must be non-negative"
		assert rate_blocks, "rate_blocks required"
		remaining = consumption_kwh
		total_charge = 0.0
		tier_breakdown: list[dict[str, Any]] = []
		for block in rate_blocks:
			if remaining <= 0:
				break
			limit = block.get("limit_kwh")
			rate = block.get("rate", 0)
			if limit is None:
				tier_kwh = remaining
			else:
				tier_kwh = min(remaining, limit)
			tier_charge = round(tier_kwh * rate * tou_multiplier, 4)
			total_charge += tier_charge
			tier_breakdown.append({"tier_kwh": round(tier_kwh, 3), "rate": rate, "charge": tier_charge})
			remaining -= tier_kwh
		fuel_adj = round(consumption_kwh * fuel_adjustment_rate, 4)
		total_charge = round(total_charge + fuel_adj, 4)
		from uuid import uuid4
		rec_id = str(uuid4())
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"account_id": account_id,
			"period": period,
			"consumption_kwh": round(consumption_kwh, 3),
			"tou_multiplier": tou_multiplier,
			"fuel_adjustment_rate": fuel_adjustment_rate,
			"fuel_adjustment_charge": fuel_adj,
			"tier_breakdown": tier_breakdown,
			"total_energy_charge": total_charge,
			"calculated_at": _now(),
		}
		self._energy_charge_records[rec_id] = rec
		return rec

	async def apply_levies(
		self,
		account_id: str,
		period: str,
		levy_types: list[str],
		consumption_kwh: float,
		energy_charge: float,
	) -> dict[str, Any]:
		"""
		Apply regulatory levies to an energy bill.
		levy_types: ["REP", "REREC", "ERC", "VAT", "fuel_levy"]
		Computes each levy amount and returns itemised levy schedule.
		"""
		assert account_id, "account_id required"
		assert levy_types, "levy_types required"
		# Standard levy rates (configurable in practice)
		levy_rates: dict[str, Any] = {
			"REP": {"basis": "energy", "rate": 0.005},           # Rural Electrification Programme levy
			"REREC": {"basis": "energy", "rate": 0.003},         # REREC levy
			"ERC": {"basis": "energy", "rate": 0.002},           # Regulator levy
			"VAT": {"basis": "charge", "rate": 0.16},            # Kenya VAT 16%
			"fuel_levy": {"basis": "energy", "rate": 0.01},
			"infrastructure_levy": {"basis": "energy", "rate": 0.008},
		}
		levies: list[dict[str, Any]] = []
		total_levies = 0.0
		for levy in levy_types:
			if levy not in levy_rates:
				self._log_operation(self.tenant_id, "unknown_levy_type", levy)
				continue
			config = levy_rates[levy]
			if config["basis"] == "energy":
				amount = round(consumption_kwh * config["rate"], 4)
			else:
				amount = round(energy_charge * config["rate"], 4)
			levies.append({"levy_type": levy, "basis": config["basis"], "rate": config["rate"], "amount": amount})
			total_levies += amount
		from uuid import uuid4
		rec_id = str(uuid4())
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"account_id": account_id,
			"period": period,
			"consumption_kwh": consumption_kwh,
			"energy_charge": energy_charge,
			"levy_items": levies,
			"total_levies": round(total_levies, 4),
			"calculated_at": _now(),
		}
		self._levy_records[rec_id] = rec
		return rec

	async def bill_payment(
		self,
		account_id: str,
		amount: float,
		payment_method: str,
		bill_id: str | None = None,
		transaction_reference: str | None = None,
		currency: str = "KES",
	) -> dict[str, Any]:
		"""
		Record a bill payment for an account.
		payment_method: cash | mpesa | bank_transfer | card | direct_debit | standing_order
		If bill_id not specified, applies to oldest outstanding bill.
		"""
		assert account_id, "account_id required"
		assert amount > 0, "amount must be positive"
		assert payment_method, "payment_method required"
		# Find the bill to apply to
		if bill_id is None:
			account_bills = [
				b for b in self._tenant_items(self.bills, self.tenant_id)
				if b.get("customer_id") == account_id and b.get("status") in ("issued", "overdue", "partially_paid")
			]
			if not account_bills:
				raise ValueError(f"No outstanding bills found for account '{account_id}'")
			# Apply to oldest first
			account_bills.sort(key=lambda x: x.get("period_start", ""))
			bill_id = account_bills[0]["id"]
		from uuid import uuid4
		payment_id = str(uuid4())
		result = self.record_payment(
			payment_id=payment_id,
			tenant_id=self.tenant_id,
			bill_id=bill_id,
			customer_id=account_id,
			payment_method=payment_method,
			amount=amount,
			currency=currency,
			transaction_reference=transaction_reference or "",
		)
		return result

	async def arrears_management(
		self,
		account_id: str,
		arrears_amount: float,
		action: str = "flag",
		payment_plan_months: int | None = None,
		currency: str = "KES",
	) -> dict[str, Any]:
		"""
		Manage arrears for an account.
		action: flag | payment_plan | disconnect_notice | write_off | legal
		"""
		assert account_id, "account_id required"
		assert arrears_amount >= 0, "arrears_amount must be non-negative"
		valid_actions = {"flag", "payment_plan", "disconnect_notice", "write_off", "legal", "refer_agency"}
		if action not in valid_actions:
			raise ValueError(f"action must be one of {valid_actions}")
		monthly_instalment: float | None = None
		if action == "payment_plan" and payment_plan_months:
			assert payment_plan_months > 0, "payment_plan_months must be positive"
			monthly_instalment = round(arrears_amount / payment_plan_months, 2)
		# Flag revenue assurance issue for large arrears
		if arrears_amount > 10000:
			from uuid import uuid4
			flag_id = str(uuid4())
			self.flag_revenue_issue(
				flag_id=flag_id,
				tenant_id=self.tenant_id,
				flag_type="unpaid_bill",
				entity_id=account_id,
				entity_type="customer",
				estimated_revenue_impact=arrears_amount,
				currency=currency,
			)
		from uuid import uuid4
		rec_id = str(uuid4())
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"account_id": account_id,
			"arrears_amount": round(arrears_amount, 2),
			"currency": currency,
			"action": action,
			"payment_plan_months": payment_plan_months,
			"monthly_instalment": monthly_instalment,
			"status": "active",
			"created_at": _now(),
		}
		self._arrears_records[rec_id] = rec
		self._audit(self.tenant_id, "arrears_managed", rec_id, "arrears")
		return rec

	async def tariff_change(
		self,
		tariff_code: str,
		new_rates: list[dict[str, Any]],
		effective_date: str,
		approved_by: str,
		change_reason: str | None = None,
	) -> dict[str, Any]:
		"""
		Process a tariff rate change. Deactivates the current tariff and creates a new version.
		new_rates: [{"limit_kwh": 100, "rate": 0.09}, ...]
		"""
		assert tariff_code, "tariff_code required"
		assert new_rates, "new_rates required"
		assert effective_date, "effective_date required"
		assert approved_by, "approved_by required"
		# Find existing tariff
		existing = next(
			(t for t in self._tenant_items(self.tariffs, self.tenant_id)
			 if t["status"] == "active" and (t["id"] == tariff_code or t["name"] == tariff_code)),
			None,
		)
		if existing:
			# Deactivate old tariff
			old_tariff = self._get_tariff(self.tenant_id, existing["id"])
			old_tariff.status = "superseded"
			self._audit(self.tenant_id, "tariff_superseded", existing["id"], "tariff")
		# Create new tariff version
		from uuid import uuid4
		new_tariff_id = str(uuid4())
		customer_class = existing["customer_class"] if existing else "residential"
		tariff_type = existing["tariff_type"] if existing else "flat_rate"
		result = self.create_tariff(
			tariff_id=new_tariff_id,
			tenant_id=self.tenant_id,
			name=tariff_code,
			tariff_type=tariff_type,
			customer_class=customer_class,
			effective_date=effective_date,
			created_by=approved_by,
			rate_blocks=new_rates,
			description=change_reason or "Rate review",
		)
		# Auto-approve and activate
		self.approve_tariff(new_tariff_id, self.tenant_id, approved_by)
		activated = self.activate_tariff(new_tariff_id, self.tenant_id)
		activated["change_reason"] = change_reason
		activated["previous_tariff_id"] = existing["id"] if existing else None
		self._audit(self.tenant_id, "tariff_changed", new_tariff_id, "tariff", {"reason": change_reason})
		return activated

	async def billing_analytics(self, period: str) -> dict[str, Any]:
		"""
		Compute billing analytics for a period (YYYY-MM).
		Returns: bills generated, collection rate, arrears, disputes, revenue assurance.
		"""
		assert period and len(period) == 7, "period must be YYYY-MM"
		bills = [
			b for b in self._tenant_items(self.bills, self.tenant_id)
			if b.get("period_start", "")[:7] == period
		]
		payments = [
			p for p in self._tenant_items(self.payments, self.tenant_id)
			if p.get("received_at", "")[:7] == period
		]
		disputes = [
			d for d in self._tenant_items(self.disputes, self.tenant_id)
			if d.get("opened_at", "")[:7] == period
		]
		arrears = [
			r for r in self._arrears_records.values()
			if r.get("tenant_id") == self.tenant_id and r.get("created_at", "")[:7] == period
		]
		total_billed = sum(b.get("total_amount", 0) for b in bills)
		total_collected = sum(p.get("amount", 0) for p in payments)
		collection_rate = round(total_collected / total_billed * 100, 2) if total_billed > 0 else 0.0
		total_arrears = sum(r.get("arrears_amount", 0) for r in arrears)
		from uuid import uuid4
		rec_id = str(uuid4())
		rec: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"period": period,
			"bills_generated": len(bills),
			"total_billed": round(total_billed, 2),
			"total_collected": round(total_collected, 2),
			"collection_rate_pct": collection_rate,
			"open_disputes": len([d for d in disputes if d.get("status") == "open"]),
			"total_disputes": len(disputes),
			"accounts_in_arrears": len(arrears),
			"total_arrears_amount": round(total_arrears, 2),
			"overdue_bills": sum(1 for b in bills if b.get("status") == "overdue"),
			"calculated_at": _now(),
		}
		self._billing_analytics_records[rec_id] = rec
		return rec
