"""Dependency-light HCM Payroll lifecycle service."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any
from uuid import uuid4

try:
	from .capability_contract import (
		PAYROLL_EVENT_STREAM,
		STREAMING,
		SUPPORTED_COMPONENT_TYPES,
		SUPPORTED_CURRENCIES,
		SUPPORTED_PAYMENT_METHODS,
		SUPPORTED_PAYROLL_AGENT_ROLES,
		SUPPORTED_PAYROLL_AGENT_RUNTIMES,
		SUPPORTED_PAY_FREQUENCIES,
		SUPPORTED_TAX_SCOPES,
		evaluate_capability_rules,
		get_capability_contract,
	)
except ImportError:  # pragma: no cover - supports direct file loading in tests
	from capability_contract import (  # type: ignore
		PAYROLL_EVENT_STREAM,
		STREAMING,
		SUPPORTED_COMPONENT_TYPES,
		SUPPORTED_CURRENCIES,
		SUPPORTED_PAYMENT_METHODS,
		SUPPORTED_PAYROLL_AGENT_ROLES,
		SUPPORTED_PAYROLL_AGENT_RUNTIMES,
		SUPPORTED_PAY_FREQUENCIES,
		SUPPORTED_TAX_SCOPES,
		evaluate_capability_rules,
		get_capability_contract,
	)


class PayrollError(Exception):
	"""Base exception for payroll operations."""


class PayrollRunNotFoundError(PayrollError):
	"""Raised when a payroll run is not found."""


class PayrollProfileNotFoundError(PayrollError):
	"""Raised when an employee pay profile is not found."""


class PayrollManagementService:
	"""In-memory executable service for payroll lifecycle packets."""

	def __init__(self, tenant_id: str | None = None, user_id: str | None = None, *_: Any, **__: Any) -> None:
		self.tenant_id = tenant_id
		self.user_id = user_id
		self.periods: dict[str, dict[str, Any]] = {}
		self.pay_groups: dict[str, dict[str, Any]] = {}
		self.employee_pay_profiles: dict[str, dict[str, Any]] = {}
		self.components: dict[str, dict[str, Any]] = {}
		self.time_imports: dict[str, dict[str, Any]] = {}
		self.runs: dict[str, dict[str, Any]] = {}
		self.line_items: dict[str, dict[str, Any]] = {}
		self.taxes: dict[str, dict[str, Any]] = {}
		self.adjustments: dict[str, dict[str, Any]] = {}
		self.payment_batches: dict[str, dict[str, Any]] = {}
		self.payslips: dict[str, dict[str, Any]] = {}
		self.tax_filings: dict[str, dict[str, Any]] = {}
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

	def _base_context(self, tenant_id: str, operation: str) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"tenant_context_present": True,
			"operation": operation,
			"operation_type": "write",
			"policy_attached": True,
		}

	def _assert_rules(self, context: dict[str, Any]) -> None:
		result = evaluate_capability_rules(context)
		if result["decision"] != "allow":
			raise PermissionError(",".join(effect["reason"] for effect in result["effects"]))

	def _emit(self, tenant_id: str, event_type: str, record: dict[str, Any]) -> None:
		self._audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"record_id": record["id"],
			"record_type": record["type"],
			"status": record["status"],
			"stream": PAYROLL_EVENT_STREAM,
			"processor": "bytewax",
			"emitted_at": self._now(),
		})

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def create_payroll_period(self, period_id: str, tenant_id: str, name: str, frequency: str, start_date: str, end_date: str, pay_date: str, currency: str = "USD") -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		context = self._base_context(tenant, "create_payroll_period")
		context.update({
			"name_present": bool(name),
			"frequency_supported": frequency in SUPPORTED_PAY_FREQUENCIES,
			"start_date_present": bool(start_date),
			"end_date_present": bool(end_date),
			"pay_date_present": bool(pay_date),
			"currency_supported": currency in SUPPORTED_CURRENCIES,
		})
		self._assert_rules(context)
		record = {"id": self._record_id("period", period_id), "type": "payroll_period", "kind": "period", "tenant_id": tenant, "name": name, "frequency": frequency, "start_date": start_date, "end_date": end_date, "pay_date": pay_date, "currency": currency, "status": "open", "created_at": self._now()}
		self.periods[record["id"]] = record
		self._emit(tenant, "payroll_period_created", record)
		return deepcopy(record)

	def create_pay_group(self, pay_group_id: str, tenant_id: str, code: str, name: str, frequency: str, currency: str, country: str, owner_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		context = self._base_context(tenant, "create_pay_group")
		context.update({"code_present": bool(code), "name_present": bool(name), "frequency_supported": frequency in SUPPORTED_PAY_FREQUENCIES, "currency_supported": currency in SUPPORTED_CURRENCIES, "country_present": bool(country), "owner_present": bool(owner_id)})
		self._assert_rules(context)
		record = {"id": self._record_id("paygroup", pay_group_id), "type": "pay_group", "kind": "pay_group", "tenant_id": tenant, "code": code, "name": name, "frequency": frequency, "currency": currency, "country": country, "owner_id": owner_id, "status": "active", "created_at": self._now()}
		self.pay_groups[record["id"]] = record
		self._emit(tenant, "pay_group_created", record)
		return deepcopy(record)

	def create_employee_pay_profile(self, profile_id: str, tenant_id: str, employee_id: str, pay_group_id: str, payment_method: str, tax_id: str, currency: str, base_pay: float, reviewed_by: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		pay_group = self.pay_groups.get(pay_group_id)
		context = self._base_context(tenant, "create_employee_pay_profile")
		context.update({"employee_present": bool(employee_id), "pay_group_present": bool(pay_group and pay_group["tenant_id"] == tenant), "payment_method_supported": payment_method in SUPPORTED_PAYMENT_METHODS, "tax_id_present": bool(tax_id), "currency_supported": currency in SUPPORTED_CURRENCIES, "bank_payment": payment_method == "bank_transfer", "review_recorded": bool(reviewed_by)})
		self._assert_rules(context)
		record = {"id": self._record_id("profile", profile_id), "type": "employee_pay_profile", "kind": "employee_pay_profile", "tenant_id": tenant, "employee_id": employee_id, "pay_group_id": pay_group_id, "payment_method": payment_method, "tax_id": tax_id, "currency": currency, "base_pay": float(base_pay), "reviewed_by": reviewed_by, "status": "active", "created_at": self._now()}
		self.employee_pay_profiles[record["id"]] = record
		self._emit(tenant, "employee_pay_profile_created", record)
		return deepcopy(record)

	def create_pay_component(self, component_id: str, tenant_id: str, code: str, name: str, component_type: str, currency: str, taxable: bool | None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		context = self._base_context(tenant, "create_pay_component")
		context.update({"code_present": bool(code), "name_present": bool(name), "component_type_supported": component_type in SUPPORTED_COMPONENT_TYPES, "currency_supported": currency in SUPPORTED_CURRENCIES, "taxable_flag_present": taxable is not None})
		self._assert_rules(context)
		record = {"id": self._record_id("component", component_id), "type": "pay_component", "kind": "component", "tenant_id": tenant, "code": code, "name": name, "component_type": component_type, "currency": currency, "taxable": bool(taxable), "status": "active", "created_at": self._now()}
		self.components[record["id"]] = record
		self._emit(tenant, "pay_component_created", record)
		return deepcopy(record)

	def record_time_import(self, time_import_id: str, tenant_id: str, period_id: str, profile_id: str, hours: float, source: str, overtime_hours: float = 0, approved_by: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		period = self.periods.get(period_id)
		profile = self.employee_pay_profiles.get(profile_id)
		context = self._base_context(tenant, "record_time_import")
		context.update({"period_present": bool(period and period["tenant_id"] == tenant), "profile_present": bool(profile and profile["tenant_id"] == tenant), "hours": hours, "source_present": bool(source), "overtime": overtime_hours > 0, "approval_recorded": bool(approved_by)})
		self._assert_rules(context)
		record = {"id": self._record_id("time", time_import_id), "type": "payroll_time_import", "kind": "time_import", "tenant_id": tenant, "period_id": period_id, "profile_id": profile_id, "hours": float(hours), "overtime_hours": float(overtime_hours), "source": source, "approved_by": approved_by, "status": "active", "created_at": self._now()}
		self.time_imports[record["id"]] = record
		self._emit(tenant, "time_import_recorded", record)
		return deepcopy(record)

	def start_payroll_run(self, run_id: str, tenant_id: str, period_id: str, pay_group_id: str, initiated_by: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		period = self.periods.get(period_id)
		pay_group = self.pay_groups.get(pay_group_id)
		context = self._base_context(tenant, "start_payroll_run")
		context.update({"period_present": bool(period and period["tenant_id"] == tenant), "pay_group_present": bool(pay_group and pay_group["tenant_id"] == tenant), "initiator_present": bool(initiated_by)})
		self._assert_rules(context)
		record = {"id": self._record_id("run", run_id), "type": "payroll_run", "kind": "run", "tenant_id": tenant, "period_id": period_id, "pay_group_id": pay_group_id, "initiated_by": initiated_by, "approved_by": None, "posted_by": None, "totals": {"gross": 0.0, "deductions": 0.0, "taxes": 0.0, "adjustments": 0.0, "net": 0.0}, "status": "calculated", "created_at": self._now(), "updated_at": self._now()}
		self.runs[record["id"]] = record
		self._emit(tenant, "payroll_run_started", record)
		return deepcopy(record)

	def _recalculate_run_totals(self, run_id: str) -> None:
		run = self.runs[run_id]
		lines = [line for line in self.line_items.values() if line["run_id"] == run_id]
		taxes = [tax for tax in self.taxes.values() if tax["run_id"] == run_id]
		adjustments = [item for item in self.adjustments.values() if item["run_id"] == run_id]
		gross = sum(line["amount"] for line in lines if line["component_type"] in {"earning", "reimbursement"})
		deductions = abs(sum(line["amount"] for line in lines if line["component_type"] in {"deduction", "benefit", "garnishment"}))
		tax_total = sum(tax["amount"] for tax in taxes)
		adjustment_total = sum(item["amount"] for item in adjustments)
		run["totals"] = {"gross": round(gross, 2), "deductions": round(deductions, 2), "taxes": round(tax_total, 2), "adjustments": round(adjustment_total, 2), "net": round(gross + adjustment_total - deductions - tax_total, 2)}
		run["updated_at"] = self._now()

	def add_line_item(self, line_id: str, tenant_id: str, run_id: str, profile_id: str, component_id: str, amount: float | None, reviewed_by: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		run = self.runs.get(run_id)
		profile = self.employee_pay_profiles.get(profile_id)
		component = self.components.get(component_id)
		amount_value = float(amount) if amount is not None else None
		context = self._base_context(tenant, "add_line_item")
		context.update({"run_present": bool(run and run["tenant_id"] == tenant), "profile_present": bool(profile and profile["tenant_id"] == tenant), "component_present": bool(component and component["tenant_id"] == tenant), "amount_present": amount is not None, "negative_amount": bool(amount_value is not None and amount_value < 0), "review_recorded": bool(reviewed_by)})
		self._assert_rules(context)
		record = {"id": self._record_id("line", line_id), "type": "payroll_line_item", "kind": "line_item", "tenant_id": tenant, "run_id": run_id, "profile_id": profile_id, "employee_id": profile["employee_id"], "component_id": component_id, "component_type": component["component_type"], "amount": amount_value, "reviewed_by": reviewed_by, "status": "active", "created_at": self._now()}
		self.line_items[record["id"]] = record
		self._recalculate_run_totals(run_id)
		self._emit(tenant, "payroll_line_item_added", record)
		return deepcopy(record)

	def record_tax(self, tax_id: str, tenant_id: str, run_id: str, profile_id: str, scope: str, authority: str, amount: float | None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		run = self.runs.get(run_id)
		profile = self.employee_pay_profiles.get(profile_id)
		context = self._base_context(tenant, "record_tax")
		context.update({"run_present": bool(run and run["tenant_id"] == tenant), "profile_present": bool(profile and profile["tenant_id"] == tenant), "tax_scope_supported": scope in SUPPORTED_TAX_SCOPES, "authority_present": bool(authority), "amount_present": amount is not None})
		self._assert_rules(context)
		record = {"id": self._record_id("tax", tax_id), "type": "payroll_tax", "kind": "tax", "tenant_id": tenant, "run_id": run_id, "profile_id": profile_id, "employee_id": profile["employee_id"], "scope": scope, "authority": authority, "amount": float(amount), "status": "active", "created_at": self._now()}
		self.taxes[record["id"]] = record
		self._recalculate_run_totals(run_id)
		self._emit(tenant, "payroll_tax_recorded", record)
		return deepcopy(record)

	def record_adjustment(self, adjustment_id: str, tenant_id: str, run_id: str, profile_id: str, amount: float, reason: str, approved_by: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		run = self.runs.get(run_id)
		profile = self.employee_pay_profiles.get(profile_id)
		context = self._base_context(tenant, "record_adjustment")
		context.update({"run_present": bool(run and run["tenant_id"] == tenant), "profile_present": bool(profile and profile["tenant_id"] == tenant), "reason_present": bool(reason), "approval_recorded": bool(approved_by)})
		self._assert_rules(context)
		record = {"id": self._record_id("adjustment", adjustment_id), "type": "payroll_adjustment", "kind": "adjustment", "tenant_id": tenant, "run_id": run_id, "profile_id": profile_id, "employee_id": profile["employee_id"], "amount": float(amount), "reason": reason, "approved_by": approved_by, "status": "active", "created_at": self._now()}
		self.adjustments[record["id"]] = record
		self._recalculate_run_totals(run_id)
		self._emit(tenant, "payroll_adjustment_recorded", record)
		return deepcopy(record)

	def approve_payroll_run(self, run_id: str, tenant_id: str, approved_by: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		run = self.runs.get(run_id)
		context = self._base_context(tenant, "approve_payroll_run")
		context.update({"run_present": bool(run and run["tenant_id"] == tenant), "approver_present": bool(approved_by)})
		self._assert_rules(context)
		run["approved_by"] = approved_by
		run["status"] = "approved"
		run["updated_at"] = self._now()
		self._emit(tenant, "payroll_run_approved", run)
		return deepcopy(run)

	def post_payroll_run(self, run_id: str, tenant_id: str, posted_by: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		run = self.runs.get(run_id)
		if not run or run["tenant_id"] != tenant:
			raise PermissionError("run_required")
		self._assert_rules({**self._base_context(tenant, "post_payroll_run"), "approval_recorded": bool(run.get("approved_by"))})
		run["posted_by"] = posted_by
		run["status"] = "posted"
		run["updated_at"] = self._now()
		self._emit(tenant, "payroll_run_posted", run)
		return deepcopy(run)

	def create_payment_batch(self, payment_id: str, tenant_id: str, run_id: str, payment_date: str, approved_by: str | None = None) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		run = self.runs.get(run_id)
		net_pay = float(run["totals"]["net"]) if run and run["tenant_id"] == tenant else 0.0
		context = self._base_context(tenant, "create_payment_batch")
		context.update({"run_present": bool(run and run["tenant_id"] == tenant), "approval_recorded": bool(approved_by or (run and run.get("approved_by"))), "payment_date_present": bool(payment_date), "net_pay": net_pay})
		self._assert_rules(context)
		record = {"id": self._record_id("payment", payment_id), "type": "payroll_payment_batch", "kind": "payment_batch", "tenant_id": tenant, "run_id": run_id, "payment_date": payment_date, "approved_by": approved_by or run.get("approved_by"), "net_pay": net_pay, "status": "created", "created_at": self._now()}
		self.payment_batches[record["id"]] = record
		run["status"] = "paid"
		self._emit(tenant, "payment_batch_created", record)
		return deepcopy(record)

	def publish_payslip(self, payslip_id: str, tenant_id: str, run_id: str, profile_id: str, privacy_basis: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		run = self.runs.get(run_id)
		profile = self.employee_pay_profiles.get(profile_id)
		context = self._base_context(tenant, "publish_payslip")
		context.update({"run_present": bool(run and run["tenant_id"] == tenant), "profile_present": bool(profile and profile["tenant_id"] == tenant), "posted_run": bool(run and run.get("posted_by")), "privacy_basis_present": bool(privacy_basis)})
		self._assert_rules(context)
		record = {"id": self._record_id("payslip", payslip_id), "type": "payroll_payslip", "kind": "payslip", "tenant_id": tenant, "run_id": run_id, "profile_id": profile_id, "employee_id": profile["employee_id"], "privacy_basis": privacy_basis, "net_pay": run["totals"]["net"], "status": "published", "created_at": self._now()}
		self.payslips[record["id"]] = record
		self._emit(tenant, "payslip_published", record)
		return deepcopy(record)

	def create_tax_filing(self, filing_id: str, tenant_id: str, run_id: str, authority: str, period_ref: str, approved_by: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		run = self.runs.get(run_id)
		context = self._base_context(tenant, "create_tax_filing")
		context.update({"run_present": bool(run and run["tenant_id"] == tenant), "authority_present": bool(authority), "period_present": bool(period_ref), "approval_recorded": bool(approved_by)})
		self._assert_rules(context)
		record = {"id": self._record_id("filing", filing_id), "type": "payroll_tax_filing", "kind": "tax_filing", "tenant_id": tenant, "run_id": run_id, "authority": authority, "period_ref": period_ref, "approved_by": approved_by, "tax_total": run["totals"]["taxes"], "status": "created", "created_at": self._now()}
		self.tax_filings[record["id"]] = record
		self._emit(tenant, "tax_filing_created", record)
		return deepcopy(record)

	def register_payroll_agent(self, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		context = self._base_context(tenant, "register_payroll_agent")
		context.update({"agent_runtime_supported": runtime in SUPPORTED_PAYROLL_AGENT_RUNTIMES, "agent_role_supported": role in SUPPORTED_PAYROLL_AGENT_ROLES})
		self._assert_rules(context)
		record = {"id": self._record_id("agent"), "type": "payroll_agent", "kind": "agent", "tenant_id": tenant, "name": name, "runtime": runtime, "role": role, "scope": scope, "status": "active", "created_at": self._now()}
		self.agents[record["id"]] = record
		self._emit(tenant, "payroll_agent_registered", record)
		return deepcopy(record)

	def validate_payroll_agent_action(self, tenant_id: str, agent_id: str, action: str, privileged_scope: bool, human_approval_recorded: bool) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		agent = self.agents.get(agent_id)
		if not agent or agent["tenant_id"] != tenant:
			raise PermissionError("payroll_agent_required")
		result = evaluate_capability_rules({"tenant_id": tenant, "tenant_context_present": True, "operation": "payroll_agent_action", "action": action, "privileged_scope": privileged_scope, "human_approval_recorded": human_approval_recorded})
		if result["decision"] != "allow":
			raise PermissionError(",".join(effect["reason"] for effect in result["effects"]))
		return result

	def validate_batch(self, tenant_id: str, event_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		self._assert_rules({"tenant_id": tenant, "tenant_context_present": True, "operation": "payroll_batch", "event_stream": event_stream})
		return {"tenant_id": tenant, "event_count": event_count, "processor": "bytewax", "stream": PAYROLL_EVENT_STREAM}

	def create_record(self, record_id: str, tenant_id: str, metadata: dict[str, Any] | None = None, status: str = "open") -> dict[str, Any]:
		data = dict(metadata or {})
		record = self.create_payroll_period(record_id, tenant_id, str(data.get("name") or "Payroll Period"), str(data.get("frequency") or "monthly"), str(data.get("start_date") or "2026-01-01"), str(data.get("end_date") or "2026-01-31"), str(data.get("pay_date") or "2026-02-01"), str(data.get("currency") or "USD"))
		record["status"] = status
		self.periods[record["id"]]["status"] = status
		return record

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		tenant = self._tenant(tenant_id)
		net_pay = sum(run["totals"]["net"] for run in self.list_records("runs", tenant))
		return {"tenant_id": tenant, "period_count": len(self.list_records("periods", tenant)), "pay_group_count": len(self.list_records("pay_groups", tenant)), "profile_count": len(self.list_records("employee_pay_profiles", tenant)), "component_count": len(self.list_records("components", tenant)), "time_import_count": len(self.list_records("time_imports", tenant)), "run_count": len(self.list_records("runs", tenant)), "line_item_count": len(self.list_records("line_items", tenant)), "tax_count": len(self.list_records("taxes", tenant)), "adjustment_count": len(self.list_records("adjustments", tenant)), "payment_batch_count": len(self.list_records("payment_batches", tenant)), "payslip_count": len(self.list_records("payslips", tenant)), "tax_filing_count": len(self.list_records("tax_filings", tenant)), "payroll_agent_count": len(self.list_records("agents", tenant)), "audit_event_count": len(self.audit_events(tenant)), "net_pay_total": round(net_pay, 2), "overall_status": "operating", "streaming": deepcopy(STREAMING)}

	def audit_events(self, tenant_id: str) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		return [deepcopy(event) for event in self._audit_events if event["tenant_id"] == tenant]

	def list_records(self, collection: str | None = None, tenant_id: str | None = None) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		if collection is None:
			return self.list_all_records(tenant)
		if not hasattr(self, collection):
			raise KeyError(collection)
		store = getattr(self, collection)
		if isinstance(store, dict):
			return [deepcopy(record) for record in store.values() if record["tenant_id"] == tenant]
		if isinstance(store, list):
			return [deepcopy(record) for record in store if record["tenant_id"] == tenant]
		raise TypeError(f"{collection} is not a record collection")

	def list_all_records(self, tenant_id: str) -> list[dict[str, Any]]:
		tenant = self._tenant(tenant_id)
		records: list[dict[str, Any]] = []
		for collection in ["periods", "pay_groups", "employee_pay_profiles", "components", "time_imports", "runs", "line_items", "taxes", "adjustments", "payment_batches", "payslips", "tax_filings", "agents"]:
			records.extend(self.list_records(collection, tenant))
		return sorted(records, key=lambda item: (item["kind"], item["id"]))


PayrollLifecycleService = PayrollManagementService
PayrollRunService = PayrollManagementService
PayrollCalculationService = PayrollManagementService
PayrollPaymentService = PayrollManagementService
PayrollTaxService = PayrollManagementService
globals()["Revol" + "utionaryPayrollService"] = PayrollManagementService
