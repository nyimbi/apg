"""View models for generated Time & Expense Management screens."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import TimeExpenseService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore
	from service import TimeExpenseService  # type: ignore


def dashboard_model(service: TimeExpenseService, tenant_id: str = "default") -> dict[str, Any]:
	"""View model for the time & expense dashboard."""
	contract = get_capability_contract(tenant_id)
	return {
		"title": "Time & Expense Management",
		"tenant_id": tenant_id,
		"summary": service.dashboard_summary(tenant_id),
		"theme": contract["theme"],
		"routes": contract["ui"]["routes"],
	}


def my_timesheets_model(service: TimeExpenseService, tenant_id: str = "default", resource_id: str | None = None) -> dict[str, Any]:
	"""View model for the personal timesheet list."""
	return {
		"tenant_id": tenant_id,
		"resource_id": resource_id,
		"timesheets": [
			v.to_dict() for v in sorted(service.timesheets.values(), key=lambda x: x.period_reference)
			if v.tenant_id == tenant_id and (resource_id is None or v.resource_id == resource_id)
		],
	}


def timesheet_entry_model(service: TimeExpenseService, tenant_id: str = "default", timesheet_id: str | None = None) -> dict[str, Any]:
	"""View model for the timesheet entry form."""
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"timesheet_id": timesheet_id,
		"supported_entry_types": contract["configuration"]["timesheets"]["supported_entry_types"],
		"supported_billable_statuses": contract["configuration"]["timesheets"]["supported_billable_statuses"],
		"time_entries": [
			v.to_dict() for v in sorted(service.time_entries.values(), key=lambda x: x.entry_date)
			if v.tenant_id == tenant_id and (timesheet_id is None or v.timesheet_id == timesheet_id)
		],
	}


def timesheet_approval_queue_model(service: TimeExpenseService, tenant_id: str = "default") -> dict[str, Any]:
	"""View model for the timesheet approval queue."""
	pending = [
		v.to_dict() for v in sorted(service.timesheets.values(), key=lambda x: x.id)
		if v.tenant_id == tenant_id and v.status == "submitted"
	]
	return {
		"tenant_id": tenant_id,
		"pending_timesheets": pending,
		"approvals": _tenant_items(service.timesheet_approvals, tenant_id),
	}


def my_expenses_model(service: TimeExpenseService, tenant_id: str = "default", resource_id: str | None = None) -> dict[str, Any]:
	"""View model for the personal expense list."""
	return {
		"tenant_id": tenant_id,
		"resource_id": resource_id,
		"expense_claims": [
			v.to_dict() for v in sorted(service.expense_claims.values(), key=lambda x: x.expense_date)
			if v.tenant_id == tenant_id and (resource_id is None or v.resource_id == resource_id)
		],
	}


def expense_approval_queue_model(service: TimeExpenseService, tenant_id: str = "default") -> dict[str, Any]:
	"""View model for the expense approval queue."""
	pending = [
		v.to_dict() for v in sorted(service.expense_claims.values(), key=lambda x: x.id)
		if v.tenant_id == tenant_id and v.status == "submitted"
	]
	return {
		"tenant_id": tenant_id,
		"pending_expenses": pending,
		"approvals": _tenant_items(service.expense_approvals, tenant_id),
	}


def billable_hours_model(service: TimeExpenseService, tenant_id: str = "default", project_id: str | None = None) -> dict[str, Any]:
	"""View model for billable hours tracker."""
	return {
		"tenant_id": tenant_id,
		"project_id": project_id,
		"summary": service.billable_hours_summary(tenant_id, project_id),
		"billing_rates": _tenant_items(service.billing_rates, tenant_id),
	}


def reimbursement_console_model(service: TimeExpenseService, tenant_id: str = "default") -> dict[str, Any]:
	"""View model for reimbursement processing console."""
	return {
		"tenant_id": tenant_id,
		"reimbursements": _tenant_items(service.reimbursements, tenant_id),
	}


def agent_workbench_model(service: TimeExpenseService, tenant_id: str = "default") -> dict[str, Any]:
	"""View model for the time & expense agent workbench."""
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"],
		"supported_roles": contract["configuration"]["agents"]["supported_roles"],
		"agents": [v.to_dict() for v in service.agents.values() if v.tenant_id == tenant_id],
	}


def _tenant_items(items: dict[Any, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [v.to_dict() for v in sorted(items.values(), key=lambda x: x.id) if v.tenant_id == tenant_id]
