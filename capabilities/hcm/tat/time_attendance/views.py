"""View models for the HCM Time and Attendance capability."""

from __future__ import annotations

from typing import Any


def dashboard_model(service: Any, tenant_id: str) -> dict[str, Any]:
	"""Return a compact dashboard model for composed APG applications."""
	summary = service.dashboard_summary(tenant_id)
	return {
		"title": "Time and Attendance",
		"tenant_id": tenant_id,
		"cards": [
			{"label": "Policies", "value": summary["policy_count"], "icon": "shield-check"},
			{"label": "Schedules", "value": summary["schedule_count"], "icon": "calendar-days"},
			{"label": "Entries", "value": summary["time_entry_count"], "icon": "timer"},
			{"label": "Timesheets", "value": summary["timesheet_count"], "icon": "clipboard-check"},
			{"label": "Exceptions", "value": summary["exception_count"], "icon": "triangle-alert"},
			{"label": "Agents", "value": summary["agent_count"], "icon": "bot"},
		],
		"streaming": summary["streaming"],
	}


def policy_model(service: Any, tenant_id: str) -> dict[str, Any]:
	return {"name": "Policies", "records": service.list_records(tenant_id, "policy")}


def schedule_model(service: Any, tenant_id: str) -> dict[str, Any]:
	return {"name": "Schedules", "records": service.list_records(tenant_id, "schedule")}


def shift_model(service: Any, tenant_id: str) -> dict[str, Any]:
	return {"name": "Shifts", "records": service.list_records(tenant_id, "shift")}


def time_entry_model(service: Any, tenant_id: str) -> dict[str, Any]:
	return {"name": "Time Entries", "records": service.list_records(tenant_id, "time_entry")}


def timesheet_model(service: Any, tenant_id: str) -> dict[str, Any]:
	return {"name": "Timesheets", "records": service.list_records(tenant_id, "timesheet")}


def leave_model(service: Any, tenant_id: str) -> dict[str, Any]:
	return {"name": "Leave Requests", "records": service.list_records(tenant_id, "leave_request")}


def exception_model(service: Any, tenant_id: str) -> dict[str, Any]:
	return {"name": "Exceptions", "records": service.list_records(tenant_id, "exception")}


def export_model(service: Any, tenant_id: str) -> dict[str, Any]:
	return {"name": "Payroll Exports", "records": service.list_records(tenant_id, "payroll_export")}


def agent_model(service: Any, tenant_id: str) -> dict[str, Any]:
	return {
		"name": "Attendance Agents",
		"records": service.list_records(tenant_id, "agent"),
		"policy": {
			"max_autonomous_scope": "inspect_prepare_and_recommend",
			"human_approval_required": True,
		},
	}


def rules_model(contract: dict[str, Any]) -> dict[str, Any]:
	return {
		"name": "Attendance Rules",
		"rule_count": len(contract["rule_engine"]["rules"]),
		"rules": contract["rule_engine"]["rules"],
	}


def settings_model(contract: dict[str, Any]) -> dict[str, Any]:
	return {
		"name": "Attendance Settings",
		"configuration": contract["configuration"],
		"theme": contract["theme"],
		"routes": contract["ui"]["routes"],
	}
