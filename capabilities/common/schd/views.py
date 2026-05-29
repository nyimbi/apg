"""Composable view models for APG Scheduling and Job Orchestration."""

from __future__ import annotations

from typing import Any

from .capability_contract import get_capability_contract
from .service import SchdService


def capability_routes() -> list[dict[str, str]]:
	return get_capability_contract()["ui"]["routes"]


def dashboard_model(service: SchdService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"title": "Scheduling and Job Orchestration",
		"summary": service.dashboard_summary(tenant_id),
		"routes": contract["ui"]["routes"],
		"theme": contract["theme"],
	}


def schedule_console_model(service: SchdService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"schedules": service.list_schedules(tenant_id),
		"calendars": service.list_calendars(tenant_id),
		"worker_pools": service.list_worker_pools(tenant_id),
		"actions": ["create_schedule", "disable_schedule", "trigger_run"],
	}


def job_library_model(service: SchdService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"jobs": service.list_jobs(tenant_id),
		"guardrails": ["critical_job_monitoring_required", "external_job_approval_required", "long_running_job_review_required"],
		"actions": ["define_job"],
	}


def run_monitor_model(service: SchdService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"runs": service.list_runs(tenant_id),
		"audit_events": service.audit_events(tenant_id),
		"actions": ["trigger_run", "complete_run"],
	}


def worker_dashboard_model(service: SchdService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"worker_pools": service.list_worker_pools(tenant_id),
		"guardrails": ["worker_pool_required", "health_check_required", "capacity_limits_required"],
	}


def calendar_manager_model(service: SchdService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"calendars": service.list_calendars(tenant_id),
		"guardrails": ["timezone_required", "calendar_policy_required"],
		"actions": ["create_calendar_policy"],
	}


def analytics_model(service: SchdService, tenant_id: str = "default") -> dict[str, Any]:
	summary = service.dashboard_summary(tenant_id)
	return {
		"summary": summary,
		"run_health": {
			"succeeded": summary["succeeded_run_count"],
			"failed": summary["failed_run_count"],
			"total": summary["run_count"],
		},
		"theme": get_capability_contract(tenant_id)["theme"],
	}


def settings_model(service: SchdService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"configuration": contract["configuration"],
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
		"permissions": ["schd:view", "schd:schedule", "schd:run_jobs", "schd:manage_workers", "schd:admin"],
	}
