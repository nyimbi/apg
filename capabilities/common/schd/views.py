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
		"streaming": contract["streaming"],
		"agents": contract["agents"],
		"routes": contract["ui"]["routes"],
		"theme": contract["theme"],
	}


def schedule_console_model(service: SchdService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"schedules": service.list_schedules(tenant_id),
		"calendars": service.list_calendars(tenant_id),
		"worker_pools": service.list_worker_pools(tenant_id),
		"guardrails": ["schedule_owner_required", "timezone_required", "calendar_policy_required", "worker_pool_required", "manual_run_reason_required"],
		"actions": ["create_schedule", "pause_schedule", "resume_schedule", "disable_schedule", "trigger_run"],
	}


def job_library_model(service: SchdService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"jobs": service.list_jobs(tenant_id),
		"guardrails": ["job_owner_required", "job_command_required", "retry_policy_required", "critical_job_monitoring_required", "external_job_approval_required", "long_running_job_review_required"],
		"actions": ["define_job"],
	}


def run_monitor_model(service: SchdService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"runs": service.list_runs(tenant_id),
		"audit_events": service.audit_events(tenant_id),
		"guardrails": ["schedule_not_runnable", "worker_pool_not_ready", "manual_run_reason_required", "bytewax_event_stream_required", "run_audit_required", "run_not_retryable"],
		"actions": ["trigger_run", "complete_run", "retry_run", "dead_letter_run", "cancel_run"],
	}


def worker_dashboard_model(service: SchdService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"worker_pools": service.list_worker_pools(tenant_id),
		"guardrails": ["worker_pool_required", "health_check_required", "capacity_limits_required", "worker_drain_reason_required"],
		"actions": ["register_worker_pool", "change_worker_state"],
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


def scheduler_agent_panel_model(service: SchdService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"agents": service.list_agents(tenant_id),
		"supported_runtimes": contract["agents"]["supported_runtimes"],
		"supported_roles": contract["agents"]["supported_roles"],
		"privileged_roles": contract["agents"]["privileged_roles"],
		"guardrails": ["scheduler_agent_id_required", "scheduler_agent_name_required", "scheduler_agent_runtime_not_supported", "scheduler_agent_role_not_supported", "scheduler_agent_scope_required", "scheduler_agent_owner_required", "scheduler_agent_purpose_required", "scheduler_agent_disclosure_required", "scheduler_agent_human_approval_required"],
		"required_controls": ["registered_by", "owner_ref", "purpose", "scope_ref", "contribution_disclosed", "human_approval_required"],
		"theme_component": contract["theme"]["components"]["scheduler_agent_roster"],
		"actions": ["register_scheduler_agent"],
	}


def lifecycle_batch_model(service: SchdService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"batches": service.list_lifecycle_batches(tenant_id),
		"streaming": contract["streaming"],
		"required_processor": contract["streaming"]["required_processor"],
		"required_operations": contract["streaming"]["required_operations"],
		"guardrails": ["schd_lifecycle_batch_empty", "unsupported_schd_lifecycle_operation", "bytewax_lifecycle_stream_required"],
		"theme_component": contract["theme"]["components"]["bytewax_lifecycle_panel"],
		"actions": ["validate_lifecycle_batch"],
	}


def audit_trail_model(service: SchdService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"audit_events": service.audit_events(tenant_id),
		"streaming_topic": get_capability_contract(tenant_id)["streaming"]["topic"],
		"guardrails": ["scheduler_audit_event_required", "cross_tenant_scheduler_access_denied"],
	}


def settings_model(service: SchdService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"configuration": contract["configuration"],
		"rules": contract["rule_engine"]["rules"],
		"agents": contract["agents"],
		"streaming": contract["streaming"],
		"theme": contract["theme"],
		"permissions": ["schd:view", "schd:schedule", "schd:run_jobs", "schd:manage_workers", "schd:audit", "schd:admin"],
	}
