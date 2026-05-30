"""UI metadata and dashboard helpers for the Distributed Computing capability."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .service import DistService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(
	service: DistService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or DistService()
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"routes": capability_routes(tenant_id),
		"summary": service.dashboard_summary(tenant_id),
		"job_console": service.list_jobs(tenant_id),
		"worker_pools": service.list_worker_pools(tenant_id),
		"worker_nodes": service.list_workers(tenant_id),
		"partition_monitor": service.list_partitions(tenant_id),
		"queue_monitor": [item for item in service.list_partitions(tenant_id) if item["status"] in {"queued", "running"}],
		"scaling_panel": service.list_scaling_decisions(tenant_id),
		"result_aggregations": service.list_aggregations(tenant_id),
		"compute_agents": service.list_compute_agents(tenant_id),
		"audit_timeline": service.list_audit_events(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"streaming": contract["streaming"],
		"theme": contract["theme"],
	}


def job_detail_model(service: DistService, tenant_id: str, job_id: str) -> dict[str, object]:
	return {
		"tenant_id": tenant_id,
		"job_id": job_id,
		"job": next((item for item in service.list_jobs(tenant_id) if item["id"] == job_id), None),
		"partitions": [item for item in service.list_partitions(tenant_id) if item["job_id"] == job_id],
		"aggregations": [item for item in service.list_aggregations(tenant_id) if item["job_id"] == job_id],
		"audit_events": [item for item in service.list_audit_events(tenant_id) if item["subject_id"] == job_id],
	}


def compute_agents_model(
	service: DistService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or DistService()
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"agents": service.list_compute_agents(tenant_id),
		"supported_runtimes": contract["configuration"]["compute_agents"]["supported_runtimes"],
		"allowed_roles": contract["configuration"]["compute_agents"]["allowed_roles"],
		"actions": ["register_compute_agent"],
		"guardrails": [
			"compute_agent_requires_registration",
			"compute_agent_runtime_supported",
			"compute_agent_role_supported",
			"compute_agent_requires_scope",
			"compute_agent_requires_disclosure",
		],
		"theme_component": contract["theme"]["components"]["agent_panel"],
	}


def audit_trail_model(
	service: DistService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or DistService()
	return {
		"tenant_id": tenant_id,
		"audit_events": service.list_audit_events(tenant_id),
		"guardrails": ["dist_state_change_requires_reason", "dist_state_change_requires_audit", "cross_tenant_compute_access_denied"],
		"actions": ["change_job_state", "validate_batch_compute_mutation"],
	}


def analytics_model(
	service: DistService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or DistService()
	summary = service.dashboard_summary(tenant_id)
	total_partitions = summary["queued_partition_count"] + summary["running_partition_count"] + summary["completed_partition_count"] + summary["failed_partition_count"]
	return {
		"tenant_id": tenant_id,
		"summary": summary,
		"signals": {
			"completion_rate": _safe_ratio(summary["completed_partition_count"], total_partitions),
			"failure_rate": _safe_ratio(summary["failed_partition_count"], total_partitions),
			"worker_utilization_signal": _safe_ratio(summary["running_partition_count"], max(summary["worker_count"], 1)),
		},
	}


def settings_model(tenant_id: str = "default") -> dict[str, object]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"configuration": contract["configuration"],
		"streaming": contract["streaming"],
		"theme": contract["theme"],
		"routes": contract["ui"]["routes"],
	}


def _safe_ratio(numerator: int, denominator: int) -> float:
	if denominator <= 0:
		return 0.0
	return round(numerator / denominator, 4)
