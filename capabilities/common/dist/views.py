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
		"audit_timeline": service.list_audit_events(tenant_id),
		"rules": contract["rule_engine"]["rules"],
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
