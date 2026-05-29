"""UI metadata and dashboard helpers for the Deployment Management capability."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .service import DeplService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(
	service: DeplService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or DeplService()
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"routes": capability_routes(tenant_id),
		"summary": service.dashboard_summary(tenant_id),
		"environment_console": service.list_environments(tenant_id),
		"release_console": service.list_releases(tenant_id),
		"rollout_strategies": service.list_deployment_plans(tenant_id),
		"deployment_monitor": service.list_deployment_runs(tenant_id),
		"health_gates": service.list_health_gates(tenant_id),
		"rollback_center": service.list_rollback_events(tenant_id),
		"rollback_plans": service.list_rollback_plans(tenant_id),
		"audit_timeline": service.list_audit_events(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
	}


def release_detail_model(service: DeplService, tenant_id: str, release_id: str) -> dict[str, object]:
	return {
		"tenant_id": tenant_id,
		"release_id": release_id,
		"release": next((item for item in service.list_releases(tenant_id) if item["id"] == release_id), None),
		"rollback_plans": [item for item in service.list_rollback_plans(tenant_id) if item["release_id"] == release_id],
		"health_gates": [item for item in service.list_health_gates(tenant_id) if item["release_id"] == release_id],
		"deployment_plans": [item for item in service.list_deployment_plans(tenant_id) if item["release_id"] == release_id],
		"deployment_runs": [item for item in service.list_deployment_runs(tenant_id) if item["release_id"] == release_id],
	}
