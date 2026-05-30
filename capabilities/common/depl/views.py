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
		"deployment_agents": service.list_deployment_agents(tenant_id),
		"audit_timeline": service.list_audit_events(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"streaming": contract["streaming"],
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


def deployment_agents_model(
	service: DeplService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or DeplService()
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"agents": service.list_deployment_agents(tenant_id),
		"supported_runtimes": contract["configuration"]["deployment_agents"]["supported_runtimes"],
		"allowed_roles": contract["configuration"]["deployment_agents"]["allowed_roles"],
		"actions": ["register_deployment_agent"],
		"guardrails": [
			"deployment_agent_requires_registration",
			"deployment_agent_runtime_supported",
			"deployment_agent_role_supported",
			"deployment_agent_requires_scope",
			"deployment_agent_requires_disclosure",
		],
		"theme_component": contract["theme"]["components"]["agent_panel"],
	}


def audit_trail_model(
	service: DeplService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or DeplService()
	return {
		"tenant_id": tenant_id,
		"audit_events": service.list_audit_events(tenant_id),
		"guardrails": ["depl_state_change_requires_reason", "depl_state_change_requires_audit", "cross_tenant_deployment_access_denied"],
		"actions": ["change_deployment_plan_state", "validate_batch_deployment_mutation"],
	}


def analytics_model(
	service: DeplService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or DeplService()
	summary = service.dashboard_summary(tenant_id)
	return {
		"tenant_id": tenant_id,
		"summary": summary,
		"signals": {
			"release_to_run_ratio": _safe_ratio(summary["deployed_run_count"], summary["release_count"]),
			"rollback_rate": _safe_ratio(summary["rollback_count"], max(summary["deployed_run_count"], 1)),
			"health_gate_pass_rate": _safe_ratio(summary["passing_health_gate_count"], len(service.list_health_gates(tenant_id))),
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
