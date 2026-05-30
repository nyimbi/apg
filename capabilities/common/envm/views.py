"""UI metadata helpers for APG Environment Management."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .service import EnvmService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(
	service: EnvmService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or EnvmService()
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"routes": capability_routes(tenant_id),
		"summary": service.dashboard_summary(tenant_id),
		"environments": service.list_environments(tenant_id),
		"promotion_paths": service.list_promotion_paths(tenant_id),
		"promotion_runs": service.list_promotion_runs(tenant_id),
		"drift_reports": service.list_drift_reports(tenant_id),
		"secret_scopes": service.list_secret_scopes(tenant_id),
		"envm_agents": service.list_envm_agents(tenant_id),
		"audit_events": service.list_audit_events(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
	}


def environment_inventory_model(service: EnvmService, tenant_id: str = "default") -> dict[str, object]:
	return {
		"tenant_id": tenant_id,
		"environments": service.list_environments(tenant_id),
		"secret_scopes": service.list_secret_scopes(tenant_id),
	}


def promotion_console_model(service: EnvmService, tenant_id: str = "default") -> dict[str, object]:
	return {
		"tenant_id": tenant_id,
		"promotion_paths": service.list_promotion_paths(tenant_id),
		"promotion_runs": service.list_promotion_runs(tenant_id),
	}


def drift_dashboard_model(service: EnvmService, tenant_id: str = "default") -> dict[str, object]:
	return {
		"tenant_id": tenant_id,
		"drift_reports": service.list_drift_reports(tenant_id),
		"review_required": [
			report for report in service.list_drift_reports(tenant_id)
			if report["status"] == "review_required"
		],
	}


def envm_agent_model(service: EnvmService, tenant_id: str = "default") -> dict[str, object]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"envm_agents": service.list_envm_agents(tenant_id),
		"supported_runtimes": contract["configuration"]["envm_agents"]["supported_runtimes"],
		"allowed_roles": contract["configuration"]["envm_agents"]["allowed_roles"],
		"route": "/envm/agents",
		"permissions": ["envm:view", "envm:govern"],
	}
