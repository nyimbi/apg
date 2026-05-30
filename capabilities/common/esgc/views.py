"""UI metadata helpers for APG ESG/Carbon Tracking."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .service import EsgcService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(
	service: EsgcService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or EsgcService()
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"routes": capability_routes(tenant_id),
		"summary": service.dashboard_summary(tenant_id),
		"inventories": service.list_inventories(tenant_id),
		"factors": service.list_factors(tenant_id),
		"activities": service.list_activities(tenant_id),
		"reports": service.list_reports(tenant_id),
		"targets": service.list_targets(tenant_id),
		"esgc_agents": service.list_esgc_agents(tenant_id),
		"audit_events": service.list_audit_events(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
	}


def emissions_inventory_model(service: EsgcService, tenant_id: str = "default") -> dict[str, object]:
	return {
		"tenant_id": tenant_id,
		"inventories": service.list_inventories(tenant_id),
		"activities": service.list_activities(tenant_id),
		"summary": service.dashboard_summary(tenant_id),
	}


def report_builder_model(service: EsgcService, tenant_id: str = "default") -> dict[str, object]:
	return {
		"tenant_id": tenant_id,
		"reports": service.list_reports(tenant_id),
		"audit_events": [
			event for event in service.list_audit_events(tenant_id)
			if event["event_type"] == "report_published"
		],
	}


def target_tracker_model(service: EsgcService, tenant_id: str = "default") -> dict[str, object]:
	return {
		"tenant_id": tenant_id,
		"targets": service.list_targets(tenant_id),
		"total_co2e_tonnes": service.dashboard_summary(tenant_id)["total_co2e_tonnes"],
	}


def esgc_agent_model(service: EsgcService, tenant_id: str = "default") -> dict[str, object]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"esgc_agents": service.list_esgc_agents(tenant_id),
		"supported_runtimes": contract["configuration"]["esgc_agents"]["supported_runtimes"],
		"allowed_roles": contract["configuration"]["esgc_agents"]["allowed_roles"],
		"route": "/esgc/agents",
		"permissions": ["esgc:view", "esgc:govern"],
	}
