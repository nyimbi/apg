"""UI metadata helpers for APG Multi-Channel Output."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .service import MchnService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(
	service: MchnService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or MchnService()
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"routes": capability_routes(tenant_id),
		"summary": service.dashboard_summary(tenant_id),
		"channels": service.list_channels(tenant_id),
		"templates": service.list_templates(tenant_id),
		"policies": service.list_policies(tenant_id),
		"delivery_routes": service.list_routes(tenant_id),
		"rendered_outputs": service.list_rendered_outputs(tenant_id),
		"delivery_batches": service.list_batches(tenant_id),
		"receipts": service.list_receipts(tenant_id),
		"audit_events": service.list_audit_events(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
		"streaming": contract["streaming"],
	}


def render_console_model(service: MchnService, tenant_id: str = "default") -> dict[str, object]:
	return {
		"tenant_id": tenant_id,
		"templates": service.list_templates(tenant_id),
		"delivery_routes": service.list_routes(tenant_id),
		"rendered_outputs": service.list_rendered_outputs(tenant_id),
	}


def template_manager_model(service: MchnService, tenant_id: str = "default") -> dict[str, object]:
	return {
		"tenant_id": tenant_id,
		"templates": service.list_templates(tenant_id),
		"theme": service.describe(tenant_id)["theme"],
	}


def route_console_model(service: MchnService, tenant_id: str = "default") -> dict[str, object]:
	return {
		"tenant_id": tenant_id,
		"delivery_routes": service.list_routes(tenant_id),
		"channels": service.list_channels(tenant_id),
		"policies": service.list_policies(tenant_id),
	}


def channel_monitor_model(service: MchnService, tenant_id: str = "default") -> dict[str, object]:
	return {
		"tenant_id": tenant_id,
		"channels": service.list_channels(tenant_id),
		"unhealthy_channels": [
			channel for channel in service.list_channels(tenant_id)
			if channel["health"] == "unhealthy"
		],
	}


def analytics_model(service: MchnService, tenant_id: str = "default") -> dict[str, object]:
	return {
		"tenant_id": tenant_id,
		"summary": service.dashboard_summary(tenant_id),
		"delivery_batches": service.list_batches(tenant_id),
		"failed_receipts": [
			receipt for receipt in service.list_receipts(tenant_id)
			if receipt["delivery_state"] in {"failed", "bounced"}
		],
	}


def policy_model(service: MchnService, tenant_id: str = "default") -> dict[str, object]:
	return {
		"tenant_id": tenant_id,
		"policies": service.list_policies(tenant_id),
		"rules": service.describe(tenant_id)["rule_engine"]["rules"],
	}


def mchn_agent_model(service: MchnService, tenant_id: str = "default") -> dict[str, object]:
	contract = service.describe(tenant_id)
	return {
		"route": "/mchn/agents",
		"mchn_agents": service.list_mchn_agents(tenant_id),
		"supported_runtimes": contract["configuration"]["mchn_agents"]["supported_runtimes"],
		"allowed_roles": contract["configuration"]["mchn_agents"]["allowed_roles"],
		"permissions": ["mchn:view", "mchn:admin"],
	}


def audit_trail_model(service: MchnService, tenant_id: str = "default") -> dict[str, object]:
	return {
		"route": "/mchn/audit",
		"audit_events": service.list_audit_events(tenant_id),
		"permissions": ["mchn:admin"],
	}


def delivery_governance_model(service: MchnService, tenant_id: str = "default") -> dict[str, object]:
	contract = service.describe(tenant_id)
	return {
		"route": "/mchn/policies",
		"policies": service.list_policies(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"streaming": contract["streaming"],
		"configuration": contract["configuration"],
	}
