"""UI metadata helpers for the API Service Mesh capability."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .service import CompositionGatewayService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(service: CompositionGatewayService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or CompositionGatewayService()
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"routes": capability_routes(tenant_id),
		"summary": service.dashboard_summary(tenant_id),
		"services": service.list_services(tenant_id),
		"mesh_routes": service.list_routes(tenant_id),
		"policies": service.list_policies(tenant_id),
		"traffic_shifts": service.list_traffic_shifts(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"streaming": contract["streaming"],
		"theme": contract["theme"],
	}


def service_registry_model(service: CompositionGatewayService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or CompositionGatewayService()
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"services": service.list_services(tenant_id),
		"guardrails": [rule for rule in contract["rule_engine"]["rules"] if "service" in rule["name"]],
		"theme": contract["theme"]["components"]["service_registry"],
	}


def route_console_model(service: CompositionGatewayService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or CompositionGatewayService()
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"routes": service.list_routes(tenant_id),
		"services": service.list_services(tenant_id),
		"guardrails": [rule for rule in contract["rule_engine"]["rules"] if "route" in rule["name"]],
		"streaming": contract["streaming"],
		"theme": contract["theme"]["components"]["route_console"],
	}


def policy_center_model(service: CompositionGatewayService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or CompositionGatewayService()
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"policies": service.list_policies(tenant_id),
		"guardrails": [rule for rule in contract["rule_engine"]["rules"] if "policy" in rule["name"] or "rate_limit" in rule["name"]],
		"theme": contract["theme"]["components"]["policy_center"],
	}


def traffic_console_model(service: CompositionGatewayService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or CompositionGatewayService()
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"traffic_shifts": service.list_traffic_shifts(tenant_id),
		"guardrails": [rule for rule in contract["rule_engine"]["rules"] if "traffic" in rule["name"] or "canary" in rule["name"]],
		"streaming": contract["streaming"],
		"theme": contract["theme"]["components"]["traffic_console"],
	}


def certificate_console_model(service: CompositionGatewayService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or CompositionGatewayService()
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"certificates": service.list_certificates(tenant_id),
		"guardrails": [rule for rule in contract["rule_engine"]["rules"] if "certificate" in rule["name"]],
		"theme": contract["theme"]["components"]["certificate_console"],
	}


def agent_workbench_model(service: CompositionGatewayService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or CompositionGatewayService()
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"agents": service.list_gateway_agents(tenant_id),
		"supported_runtimes": contract["configuration"]["gateway_agents"]["supported_runtimes"],
		"supported_roles": contract["configuration"]["gateway_agents"]["supported_roles"],
		"guardrails": [rule for rule in contract["rule_engine"]["rules"] if "agent" in rule["name"]],
		"theme": contract["theme"]["components"]["agent_workbench"],
	}
