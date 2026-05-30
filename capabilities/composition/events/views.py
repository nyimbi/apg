"""UI metadata helpers for the Event Streaming Bus capability."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .service import CompositionEventsService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(service: CompositionEventsService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or CompositionEventsService()
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"routes": capability_routes(tenant_id),
		"summary": service.dashboard_summary(tenant_id),
		"streams": service.list_streams(tenant_id),
		"schemas": service.list_schemas(tenant_id),
		"subscriptions": service.list_subscriptions(tenant_id),
		"processors": service.list_processors(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"streaming": contract["streaming"],
		"theme": contract["theme"],
	}


def stream_console_model(service: CompositionEventsService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or CompositionEventsService()
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"streams": service.list_streams(tenant_id),
		"guardrails": [rule for rule in contract["rule_engine"]["rules"] if "stream" in rule["name"]],
		"theme": contract["theme"]["components"]["stream_console"],
	}


def schema_registry_model(service: CompositionEventsService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or CompositionEventsService()
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"schemas": service.list_schemas(tenant_id),
		"guardrails": [rule for rule in contract["rule_engine"]["rules"] if "schema" in rule["name"]],
		"theme": contract["theme"]["components"]["schema_registry"],
	}


def subscription_console_model(service: CompositionEventsService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or CompositionEventsService()
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"subscriptions": service.list_subscriptions(tenant_id),
		"guardrails": [rule for rule in contract["rule_engine"]["rules"] if "subscription" in rule["name"]],
		"theme": contract["theme"]["components"]["subscription_console"],
	}


def processor_topology_model(service: CompositionEventsService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or CompositionEventsService()
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"processors": service.list_processors(tenant_id),
		"guardrails": [rule for rule in contract["rule_engine"]["rules"] if "processor" in rule["name"]],
		"streaming": contract["streaming"],
		"theme": contract["theme"]["components"]["processor_topology"],
	}


def agent_workbench_model(service: CompositionEventsService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or CompositionEventsService()
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"agents": service.list_event_agents(tenant_id),
		"supported_runtimes": contract["configuration"]["event_agents"]["supported_runtimes"],
		"supported_roles": contract["configuration"]["event_agents"]["supported_roles"],
		"guardrails": [rule for rule in contract["rule_engine"]["rules"] if "agent" in rule["name"]],
		"theme": contract["theme"]["components"]["agent_workbench"],
	}
