"""UI metadata helpers for the Access Control Integration Hub capability."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .service import CompositionAccessService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(
	service: CompositionAccessService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or CompositionAccessService()
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"routes": capability_routes(tenant_id),
		"summary": service.dashboard_summary(tenant_id),
		"providers": service.list_providers(tenant_id),
		"resources": service.list_resources(tenant_id),
		"policies": service.list_policies(tenant_id),
		"grants": service.list_grants(tenant_id),
		"decisions": service.list_decisions(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"streaming": contract["streaming"],
		"theme": contract["theme"],
	}


def provider_console_model(
	service: CompositionAccessService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or CompositionAccessService()
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"providers": service.list_providers(tenant_id),
		"supported_types": contract["configuration"]["identity_providers"]["supported_types"],
		"guardrails": [
			rule for rule in contract["rule_engine"]["rules"]
			if rule["name"].startswith("provider_")
		],
		"theme": contract["theme"]["components"]["provider_console"],
	}


def policy_studio_model(
	service: CompositionAccessService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or CompositionAccessService()
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"resources": service.list_resources(tenant_id),
		"policies": service.list_policies(tenant_id),
		"guardrails": [
			rule for rule in contract["rule_engine"]["rules"]
			if "policy" in rule["name"]
		],
		"theme": contract["theme"]["components"]["policy_studio"],
	}


def grant_workbench_model(
	service: CompositionAccessService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or CompositionAccessService()
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"grants": service.list_grants(tenant_id),
		"resources": service.list_resources(tenant_id),
		"guardrails": [
			rule for rule in contract["rule_engine"]["rules"]
			if "grant" in rule["name"]
		],
		"streaming": contract["streaming"],
		"theme": contract["theme"]["components"]["grant_workbench"],
	}


def decision_explorer_model(
	service: CompositionAccessService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or CompositionAccessService()
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"sessions": service.list_sessions(tenant_id),
		"decisions": service.list_decisions(tenant_id),
		"streaming": contract["streaming"],
		"theme": contract["theme"]["components"]["decision_explorer"],
	}


def agent_workbench_model(
	service: CompositionAccessService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or CompositionAccessService()
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"agents": service.list_access_agents(tenant_id),
		"supported_runtimes": contract["configuration"]["access_agents"]["supported_runtimes"],
		"supported_roles": contract["configuration"]["access_agents"]["supported_roles"],
		"guardrails": [
			rule for rule in contract["rule_engine"]["rules"]
			if "agent" in rule["name"]
		],
		"theme": contract["theme"]["components"]["agent_workbench"],
	}


def audit_console_model(
	service: CompositionAccessService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or CompositionAccessService()
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"audit_events": service.audit_events(tenant_id),
		"streaming": contract["streaming"],
		"theme": contract["theme"]["components"]["decision_explorer"],
	}
