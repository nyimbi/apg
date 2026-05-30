"""UI metadata helpers for the Central Configuration Management capability."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .service import CompositionConfigService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(service: CompositionConfigService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or CompositionConfigService()
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"routes": capability_routes(tenant_id),
		"summary": service.dashboard_summary(tenant_id),
		"namespaces": service.list_namespaces(tenant_id),
		"configurations": service.list_configurations(tenant_id),
		"deployments": service.list_deployments(tenant_id),
		"templates": service.list_templates(tenant_id),
		"drift": service.list_drift_records(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"streaming": contract["streaming"],
		"theme": contract["theme"],
	}


def namespace_console_model(service: CompositionConfigService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or CompositionConfigService()
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"namespaces": service.list_namespaces(tenant_id),
		"guardrails": [rule for rule in contract["rule_engine"]["rules"] if "namespace" in rule["name"]],
		"theme": contract["theme"]["components"]["namespace_console"],
	}


def config_editor_model(service: CompositionConfigService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or CompositionConfigService()
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"namespaces": service.list_namespaces(tenant_id),
		"configurations": service.list_configurations(tenant_id),
		"guardrails": [rule for rule in contract["rule_engine"]["rules"] if "configuration" in rule["name"] or "secret" in rule["name"]],
		"theme": contract["theme"]["components"]["config_editor"],
	}


def release_board_model(service: CompositionConfigService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or CompositionConfigService()
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"deployments": service.list_deployments(tenant_id),
		"guardrails": [rule for rule in contract["rule_engine"]["rules"] if "deployment" in rule["name"] or "rollback" in rule["name"]],
		"streaming": contract["streaming"],
		"theme": contract["theme"]["components"]["release_board"],
	}


def template_library_model(service: CompositionConfigService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or CompositionConfigService()
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"templates": service.list_templates(tenant_id),
		"guardrails": [rule for rule in contract["rule_engine"]["rules"] if "template" in rule["name"]],
		"theme": contract["theme"]["components"]["template_library"],
	}


def drift_monitor_model(service: CompositionConfigService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or CompositionConfigService()
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"drift": service.list_drift_records(tenant_id),
		"streaming": contract["streaming"],
		"theme": contract["theme"]["components"]["drift_monitor"],
	}


def agent_workbench_model(service: CompositionConfigService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or CompositionConfigService()
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"agents": service.list_config_agents(tenant_id),
		"supported_runtimes": contract["configuration"]["config_agents"]["supported_runtimes"],
		"supported_roles": contract["configuration"]["config_agents"]["supported_roles"],
		"guardrails": [rule for rule in contract["rule_engine"]["rules"] if "agent" in rule["name"]],
		"theme": contract["theme"]["components"]["agent_workbench"],
	}
