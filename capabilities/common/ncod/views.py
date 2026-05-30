"""UI metadata helpers for the No-Code/Low-Code Builder capability."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .service import NcodService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(
	service: NcodService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or NcodService()
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"summary": service.dashboard_summary(tenant_id),
		"routes": capability_routes(tenant_id),
		"apps": service.list_apps(tenant_id),
		"releases": service.list_releases(tenant_id),
		"deployments": service.list_deployments(tenant_id),
		"builder_agents": service.list_builder_agents(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
		"streaming": contract["streaming"],
	}


def app_library_model(
	service: NcodService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or NcodService()
	return {
		"tenant_id": tenant_id,
		"apps": service.list_apps(tenant_id),
		"validations": service.list_validations(tenant_id),
		"releases": service.list_releases(tenant_id),
		"route": "/ncod/apps",
	}


def builder_model(
	service: NcodService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or NcodService()
	return {
		"tenant_id": tenant_id,
		"apps": service.list_apps(tenant_id),
		"pages": service.list_pages(tenant_id),
		"components": service.list_components(tenant_id),
		"data_models": service.list_data_models(tenant_id),
		"data_bindings": service.list_data_bindings(tenant_id),
		"workflow_bindings": service.list_workflow_bindings(tenant_id),
		"theme_variants": service.list_theme_variants(tenant_id),
		"builder_agents": service.list_builder_agents(tenant_id),
		"route": "/ncod/builder",
	}


def page_composer_model(
	service: NcodService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or NcodService()
	return {
		"tenant_id": tenant_id,
		"pages": service.list_pages(tenant_id),
		"components": service.list_components(tenant_id),
		"route": "/ncod/pages",
	}


def component_catalog_model(
	service: NcodService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or NcodService()
	return {
		"tenant_id": tenant_id,
		"component_types": ["text", "input", "select", "table", "chart", "button", "form", "metric", "workflow_action", "agent_panel", "kanban", "timeline"],
		"components": service.list_components(tenant_id),
		"route": "/ncod/components",
	}


def data_modeler_model(
	service: NcodService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or NcodService()
	return {
		"tenant_id": tenant_id,
		"data_models": service.list_data_models(tenant_id),
		"data_bindings": service.list_data_bindings(tenant_id),
		"route": "/ncod/data-models",
	}


def workflow_designer_model(
	service: NcodService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or NcodService()
	return {
		"tenant_id": tenant_id,
		"workflow_bindings": service.list_workflow_bindings(tenant_id),
		"script_extensions": service.list_script_extensions(tenant_id),
		"route": "/ncod/workflows",
	}


def publish_center_model(
	service: NcodService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or NcodService()
	return {
		"tenant_id": tenant_id,
		"validations": service.list_validations(tenant_id),
		"releases": service.list_releases(tenant_id),
		"route": "/ncod/publishing",
	}


def deployment_center_model(
	service: NcodService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or NcodService()
	return {
		"tenant_id": tenant_id,
		"releases": service.list_releases(tenant_id),
		"deployments": service.list_deployments(tenant_id),
		"route": "/ncod/deployments",
	}


def connector_bindings_model(
	service: NcodService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or NcodService()
	return {
		"tenant_id": tenant_id,
		"data_bindings": service.list_data_bindings(tenant_id),
		"workflow_bindings": service.list_workflow_bindings(tenant_id),
		"script_extensions": service.list_script_extensions(tenant_id),
		"connector_bindings": service.list_connector_bindings(tenant_id),
		"route": "/ncod/connectors",
	}


def builder_agents_model(
	service: NcodService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or NcodService()
	return {
		"tenant_id": tenant_id,
		"builder_agents": service.list_builder_agents(tenant_id),
		"route": "/ncod/agents",
	}


def audit_trail_model(
	service: NcodService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or NcodService()
	return {
		"tenant_id": tenant_id,
		"audit_events": service.list_audit_events(tenant_id),
		"route": "/ncod/audit",
	}


def analytics_model(
	service: NcodService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or NcodService()
	return {
		"tenant_id": tenant_id,
		"summary": service.dashboard_summary(tenant_id),
		"route": "/ncod/analytics",
	}


def settings_model(
	service: NcodService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or NcodService()
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"configuration": contract["configuration"],
		"rules": contract["rule_engine"]["rules"],
		"streaming": contract["streaming"],
		"audit_events": service.list_audit_events(tenant_id),
		"route": "/ncod/settings",
	}
