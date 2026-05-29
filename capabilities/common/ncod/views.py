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
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
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
		"data_bindings": service.list_data_bindings(tenant_id),
		"workflow_bindings": service.list_workflow_bindings(tenant_id),
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
		"component_types": ["text", "input", "select", "table", "chart", "button", "form", "metric", "workflow_action"],
		"components": service.list_components(tenant_id),
		"route": "/ncod/components",
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
		"audit_events": service.list_audit_events(tenant_id),
		"route": "/ncod/settings",
	}
