"""UI metadata helpers for the Website Builder capability."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .service import WsblService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(
	service: WsblService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or WsblService()
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"summary": service.dashboard_summary(tenant_id),
		"routes": capability_routes(tenant_id),
		"recent_audit_events": service.list_audit_events(tenant_id)[-10:],
		"pending_reviews": service.list_pending_reviews(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
		"streaming": contract["streaming"],
	}


def site_console_model(service: WsblService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or WsblService()
	return {
		"route": "/wsbl/sites",
		"tenant_id": tenant_id,
		"sites": service.list_sites(tenant_id),
		"domains": service.list_domains(tenant_id),
		"actions": ["create_site", "validate_domain", "archive_site"],
	}


def page_library_model(service: WsblService | None = None, tenant_id: str = "default", site_id: str | None = None) -> dict[str, object]:
	service = service or WsblService()
	return {
		"route": "/wsbl/pages",
		"tenant_id": tenant_id,
		"site_id": site_id,
		"pages": service.list_pages(tenant_id, site_id),
		"actions": ["create_page", "edit_page", "preview_page"],
	}


def page_editor_model(service: WsblService | None = None, tenant_id: str = "default", page_id: str | None = None) -> dict[str, object]:
	service = service or WsblService()
	pages = service.list_pages(tenant_id)
	selected = next((page for page in pages if page["id"] == page_id), None) if page_id else None
	return {
		"route": "/wsbl/editor",
		"tenant_id": tenant_id,
		"selected_page": selected,
		"component_palette": service.list_components(tenant_id),
		"actions": ["add_section", "autosave_draft", "mark_review_ready"],
	}


def component_library_model(service: WsblService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or WsblService()
	components = service.list_components(tenant_id)
	return {
		"route": "/wsbl/components",
		"tenant_id": tenant_id,
		"components": components,
		"pending_review": [component for component in components if component["status"] == "review_required"],
		"pending_reviews": [
			component
			for component in service.list_pending_reviews(tenant_id)
			if component.get("component_type")
		],
		"actions": ["create_component", "review_component", "retire_component"],
	}


def publish_queue_model(service: WsblService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or WsblService()
	requests = service.list_publish_requests(tenant_id)
	return {
		"route": "/wsbl/publishing",
		"tenant_id": tenant_id,
		"publish_requests": requests,
		"review_required": [request for request in requests if request["status"] == "review_required"],
		"denied": [request for request in requests if request["status"] == "denied"],
		"pending_reviews": [
			request
			for request in service.list_pending_reviews(tenant_id)
			if request.get("environment")
		],
		"actions": ["request_publish", "publish_site", "rollback_site"],
		"streaming": service.describe(tenant_id)["streaming"],
	}


def analytics_model(service: WsblService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or WsblService()
	summary = service.dashboard_summary(tenant_id)
	return {
		"route": "/wsbl/analytics",
		"tenant_id": tenant_id,
		"summary": summary,
		"signals": {
			"published_site_ratio": _ratio(summary["published_site_count"], summary["site_count"]),
			"custom_component_ratio": _ratio(summary["custom_component_count"], summary["component_count"]),
		},
		"streaming": service.describe(tenant_id)["streaming"],
	}


def agent_workbench_model(service: WsblService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or WsblService()
	contract = service.describe(tenant_id)
	return {
		"route": "/wsbl/agents",
		"tenant_id": tenant_id,
		"agents": service.list_wsbl_agents(tenant_id),
		"supported_runtimes": contract["configuration"]["wsbl_agents"]["supported_runtimes"],
		"supported_roles": contract["configuration"]["wsbl_agents"]["supported_roles"],
		"human_approval_required": contract["configuration"]["wsbl_agents"]["human_approval_required"],
	}


def policy_center_model(service: WsblService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or WsblService()
	contract = service.describe(tenant_id)
	return {
		"route": "/wsbl/policy",
		"tenant_id": tenant_id,
		"rules": contract["rule_engine"]["rules"],
		"streaming": contract["streaming"],
		"publish_requests": service.list_publish_requests(tenant_id),
		"pending_reviews": service.list_pending_reviews(tenant_id),
		"denied_publish_requests": [
			request
			for request in service.list_publish_requests(tenant_id)
			if request["status"] == "denied"
		],
		"pending_components": [
			component
			for component in service.list_components(tenant_id)
			if component["status"] == "review_required"
		],
	}


def settings_model(tenant_id: str = "default") -> dict[str, object]:
	contract = get_capability_contract(tenant_id)
	return {
		"route": "/wsbl/settings",
		"tenant_id": tenant_id,
		"configuration": contract["configuration"],
		"theme": contract["theme"],
		"streaming": contract["streaming"],
		"permissions": [route["permission"] for route in contract["ui"]["routes"]],
	}


def _ratio(numerator: int, denominator: int) -> float:
	return round(numerator / denominator, 4) if denominator else 0.0
