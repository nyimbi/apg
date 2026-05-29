"""UI metadata and view-model helpers for Plugin/Extension Framework."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .service import PlgnService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(
	service: PlgnService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or PlgnService()
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"summary": service.dashboard_summary(tenant_id),
		"routes": capability_routes(tenant_id),
		"plugins": service.list_plugins(tenant_id),
		"marketplace": service.list_marketplace_listings(tenant_id),
		"releases": service.list_releases(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
	}


def marketplace_model(service: PlgnService, tenant_id: str = "default") -> dict[str, object]:
	return {
		"route": "/plgn/marketplace",
		"listings": service.list_marketplace_listings(tenant_id),
		"plugins": service.list_plugins(tenant_id),
	}


def plugin_registry_model(service: PlgnService, tenant_id: str = "default") -> dict[str, object]:
	return {
		"route": "/plgn/plugins",
		"plugins": service.list_plugins(tenant_id),
		"installations": service.list_installations(tenant_id),
	}


def permission_review_model(service: PlgnService, tenant_id: str = "default") -> dict[str, object]:
	return {
		"route": "/plgn/permissions",
		"reviews": service.list_permission_reviews(tenant_id),
		"plugins": service.list_plugins(tenant_id),
	}


def sandbox_policy_model(service: PlgnService, tenant_id: str = "default") -> dict[str, object]:
	return {
		"route": "/plgn/sandbox",
		"policies": service.list_sandbox_policies(tenant_id),
	}


def release_manager_model(service: PlgnService, tenant_id: str = "default") -> dict[str, object]:
	return {
		"route": "/plgn/releases",
		"releases": service.list_releases(tenant_id),
		"listings": service.list_marketplace_listings(tenant_id),
	}


def governance_model(service: PlgnService, tenant_id: str = "default") -> dict[str, object]:
	return {
		"route": "/plgn/settings",
		"audit_events": service.list_audit_events(tenant_id),
		"rules": service.describe(tenant_id)["rule_engine"]["rules"],
	}
