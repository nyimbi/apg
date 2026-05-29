"""UI metadata helpers for the UI/UX Theming and Branding capability."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .service import ThemService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(
	service: ThemService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or ThemService()
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"routes": capability_routes(tenant_id),
		"summary": service.dashboard_summary(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
	}


def theme_console_model(
	service: ThemService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or ThemService()
	return {
		"route": "/them/themes",
		"tenant_id": tenant_id,
		"themes": service.list_themes(tenant_id),
		"statuses": ["draft", "preview_ready", "approved", "published", "review_required", "blocked"],
	}


def token_editor_model(
	service: ThemService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or ThemService()
	return {
		"route": "/them/tokens",
		"tenant_id": tenant_id,
		"tokens": service.list_tokens(tenant_id),
		"governed_groups": ["color", "typography", "spacing", "density", "component"],
		"contrast_validation_required": True,
	}


def brand_guidelines_model(
	service: ThemService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or ThemService()
	return {
		"route": "/them/branding",
		"tenant_id": tenant_id,
		"themes": service.list_themes(tenant_id),
		"guidelines_required": True,
	}


def brand_asset_manager_model(
	service: ThemService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or ThemService()
	return {
		"route": "/them/assets",
		"tenant_id": tenant_id,
		"assets": service.list_assets(tenant_id),
		"license_required": True,
	}


def preview_model(
	service: ThemService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or ThemService()
	return {
		"route": "/them/preview",
		"tenant_id": tenant_id,
		"previews": service.list_previews(tenant_id),
		"viewports": ["mobile", "tablet", "desktop"],
	}


def policies_model(
	service: ThemService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or ThemService()
	return {
		"route": "/them/policies",
		"tenant_id": tenant_id,
		"publications": service.list_publications(tenant_id),
		"review_required": [
			publication
			for publication in service.list_publications(tenant_id)
			if publication["status"] == "review_required"
		],
	}


def settings_model(tenant_id: str = "default") -> dict[str, object]:
	contract = get_capability_contract(tenant_id)
	return {
		"route": "/them/settings",
		"tenant_id": tenant_id,
		"configuration": contract["configuration"],
		"theme": contract["theme"],
	}
