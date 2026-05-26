"""APG UI/UX Theming and Branding capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract

__version__ = "1.0.0"
__capability_id__ = "them"
__capability_name__ = "UI/UX Theming and Branding"
__apg_dependencies__ = ["conf", "auth", "i18n"]

capability_metadata: dict[str, Any] = {
	"name": "them",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Tenant-aware theme tokens, brand systems, preview workflows, and governed visual publishing",
	"category": "experience",
	"subcategory": "design_system",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": ["theme_tokens", "brand_governance", "asset_libraries", "preview_workflows", "visual_theming"],
	"permissions": ["them:view", "them:design", "them:manage_brand", "them:publish", "them:admin"]
}


def register_capability() -> dict[str, Any]:
	"""Register THEM with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "them",
		"aliases": ["theming", "branding", "design-system"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": ["accs", "mchn", "ncod", "wsbl"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"theme_tokens": "Manage governed color, typography, density, and component tokens",
			"brand_governance": "Control brand identity, licensing, review, and publishing policy",
			"asset_libraries": "Curate tenant-approved logos, imagery, iconography, and design assets",
			"preview_workflows": "Preview themes across APG shells and composed capabilities",
			"capability_rules": "Evaluate deterministic visual-governance rules",
			"visual_theming": "Apply APG brand-system theme tokens and components"
		},
		"endpoints": {"themes": "/them/api/v1/themes", "tokens": "/them/api/v1/tokens", "branding": "/them/api/v1/branding", "assets": "/them/api/v1/assets", "publishing": "/them/api/v1/publishing"},
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"permissions": capability_metadata["permissions"]
	}


def get_capability_info() -> dict[str, Any]:
	"""Get THEM capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["contract"] = get_capability_contract()
	return info


__all__ = ["capability_metadata", "register_capability", "get_capability_info", "get_capability_contract", "evaluate_capability_rules", "__version__", "__capability_id__", "__capability_name__", "__apg_dependencies__"]
