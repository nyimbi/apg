"""APG Plugin/Extension Framework capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract

__version__ = "1.0.0"
__capability_id__ = "plgn"
__capability_name__ = "Plugin/Extension Framework"
__apg_dependencies__ = ["auth", "secu", "conf"]

capability_metadata: dict[str, Any] = {
	"name": "plgn",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Tenant plugin manifests, marketplace governance, extension permissions, sandbox policy, and release lifecycle",
	"category": "platform",
	"subcategory": "extensibility",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": ["plugin_registry", "extension_marketplace", "permission_review", "sandbox_policy", "plugin_release_lifecycle"],
	"permissions": ["plgn:view", "plgn:install", "plgn:publish", "plgn:review", "plgn:admin"]
}


def register_capability() -> dict[str, Any]:
	"""Register PLGN with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "plgn",
		"aliases": ["plugins", "extensions", "marketplace"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": ["regy", "agnt", "sbox", "wflo"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"plugin_registry": "Register plugin manifests, owners, versions, signatures, and dependencies",
			"extension_marketplace": "Manage approved extension listings, installation policy, and release channels",
			"permission_review": "Review requested scopes, capabilities, and extension trust boundaries",
			"sandbox_policy": "Bind plugin execution to sandbox and security policy",
			"capability_rules": "Evaluate deterministic plugin-governance rules",
			"visual_theming": "Apply plugin marketplace theme tokens and components"
		},
		"endpoints": {"plugins": "/plgn/api/v1/plugins", "marketplace": "/plgn/api/v1/marketplace", "permissions": "/plgn/api/v1/permissions", "sandbox": "/plgn/api/v1/sandbox", "releases": "/plgn/api/v1/releases"},
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"permissions": capability_metadata["permissions"]
	}


def get_capability_info() -> dict[str, Any]:
	"""Get PLGN capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["contract"] = get_capability_contract()
	return info


__all__ = ["capability_metadata", "register_capability", "get_capability_info", "get_capability_contract", "evaluate_capability_rules", "__version__", "__capability_id__", "__capability_name__", "__apg_dependencies__"]
