"""APG Multi-Channel Output (MCHN) capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract

__version__ = "1.0.0"
__capability_id__ = "mchn"
__capability_name__ = "Multi-Channel Output"
__apg_dependencies__ = ["ntfy", "auth", "conf"]

capability_metadata: dict[str, Any] = {
	"name": "mchn",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Tenant-aware output routing, channel rendering, format conversion, delivery policy, and omnichannel governance",
	"category": "specialized_ai_analytics",
	"subcategory": "multi_channel_output",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": ["channel_routing", "format_rendering", "output_templates", "delivery_policy", "omnichannel_analytics"],
	"permissions": ["mchn:view", "mchn:render", "mchn:route", "mchn:manage_templates", "mchn:admin"]
}


def register_capability() -> dict[str, Any]:
	"""Register MCHN with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "mchn",
		"aliases": ["multi_channel_output", "omnichannel_output", "output_routing"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": ["i18n", "them", "audl", "wflo"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"channel_routing": "Route generated output to tenant-governed channels and fallbacks",
			"format_rendering": "Render messages, documents, reports, and media for channel constraints",
			"output_templates": "Manage reusable output templates with localization and themes",
			"delivery_policy": "Apply recipient, channel, compliance, and throttling policies",
			"capability_rules": "Evaluate deterministic multi-channel output rules",
			"visual_theming": "Apply omnichannel-output theme tokens and components"
		},
		"endpoints": {"render": "/mchn/api/v1/render", "routes": "/mchn/api/v1/routes", "templates": "/mchn/api/v1/templates", "channels": "/mchn/api/v1/channels", "analytics": "/mchn/api/v1/analytics"},
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"permissions": capability_metadata["permissions"]
	}


def get_capability_info() -> dict[str, Any]:
	"""Get MCHN capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["contract"] = get_capability_contract()
	return info


__all__ = ["capability_metadata", "register_capability", "get_capability_info", "get_capability_contract", "evaluate_capability_rules", "__version__", "__capability_id__", "__capability_name__", "__apg_dependencies__"]
