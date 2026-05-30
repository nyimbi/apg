"""APG Notifications and Alerts (NTFY) capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract

__version__ = "1.0.0"
__capability_id__ = "ntfy"
__capability_name__ = "Notifications and Alerts"
__apg_dependencies__ = ["mqeb", "auth", "mten"]

capability_metadata: dict[str, Any] = {
	"name": "ntfy",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Tenant-aware notifications, alerts, campaigns, delivery channels, preferences, and engagement analytics",
	"category": "collaboration_communication",
	"subcategory": "notifications",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": [
		"multi_channel_notifications",
		"alert_routing",
		"campaign_delivery",
		"user_preferences",
		"template_governance",
		"channel_health",
		"delivery_audit",
		"engagement_analytics",
	],
	"permissions": ["ntfy:view", "ntfy:send", "ntfy:manage_templates", "ntfy:manage_campaigns", "ntfy:audit", "ntfy:admin"]
}

APG_CAPABILITY_INFO = capability_metadata


def register_capability() -> dict[str, Any]:
	"""Register NTFY with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "ntfy",
		"aliases": ["notifications", "alerts", "notification_engine"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": ["audl", "aicr", "colb", "mchn", "secu", "cach"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"multi_channel_notifications": "Deliver messages across governed tenant channels",
			"alert_routing": "Route operational and business alerts by policy, severity, and preference",
			"campaign_delivery": "Manage scheduled and event-driven notification campaigns",
			"user_preferences": "Respect recipient opt-in, quiet hours, and channel preference controls",
			"template_governance": "Version and approve tenant notification templates",
			"channel_health": "Track provider health, fallback routes, and channel ownership",
			"delivery_audit": "Record notification delivery, preference, template, campaign, and channel audit events",
			"capability_rules": "Evaluate deterministic notification-governance rules",
			"visual_theming": "Apply notification-operations theme tokens and components"
		},
		"endpoints": {
			"status": "/ntfy/api/v1/status",
			"messages": "/ntfy/api/v1/messages",
			"templates": "/ntfy/api/v1/templates",
			"campaigns": "/ntfy/api/v1/campaigns",
			"preferences": "/ntfy/api/v1/preferences",
			"suppression": "/ntfy/api/v1/suppression",
			"channels": "/ntfy/api/v1/channels",
			"analytics": "/ntfy/api/v1/analytics",
			"audit": "/ntfy/api/v1/audit"
		},
		"adapters": contract["configuration"]["adapters"],
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"permissions": capability_metadata["permissions"]
	}


def get_capability_info() -> dict[str, Any]:
	"""Get NTFY capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["contract"] = get_capability_contract()
	return info


def register_notification_capability() -> dict[str, Any]:
	"""Compatibility alias for older notification registration callers."""
	return register_capability()


__all__ = ["APG_CAPABILITY_INFO", "capability_metadata", "register_capability", "register_notification_capability", "get_capability_info", "get_capability_contract", "evaluate_capability_rules", "__version__", "__capability_id__", "__capability_name__", "__apg_dependencies__"]
