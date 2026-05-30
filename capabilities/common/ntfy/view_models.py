"""Dependency-light view models for package-backed NTFY generated UIs."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .notification_runtime import NotificationRuntime


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(runtime: NotificationRuntime | None = None, tenant_id: str = "default") -> dict[str, object]:
	runtime = runtime or NotificationRuntime()
	contract = runtime.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"summary": runtime.dashboard_summary(tenant_id),
		"routes": capability_routes(tenant_id),
		"recent_audit_events": runtime.list_audit_events(tenant_id)[-10:],
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
	}


def message_console_model(runtime: NotificationRuntime | None = None, tenant_id: str = "default") -> dict[str, object]:
	runtime = runtime or NotificationRuntime()
	deliveries = runtime.list_deliveries(tenant_id)
	return {
		"route": "/ntfy/messages",
		"tenant_id": tenant_id,
		"deliveries": deliveries,
		"review_required": [delivery for delivery in deliveries if delivery["status"] == "review_required"],
		"failed": [delivery for delivery in deliveries if delivery["status"] == "failed"],
		"actions": ["send_message", "send_campaign"],
		"theme_component": "delivery_timeline",
	}


def template_studio_model(runtime: NotificationRuntime | None = None, tenant_id: str = "default") -> dict[str, object]:
	runtime = runtime or NotificationRuntime()
	templates = runtime.list_templates(tenant_id)
	return {
		"route": "/ntfy/templates",
		"tenant_id": tenant_id,
		"templates": templates,
		"pending_approval": [template for template in templates if not template["approved"]],
		"actions": ["register_template", "approve_template"],
		"theme_component": "template_studio",
	}


def campaign_console_model(runtime: NotificationRuntime | None = None, tenant_id: str = "default") -> dict[str, object]:
	runtime = runtime or NotificationRuntime()
	campaigns = runtime.list_campaigns(tenant_id)
	return {
		"route": "/ntfy/campaigns",
		"tenant_id": tenant_id,
		"campaigns": campaigns,
		"drafts": [campaign for campaign in campaigns if campaign["status"] == "draft"],
		"review_required": [campaign for campaign in campaigns if campaign["status"] == "review_required"],
		"sent": [campaign for campaign in campaigns if campaign["status"] == "sent"],
		"actions": ["create_campaign", "approve_campaign", "send_campaign"],
		"theme_component": "campaign_table",
	}


def preference_center_model(runtime: NotificationRuntime | None = None, tenant_id: str = "default") -> dict[str, object]:
	runtime = runtime or NotificationRuntime()
	preferences = runtime.list_preferences(tenant_id)
	return {
		"route": "/ntfy/preferences",
		"tenant_id": tenant_id,
		"preferences": preferences,
		"opted_in": [preference for preference in preferences if preference["opted_in"]],
		"unsubscribed": [preference for preference in preferences if preference["unsubscribed"]],
		"actions": ["register_preference"],
		"theme_component": "preference_panel",
	}


def channel_health_model(runtime: NotificationRuntime | None = None, tenant_id: str = "default") -> dict[str, object]:
	runtime = runtime or NotificationRuntime()
	channels = runtime.list_channels(tenant_id)
	return {
		"route": "/ntfy/channels",
		"tenant_id": tenant_id,
		"channels": channels,
		"unhealthy": [channel for channel in channels if not channel["healthy"]],
		"without_fallback": [channel for channel in channels if not channel["fallback_channel"]],
		"actions": ["register_channel"],
		"theme_component": "channel_matrix",
	}


def analytics_model(runtime: NotificationRuntime | None = None, tenant_id: str = "default") -> dict[str, object]:
	runtime = runtime or NotificationRuntime()
	summary = runtime.dashboard_summary(tenant_id)
	delivery_count = summary["delivery_count"]
	delivered = summary["delivered_count"]
	return {
		"route": "/ntfy/analytics",
		"tenant_id": tenant_id,
		"summary": summary,
		"delivery_rate": round(delivered / delivery_count, 4) if delivery_count else 0.0,
		"theme_component": "delivery_timeline",
	}


def audit_model(runtime: NotificationRuntime | None = None, tenant_id: str = "default") -> dict[str, object]:
	runtime = runtime or NotificationRuntime()
	return {
		"route": "/ntfy/audit",
		"tenant_id": tenant_id,
		"audit_events": runtime.list_audit_events(tenant_id),
		"theme_component": "audit_timeline",
	}


def settings_model(tenant_id: str = "default") -> dict[str, object]:
	contract = get_capability_contract(tenant_id)
	return {
		"route": "/ntfy/settings",
		"tenant_id": tenant_id,
		"configuration": contract["configuration"],
		"theme": contract["theme"],
		"permissions": [route["permission"] for route in contract["ui"]["routes"]],
	}
