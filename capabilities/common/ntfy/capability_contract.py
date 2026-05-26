"""Executable capability contract for APG Notifications and Alerts."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"channels": {
		"enabled": ["email", "sms", "push", "websocket", "webhook", "slack", "teams"],
		"fallback_routing_enabled": True,
		"delivery_retry_attempts": 3,
		"provider_health_required": True
	},
	"delivery": {
		"event_bus_required": True,
		"max_batch_size": 5000,
		"quiet_hours_enforced": True,
		"priority_override_allowed": True
	},
	"preferences": {
		"recipient_opt_in_required": True,
		"channel_preferences_required": True,
		"unsubscribe_supported": True,
		"consent_audit_required": True
	},
	"governance": {
		"require_tenant_context": True,
		"audit_delivery": True,
		"template_approval_required": True,
		"sensitive_payload_encryption_required": True
	},
	"ui": {
		"enable_notification_dashboard": True,
		"enable_template_studio": True,
		"enable_campaign_console": True,
		"enable_delivery_analytics": True
	},
	"theme": {
		"default_theme": "ntfy_notification_ops",
		"allow_tenant_overrides": True
	}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": ["tenant_id", "channels", "delivery", "preferences", "governance", "ui", "theme"],
	"properties": {key: {"type": "object"} for key in ["channels", "delivery", "preferences", "governance", "ui", "theme"]} | {
		"tenant_id": {"type": "string", "minLength": 1}
	}
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All notification operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "recipient_opt_in_required", "description": "Non-operational notifications require recipient opt-in.", "condition": {"message_class": "marketing", "recipient_opted_in": False}, "effect": {"decision": "deny", "reason": "recipient_opt_in_required", "required_action": "record_recipient_opt_in"}},
	{"name": "approved_template_required", "description": "Campaign sends require approved templates.", "condition": {"operation": "send_campaign", "template_approved": False}, "effect": {"decision": "deny", "reason": "template_approval_required", "required_action": "approve_template"}},
	{"name": "sensitive_payload_requires_encryption", "description": "Sensitive notification payloads require encryption.", "condition": {"sensitive_payload": True, "payload_encrypted": False}, "effect": {"decision": "deny", "reason": "payload_encryption_required", "required_action": "encrypt_payload"}},
	{"name": "provider_health_required", "description": "Messages cannot route to unhealthy providers.", "condition": {"provider_health": "unhealthy", "delivery_requested": True}, "effect": {"decision": "deny", "reason": "provider_unhealthy", "required_action": "reroute_or_restore_provider"}},
	{"name": "large_batch_requires_review", "description": "Large notification batches require review.", "condition": {"recipient_count_gt": 5000, "batch_review_recorded": False}, "effect": {"decision": "require_review", "reason": "large_batch_review_required", "required_action": "review_batch"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/ntfy/dashboard", "component": "NTFYDashboard", "permission": "ntfy:view", "nav_group": "Overview"},
	{"name": "messages", "path": "/ntfy/messages", "component": "MessageConsole", "permission": "ntfy:send", "nav_group": "Delivery"},
	{"name": "templates", "path": "/ntfy/templates", "component": "TemplateStudio", "permission": "ntfy:manage_templates", "nav_group": "Design"},
	{"name": "campaigns", "path": "/ntfy/campaigns", "component": "CampaignConsole", "permission": "ntfy:manage_campaigns", "nav_group": "Campaigns"},
	{"name": "preferences", "path": "/ntfy/preferences", "component": "PreferenceCenter", "permission": "ntfy:view", "nav_group": "Recipients"},
	{"name": "channels", "path": "/ntfy/channels", "component": "ChannelHealth", "permission": "ntfy:admin", "nav_group": "Operations"},
	{"name": "analytics", "path": "/ntfy/analytics", "component": "DeliveryAnalytics", "permission": "ntfy:view", "nav_group": "Operations"},
	{"name": "settings", "path": "/ntfy/settings", "component": "NTFYSettings", "permission": "ntfy:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {
	"name": "ntfy_notification_ops",
	"tokens": {
		"color.primary": "#28536B",
		"color.accent": "#D69E2E",
		"color.success": "#2F855A",
		"color.warning": "#B7791F",
		"color.danger": "#C53030",
		"surface.canvas": "#F7F8FA",
		"surface.panel": "#FFFFFF",
		"text.primary": "#172033",
		"text.secondary": "#52606D",
		"border.radius": "8px",
		"density": "compact"
	},
	"components": {
		"channel_matrix": {"icon": "send", "status_indicator": "channel-pill", "risk_style": "health-band"},
		"delivery_timeline": {"visual": "event-timeline", "highlight": "latency-chip"},
		"campaign_table": {"visual": "campaign-list", "status_style": "approval-chip"},
		"preference_panel": {"visual": "recipient-controls", "status_style": "consent-chip"}
	}
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable NTFY capability contract."""
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "ntfy",
		"display_name": "Notifications and Alerts",
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "flask_appbuilder",
			"view_module": "views.py",
			"api_prefix": "/ntfy/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True
		},
		"theme": deepcopy(THEME)
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate default NTFY governance rules."""
	matched: list[str] = []
	actions: list[dict[str, Any]] = []
	decision = "allow"
	for rule in RULES:
		if _matches(rule["condition"], context):
			matched.append(rule["name"])
			effect = rule["effect"]
			actions.append(effect)
			if effect["decision"] == "deny":
				decision = "deny"
			elif effect["decision"] == "require_review" and decision != "deny":
				decision = "require_review"
	return {"decision": decision, "matched_rules": matched, "actions": actions, "context": context}


def _matches(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if key.endswith("_lt"):
			if not context.get(key[:-3], 0) < expected:
				return False
		elif key.endswith("_gt"):
			if not context.get(key[:-3], 0) > expected:
				return False
		elif context.get(key) != expected:
			return False
	return True


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
	for key, value in source.items():
		if isinstance(value, dict) and isinstance(target.get(key), dict):
			_deep_merge(target[key], value)
		else:
			target[key] = value
