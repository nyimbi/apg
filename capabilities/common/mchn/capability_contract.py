"""Executable capability contract for APG Multi-Channel Output."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"channels": {"enabled": ["email", "sms", "push", "pdf", "web", "api", "print"], "fallback_required": True, "channel_owner_required": True, "health_required": True},
	"rendering": {"template_approval_required": True, "localization_supported": True, "theme_policy_required": True, "format_validation_required": True},
	"delivery": {"recipient_policy_required": True, "throttle_policy_required": True, "delivery_receipts_enabled": True, "sensitive_output_encryption_required": True},
	"governance": {"require_tenant_context": True, "audit_output_events": True, "restricted_content_filtering": True, "compliance_policy_required": True},
	"ui": {"enable_output_dashboard": True, "enable_template_manager": True, "enable_route_console": True, "enable_channel_monitor": True},
	"theme": {"default_theme": "mchn_omnichannel_output", "allow_tenant_overrides": True}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {"type": "object", "required": ["tenant_id", "channels", "rendering", "delivery", "governance", "ui", "theme"], "properties": {key: {"type": "object"} for key in ["channels", "rendering", "delivery", "governance", "ui", "theme"]} | {"tenant_id": {"type": "string", "minLength": 1}}}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All output operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "channel_requires_owner", "description": "Output channels require an accountable owner.", "condition": {"operation": "create_channel", "channel_owner_assigned": False}, "effect": {"decision": "deny", "reason": "channel_owner_required", "required_action": "assign_channel_owner"}},
	{"name": "template_requires_approval", "description": "Output templates require approval.", "condition": {"operation": "publish_template", "template_approved": False}, "effect": {"decision": "deny", "reason": "template_approval_required", "required_action": "approve_template"}},
	{"name": "sensitive_output_requires_encryption", "description": "Sensitive output requires encryption.", "condition": {"sensitive_output": True, "output_encrypted": False}, "effect": {"decision": "deny", "reason": "output_encryption_required", "required_action": "encrypt_output"}},
	{"name": "unhealthy_channel_blocks_delivery", "description": "Unhealthy channels cannot receive delivery.", "condition": {"channel_health": "unhealthy", "delivery_requested": True}, "effect": {"decision": "deny", "reason": "channel_unhealthy", "required_action": "reroute_or_restore_channel"}},
	{"name": "large_delivery_requires_review", "description": "Large deliveries require review.", "condition": {"recipient_count_gt": 10000, "delivery_review_recorded": False}, "effect": {"decision": "require_review", "reason": "large_delivery_review_required", "required_action": "review_delivery"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/mchn/dashboard", "component": "MCHNDashboard", "permission": "mchn:view", "nav_group": "Overview"},
	{"name": "render", "path": "/mchn/render", "component": "RenderConsole", "permission": "mchn:render", "nav_group": "Rendering"},
	{"name": "templates", "path": "/mchn/templates", "component": "TemplateManager", "permission": "mchn:manage_templates", "nav_group": "Rendering"},
	{"name": "routes", "path": "/mchn/routes", "component": "RouteConsole", "permission": "mchn:route", "nav_group": "Routing"},
	{"name": "channels", "path": "/mchn/channels", "component": "ChannelMonitor", "permission": "mchn:admin", "nav_group": "Channels"},
	{"name": "analytics", "path": "/mchn/analytics", "component": "OutputAnalytics", "permission": "mchn:view", "nav_group": "Operations"},
	{"name": "policies", "path": "/mchn/policies", "component": "OutputPolicies", "permission": "mchn:admin", "nav_group": "Governance"},
	{"name": "settings", "path": "/mchn/settings", "component": "MCHNSettings", "permission": "mchn:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {"name": "mchn_omnichannel_output", "tokens": {"color.primary": "#28536B", "color.accent": "#DD6B20", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F7F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"}, "components": {"route_console": {"icon": "route", "status_indicator": "route-pill", "risk_style": "policy-band"}, "template_manager": {"visual": "template-grid", "highlight": "locale-chip"}, "channel_monitor": {"visual": "channel-health-table", "status_style": "health-chip"}, "render_preview": {"visual": "format-preview", "status_style": "validation-chip"}}}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {"capability": "mchn", "display_name": "Multi-Channel Output", "configuration": config, "configuration_schema": CONFIGURATION_SCHEMA, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "flask_appbuilder", "view_module": "views.py", "api_prefix": "/mchn/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True}, "theme": deepcopy(THEME)}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
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
