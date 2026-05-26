"""Executable capability contract for APG Chat and Messaging."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"rooms": {
		"private_rooms_enabled": True,
		"public_rooms_enabled": True,
		"max_members_per_room": 5000,
		"room_owner_required": True
	},
	"messaging": {
		"event_bus_required": True,
		"attachments_enabled": True,
		"max_message_length": 8000,
		"delivery_receipts_enabled": True
	},
	"moderation": {
		"moderation_policy_required": True,
		"restricted_content_filtering": True,
		"nlp_assisted_moderation": True,
		"moderator_review_supported": True
	},
	"governance": {
		"require_tenant_context": True,
		"audit_moderation_actions": True,
		"retention_policy_required": True,
		"external_guest_policy_required": True
	},
	"ui": {
		"enable_chat_console": True,
		"enable_room_manager": True,
		"enable_moderation_queue": True,
		"enable_presence_panel": True
	},
	"theme": {
		"default_theme": "chat_team_messaging",
		"allow_tenant_overrides": True
	}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": ["tenant_id", "rooms", "messaging", "moderation", "governance", "ui", "theme"],
	"properties": {key: {"type": "object"} for key in ["rooms", "messaging", "moderation", "governance", "ui", "theme"]} | {
		"tenant_id": {"type": "string", "minLength": 1}
	}
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All chat operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "room_requires_owner", "description": "Rooms require an accountable owner.", "condition": {"operation": "create_room", "room_owner_assigned": False}, "effect": {"decision": "deny", "reason": "room_owner_required", "required_action": "assign_room_owner"}},
	{"name": "retention_policy_required", "description": "Message rooms require a retention policy.", "condition": {"retention_policy_attached": False}, "effect": {"decision": "deny", "reason": "retention_policy_required", "required_action": "attach_retention_policy"}},
	{"name": "external_guest_requires_policy", "description": "External guests require a guest policy.", "condition": {"external_guest_present": True, "guest_policy_attached": False}, "effect": {"decision": "deny", "reason": "guest_policy_required", "required_action": "attach_guest_policy"}},
	{"name": "restricted_content_requires_moderation", "description": "Restricted content requires moderation before release.", "condition": {"restricted_content_detected": True, "moderation_completed": False}, "effect": {"decision": "deny", "reason": "moderation_required", "required_action": "complete_moderation"}},
	{"name": "large_room_requires_review", "description": "Large rooms require access review.", "condition": {"member_count_gt": 5000, "access_review_recorded": False}, "effect": {"decision": "require_review", "reason": "large_room_review_required", "required_action": "review_room_access"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/chat/dashboard", "component": "CHATDashboard", "permission": "chat:view", "nav_group": "Overview"},
	{"name": "rooms", "path": "/chat/rooms", "component": "RoomManager", "permission": "chat:manage_rooms", "nav_group": "Rooms"},
	{"name": "messages", "path": "/chat/messages", "component": "MessageConsole", "permission": "chat:send", "nav_group": "Messaging"},
	{"name": "presence", "path": "/chat/presence", "component": "PresencePanel", "permission": "chat:view", "nav_group": "Messaging"},
	{"name": "moderation", "path": "/chat/moderation", "component": "ModerationQueue", "permission": "chat:moderate", "nav_group": "Governance"},
	{"name": "retention", "path": "/chat/retention", "component": "RetentionPolicy", "permission": "chat:admin", "nav_group": "Governance"},
	{"name": "analytics", "path": "/chat/analytics", "component": "ChatAnalytics", "permission": "chat:view", "nav_group": "Operations"},
	{"name": "settings", "path": "/chat/settings", "component": "CHATSettings", "permission": "chat:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {
	"name": "chat_team_messaging",
	"tokens": {
		"color.primary": "#2C5282",
		"color.accent": "#319795",
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
		"room_list": {"icon": "messages-square", "status_indicator": "room-pill", "risk_style": "membership-band"},
		"message_thread": {"visual": "threaded-list", "highlight": "receipt-chip"},
		"presence_panel": {"visual": "availability-grid", "status_style": "presence-chip"},
		"moderation_queue": {"visual": "review-lanes", "status_style": "policy-chip"}
	}
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable CHAT capability contract."""
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "chat",
		"display_name": "Chat and Messaging",
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "flask_appbuilder",
			"view_module": "views.py",
			"api_prefix": "/chat/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True
		},
		"theme": deepcopy(THEME)
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate default CHAT governance rules."""
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
