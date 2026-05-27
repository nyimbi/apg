"""Executable capability contract for APG Collaboration Tools."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"workspaces": {
		"workspace_owner_required": True,
		"guest_access_supported": True,
		"max_participants_per_workspace": 1000,
		"artifact_policy_required": True
	},
	"sessions": {
		"realtime_sync_enabled": True,
		"session_recording_supported": True,
		"presence_required": True,
		"conflict_resolution": "operational_transform"
	},
	"protocols": {
		"enabled": ["websocket", "webrtc", "mqtt", "grpc"],
		"secure_transport_required": True,
		"protocol_health_required": True,
		"fallback_protocol_enabled": True
	},
	"governance": {
		"require_tenant_context": True,
		"audit_collaboration_events": True,
		"external_collaboration_policy_required": True,
		"retention_policy_required": True
	},
	"ui": {
		"enable_workspace_dashboard": True,
		"enable_session_console": True,
		"enable_annotation_panel": True,
		"enable_protocol_monitor": True
	},
	"theme": {
		"default_theme": "colb_collaboration_workspace",
		"allow_tenant_overrides": True
	}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": ["tenant_id", "workspaces", "sessions", "protocols", "governance", "ui", "theme"],
	"properties": {key: {"type": "object"} for key in ["workspaces", "sessions", "protocols", "governance", "ui", "theme"]} | {
		"tenant_id": {"type": "string", "minLength": 1}
	}
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All collaboration operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "workspace_requires_owner", "description": "Workspaces require an accountable owner.", "condition": {"operation": "create_workspace", "workspace_owner_assigned": False}, "effect": {"decision": "deny", "reason": "workspace_owner_required", "required_action": "assign_workspace_owner"}},
	{"name": "external_collaboration_requires_policy", "description": "External collaboration requires a tenant policy.", "condition": {"external_participant_present": True, "external_policy_attached": False}, "effect": {"decision": "deny", "reason": "external_policy_required", "required_action": "attach_external_collaboration_policy"}},
	{"name": "secure_transport_required", "description": "Realtime protocols require secure transport.", "condition": {"realtime_session": True, "secure_transport": False}, "effect": {"decision": "deny", "reason": "secure_transport_required", "required_action": "enable_secure_transport"}},
	{"name": "artifact_policy_required", "description": "Shared artifacts require an artifact policy.", "condition": {"shared_artifact_present": True, "artifact_policy_attached": False}, "effect": {"decision": "deny", "reason": "artifact_policy_required", "required_action": "attach_artifact_policy"}},
	{"name": "large_workspace_requires_review", "description": "Large workspaces require membership review.", "condition": {"participant_count_gt": 1000, "membership_review_recorded": False}, "effect": {"decision": "require_review", "reason": "large_workspace_review_required", "required_action": "review_workspace_membership"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/colb/dashboard", "component": "COLBDashboard", "permission": "colb:view", "nav_group": "Overview"},
	{"name": "workspaces", "path": "/colb/workspaces", "component": "WorkspaceManager", "permission": "colb:create_workspace", "nav_group": "Workspaces"},
	{"name": "sessions", "path": "/colb/sessions", "component": "SessionConsole", "permission": "colb:manage_sessions", "nav_group": "Realtime"},
	{"name": "presence", "path": "/colb/presence", "component": "PresenceSync", "permission": "colb:view", "nav_group": "Realtime"},
	{"name": "annotations", "path": "/colb/annotations", "component": "AnnotationThreads", "permission": "colb:collaborate", "nav_group": "Artifacts"},
	{"name": "protocols", "path": "/colb/protocols", "component": "ProtocolMonitor", "permission": "colb:admin", "nav_group": "Operations"},
	{"name": "analytics", "path": "/colb/analytics", "component": "CollaborationAnalytics", "permission": "colb:view", "nav_group": "Operations"},
	{"name": "settings", "path": "/colb/settings", "component": "COLBSettings", "permission": "colb:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {
	"name": "colb_collaboration_workspace",
	"tokens": {
		"color.primary": "#2B6CB0",
		"color.accent": "#DD6B20",
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
		"workspace_grid": {"icon": "users", "status_indicator": "workspace-pill", "risk_style": "membership-band"},
		"session_canvas": {"visual": "collaborative-surface", "highlight": "presence-chip"},
		"annotation_panel": {"visual": "threaded-comments", "status_style": "decision-chip"},
		"protocol_monitor": {"visual": "protocol-health-table", "status_style": "transport-chip"}
	}
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable COLB capability contract."""
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "colb",
		"display_name": "Collaboration Tools",
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"view_module": "views.py",
			"api_prefix": "/colb/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True
		},
		"theme": deepcopy(THEME)
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate default COLB governance rules."""
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
