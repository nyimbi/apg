"""Executable capability contract for APG Video Conferencing."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"meetings": {
		"waiting_room_enabled": True,
		"host_required": True,
		"max_participants": 500,
		"screen_share_policy_required": True
	},
	"media": {
		"secure_transport_required": True,
		"recording_encryption_required": True,
		"captioning_supported": True,
		"computer_vision_assist_enabled": True
	},
	"recordings": {
		"retention_policy_required": True,
		"recording_consent_required": True,
		"transcript_export_enabled": True,
		"access_audit_required": True
	},
	"governance": {
		"require_tenant_context": True,
		"audit_meeting_events": True,
		"external_guest_policy_required": True,
		"moderation_policy_required": True
	},
	"ui": {
		"enable_meeting_dashboard": True,
		"enable_room_console": True,
		"enable_recording_library": True,
		"enable_caption_workbench": True
	},
	"theme": {
		"default_theme": "vidc_meeting_room",
		"allow_tenant_overrides": True
	}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": ["tenant_id", "meetings", "media", "recordings", "governance", "ui", "theme"],
	"properties": {key: {"type": "object"} for key in ["meetings", "media", "recordings", "governance", "ui", "theme"]} | {
		"tenant_id": {"type": "string", "minLength": 1}
	}
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All video operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "meeting_requires_host", "description": "Meetings require an accountable host.", "condition": {"operation": "start_meeting", "host_present": False}, "effect": {"decision": "deny", "reason": "host_required", "required_action": "assign_or_wait_for_host"}},
	{"name": "external_guest_requires_policy", "description": "External guests require an access policy.", "condition": {"external_guest_present": True, "guest_policy_attached": False}, "effect": {"decision": "deny", "reason": "guest_policy_required", "required_action": "attach_guest_policy"}},
	{"name": "recording_requires_consent", "description": "Meeting recordings require consent.", "condition": {"recording_requested": True, "recording_consent_recorded": False}, "effect": {"decision": "deny", "reason": "recording_consent_required", "required_action": "record_meeting_consent"}},
	{"name": "recording_requires_encryption", "description": "Meeting recordings must be encrypted.", "condition": {"recording_requested": True, "recording_encrypted": False}, "effect": {"decision": "deny", "reason": "recording_encryption_required", "required_action": "encrypt_recording"}},
	{"name": "large_meeting_requires_review", "description": "Large meetings require capacity and moderation review.", "condition": {"participant_count_gt": 500, "capacity_review_recorded": False}, "effect": {"decision": "require_review", "reason": "large_meeting_review_required", "required_action": "review_meeting_capacity"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/vidc/dashboard", "component": "VIDCDashboard", "permission": "vidc:view", "nav_group": "Overview"},
	{"name": "meetings", "path": "/vidc/meetings", "component": "MeetingConsole", "permission": "vidc:schedule", "nav_group": "Meetings"},
	{"name": "rooms", "path": "/vidc/rooms", "component": "RoomManager", "permission": "vidc:moderate", "nav_group": "Meetings"},
	{"name": "participants", "path": "/vidc/participants", "component": "ParticipantPanel", "permission": "vidc:moderate", "nav_group": "Meetings"},
	{"name": "recordings", "path": "/vidc/recordings", "component": "RecordingLibrary", "permission": "vidc:manage_recordings", "nav_group": "Artifacts"},
	{"name": "captions", "path": "/vidc/captions", "component": "CaptionWorkbench", "permission": "vidc:view", "nav_group": "Artifacts"},
	{"name": "analytics", "path": "/vidc/analytics", "component": "MeetingAnalytics", "permission": "vidc:view", "nav_group": "Operations"},
	{"name": "settings", "path": "/vidc/settings", "component": "VIDCSettings", "permission": "vidc:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {
	"name": "vidc_meeting_room",
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
		"meeting_grid": {"icon": "video", "status_indicator": "meeting-pill", "risk_style": "capacity-band"},
		"participant_strip": {"visual": "participant-tiles", "highlight": "host-chip"},
		"recording_library": {"visual": "recording-list", "status_style": "retention-chip"},
		"caption_panel": {"visual": "transcript-lines", "status_style": "language-chip"}
	}
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable VIDC capability contract."""
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "vidc",
		"display_name": "Video Conferencing",
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "flask_appbuilder",
			"view_module": "views.py",
			"api_prefix": "/vidc/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True
		},
		"theme": deepcopy(THEME)
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate default VIDC governance rules."""
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
