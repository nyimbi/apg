"""Executable capability contract for APG Video Conferencing."""

from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Any


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"meetings": {
		"waiting_room_enabled": True,
		"host_required": True,
		"max_participants": 500,
		"capacity_review_threshold": 500,
		"screen_share_policy_required": True,
		"secure_transport_required": True,
		"meeting_state_audit_required": True,
	},
	"media": {
		"secure_transport_required": True,
		"recording_encryption_required": True,
		"captioning_supported": True,
		"supported_caption_languages": ["en", "fr", "sw", "ar"],
		"computer_vision_assist_enabled": True,
		"computer_vision_policy_required": True,
	},
	"recordings": {
		"retention_policy_required": True,
		"recording_consent_required": True,
		"transcript_export_enabled": True,
		"access_audit_required": True,
	},
	"meeting_agents": {
		"agent_assist_enabled": True,
		"agent_registration_required": True,
		"agent_scope_required": True,
		"agent_contribution_disclosure_required": True,
		"supported_runtimes": ["codex", "claude_code", "opencode", "pi"],
		"allowed_roles": ["captioner", "summarizer", "moderator", "action_tracker"],
	},
	"governance": {
		"require_tenant_context": True,
		"audit_meeting_events": True,
		"external_guest_policy_required": True,
		"moderation_policy_required": True,
		"tenant_isolation_required": True,
		"batch_event_stream": "bytewax",
	},
	"observability": {
		"audit_required": True,
		"quality_metrics_required": True,
		"participant_metrics_required": True,
		"event_stream": "bytewax",
	},
	"adapters": {
		"generated_app_runtime": "service.VidcService",
		"runtime_models": "video_runtime.py",
		"api_helpers": "api.py",
		"view_models": "views.py",
		"event_stream": "bytewax",
		"message_bus": "mqeb",
		"collaboration": "colb",
		"computer_vision": "cvsn",
		"notifications": "ntfy",
		"identity": "auth",
		"audit_sink": "audl",
		"speech_and_language": "nlpc",
		"theme": "them",
	},
	"ui": {
		"enable_meeting_dashboard": True,
		"enable_room_console": True,
		"enable_recording_library": True,
		"enable_caption_workbench": True,
		"enable_agent_panel": True,
		"enable_audit": True,
	},
	"theme": {"default_theme": "vidc_meeting_room", "allow_tenant_overrides": True},
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"meetings",
		"media",
		"recordings",
		"meeting_agents",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {key: {"type": "object"} for key in [
		"meetings",
		"media",
		"recordings",
		"meeting_agents",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	]} | {"tenant_id": {"type": "string", "minLength": 1}},
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All video operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "room_requires_name", "description": "Meeting rooms require a readable name.", "condition": {"operation": "create_room", "room_name_present": False}, "effect": {"decision": "deny", "reason": "room_name_required", "required_action": "name_room"}},
	{"name": "room_requires_owner", "description": "Meeting rooms require an accountable owner.", "condition": {"operation": "create_room", "room_owner_present": False}, "effect": {"decision": "deny", "reason": "room_owner_required", "required_action": "assign_room_owner"}},
	{"name": "room_requires_moderation_policy", "description": "Meeting rooms require a moderation policy.", "condition": {"operation": "create_room", "moderation_policy_attached": False}, "effect": {"decision": "deny", "reason": "moderation_policy_required", "required_action": "attach_moderation_policy"}},
	{"name": "meeting_requires_room", "description": "Meetings require a tenant-local room.", "condition": {"operation": "start_meeting", "room_present": False}, "effect": {"decision": "deny", "reason": "room_required", "required_action": "select_room"}},
	{"name": "meeting_requires_host", "description": "Meetings require an accountable host.", "condition": {"operation": "start_meeting", "host_present": False}, "effect": {"decision": "deny", "reason": "host_required", "required_action": "assign_or_wait_for_host"}},
	{"name": "meeting_requires_secure_transport", "description": "Video meetings require secure media transport.", "condition": {"operation": "start_meeting", "secure_transport": False}, "effect": {"decision": "deny", "reason": "secure_transport_required", "required_action": "enable_secure_media_transport"}},
	{"name": "meeting_requires_screen_share_policy", "description": "Screen sharing requires a tenant policy.", "condition": {"screen_share_requested": True, "screen_share_policy_attached": False}, "effect": {"decision": "deny", "reason": "screen_share_policy_required", "required_action": "attach_screen_share_policy"}},
	{"name": "external_guest_requires_policy", "description": "External guests require an access policy.", "condition": {"external_guest_present": True, "guest_policy_attached": False}, "effect": {"decision": "deny", "reason": "guest_policy_required", "required_action": "attach_guest_policy"}},
	{"name": "external_guest_requires_waiting_room", "description": "External guests require waiting-room control.", "condition": {"external_guest_present": True, "waiting_room_enabled": False}, "effect": {"decision": "require_review", "reason": "waiting_room_review_required", "required_action": "enable_waiting_room_or_approve_exception"}},
	{"name": "recording_requires_consent", "description": "Meeting recordings require consent.", "condition": {"recording_requested": True, "recording_consent_recorded": False}, "effect": {"decision": "deny", "reason": "recording_consent_required", "required_action": "record_meeting_consent"}},
	{"name": "recording_requires_encryption", "description": "Meeting recordings must be encrypted.", "condition": {"recording_requested": True, "recording_encrypted": False}, "effect": {"decision": "deny", "reason": "recording_encryption_required", "required_action": "encrypt_recording"}},
	{"name": "recording_requires_retention", "description": "Meeting recordings require retention policy.", "condition": {"recording_requested": True, "recording_retention_policy_attached": False}, "effect": {"decision": "deny", "reason": "recording_retention_required", "required_action": "attach_recording_retention_policy"}},
	{"name": "recording_requires_access_audit", "description": "Recording access requires audit controls.", "condition": {"recording_requested": True, "recording_access_audit_enabled": False}, "effect": {"decision": "deny", "reason": "recording_access_audit_required", "required_action": "enable_recording_access_audit"}},
	{"name": "large_meeting_requires_review", "description": "Large meetings require capacity and moderation review.", "condition": {"participant_count_gt": 500, "capacity_review_recorded": False}, "effect": {"decision": "require_review", "reason": "large_meeting_review_required", "required_action": "review_meeting_capacity"}},
	{"name": "participant_requires_meeting", "description": "Participants require a tenant-local meeting.", "condition": {"operation": "add_participant", "meeting_present": False}, "effect": {"decision": "deny", "reason": "meeting_required", "required_action": "select_meeting"}},
	{"name": "participant_requires_user", "description": "Participants require a user reference.", "condition": {"operation": "add_participant", "user_ref_present": False}, "effect": {"decision": "deny", "reason": "participant_user_ref_required", "required_action": "identify_participant"}},
	{"name": "caption_requires_transcript", "description": "Captions require transcript storage reference.", "condition": {"operation": "generate_captions", "transcript_ref_present": False}, "effect": {"decision": "deny", "reason": "transcript_ref_required", "required_action": "attach_transcript_reference"}},
	{"name": "caption_language_supported", "description": "Captions must use an enabled language.", "condition": {"operation": "generate_captions", "caption_language_supported": False}, "effect": {"decision": "deny", "reason": "caption_language_not_supported", "required_action": "choose_supported_caption_language"}},
	{"name": "computer_vision_assist_requires_policy", "description": "Computer-vision meeting assistance requires policy evidence.", "condition": {"computer_vision_assist_requested": True, "computer_vision_policy_attached": False}, "effect": {"decision": "deny", "reason": "computer_vision_policy_required", "required_action": "attach_computer_vision_policy"}},
	{"name": "meeting_agent_requires_registration", "description": "AI meeting agents must be registered.", "condition": {"meeting_agent_present": True, "agent_registered": False}, "effect": {"decision": "deny", "reason": "meeting_agent_registration_required", "required_action": "register_meeting_agent"}},
	{"name": "meeting_agent_runtime_supported", "description": "AI meeting agents must use a configured runtime.", "condition": {"meeting_agent_present": True, "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "meeting_agent_runtime_not_supported", "required_action": "choose_supported_meeting_agent_runtime"}},
	{"name": "meeting_agent_requires_scope", "description": "AI meeting agents require meeting scope.", "condition": {"meeting_agent_present": True, "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "meeting_agent_scope_required", "required_action": "set_meeting_agent_scope"}},
	{"name": "meeting_agent_requires_disclosure", "description": "AI-generated meeting contributions require disclosure.", "condition": {"meeting_agent_present": True, "agent_contribution_disclosed": False}, "effect": {"decision": "deny", "reason": "meeting_agent_disclosure_required", "required_action": "disclose_meeting_agent"}},
	{"name": "meeting_state_change_requires_audit", "description": "Meeting state changes require audit evidence.", "condition": {"state_change_requested": True, "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "meeting_audit_event_required", "required_action": "record_meeting_audit_event"}},
	{"name": "cross_tenant_video_access_denied", "description": "Video records may not cross tenant boundaries.", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_video_access_denied", "required_action": "use_tenant_local_context"}},
	{"name": "batch_video_mutation_requires_bytewax", "description": "Batch video meeting mutations must use Bytewax event streams.", "condition": {"operation": "batch_video_mutation", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}},
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/vidc/dashboard", "component": "VIDCDashboard", "permission": "vidc:view", "nav_group": "Overview"},
	{"name": "meetings", "path": "/vidc/meetings", "component": "MeetingConsole", "permission": "vidc:schedule", "nav_group": "Meetings"},
	{"name": "rooms", "path": "/vidc/rooms", "component": "RoomManager", "permission": "vidc:moderate", "nav_group": "Meetings"},
	{"name": "participants", "path": "/vidc/participants", "component": "ParticipantPanel", "permission": "vidc:moderate", "nav_group": "Meetings"},
	{"name": "recordings", "path": "/vidc/recordings", "component": "RecordingLibrary", "permission": "vidc:manage_recordings", "nav_group": "Artifacts"},
	{"name": "captions", "path": "/vidc/captions", "component": "CaptionWorkbench", "permission": "vidc:view", "nav_group": "Artifacts"},
	{"name": "agents", "path": "/vidc/agents", "component": "MeetingAgentPanel", "permission": "vidc:moderate", "nav_group": "Meetings"},
	{"name": "analytics", "path": "/vidc/analytics", "component": "MeetingAnalytics", "permission": "vidc:view", "nav_group": "Operations"},
	{"name": "audit", "path": "/vidc/audit", "component": "MeetingAuditTrail", "permission": "vidc:audit", "nav_group": "Governance"},
	{"name": "settings", "path": "/vidc/settings", "component": "VIDCSettings", "permission": "vidc:admin", "nav_group": "Administration"},
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
		"density": "compact",
	},
	"components": {
		"meeting_grid": {"icon": "video", "status_indicator": "meeting-pill", "risk_style": "capacity-band"},
		"participant_strip": {"visual": "participant-tiles", "highlight": "host-chip"},
		"recording_library": {"visual": "recording-list", "status_style": "retention-chip"},
		"caption_panel": {"visual": "transcript-lines", "status_style": "language-chip"},
		"agent_panel": {"visual": "agent-roster", "status_style": "scope-chip"},
		"audit_timeline": {"visual": "event-timeline", "status_style": "meeting-chip"},
	},
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
			"shell": "apg_python",
			"view_module": config["adapters"]["view_models"],
			"api_prefix": "/vidc/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True,
		},
		"theme": deepcopy(THEME),
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
		if key.endswith("_lte"):
			actual = context.get(key[:-4])
			if not isinstance(actual, Number) or not actual <= expected:
				return False
		elif key.endswith("_lt"):
			actual = context.get(key[:-3])
			if not isinstance(actual, Number) or not actual < expected:
				return False
		elif key.endswith("_gte"):
			actual = context.get(key[:-4])
			if not isinstance(actual, Number) or not actual >= expected:
				return False
		elif key.endswith("_gt"):
			actual = context.get(key[:-3])
			if not isinstance(actual, Number) or not actual > expected:
				return False
		elif key.endswith("_ne"):
			if context.get(key[:-3]) == expected:
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
