"""Executable capability contract for APG CKM Real-Time Collaboration."""

from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Any


SUPPORTED_RTC_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_RTC_AGENT_ROLES = [
	"session_facilitator",
	"decision_reviewer",
	"transcript_reviewer",
	"risk_moderator",
	"workflow_assistant",
]
SUPPORTED_RTC_MODES = ["chat", "presence", "voice", "video", "screen_share", "co_edit", "whiteboard"]
SUPPORTED_RTC_PROTOCOLS = ["websocket", "webrtc", "grpc", "sip", "rtmp", "socketio"]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"sessions": {
		"owner_required": True,
		"participant_policy_required": True,
		"join_policy_required": True,
		"max_participants": 250,
	},
	"presence": {
		"tenant_scoped": True,
		"heartbeat_required": True,
		"stale_after_seconds": 90,
		"context_disclosure_required": True,
	},
	"messaging": {
		"message_audit_required": True,
		"retention_policy_required": True,
		"sensitive_content_review_supported": True,
	},
	"media": {
		"recording_requires_consent": True,
		"screen_share_requires_permission": True,
		"protocols": SUPPORTED_RTC_PROTOCOLS,
	},
	"collaboration": {
		"page_context_required": True,
		"form_delegation_supported": True,
		"co_edit_locking_required": True,
		"decision_capture_required": True,
	},
	"rtc_agents": {
		"agent_assist_enabled": True,
		"agent_registration_required": True,
		"agent_runtime_required": True,
		"agent_scope_required": True,
		"agent_contribution_disclosure_required": True,
		"supported_runtimes": SUPPORTED_RTC_AGENT_RUNTIMES,
		"allowed_roles": SUPPORTED_RTC_AGENT_ROLES,
	},
	"governance": {
		"audit_collaboration_events": True,
		"decision_trace_required": True,
		"state_change_requires_audit": True,
		"batch_event_stream": "bytewax",
	},
	"observability": {
		"audit_required": True,
		"trace_required": True,
		"session_metrics_required": True,
		"agent_activity_required": True,
		"event_stream": "bytewax",
	},
	"adapters": {
		"generated_app_runtime": "lifecycle.RtcLifecycleService",
		"legacy_runtime": "runtime_app.py",
		"api_helpers": "api.py",
		"view_models": "views.py",
		"event_stream": "bytewax",
		"audit_sink": "audl",
		"identity": "auth",
		"notification": "ckm_not",
		"configuration": "conf",
		"scheduler": "schd",
		"monitoring": "moni",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_room_console": True,
		"enable_presence_panel": True,
		"enable_message_stream": True,
		"enable_media_controls": True,
		"enable_decision_log": True,
		"enable_agent_panel": True,
		"enable_rules": True,
		"enable_audit": True,
		"enable_analytics": True,
	},
	"theme": {
		"default_theme": "ckm_rtc_collaboration_ops",
		"allow_tenant_overrides": True,
	},
}


CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"sessions",
		"presence",
		"messaging",
		"media",
		"collaboration",
		"rtc_agents",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {
		key: {"type": "object"}
		for key in [
			"sessions",
			"presence",
			"messaging",
			"media",
			"collaboration",
			"rtc_agents",
			"governance",
			"observability",
			"adapters",
			"ui",
			"theme",
		]
	}
	| {"tenant_id": {"type": "string", "minLength": 1}},
}


RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "RTC operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "session_requires_owner", "description": "Collaboration sessions require an accountable owner.", "condition": {"operation": "create_session", "owner_present": False}, "effect": {"decision": "deny", "reason": "session_owner_required", "required_action": "assign_session_owner"}},
	{"name": "session_requires_participant_policy", "description": "Collaboration sessions require a participant policy.", "condition": {"operation": "create_session", "participant_policy_attached": False}, "effect": {"decision": "deny", "reason": "participant_policy_required", "required_action": "attach_participant_policy"}},
	{"name": "join_requires_allowed_participant", "description": "Participants must be allowed by the session policy.", "condition": {"operation": "join_session", "participant_allowed": False}, "effect": {"decision": "deny", "reason": "participant_not_allowed", "required_action": "update_participant_policy"}},
	{"name": "presence_requires_heartbeat", "description": "Presence updates require heartbeat evidence.", "condition": {"operation": "update_presence", "heartbeat_present": False}, "effect": {"decision": "deny", "reason": "presence_heartbeat_required", "required_action": "send_presence_heartbeat"}},
	{"name": "message_requires_active_session", "description": "Messages require an active collaboration session.", "condition": {"operation": "post_message", "session_active": False}, "effect": {"decision": "deny", "reason": "active_session_required", "required_action": "reactivate_or_create_session"}},
	{"name": "sensitive_message_requires_review", "description": "Sensitive messages require review before broad sharing.", "condition": {"operation": "post_message", "sensitive_content_detected": True, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "sensitive_content_review_required", "required_action": "record_content_review"}},
	{"name": "screen_share_requires_permission", "description": "Screen sharing requires explicit permission.", "condition": {"operation": "start_screen_share", "screen_share_permission": False}, "effect": {"decision": "deny", "reason": "screen_share_permission_required", "required_action": "grant_screen_share_permission"}},
	{"name": "recording_requires_consent", "description": "Session recording requires consent.", "condition": {"operation": "start_recording", "recording_consent_present": False}, "effect": {"decision": "deny", "reason": "recording_consent_required", "required_action": "capture_recording_consent"}},
	{"name": "decision_requires_trace", "description": "Captured decisions require trace evidence.", "condition": {"operation": "capture_decision", "decision_trace_present": False}, "effect": {"decision": "deny", "reason": "decision_trace_required", "required_action": "attach_decision_trace"}},
	{"name": "rtc_agent_requires_registration", "description": "AI RTC agents must be registered.", "condition": {"rtc_agent_present": True, "agent_registered": False}, "effect": {"decision": "deny", "reason": "rtc_agent_registration_required", "required_action": "register_rtc_agent"}},
	{"name": "rtc_agent_runtime_supported", "description": "AI RTC agents must use a supported runtime.", "condition": {"rtc_agent_present": True, "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "rtc_agent_runtime_not_supported", "required_action": "choose_supported_rtc_agent_runtime"}},
	{"name": "rtc_agent_role_supported", "description": "AI RTC agents must use a supported role.", "condition": {"rtc_agent_present": True, "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "rtc_agent_role_not_supported", "required_action": "choose_supported_rtc_agent_role"}},
	{"name": "rtc_agent_requires_scope", "description": "AI RTC agents require explicit scope.", "condition": {"rtc_agent_present": True, "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "rtc_agent_scope_required", "required_action": "set_rtc_agent_scope"}},
	{"name": "rtc_agent_requires_disclosure", "description": "AI RTC-agent contributions require disclosure.", "condition": {"rtc_agent_present": True, "agent_contribution_disclosed": False}, "effect": {"decision": "deny", "reason": "rtc_agent_disclosure_required", "required_action": "disclose_rtc_agent"}},
	{"name": "rtc_state_change_requires_audit", "description": "RTC lifecycle state changes require audit evidence.", "condition": {"state_change_requested": True, "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "rtc_audit_event_required", "required_action": "record_rtc_audit_event"}},
	{"name": "batch_rtc_mutation_requires_bytewax", "description": "Batch RTC mutations must use Bytewax event streams.", "condition": {"requested_operation": "batch_rtc_mutation", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}},
]


UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/ckm-rtc/dashboard", "component": "RtcDashboard", "permission": "ckm_rtc:view", "nav_group": "Overview"},
	{"name": "rooms", "path": "/ckm-rtc/rooms", "component": "RtcRoomConsole", "permission": "ckm_rtc:manage_rooms", "nav_group": "Collaboration"},
	{"name": "presence", "path": "/ckm-rtc/presence", "component": "RtcPresencePanel", "permission": "ckm_rtc:view", "nav_group": "Collaboration"},
	{"name": "messages", "path": "/ckm-rtc/messages", "component": "RtcMessageStream", "permission": "ckm_rtc:participate", "nav_group": "Collaboration"},
	{"name": "media", "path": "/ckm-rtc/media", "component": "RtcMediaControls", "permission": "ckm_rtc:participate", "nav_group": "Media"},
	{"name": "decisions", "path": "/ckm-rtc/decisions", "component": "RtcDecisionLog", "permission": "ckm_rtc:participate", "nav_group": "Governance"},
	{"name": "agents", "path": "/ckm-rtc/agents", "component": "RtcAgentPanel", "permission": "ckm_rtc:govern", "nav_group": "Governance"},
	{"name": "rules", "path": "/ckm-rtc/rules", "component": "RtcRules", "permission": "ckm_rtc:govern", "nav_group": "Governance"},
	{"name": "analytics", "path": "/ckm-rtc/analytics", "component": "RtcAnalytics", "permission": "ckm_rtc:view", "nav_group": "Insights"},
	{"name": "audit", "path": "/ckm-rtc/audit", "component": "RtcAudit", "permission": "ckm_rtc:view", "nav_group": "Governance"},
	{"name": "settings", "path": "/ckm-rtc/settings", "component": "RtcSettings", "permission": "ckm_rtc:admin", "nav_group": "Administration"},
]


THEME: dict[str, Any] = {
	"name": "ckm_rtc_collaboration_ops",
	"tokens": {
		"color.primary": "#1F4E5F",
		"color.accent": "#2A9D8F",
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
		"room_console": {"icon": "messages-square", "status_indicator": "session-pill", "risk_style": "participant-band"},
		"presence_panel": {"icon": "users", "status_indicator": "heartbeat-chip"},
		"message_stream": {"icon": "message-circle", "status_indicator": "retention-chip"},
		"media_controls": {"icon": "video", "status_indicator": "consent-chip"},
		"decision_log": {"icon": "clipboard-check", "status_indicator": "trace-chip"},
		"rtc_agent_panel": {"icon": "bot", "status_indicator": "scope-chip"},
		"stream_health": {"visual": "event-lane", "status_style": "stream-chip"},
		"audit": {"visual": "event-ledger", "status_style": "decision-chip"},
	},
}


def streaming_manifest() -> dict[str, Any]:
	return {
		"processor": "bytewax",
		"topic": "apg.ckm_rtc.lifecycle",
		"state": ["sessions", "participants", "presence", "messages", "media_events", "decisions", "rtc_agents", "audit_events"],
		"events": [
			"rtc_session_created",
			"rtc_participant_joined",
			"rtc_presence_updated",
			"rtc_message_posted",
			"rtc_screen_share_started",
			"rtc_recording_started",
			"rtc_decision_captured",
			"rtc_agent_registered",
		],
		"batch_mutation_guardrail": "batch_rtc_mutation_requires_bytewax",
	}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "ckm_rtc",
		"display_name": "Real-Time Collaboration",
		"version": "1.0.0",
		"provides": [
			"collaboration_sessions",
			"presence_awareness",
			"real_time_messaging",
			"media_collaboration",
			"decision_capture",
			"page_collaboration",
			"rtc_agents",
		],
		"requires": ["auth", "conf", "audl", "ckm_not"],
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"view_module": "views.py",
			"api_prefix": "/ckm-rtc/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True,
		},
		"theme": deepcopy(THEME),
		"streaming": streaming_manifest(),
	}


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
		if key.endswith("_lte"):
			actual = context.get(key[:-4])
			if not isinstance(actual, Number) or not actual <= expected:
				return False
		elif key.endswith("_lt"):
			actual = context.get(key[:-3])
			if not isinstance(actual, Number) or not actual < expected:
				return False
		elif key.endswith("_gt"):
			actual = context.get(key[:-3])
			if not isinstance(actual, Number) or not actual > expected:
				return False
		elif key.endswith("_gte"):
			actual = context.get(key[:-4])
			if not isinstance(actual, Number) or not actual >= expected:
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
