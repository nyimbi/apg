"""Executable capability contract for APG Chat and Messaging."""

from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Any


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"rooms": {
		"private_rooms_enabled": True,
		"public_rooms_enabled": True,
		"direct_rooms_enabled": True,
		"max_members_per_room": 5000,
		"room_owner_required": True,
		"retention_policy_required": True,
		"external_guest_policy_required": True,
	},
	"messaging": {
		"event_bus_required": True,
		"event_stream": "bytewax",
		"attachments_enabled": True,
		"max_message_length": 8000,
		"delivery_receipts_enabled": True,
		"threading_enabled": True,
		"reactions_enabled": True,
		"edits_enabled": True,
		"deletions_require_audit": True,
	},
	"presence": {
		"room_presence_enabled": True,
		"typing_indicators_enabled": True,
		"availability_status_enabled": True,
		"presence_ttl_seconds": 90,
	},
	"moderation": {
		"moderation_policy_required": True,
		"restricted_content_filtering": True,
		"nlpc_assisted_moderation": True,
		"moderator_review_supported": True,
		"attachment_scan_required": True,
		"restricted_terms": ["secret", "credential", "restricted"],
	},
	"ai_agents": {
		"agent_participants_enabled": True,
		"agent_registration_required": True,
		"agent_scope_required": True,
		"agent_response_disclosure_required": True,
		"supported_runtimes": ["codex", "claude_code", "opencode", "pi"],
	},
	"security": {
		"tenant_isolation_required": True,
		"authenticated_sender_required": True,
		"dlp_for_external_sharing": True,
		"audit_message_delivery": True,
		"secret_redaction_required": True,
	},
	"governance": {
		"require_tenant_context": True,
		"audit_moderation_actions": True,
		"retention_policy_required": True,
		"external_guest_policy_required": True,
		"large_room_review_threshold": 5000,
	},
	"retention": {
		"default_policy": "retain-90-days",
		"legal_hold_supported": True,
		"export_requires_approval": True,
		"delete_requires_audit": True,
	},
	"observability": {
		"delivery_metrics_required": True,
		"presence_metrics_required": True,
		"moderation_metrics_required": True,
		"audit_required": True,
		"event_stream": "bytewax",
	},
	"adapters": {
		"generated_app_runtime": "service.ChatService",
		"helper_runtime": "service.py",
		"api_helpers": "api.py",
		"view_models": "views.py",
		"event_stream": "bytewax",
		"message_bus": "mqeb",
		"notification": "ntfy",
		"authentication": "auth",
		"multi_tenancy": "mten",
		"audit_sink": "audl",
		"nlp": "nlpc",
		"collaboration": "colb",
		"security": "secu",
		"cache": "cach",
	},
	"ui": {
		"enable_chat_console": True,
		"enable_room_manager": True,
		"enable_direct_messages": True,
		"enable_moderation_queue": True,
		"enable_presence_panel": True,
		"enable_agent_panel": True,
		"enable_retention_console": True,
		"enable_audit": True,
		"enable_analytics": True,
		"enable_settings": True,
	},
	"theme": {"default_theme": "chat_team_messaging", "allow_tenant_overrides": True},
}


CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"rooms",
		"messaging",
		"presence",
		"moderation",
		"ai_agents",
		"security",
		"governance",
		"retention",
		"observability",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {key: {"type": "object"} for key in [
		"rooms",
		"messaging",
		"presence",
		"moderation",
		"ai_agents",
		"security",
		"governance",
		"retention",
		"observability",
		"adapters",
		"ui",
		"theme",
	]} | {"tenant_id": {"type": "string", "minLength": 1}},
}


RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All chat operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "room_requires_owner", "description": "Rooms require an accountable owner.", "condition": {"operation": "create_room", "room_owner_assigned": False}, "effect": {"decision": "deny", "reason": "room_owner_required", "required_action": "assign_room_owner"}},
	{"name": "room_requires_name", "description": "Rooms require a readable name.", "condition": {"operation": "create_room", "room_name_present": False}, "effect": {"decision": "deny", "reason": "room_name_required", "required_action": "name_room"}},
	{"name": "room_requires_member", "description": "Rooms require at least one member or owner.", "condition": {"operation": "create_room", "member_present": False}, "effect": {"decision": "deny", "reason": "room_member_required", "required_action": "add_room_member"}},
	{"name": "retention_policy_required", "description": "Message rooms require a retention policy.", "condition": {"operation": "create_room", "retention_policy_attached": False}, "effect": {"decision": "deny", "reason": "retention_policy_required", "required_action": "attach_retention_policy"}},
	{"name": "external_guest_requires_policy", "description": "External guests require a guest policy.", "condition": {"external_guest_present": True, "guest_policy_attached": False}, "effect": {"decision": "deny", "reason": "guest_policy_required", "required_action": "attach_guest_policy"}},
	{"name": "external_guest_requires_expiry", "description": "External guest rooms require an access expiry.", "condition": {"external_guest_present": True, "guest_access_expiry_present": False}, "effect": {"decision": "require_review", "reason": "guest_expiry_required", "required_action": "set_guest_access_expiry"}},
	{"name": "large_room_requires_review", "description": "Large rooms require access review.", "condition": {"member_count_gt": 5000, "access_review_recorded": False}, "effect": {"decision": "require_review", "reason": "large_room_review_required", "required_action": "review_room_access"}},
	{"name": "active_room_required", "description": "Messages require an active room.", "condition": {"operation": "send_message", "room_active": False}, "effect": {"decision": "deny", "reason": "room_not_active", "required_action": "activate_room"}},
	{"name": "sender_identity_required", "description": "Messages require an authenticated sender.", "condition": {"operation": "send_message", "sender_authenticated": False}, "effect": {"decision": "deny", "reason": "sender_identity_required", "required_action": "authenticate_sender"}},
	{"name": "sender_membership_required", "description": "Senders must belong to the target room.", "condition": {"operation": "send_message", "sender_is_member": False}, "effect": {"decision": "deny", "reason": "sender_not_room_member", "required_action": "join_room_or_change_sender"}},
	{"name": "message_requires_body_or_attachment", "description": "Messages require text or an attachment.", "condition": {"operation": "send_message", "message_payload_present": False}, "effect": {"decision": "deny", "reason": "message_payload_required", "required_action": "enter_message_or_attachment"}},
	{"name": "message_length_within_limit", "description": "Messages must stay within configured length limits.", "condition": {"operation": "send_message", "message_length_within_limit": False}, "effect": {"decision": "deny", "reason": "message_length_exceeded", "required_action": "shorten_message"}},
	{"name": "restricted_content_requires_moderation", "description": "Restricted content requires moderation before release.", "condition": {"restricted_content_detected": True, "moderation_completed": False}, "effect": {"decision": "deny", "reason": "moderation_required", "required_action": "complete_moderation"}},
	{"name": "attachment_requires_scan", "description": "Attachments require scan evidence.", "condition": {"attachment_present": True, "attachment_scan_completed": False}, "effect": {"decision": "deny", "reason": "attachment_scan_required", "required_action": "scan_attachment"}},
	{"name": "external_share_requires_dlp", "description": "Messages shared externally require DLP review.", "condition": {"external_share_requested": True, "dlp_check_completed": False}, "effect": {"decision": "deny", "reason": "dlp_check_required", "required_action": "run_dlp_check"}},
	{"name": "delivery_requires_event_bus", "description": "Message delivery requires event bus evidence.", "condition": {"delivery_requested": True, "event_bus_present": False}, "effect": {"decision": "deny", "reason": "event_bus_required", "required_action": "attach_event_bus"}},
	{"name": "message_requires_audit", "description": "Message state changes require audit evidence.", "condition": {"state_change_requested": True, "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "chat_audit_event_required", "required_action": "record_chat_audit"}},
	{"name": "thread_requires_root_message", "description": "Thread replies require a root message.", "condition": {"operation": "reply_to_thread", "root_message_present": False}, "effect": {"decision": "deny", "reason": "thread_root_required", "required_action": "select_thread_root"}},
	{"name": "reaction_requires_message", "description": "Reactions require a target message.", "condition": {"operation": "react_to_message", "message_present": False}, "effect": {"decision": "deny", "reason": "reaction_message_required", "required_action": "select_message"}},
	{"name": "edit_requires_author_or_moderator", "description": "Message edits require author or moderator rights.", "condition": {"operation": "edit_message", "author_or_moderator": False}, "effect": {"decision": "deny", "reason": "edit_authorization_required", "required_action": "use_author_or_moderator"}},
	{"name": "delete_requires_author_or_moderator", "description": "Message deletes require author or moderator rights.", "condition": {"operation": "delete_message", "author_or_moderator": False}, "effect": {"decision": "deny", "reason": "delete_authorization_required", "required_action": "use_author_or_moderator"}},
	{"name": "presence_requires_authenticated_user", "description": "Presence updates require authenticated users.", "condition": {"operation": "update_presence", "user_authenticated": False}, "effect": {"decision": "deny", "reason": "presence_identity_required", "required_action": "authenticate_user"}},
	{"name": "typing_requires_room_membership", "description": "Typing indicators require room membership.", "condition": {"operation": "update_presence", "typing": True, "user_is_member": False}, "effect": {"decision": "deny", "reason": "typing_room_membership_required", "required_action": "join_room_or_disable_typing"}},
	{"name": "moderation_requires_reviewer", "description": "Moderation decisions require a reviewer.", "condition": {"operation": "review_moderation", "moderator_assigned": False}, "effect": {"decision": "deny", "reason": "moderator_required", "required_action": "assign_moderator"}},
	{"name": "moderation_decision_required", "description": "Moderation reviews require an explicit decision.", "condition": {"operation": "review_moderation", "moderation_decision_present": False}, "effect": {"decision": "deny", "reason": "moderation_decision_required", "required_action": "record_moderation_decision"}},
	{"name": "retention_export_requires_approval", "description": "Retention exports require approval.", "condition": {"operation": "export_retention", "export_approved": False}, "effect": {"decision": "deny", "reason": "retention_export_approval_required", "required_action": "approve_retention_export"}},
	{"name": "ai_agent_requires_registration", "description": "AI chat participants must be registered.", "condition": {"ai_agent_participant": True, "agent_registered": False}, "effect": {"decision": "deny", "reason": "ai_agent_registration_required", "required_action": "register_ai_agent"}},
	{"name": "ai_agent_requires_scope", "description": "AI chat participants require explicit room scope.", "condition": {"ai_agent_participant": True, "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "ai_agent_scope_required", "required_action": "set_agent_scope"}},
	{"name": "ai_response_requires_disclosure", "description": "AI responses require visible disclosure.", "condition": {"ai_agent_participant": True, "ai_response_disclosed": False}, "effect": {"decision": "deny", "reason": "ai_response_disclosure_required", "required_action": "disclose_ai_response"}},
	{"name": "duplicate_message_id_blocked", "description": "Duplicate message IDs are blocked within a tenant.", "condition": {"operation": "send_message", "duplicate_message_id": True}, "effect": {"decision": "deny", "reason": "duplicate_message_id", "required_action": "reuse_existing_message"}},
	{"name": "cross_tenant_chat_access_denied", "description": "Chat records may not cross tenant boundaries.", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_chat_access_denied", "required_action": "use_tenant_local_context"}},
	{"name": "batch_chat_mutation_requires_bytewax", "description": "Batch chat mutations must use Bytewax event streams.", "condition": {"operation": "batch_chat_mutation", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}},
]


UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/chat/dashboard", "component": "CHATDashboard", "permission": "chat:view", "nav_group": "Overview"},
	{"name": "rooms", "path": "/chat/rooms", "component": "RoomManager", "permission": "chat:manage_rooms", "nav_group": "Rooms"},
	{"name": "direct", "path": "/chat/direct", "component": "DirectMessages", "permission": "chat:send", "nav_group": "Messaging"},
	{"name": "messages", "path": "/chat/messages", "component": "MessageConsole", "permission": "chat:send", "nav_group": "Messaging"},
	{"name": "presence", "path": "/chat/presence", "component": "PresencePanel", "permission": "chat:view", "nav_group": "Messaging"},
	{"name": "agents", "path": "/chat/agents", "component": "AgentParticipants", "permission": "chat:manage_rooms", "nav_group": "Messaging"},
	{"name": "moderation", "path": "/chat/moderation", "component": "ModerationQueue", "permission": "chat:moderate", "nav_group": "Governance"},
	{"name": "retention", "path": "/chat/retention", "component": "RetentionPolicy", "permission": "chat:admin", "nav_group": "Governance"},
	{"name": "audit", "path": "/chat/audit", "component": "ChatAuditTrail", "permission": "chat:audit", "nav_group": "Governance"},
	{"name": "analytics", "path": "/chat/analytics", "component": "ChatAnalytics", "permission": "chat:view", "nav_group": "Operations"},
	{"name": "settings", "path": "/chat/settings", "component": "CHATSettings", "permission": "chat:admin", "nav_group": "Administration"},
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
		"density": "compact",
	},
	"components": {
		"room_list": {"icon": "messages-square", "status_indicator": "room-pill", "risk_style": "membership-band"},
		"message_thread": {"visual": "threaded-list", "highlight": "receipt-chip"},
		"direct_messages": {"visual": "conversation-list", "status_style": "receipt-chip"},
		"presence_panel": {"visual": "availability-grid", "status_style": "presence-chip"},
		"agent_panel": {"visual": "agent-roster", "status_style": "scope-chip"},
		"moderation_queue": {"visual": "review-lanes", "status_style": "policy-chip"},
		"audit_timeline": {"visual": "event-timeline", "status_style": "delivery-chip"},
	},
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
			"shell": "apg_python",
			"view_module": config["adapters"]["view_models"],
			"api_prefix": "/chat/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True,
		},
		"theme": deepcopy(THEME),
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
