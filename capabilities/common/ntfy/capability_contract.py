"""Executable capability contract for APG Notifications and Alerts."""

from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Any


SUPPORTED_NTFY_AGENT_RUNTIMES: list[str] = ["codex", "claude_code", "opencode", "pi"]

SUPPORTED_NTFY_AGENT_ROLES: list[str] = [
	"channel_reviewer",
	"template_reviewer",
	"campaign_reviewer",
	"delivery_reviewer",
	"preference_reviewer",
	"suppression_reviewer",
	"provider_health_reviewer",
	"alert_routing_reviewer",
	"lifecycle_batch_reviewer",
	"notification_steward",
]

PRIVILEGED_NTFY_AGENT_ROLES: list[str] = [
	"campaign_reviewer",
	"delivery_reviewer",
	"suppression_reviewer",
	"provider_health_reviewer",
	"alert_routing_reviewer",
	"lifecycle_batch_reviewer",
	"notification_steward",
]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"channels": {
		"enabled": ["email", "sms", "push", "websocket", "webhook", "slack", "teams"],
		"fallback_routing_enabled": True,
		"delivery_retry_attempts": 3,
		"provider_health_required": True,
		"channel_owner_required": True,
	},
	"delivery": {
		"event_bus_required": True,
		"max_batch_size": 5000,
		"quiet_hours_enforced": True,
		"priority_override_allowed": True,
		"delivery_ttl_minutes": 1440,
		"idempotency_required": True,
	},
	"preferences": {
		"recipient_opt_in_required": True,
		"channel_preferences_required": True,
		"unsubscribe_supported": True,
		"consent_audit_required": True,
		"quiet_hours_supported": True,
	},
	"templates": {
		"template_approval_required": True,
		"template_owner_required": True,
		"locale_required": True,
		"variables_schema_supported": True,
		"versioning_enabled": True,
	},
	"campaigns": {
		"campaign_approval_required": True,
		"large_batch_review_threshold": 5000,
		"audience_required": True,
		"schedule_supported": True,
		"frequency_caps_enabled": True,
	},
	"security": {
		"sensitive_payload_encryption_required": True,
		"webhook_signatures_required": True,
		"tenant_isolation_required": True,
		"audit_delivery": True,
		"secret_redaction_required": True,
	},
	"governance": {
		"require_tenant_context": True,
		"audit_delivery": True,
		"template_approval_required": True,
		"sensitive_payload_encryption_required": True,
		"delivery_consent_required": True,
	},
	"observability": {
		"delivery_metrics_required": True,
		"trace_required": True,
		"audit_required": True,
		"event_stream": "bytewax",
	},
	"agents": {
		"first_class": True,
		"supported_runtimes": SUPPORTED_NTFY_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_NTFY_AGENT_ROLES,
		"privileged_roles": PRIVILEGED_NTFY_AGENT_ROLES,
		"require_scope": True,
		"require_owner": True,
		"require_purpose": True,
		"require_contribution_disclosure": True,
		"require_human_approval_for_privileged_roles": True,
		"adapter_contract": "aicr_provider_neutral_notification_agent_adapter",
	},
	"streaming": {
		"engine": "bytewax",
		"lifecycle_stream": "ntfy.lifecycle",
		"watermark": "event_time",
		"required_processor": "bytewax",
		"required_operations": [
			"channel_batch",
			"preference_batch",
			"template_batch",
			"message_batch",
			"delivery_batch",
			"campaign_batch",
			"suppression_batch",
			"provider_health_batch",
			"notification_agent_batch",
		],
		"topics": [
			"ntfy.channels",
			"ntfy.preferences",
			"ntfy.templates",
			"ntfy.messages",
			"ntfy.deliveries",
			"ntfy.campaigns",
			"ntfy.suppression",
			"ntfy.providers",
			"ntfy.agents",
		],
		"broker_core_dependency_allowed": False,
	},
	"adapters": {
		"generated_app_runtime": "notification_runtime.NotificationRuntime",
		"helper_runtime": "notification_runtime.py",
		"api_helpers": "package_api.py",
		"view_models": "view_models.py",
		"production_runtime": "service.py",
		"production_api": "api.py",
		"production_views": "views.py",
		"event_stream": "bytewax",
		"message_bus": "mqeb",
		"authentication": "auth",
		"multi_tenancy": "mten",
		"audit_sink": "audl",
		"ai_orchestration": "aicr",
		"collaboration": "colb",
		"machine_channel": "mchn",
		"security": "secu",
		"cache": "cach",
		"agent_adapter": "aicr_provider_neutral_notification_agent_adapter",
	},
	"ui": {
		"enable_notification_dashboard": True,
		"enable_message_console": True,
		"enable_template_studio": True,
		"enable_campaign_console": True,
		"enable_preference_center": True,
		"enable_channel_health": True,
		"enable_delivery_analytics": True,
		"enable_suppression_lists": True,
		"enable_notification_agent_roster": True,
		"enable_lifecycle_batch_monitor": True,
		"enable_audit": True,
		"enable_settings": True,
	},
	"theme": {"default_theme": "ntfy_notification_ops", "allow_tenant_overrides": True},
}


CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"channels",
		"delivery",
		"preferences",
		"templates",
		"campaigns",
		"security",
		"governance",
		"observability",
		"agents",
		"streaming",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {key: {"type": "object"} for key in [
		"channels",
		"delivery",
		"preferences",
		"templates",
		"campaigns",
		"security",
		"governance",
		"observability",
		"agents",
		"streaming",
		"adapters",
		"ui",
		"theme",
	]} | {"tenant_id": {"type": "string", "minLength": 1}},
}


RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All notification operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "recipient_requires_address", "description": "Recipient preferences require at least one address.", "condition": {"operation": "register_preference", "recipient_address_present": False}, "effect": {"decision": "deny", "reason": "recipient_address_required", "required_action": "record_recipient_address"}},
	{"name": "recipient_requires_channel_preferences", "description": "Recipients require channel preferences.", "condition": {"operation": "register_preference", "channel_preferences_present": False}, "effect": {"decision": "deny", "reason": "channel_preferences_required", "required_action": "record_channel_preferences"}},
	{"name": "marketing_requires_opt_in", "description": "Marketing notifications require recipient opt-in.", "condition": {"message_class": "marketing", "recipient_opted_in": False}, "effect": {"decision": "deny", "reason": "recipient_opt_in_required", "required_action": "record_recipient_opt_in"}},
	{"name": "unsubscribe_blocks_marketing", "description": "Unsubscribed recipients cannot receive marketing messages.", "condition": {"message_class": "marketing", "recipient_unsubscribed": True}, "effect": {"decision": "deny", "reason": "recipient_unsubscribed", "required_action": "respect_unsubscribe"}},
	{"name": "quiet_hours_require_urgent_priority", "description": "Quiet-hour sends require urgent or critical priority.", "condition": {"quiet_hours_active": True, "priority_override_allowed": False}, "effect": {"decision": "require_review", "reason": "quiet_hours_review_required", "required_action": "review_quiet_hours_override"}},
	{"name": "template_requires_owner", "description": "Templates require accountable owners.", "condition": {"operation": "register_template", "template_owner_assigned": False}, "effect": {"decision": "deny", "reason": "template_owner_required", "required_action": "assign_template_owner"}},
	{"name": "template_requires_name", "description": "Templates require a name.", "condition": {"operation": "register_template", "template_name_present": False}, "effect": {"decision": "deny", "reason": "template_name_required", "required_action": "name_template"}},
	{"name": "template_requires_locale", "description": "Templates require a locale.", "condition": {"operation": "register_template", "template_locale_present": False}, "effect": {"decision": "deny", "reason": "template_locale_required", "required_action": "set_template_locale"}},
	{"name": "template_requires_content", "description": "Templates require channel content.", "condition": {"operation": "register_template", "template_content_present": False}, "effect": {"decision": "deny", "reason": "template_content_required", "required_action": "record_template_content"}},
	{"name": "send_requires_template", "description": "Message sends require a tenant-local template.", "condition": {"operation": "send_message", "template_present": False}, "effect": {"decision": "deny", "reason": "template_required", "required_action": "select_template"}},
	{"name": "approved_template_required", "description": "Notification sends require approved templates.", "condition": {"operation": "send_message", "template_approved": False}, "effect": {"decision": "deny", "reason": "template_approval_required", "required_action": "approve_template"}},
	{"name": "campaign_template_required", "description": "Campaign sends require approved templates.", "condition": {"operation": "send_campaign", "template_approved": False}, "effect": {"decision": "deny", "reason": "template_approval_required", "required_action": "approve_template"}},
	{"name": "sensitive_payload_requires_encryption", "description": "Sensitive notification payloads require encryption.", "condition": {"sensitive_payload": True, "payload_encrypted": False}, "effect": {"decision": "deny", "reason": "payload_encryption_required", "required_action": "encrypt_payload"}},
	{"name": "provider_health_required", "description": "Messages cannot route to unhealthy providers.", "condition": {"provider_health": "unhealthy", "delivery_requested": True}, "effect": {"decision": "deny", "reason": "provider_unhealthy", "required_action": "reroute_or_restore_provider"}},
	{"name": "channel_enabled_required", "description": "Messages require enabled channels.", "condition": {"delivery_requested": True, "channel_enabled": False}, "effect": {"decision": "deny", "reason": "channel_not_enabled", "required_action": "select_enabled_channel"}},
	{"name": "fallback_required_for_failed_primary", "description": "Failed primary channels require fallback routing.", "condition": {"primary_channel_failed": True, "fallback_channel_present": False}, "effect": {"decision": "require_review", "reason": "fallback_channel_required", "required_action": "configure_fallback_channel"}},
	{"name": "large_batch_requires_review", "description": "Large notification batches require review.", "condition": {"recipient_count_gt": 5000, "batch_review_recorded": False}, "effect": {"decision": "require_review", "reason": "large_batch_review_required", "required_action": "review_batch"}},
	{"name": "campaign_requires_audience", "description": "Campaigns require a recipient audience.", "condition": {"operation": "create_campaign", "audience_present": False}, "effect": {"decision": "deny", "reason": "campaign_audience_required", "required_action": "attach_campaign_audience"}},
	{"name": "campaign_requires_owner", "description": "Campaigns require an owner.", "condition": {"operation": "create_campaign", "campaign_owner_assigned": False}, "effect": {"decision": "deny", "reason": "campaign_owner_required", "required_action": "assign_campaign_owner"}},
	{"name": "campaign_requires_approval", "description": "Campaign sends require approval.", "condition": {"operation": "send_campaign", "campaign_approved": False}, "effect": {"decision": "deny", "reason": "campaign_approval_required", "required_action": "approve_campaign"}},
	{"name": "duplicate_idempotency_key_blocked", "description": "Duplicate idempotency keys are blocked.", "condition": {"operation": "send_message", "duplicate_idempotency_key": True}, "effect": {"decision": "deny", "reason": "duplicate_notification_send", "required_action": "reuse_existing_delivery"}},
	{"name": "webhook_requires_signature", "description": "Webhook deliveries require signatures.", "condition": {"channel": "webhook", "webhook_signature_present": False}, "effect": {"decision": "deny", "reason": "webhook_signature_required", "required_action": "sign_webhook_payload"}},
	{"name": "channel_requires_provider", "description": "Enabled channels require providers.", "condition": {"operation": "register_channel", "provider_present": False}, "effect": {"decision": "deny", "reason": "channel_provider_required", "required_action": "attach_channel_provider"}},
	{"name": "provider_requires_owner", "description": "Channel providers require owners.", "condition": {"operation": "register_channel", "channel_owner_assigned": False}, "effect": {"decision": "deny", "reason": "channel_owner_required", "required_action": "assign_channel_owner"}},
	{"name": "delivery_requires_event_bus", "description": "Delivery requires event bus evidence.", "condition": {"delivery_requested": True, "event_bus_present": False}, "effect": {"decision": "deny", "reason": "event_bus_required", "required_action": "attach_event_bus"}},
	{"name": "delivery_requires_audit", "description": "Delivery decisions require audit evidence.", "condition": {"delivery_requested": True, "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "delivery_audit_required", "required_action": "record_delivery_audit"}},
	{"name": "delivery_ttl_required", "description": "Scheduled messages require delivery TTL.", "condition": {"operation": "send_message", "delivery_ttl_present": False}, "effect": {"decision": "require_review", "reason": "delivery_ttl_required", "required_action": "set_delivery_ttl"}},
	{"name": "cross_tenant_notification_access_denied", "description": "Notification records may not cross tenant boundaries.", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_notification_access_denied", "required_action": "use_tenant_local_context"}},
	{"name": "notification_state_change_requires_audit", "description": "Notification state changes require audit evidence.", "condition": {"state_change_requested": True, "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "notification_audit_event_required", "required_action": "record_notification_audit"}},
	{"name": "batch_notification_mutation_requires_bytewax", "description": "Batch notification mutations must use Bytewax event streams.", "condition": {"operation": "batch_notification_mutation", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}},
	{"name": "notification_agent_runtime_supported", "description": "Notification agents must use supported provider-neutral runtimes.", "condition": {"operation": "register_notification_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_notification_agent_runtime", "required_action": "choose_supported_notification_agent_runtime"}},
	{"name": "notification_agent_role_supported", "description": "Notification agents must use supported notification-governance roles.", "condition": {"operation": "register_notification_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_notification_agent_role", "required_action": "choose_supported_notification_agent_role"}},
	{"name": "notification_agent_requires_scope", "description": "Notification agents require explicit channel, template, campaign, delivery, preference, suppression, provider, alert-routing, or lifecycle scope.", "condition": {"operation": "register_notification_agent", "scope_present": False}, "effect": {"decision": "deny", "reason": "notification_agent_scope_required", "required_action": "declare_notification_agent_scope"}},
	{"name": "notification_agent_requires_owner", "description": "Notification agents require an accountable owner.", "condition": {"operation": "register_notification_agent", "owner_present": False}, "effect": {"decision": "deny", "reason": "notification_agent_owner_required", "required_action": "assign_notification_agent_owner"}},
	{"name": "notification_agent_requires_purpose", "description": "Notification agents require a documented delivery-governance purpose.", "condition": {"operation": "register_notification_agent", "purpose_present": False}, "effect": {"decision": "deny", "reason": "notification_agent_purpose_required", "required_action": "document_notification_agent_purpose"}},
	{"name": "notification_agent_requires_contribution_disclosure", "description": "Notification agents must disclose machine-authored channel, template, campaign, delivery, suppression, routing, and lifecycle contributions.", "condition": {"operation": "register_notification_agent", "contribution_disclosed": False}, "effect": {"decision": "deny", "reason": "notification_agent_contribution_disclosure_required", "required_action": "disclose_machine_contribution"}},
	{"name": "notification_agent_privileged_role_requires_human_approval", "description": "Privileged notification-agent roles require human approval evidence.", "condition": {"operation": "register_notification_agent", "privileged_role": True, "human_approval_required": False}, "effect": {"decision": "require_review", "reason": "notification_agent_human_approval_required", "required_action": "record_human_notification_agent_approval"}},
	{"name": "ntfy_lifecycle_batch_requires_mutations", "description": "NTFY lifecycle batches must include at least one mutation.", "condition": {"operation": "validate_ntfy_lifecycle_batch", "mutation_count_lte": 0}, "effect": {"decision": "deny", "reason": "ntfy_lifecycle_batch_empty", "required_action": "include_ntfy_lifecycle_mutations"}},
	{"name": "bytewax_ntfy_stream_required", "description": "NTFY lifecycle batches must be routed through Bytewax.", "condition": {"operation": "validate_ntfy_lifecycle_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_lifecycle_stream_required", "required_action": "route_ntfy_lifecycle_batch_to_bytewax"}},
]


UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/ntfy/dashboard", "component": "NTFYDashboard", "permission": "ntfy:view", "nav_group": "Overview"},
	{"name": "messages", "path": "/ntfy/messages", "component": "MessageConsole", "permission": "ntfy:send", "nav_group": "Delivery"},
	{"name": "templates", "path": "/ntfy/templates", "component": "TemplateStudio", "permission": "ntfy:manage_templates", "nav_group": "Design"},
	{"name": "campaigns", "path": "/ntfy/campaigns", "component": "CampaignConsole", "permission": "ntfy:manage_campaigns", "nav_group": "Campaigns"},
	{"name": "preferences", "path": "/ntfy/preferences", "component": "PreferenceCenter", "permission": "ntfy:view", "nav_group": "Recipients"},
	{"name": "suppression", "path": "/ntfy/suppression", "component": "SuppressionLists", "permission": "ntfy:manage_campaigns", "nav_group": "Recipients"},
	{"name": "channels", "path": "/ntfy/channels", "component": "ChannelHealth", "permission": "ntfy:admin", "nav_group": "Operations"},
	{"name": "analytics", "path": "/ntfy/analytics", "component": "DeliveryAnalytics", "permission": "ntfy:view", "nav_group": "Operations"},
	{"name": "agents", "path": "/ntfy/agents", "component": "NotificationAgentRoster", "permission": "ntfy:admin", "nav_group": "Governance"},
	{"name": "lifecycle", "path": "/ntfy/lifecycle", "component": "NTFYLifecycleBatchMonitor", "permission": "ntfy:admin", "nav_group": "Operations"},
	{"name": "audit", "path": "/ntfy/audit", "component": "NotificationAuditTrail", "permission": "ntfy:audit", "nav_group": "Governance"},
	{"name": "settings", "path": "/ntfy/settings", "component": "NTFYSettings", "permission": "ntfy:admin", "nav_group": "Administration"},
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
		"density": "compact",
	},
	"components": {
		"channel_matrix": {"icon": "send", "status_indicator": "channel-pill", "risk_style": "health-band"},
		"delivery_timeline": {"visual": "event-timeline", "highlight": "latency-chip"},
		"campaign_table": {"visual": "campaign-list", "status_style": "approval-chip"},
		"preference_panel": {"visual": "recipient-controls", "status_style": "consent-chip"},
		"template_studio": {"visual": "template-list", "status_style": "approval-chip"},
		"suppression_list": {"visual": "recipient-table", "status_style": "suppression-chip"},
		"audit_timeline": {"visual": "event-timeline", "status_style": "delivery-chip"},
		"notification_agent_roster": {"icon": "bot", "visual": "agent-roster", "status_style": "approval-chip"},
		"bytewax_lifecycle_panel": {"icon": "activity", "visual": "lifecycle-batch-list", "status_style": "stream-chip"},
	},
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
			"shell": "apg_python",
			"view_module": config["adapters"]["view_models"],
			"api_prefix": "/ntfy/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True,
		},
		"theme": deepcopy(THEME),
		"agents": agent_manifest(),
		"streaming": streaming_manifest(),
	}


def agent_manifest() -> dict[str, Any]:
	"""Return provider-neutral notification agent composition metadata."""
	agents = DEFAULT_CONFIGURATION["agents"]
	return {
		"first_class": bool(agents["first_class"]),
		"supported_runtimes": list(agents["supported_runtimes"]),
		"supported_roles": list(agents["supported_roles"]),
		"privileged_roles": list(agents["privileged_roles"]),
		"required_fields": ["tenant_id", "agent_id", "name", "runtime", "role", "scope", "owner", "purpose"],
		"guardrails": [
			"supported_runtime",
			"supported_role",
			"explicit_scope",
			"accountable_owner",
			"declared_purpose",
			"machine_contribution_disclosure",
			"human_approval_for_privileged_roles",
		],
		"adapter_contract": agents["adapter_contract"],
	}


def streaming_manifest() -> dict[str, Any]:
	"""Return Bytewax lifecycle stream metadata for NTFY composition."""
	streaming = DEFAULT_CONFIGURATION["streaming"]
	return {
		"engine": streaming["engine"],
		"lifecycle_stream": streaming["lifecycle_stream"],
		"watermark": streaming["watermark"],
		"required_processor": streaming["required_processor"],
		"required_operations": list(streaming["required_operations"]),
		"topics": list(streaming["topics"]),
		"broker_core_dependency_allowed": bool(streaming["broker_core_dependency_allowed"]),
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
