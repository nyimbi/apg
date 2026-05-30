"""Executable capability contract for APG CKM Notification System."""

from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Any


SUPPORTED_NOTIFICATION_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_NOTIFICATION_AGENT_ROLES = [
	"template_reviewer",
	"audience_reviewer",
	"delivery_reviewer",
	"compliance_reviewer",
	"escalation_reviewer",
]
SUPPORTED_CHANNELS = [
	"email",
	"sms",
	"push",
	"in_app",
	"voice",
	"webhook",
	"whatsapp",
	"slack",
	"teams",
	"web_push",
]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"channels": {
		"supported": SUPPORTED_CHANNELS,
		"provider_registry_required": True,
		"delivery_fallback_required": True,
		"provider_secret_reference_required": True,
	},
	"templates": {
		"approval_required": True,
		"locale_required": True,
		"variable_schema_required": True,
		"channel_content_required": True,
	},
	"campaigns": {
		"audience_policy_required": True,
		"send_window_required": True,
		"approval_required_for_bulk": True,
		"experiment_governance_supported": True,
	},
	"delivery": {
		"tenant_context_required": True,
		"recipient_consent_required": True,
		"preference_enforcement_required": True,
		"quiet_hours_deferral_required": True,
		"priority_escalation_supported": True,
	},
	"preferences": {
		"user_managed": True,
		"channel_opt_out_supported": True,
		"topic_opt_out_supported": True,
		"consent_evidence_required": True,
	},
	"notification_agents": {
		"agent_assist_enabled": True,
		"agent_registration_required": True,
		"agent_runtime_required": True,
		"agent_scope_required": True,
		"agent_contribution_disclosure_required": True,
		"supported_runtimes": SUPPORTED_NOTIFICATION_AGENT_RUNTIMES,
		"allowed_roles": SUPPORTED_NOTIFICATION_AGENT_ROLES,
	},
	"governance": {
		"audit_notification_events": True,
		"consent_trace_required": True,
		"state_change_requires_audit": True,
		"batch_event_stream": "bytewax",
	},
	"observability": {
		"audit_required": True,
		"trace_required": True,
		"delivery_metrics_required": True,
		"agent_activity_required": True,
		"event_stream": "bytewax",
	},
	"adapters": {
		"generated_app_runtime": "lifecycle.NotificationLifecycleService",
		"api_helpers": "api.py",
		"view_models": "views.py",
		"event_stream": "bytewax",
		"audit_sink": "audl",
		"encryption": "encr",
		"configuration": "conf",
		"scheduler": "schd",
		"monitoring": "moni",
		"compliance": "comp",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_template_studio": True,
		"enable_campaign_console": True,
		"enable_delivery_workbench": True,
		"enable_preference_center": True,
		"enable_provider_registry": True,
		"enable_agent_panel": True,
		"enable_rules": True,
		"enable_audit": True,
		"enable_analytics": True,
	},
	"theme": {
		"default_theme": "ckm_not_notification_ops",
		"allow_tenant_overrides": True,
	},
}


CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"channels",
		"templates",
		"campaigns",
		"delivery",
		"preferences",
		"notification_agents",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {
		key: {"type": "object"}
		for key in [
			"channels",
			"templates",
			"campaigns",
			"delivery",
			"preferences",
			"notification_agents",
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
	{"name": "tenant_context_required", "description": "Notification operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "template_requires_channel_content", "description": "Templates require content for every declared delivery channel.", "condition": {"operation": "create_template", "channel_content_complete": False}, "effect": {"decision": "deny", "reason": "channel_content_required", "required_action": "add_channel_content"}},
	{"name": "template_requires_variable_schema", "description": "Templates require variable schema evidence before activation.", "condition": {"operation": "activate_template", "variable_schema_attached": False}, "effect": {"decision": "deny", "reason": "template_variable_schema_required", "required_action": "attach_variable_schema"}},
	{"name": "external_delivery_requires_consent", "description": "External notifications require recipient consent evidence.", "condition": {"delivery_requested": True, "external_channel_requested": True, "recipient_consent_present": False}, "effect": {"decision": "deny", "reason": "recipient_consent_required", "required_action": "capture_recipient_consent"}},
	{"name": "delivery_channel_must_be_allowed", "description": "Delivery channels must be allowed by recipient preferences.", "condition": {"delivery_requested": True, "channel_allowed_by_preference": False}, "effect": {"decision": "deny", "reason": "notification_channel_not_allowed", "required_action": "honor_recipient_channel_preference"}},
	{"name": "suppressed_recipient_blocks_delivery", "description": "Suppressed recipients cannot receive notifications.", "condition": {"delivery_requested": True, "recipient_suppressed": True}, "effect": {"decision": "deny", "reason": "recipient_suppressed", "required_action": "honor_recipient_preference"}},
	{"name": "quiet_hours_requires_deferral", "description": "Quiet-hour notifications require deferral unless a permitted urgent override is present.", "condition": {"delivery_requested": True, "within_quiet_hours": True, "deferral_scheduled": False, "urgent_override_present": False}, "effect": {"decision": "require_review", "reason": "quiet_hours_deferral_required", "required_action": "schedule_delivery_deferral"}},
	{"name": "campaign_requires_audience_policy", "description": "Campaign sends require an audience policy.", "condition": {"campaign_operation": True, "audience_policy_attached": False}, "effect": {"decision": "deny", "reason": "audience_policy_required", "required_action": "attach_audience_policy"}},
	{"name": "bulk_campaign_requires_approval", "description": "Bulk campaigns require approval before execution.", "condition": {"campaign_operation": True, "recipient_count_gt": 500, "approval_recorded": False}, "effect": {"decision": "require_review", "reason": "bulk_campaign_approval_required", "required_action": "record_campaign_approval"}},
	{"name": "provider_credentials_require_secret_reference", "description": "Provider credentials must be referenced through managed secrets.", "condition": {"operation": "register_provider", "provider_secret_ref_present": False}, "effect": {"decision": "deny", "reason": "provider_secret_reference_required", "required_action": "store_provider_secret_reference"}},
	{"name": "notification_agent_requires_registration", "description": "AI notification agents must be registered.", "condition": {"notification_agent_present": True, "agent_registered": False}, "effect": {"decision": "deny", "reason": "notification_agent_registration_required", "required_action": "register_notification_agent"}},
	{"name": "notification_agent_runtime_supported", "description": "AI notification agents must use a supported runtime.", "condition": {"notification_agent_present": True, "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "notification_agent_runtime_not_supported", "required_action": "choose_supported_notification_agent_runtime"}},
	{"name": "notification_agent_role_supported", "description": "AI notification agents must use a supported role.", "condition": {"notification_agent_present": True, "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "notification_agent_role_not_supported", "required_action": "choose_supported_notification_agent_role"}},
	{"name": "notification_agent_requires_scope", "description": "AI notification agents require explicit scope.", "condition": {"notification_agent_present": True, "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "notification_agent_scope_required", "required_action": "set_notification_agent_scope"}},
	{"name": "notification_agent_requires_disclosure", "description": "AI notification-agent contributions require disclosure.", "condition": {"notification_agent_present": True, "agent_contribution_disclosed": False}, "effect": {"decision": "deny", "reason": "notification_agent_disclosure_required", "required_action": "disclose_notification_agent"}},
	{"name": "notification_state_change_requires_audit", "description": "Notification lifecycle state changes require audit evidence.", "condition": {"state_change_requested": True, "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "notification_audit_event_required", "required_action": "record_notification_audit_event"}},
	{"name": "batch_notification_mutation_requires_bytewax", "description": "Batch notification mutations must use Bytewax event streams.", "condition": {"requested_operation": "batch_notification_mutation", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}},
]


UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/ckm-not/dashboard", "component": "NotificationDashboard", "permission": "ckm_not:view", "nav_group": "Overview"},
	{"name": "templates", "path": "/ckm-not/templates", "component": "TemplateStudio", "permission": "ckm_not:manage_templates", "nav_group": "Design"},
	{"name": "campaigns", "path": "/ckm-not/campaigns", "component": "CampaignConsole", "permission": "ckm_not:manage_campaigns", "nav_group": "Campaigns"},
	{"name": "deliveries", "path": "/ckm-not/deliveries", "component": "DeliveryWorkbench", "permission": "ckm_not:send", "nav_group": "Operations"},
	{"name": "preferences", "path": "/ckm-not/preferences", "component": "PreferenceCenter", "permission": "ckm_not:view_preferences", "nav_group": "Governance"},
	{"name": "providers", "path": "/ckm-not/providers", "component": "ProviderRegistry", "permission": "ckm_not:admin", "nav_group": "Administration"},
	{"name": "agents", "path": "/ckm-not/agents", "component": "NotificationAgentPanel", "permission": "ckm_not:govern", "nav_group": "Governance"},
	{"name": "rules", "path": "/ckm-not/rules", "component": "NotificationRules", "permission": "ckm_not:govern", "nav_group": "Governance"},
	{"name": "analytics", "path": "/ckm-not/analytics", "component": "NotificationAnalytics", "permission": "ckm_not:view", "nav_group": "Insights"},
	{"name": "audit", "path": "/ckm-not/audit", "component": "NotificationAudit", "permission": "ckm_not:view", "nav_group": "Governance"},
	{"name": "settings", "path": "/ckm-not/settings", "component": "NotificationSettings", "permission": "ckm_not:admin", "nav_group": "Administration"},
]


THEME: dict[str, Any] = {
	"name": "ckm_not_notification_ops",
	"tokens": {
		"color.primary": "#264653",
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
		"template_studio": {"icon": "file-text", "status_indicator": "approval-pill", "risk_style": "content-band"},
		"campaign_console": {"icon": "send", "status_indicator": "approval-pill", "risk_style": "audience-band"},
		"delivery_workbench": {"icon": "radio", "status_indicator": "delivery-chip", "risk_style": "sla-band"},
		"preference_center": {"icon": "sliders-horizontal", "status_indicator": "consent-chip"},
		"provider_registry": {"icon": "plug", "status_indicator": "secret-chip"},
		"notification_agent_panel": {"icon": "bot", "status_indicator": "scope-chip"},
		"stream_health": {"visual": "event-lane", "status_style": "stream-chip"},
		"audit": {"visual": "event-ledger", "status_style": "decision-chip"},
	},
}


def streaming_manifest() -> dict[str, Any]:
	return {
		"processor": "bytewax",
		"topic": "apg.ckm_not.lifecycle",
		"state": [
			"templates",
			"campaigns",
			"deliveries",
			"preferences",
			"providers",
			"notification_agents",
			"audit_events",
		],
		"events": [
			"notification_template_created",
			"notification_template_approved",
			"notification_campaign_requested",
			"notification_campaign_approved",
			"notification_delivery_requested",
			"notification_delivery_deferred",
			"notification_delivery_recorded",
			"notification_preference_updated",
			"notification_provider_registered",
			"notification_agent_registered",
		],
		"batch_mutation_guardrail": "batch_notification_mutation_requires_bytewax",
	}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "ckm_not",
		"display_name": "Notification System",
		"version": "1.0.0",
		"provides": [
			"notification_delivery",
			"template_management",
			"campaign_orchestration",
			"preference_center",
			"channel_provider_registry",
			"engagement_analytics",
			"notification_agents",
		],
		"requires": ["auth", "conf", "encr", "audl"],
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"view_module": "views.py",
			"api_prefix": "/ckm-not/api/v1",
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
