"""Executable capability contract for APG Multi-Channel Output."""

from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Any


SUPPORTED_MCHN_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_MCHN_AGENT_ROLES = [
	"route_reviewer",
	"template_reviewer",
	"delivery_reviewer",
	"channel_operator",
	"compliance_reviewer",
	"accessibility_reviewer",
]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"channels": {
		"enabled": ["email", "sms", "push", "pdf", "web", "api", "print"],
		"fallback_required": True,
		"channel_owner_required": True,
		"provider_required": True,
		"health_required": True,
	},
	"rendering": {
		"template_approval_required": True,
		"template_content_required": True,
		"template_channel_required": True,
		"localization_supported": True,
		"theme_policy_required": True,
		"format_validation_required": True,
	},
	"delivery": {
		"recipient_policy_required": True,
		"delivery_actor_required": True,
		"rendered_output_required": True,
		"throttle_policy_required": True,
		"delivery_receipts_enabled": True,
		"sensitive_output_encryption_required": True,
		"event_stream": "bytewax",
	},
	"mchn_agents": {
		"agent_assist_enabled": True,
		"agent_registration_required": True,
		"agent_runtime_required": True,
		"agent_role_required": True,
		"agent_scope_required": True,
		"agent_contribution_disclosure_required": True,
		"supported_runtimes": SUPPORTED_MCHN_AGENT_RUNTIMES,
		"allowed_roles": SUPPORTED_MCHN_AGENT_ROLES,
	},
	"governance": {
		"require_tenant_context": True,
		"audit_output_events": True,
		"restricted_content_filtering": True,
		"compliance_policy_required": True,
		"state_change_audit_required": True,
		"tenant_isolation_required": True,
		"batch_event_stream": "bytewax",
	},
	"observability": {
		"audit_required": True,
		"channel_metrics_required": True,
		"render_metrics_required": True,
		"delivery_metrics_required": True,
		"agent_activity_required": True,
		"event_stream": "bytewax",
	},
	"adapters": {
		"generated_app_runtime": "service.MchnService",
		"api_helpers": "api.py",
		"view_models": "views.py",
		"event_stream": "bytewax",
		"notification": "ntfy",
		"identity": "auth",
		"configuration": "conf",
		"audit_sink": "audl",
		"localization": "i18n",
		"theme": "them",
		"workflow": "wflo",
		"content_policy": "comp",
	},
	"ui": {
		"enable_output_dashboard": True,
		"enable_template_manager": True,
		"enable_route_console": True,
		"enable_channel_monitor": True,
		"enable_agent_panel": True,
		"enable_audit": True,
		"enable_policies": True,
	},
	"theme": {
		"default_theme": "mchn_omnichannel_output",
		"allow_tenant_overrides": True,
	},
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"channels",
		"rendering",
		"delivery",
		"mchn_agents",
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
			"rendering",
			"delivery",
			"mchn_agents",
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
	{"name": "tenant_context_required", "description": "All output operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "channel_requires_owner", "description": "Output channels require an accountable owner.", "condition": {"operation": "create_channel", "channel_owner_assigned": False}, "effect": {"decision": "deny", "reason": "channel_owner_required", "required_action": "assign_channel_owner"}},
	{"name": "channel_requires_provider", "description": "Output channels require provider reference.", "condition": {"operation": "create_channel", "provider_ref_present": False}, "effect": {"decision": "deny", "reason": "channel_provider_required", "required_action": "attach_channel_provider"}},
	{"name": "template_requires_approval", "description": "Output templates require approval.", "condition": {"operation": "publish_template", "template_approved": False}, "effect": {"decision": "deny", "reason": "template_approval_required", "required_action": "approve_template"}},
	{"name": "template_requires_approver", "description": "Approved output templates require approver identity.", "condition": {"operation": "publish_template", "template_approved": True, "template_approver_present": False}, "effect": {"decision": "deny", "reason": "template_approver_required", "required_action": "attach_template_approver"}},
	{"name": "template_requires_content", "description": "Output templates require subject or body content.", "condition": {"operation": "publish_template", "template_content_present": False}, "effect": {"decision": "deny", "reason": "template_content_required", "required_action": "add_template_content"}},
	{"name": "template_requires_channel", "description": "Output templates require at least one channel type.", "condition": {"operation": "publish_template", "template_channel_present": False}, "effect": {"decision": "deny", "reason": "template_channel_required", "required_action": "attach_template_channel"}},
	{"name": "policy_requires_recipient_limit", "description": "Delivery policies require recipient limits.", "condition": {"operation": "create_delivery_policy", "recipient_limit_valid": False}, "effect": {"decision": "deny", "reason": "recipient_policy_required", "required_action": "set_recipient_limit"}},
	{"name": "policy_requires_throttle", "description": "Delivery policies require throttling.", "condition": {"operation": "create_delivery_policy", "throttle_policy_valid": False}, "effect": {"decision": "deny", "reason": "throttle_policy_required", "required_action": "set_delivery_throttle"}},
	{"name": "policy_requires_compliance_ref", "description": "Delivery policies require compliance reference.", "condition": {"operation": "create_delivery_policy", "compliance_ref_present": False}, "effect": {"decision": "deny", "reason": "compliance_policy_required", "required_action": "attach_compliance_policy"}},
	{"name": "sensitive_output_requires_encryption", "description": "Sensitive output requires encryption.", "condition": {"sensitive_output": True, "output_encrypted": False}, "effect": {"decision": "deny", "reason": "output_encryption_required", "required_action": "encrypt_output"}},
	{"name": "render_requires_recipient", "description": "Rendered output requires a recipient reference.", "condition": {"operation": "render_output", "recipient_ref_present": False}, "effect": {"decision": "deny", "reason": "recipient_policy_required", "required_action": "attach_recipient"}},
	{"name": "unhealthy_channel_blocks_delivery", "description": "Unhealthy channels cannot receive delivery.", "condition": {"channel_health": "unhealthy", "delivery_requested": True}, "effect": {"decision": "deny", "reason": "channel_unhealthy", "required_action": "reroute_or_restore_channel"}},
	{"name": "delivery_requires_actor", "description": "Delivery batches require requester identity.", "condition": {"operation": "deliver_batch", "delivery_actor_present": False}, "effect": {"decision": "deny", "reason": "delivery_actor_required", "required_action": "attach_delivery_actor"}},
	{"name": "delivery_requires_rendered_output", "description": "Delivery batches require rendered outputs.", "condition": {"operation": "deliver_batch", "rendered_output_present": False}, "effect": {"decision": "deny", "reason": "rendered_output_required", "required_action": "attach_rendered_outputs"}},
	{"name": "delivery_requires_positive_recipients", "description": "Delivery batches require positive recipient count.", "condition": {"operation": "deliver_batch", "recipient_count_lte": 0}, "effect": {"decision": "deny", "reason": "recipient_policy_required", "required_action": "set_recipient_count"}},
	{"name": "delivery_requires_bytewax_stream", "description": "Delivery lifecycle events require Bytewax event streams.", "condition": {"operation": "deliver_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}},
	{"name": "large_delivery_requires_review", "description": "Large deliveries require review.", "condition": {"recipient_count_gt": 10000, "delivery_review_recorded": False}, "effect": {"decision": "require_review", "reason": "large_delivery_review_required", "required_action": "review_delivery"}},
	{"name": "receipt_requires_provider_message", "description": "Delivery receipts require provider message reference.", "condition": {"operation": "record_receipt", "provider_message_present": False}, "effect": {"decision": "deny", "reason": "provider_message_required", "required_action": "attach_provider_message"}},
	{"name": "mchn_agent_requires_registration", "description": "AI output agents must be registered.", "condition": {"mchn_agent_present": True, "agent_registered": False}, "effect": {"decision": "deny", "reason": "mchn_agent_registration_required", "required_action": "register_mchn_agent"}},
	{"name": "mchn_agent_runtime_supported", "description": "AI output agents must use a supported runtime.", "condition": {"mchn_agent_present": True, "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "mchn_agent_runtime_not_supported", "required_action": "choose_supported_mchn_agent_runtime"}},
	{"name": "mchn_agent_role_supported", "description": "AI output agents must use a supported role.", "condition": {"mchn_agent_present": True, "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "mchn_agent_role_not_supported", "required_action": "choose_supported_mchn_agent_role"}},
	{"name": "mchn_agent_requires_scope", "description": "AI output agents require explicit scope.", "condition": {"mchn_agent_present": True, "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "mchn_agent_scope_required", "required_action": "set_mchn_agent_scope"}},
	{"name": "mchn_agent_requires_disclosure", "description": "AI output-agent contributions require disclosure.", "condition": {"mchn_agent_present": True, "agent_contribution_disclosed": False}, "effect": {"decision": "deny", "reason": "mchn_agent_disclosure_required", "required_action": "disclose_mchn_agent"}},
	{"name": "mchn_state_change_requires_audit", "description": "Output lifecycle state changes require audit evidence.", "condition": {"state_change_requested": True, "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "mchn_audit_event_required", "required_action": "record_mchn_audit_event"}},
	{"name": "batch_output_mutation_requires_bytewax", "description": "Batch output mutations must use Bytewax event streams.", "condition": {"requested_operation": "batch_output_mutation", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}},
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/mchn/dashboard", "component": "MCHNDashboard", "permission": "mchn:view", "nav_group": "Overview"},
	{"name": "render", "path": "/mchn/render", "component": "RenderConsole", "permission": "mchn:render", "nav_group": "Rendering"},
	{"name": "templates", "path": "/mchn/templates", "component": "TemplateManager", "permission": "mchn:manage_templates", "nav_group": "Rendering"},
	{"name": "routes", "path": "/mchn/routes", "component": "RouteConsole", "permission": "mchn:route", "nav_group": "Routing"},
	{"name": "channels", "path": "/mchn/channels", "component": "ChannelMonitor", "permission": "mchn:admin", "nav_group": "Channels"},
	{"name": "agents", "path": "/mchn/agents", "component": "MCHNAgentPanel", "permission": "mchn:admin", "nav_group": "Operations"},
	{"name": "analytics", "path": "/mchn/analytics", "component": "OutputAnalytics", "permission": "mchn:view", "nav_group": "Operations"},
	{"name": "policies", "path": "/mchn/policies", "component": "OutputPolicies", "permission": "mchn:admin", "nav_group": "Governance"},
	{"name": "audit", "path": "/mchn/audit", "component": "OutputAuditTrail", "permission": "mchn:admin", "nav_group": "Governance"},
	{"name": "settings", "path": "/mchn/settings", "component": "MCHNSettings", "permission": "mchn:admin", "nav_group": "Administration"},
]

THEME: dict[str, Any] = {
	"name": "mchn_omnichannel_output",
	"tokens": {
		"color.primary": "#28536B",
		"color.accent": "#DD6B20",
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
		"route_console": {"icon": "route", "status_indicator": "route-pill", "risk_style": "policy-band"},
		"template_manager": {"visual": "template-grid", "highlight": "locale-chip"},
		"channel_monitor": {"visual": "channel-health-table", "status_style": "health-chip"},
		"render_preview": {"visual": "format-preview", "status_style": "validation-chip"},
		"agent_panel": {"visual": "agent-roster", "status_style": "scope-chip"},
		"audit_timeline": {"visual": "event-timeline", "status_style": "output-chip"},
	},
}


def streaming_manifest() -> dict[str, Any]:
	return {
		"processor": "bytewax",
		"topic": "apg.mchn.lifecycle",
		"state": ["channels", "templates", "policies", "routes", "rendered_outputs", "batches", "receipts", "mchn_agents", "audit_events"],
		"events": [
			"mchn_channel_created",
			"mchn_template_published",
			"mchn_policy_created",
			"mchn_route_created",
			"mchn_output_rendered",
			"mchn_delivery_queued",
			"mchn_receipt_recorded",
			"mchn_agent_registered",
		],
		"batch_mutation_guardrail": "batch_output_mutation_requires_bytewax",
	}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "mchn",
		"display_name": "Multi-Channel Output",
		"version": "1.0.0",
		"provides": [
			"channel_routing",
			"format_rendering",
			"output_templates",
			"delivery_policy",
			"delivery_receipts",
			"omnichannel_analytics",
			"mchn_agents",
		],
		"requires": ["ntfy", "auth", "conf", "audl"],
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"view_module": "views.py",
			"api_prefix": "/mchn/api/v1",
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


def event_stream_name(value: str) -> str:
	return value.strip().lower().split("://", 1)[0]


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
			actual = context.get(key[:-3])
			if actual is None or actual == expected:
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
