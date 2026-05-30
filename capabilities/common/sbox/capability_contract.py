"""Executable capability contract for APG Sandbox/Testing Environment."""

from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Any


SUPPORTED_SBOX_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_SBOX_AGENT_ROLES = [
	"isolation_reviewer",
	"dataset_reviewer",
	"run_reviewer",
	"plugin_test_reviewer",
	"security_reviewer",
	"lifecycle_reviewer",
]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"sandboxes": {
		"sandbox_owner_required": True,
		"template_required": True,
		"ttl_hours": 24,
		"environment_isolation_required": True,
		"run_event_stream": "bytewax",
	},
	"isolation": {
		"network_policy_required": True,
		"secret_redaction_required": True,
		"data_masking_required": True,
		"outbound_access_denied_by_default": True,
		"approved_profile_required": True,
	},
	"datasets": {
		"synthetic_data_supported": True,
		"production_data_review_required": True,
		"dataset_lineage_required": True,
		"retention_policy_required": True,
		"masking_required_for_sensitive_data": True,
	},
	"sbox_agents": {
		"agent_assist_enabled": True,
		"agent_registration_required": True,
		"agent_runtime_required": True,
		"agent_role_required": True,
		"agent_scope_required": True,
		"agent_contribution_disclosure_required": True,
		"supported_runtimes": SUPPORTED_SBOX_AGENT_RUNTIMES,
		"allowed_roles": SUPPORTED_SBOX_AGENT_ROLES,
	},
	"governance": {
		"require_tenant_context": True,
		"audit_sandbox_runs": True,
		"long_lived_review_hours": 48,
		"plugin_test_policy_required": True,
		"state_change_audit_required": True,
		"tenant_isolation_required": True,
		"batch_event_stream": "bytewax",
	},
	"observability": {
		"audit_required": True,
		"sandbox_metrics_required": True,
		"run_metrics_required": True,
		"dataset_metrics_required": True,
		"agent_activity_required": True,
		"event_stream": "bytewax",
	},
	"adapters": {
		"generated_app_runtime": "service.SboxService",
		"api_helpers": "api.py",
		"view_models": "views.py",
		"event_stream": "bytewax",
		"plugins": "plgn",
		"security": "secu",
		"environment": "envm",
		"audit_sink": "audl",
		"logging": "logt",
		"deployment": "depl",
	},
	"ui": {
		"enable_sandbox_console": True,
		"enable_template_library": True,
		"enable_run_monitor": True,
		"enable_policy_center": True,
		"enable_agent_panel": True,
		"enable_audit": True,
	},
	"theme": {
		"default_theme": "sbox_safe_testing",
		"allow_tenant_overrides": True,
	},
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"sandboxes",
		"isolation",
		"datasets",
		"sbox_agents",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {
		key: {"type": "object"}
		for key in [
			"sandboxes",
			"isolation",
			"datasets",
			"sbox_agents",
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
	{"name": "tenant_context_required", "description": "All sandbox operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "sandbox_requires_owner", "description": "Sandboxes require an accountable owner.", "condition": {"operation": "create_sandbox", "sandbox_owner_assigned": False}, "effect": {"decision": "deny", "reason": "sandbox_owner_required", "required_action": "assign_sandbox_owner"}},
	{"name": "sandbox_requires_template", "description": "Sandboxes require a template.", "condition": {"operation": "create_sandbox", "template_present": False}, "effect": {"decision": "deny", "reason": "sandbox_template_required", "required_action": "attach_sandbox_template"}},
	{"name": "sandbox_requires_isolation_profile", "description": "Sandboxes require an isolation profile.", "condition": {"operation": "create_sandbox", "isolation_profile_attached": False}, "effect": {"decision": "deny", "reason": "isolation_profile_required", "required_action": "attach_isolation_profile"}},
	{"name": "sandbox_requires_positive_ttl", "description": "Sandboxes require positive TTL.", "condition": {"operation": "create_sandbox", "ttl_hours_lt": 1}, "effect": {"decision": "deny", "reason": "sandbox_ttl_required", "required_action": "set_sandbox_ttl"}},
	{"name": "secrets_require_redaction", "description": "Sandbox secrets require redaction policy.", "condition": {"secret_access_requested": True, "secret_redaction_enabled": False}, "effect": {"decision": "deny", "reason": "secret_redaction_required", "required_action": "enable_secret_redaction"}},
	{"name": "outbound_network_requires_approval", "description": "Outbound sandbox network access requires approval.", "condition": {"outbound_network_requested": True, "network_approval_recorded": False}, "effect": {"decision": "deny", "reason": "outbound_network_approval_required", "required_action": "approve_outbound_network"}},
	{"name": "long_lived_sandbox_requires_review", "description": "Long-lived sandboxes require review.", "condition": {"ttl_hours_gt": 48, "lifecycle_review_recorded": False}, "effect": {"decision": "require_review", "reason": "long_lived_sandbox_review_required", "required_action": "review_sandbox_lifecycle"}},
	{"name": "dataset_requires_owner", "description": "Sandbox datasets require an owner.", "condition": {"operation": "register_dataset", "dataset_owner_assigned": False}, "effect": {"decision": "deny", "reason": "dataset_owner_required", "required_action": "assign_dataset_owner"}},
	{"name": "dataset_requires_lineage", "description": "Sandbox datasets require lineage.", "condition": {"operation": "register_dataset", "dataset_lineage_present": False}, "effect": {"decision": "deny", "reason": "dataset_lineage_required", "required_action": "attach_dataset_lineage"}},
	{"name": "dataset_requires_retention", "description": "Sandbox datasets require retention policy.", "condition": {"operation": "register_dataset", "retention_days_lt": 1}, "effect": {"decision": "deny", "reason": "retention_policy_required", "required_action": "set_dataset_retention"}},
	{"name": "production_dataset_requires_review", "description": "Production sample datasets require review.", "condition": {"operation": "register_dataset", "production_dataset": True, "production_review_recorded": False}, "effect": {"decision": "deny", "reason": "production_data_review_required", "required_action": "review_production_dataset"}},
	{"name": "sensitive_dataset_requires_masking", "description": "Sensitive sandbox datasets require masking.", "condition": {"operation": "register_dataset", "sensitive_dataset": True, "dataset_masked": False}, "effect": {"decision": "deny", "reason": "dataset_masking_required", "required_action": "mask_dataset"}},
	{"name": "run_requires_requester", "description": "Sandbox runs require requester identity.", "condition": {"operation": "start_run", "run_requester_present": False}, "effect": {"decision": "deny", "reason": "run_requester_required", "required_action": "set_run_requester"}},
	{"name": "run_requires_test_count", "description": "Sandbox runs require positive test count.", "condition": {"operation": "start_run", "tests_requested_lt": 1}, "effect": {"decision": "deny", "reason": "tests_requested_required", "required_action": "set_tests_requested"}},
	{"name": "plugin_run_requires_policy", "description": "Plugin sandbox runs require plugin test policy.", "condition": {"operation": "start_run", "plugin_run": True, "plugin_test_policy_present": False}, "effect": {"decision": "deny", "reason": "plugin_test_policy_required", "required_action": "attach_plugin_test_policy"}},
	{"name": "run_requires_bytewax_stream", "description": "Sandbox run lifecycle events require Bytewax streams.", "condition": {"operation": "start_run", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}},
	{"name": "sbox_agent_requires_registration", "description": "AI sandbox agents must be registered.", "condition": {"sbox_agent_present": True, "agent_registered": False}, "effect": {"decision": "deny", "reason": "sbox_agent_registration_required", "required_action": "register_sbox_agent"}},
	{"name": "sbox_agent_runtime_supported", "description": "AI sandbox agents must use a supported runtime.", "condition": {"sbox_agent_present": True, "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "sbox_agent_runtime_not_supported", "required_action": "choose_supported_sbox_agent_runtime"}},
	{"name": "sbox_agent_role_supported", "description": "AI sandbox agents must use a supported role.", "condition": {"sbox_agent_present": True, "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "sbox_agent_role_not_supported", "required_action": "choose_supported_sbox_agent_role"}},
	{"name": "sbox_agent_requires_scope", "description": "AI sandbox agents require explicit scope.", "condition": {"sbox_agent_present": True, "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "sbox_agent_scope_required", "required_action": "set_sbox_agent_scope"}},
	{"name": "sbox_agent_requires_disclosure", "description": "AI sandbox-agent contributions require disclosure.", "condition": {"sbox_agent_present": True, "agent_contribution_disclosed": False}, "effect": {"decision": "deny", "reason": "sbox_agent_disclosure_required", "required_action": "disclose_sbox_agent"}},
	{"name": "sbox_state_change_requires_audit", "description": "Sandbox lifecycle state changes require audit evidence.", "condition": {"state_change_requested": True, "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "sbox_audit_event_required", "required_action": "record_sbox_audit_event"}},
	{"name": "batch_sandbox_mutation_requires_bytewax", "description": "Batch sandbox mutations must use Bytewax event streams.", "condition": {"requested_operation": "batch_sandbox_mutation", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}},
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/sbox/dashboard", "component": "SBOXDashboard", "permission": "sbox:view", "nav_group": "Overview"},
	{"name": "sandboxes", "path": "/sbox/sandboxes", "component": "SandboxConsole", "permission": "sbox:create", "nav_group": "Sandboxes"},
	{"name": "templates", "path": "/sbox/templates", "component": "TemplateLibrary", "permission": "sbox:create", "nav_group": "Templates"},
	{"name": "datasets", "path": "/sbox/datasets", "component": "DatasetManager", "permission": "sbox:manage_policy", "nav_group": "Data"},
	{"name": "runs", "path": "/sbox/runs", "component": "RunMonitor", "permission": "sbox:run_tests", "nav_group": "Runs"},
	{"name": "agents", "path": "/sbox/agents", "component": "SBOXAgentPanel", "permission": "sbox:admin", "nav_group": "Operations"},
	{"name": "policies", "path": "/sbox/policies", "component": "PolicyCenter", "permission": "sbox:manage_policy", "nav_group": "Governance"},
	{"name": "audit", "path": "/sbox/audit", "component": "SandboxAuditTrail", "permission": "sbox:admin", "nav_group": "Governance"},
	{"name": "logs", "path": "/sbox/logs", "component": "SandboxLogs", "permission": "sbox:view", "nav_group": "Operations"},
	{"name": "settings", "path": "/sbox/settings", "component": "SBOXSettings", "permission": "sbox:admin", "nav_group": "Administration"},
]

THEME: dict[str, Any] = {
	"name": "sbox_safe_testing",
	"tokens": {
		"color.primary": "#234E52",
		"color.accent": "#3182CE",
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
		"sandbox_card": {"icon": "container", "status_indicator": "ttl-pill", "risk_style": "isolation-band"},
		"run_monitor": {"visual": "test-timeline", "highlight": "result-chip"},
		"dataset_manager": {"visual": "masked-data-grid", "status_style": "lineage-chip"},
		"policy_center": {"visual": "guardrail-list", "status_style": "approval-chip"},
		"agent_panel": {"visual": "agent-roster", "status_style": "scope-chip"},
		"audit_timeline": {"visual": "event-timeline", "status_style": "sandbox-chip"},
	},
}


def streaming_manifest() -> dict[str, Any]:
	return {
		"processor": "bytewax",
		"topic": "apg.sbox.lifecycle",
		"state": [
			"isolation_profiles",
			"templates",
			"datasets",
			"sandboxes",
			"runs",
			"sbox_agents",
			"audit_events",
		],
		"events": [
			"sbox_isolation_profile_created",
			"sbox_template_created",
			"sbox_dataset_registered",
			"sbox_sandbox_created",
			"sbox_run_started",
			"sbox_run_completed",
			"sbox_agent_registered",
		],
		"batch_mutation_guardrail": "batch_sandbox_mutation_requires_bytewax",
	}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "sbox",
		"display_name": "Sandbox/Testing Environment",
		"version": "1.0.0",
		"provides": [
			"sandbox_registry",
			"isolation_profiles",
			"test_runs",
			"synthetic_datasets",
			"safety_policy",
			"sbox_agents",
		],
		"requires": ["plgn", "secu", "envm", "audl"],
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"view_module": "views.py",
			"api_prefix": "/sbox/api/v1",
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
