"""Executable capability contract for APG Environment Management."""

from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Any


SUPPORTED_ENVM_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_ENVM_AGENT_ROLES = [
	"environment_reviewer",
	"promotion_reviewer",
	"drift_reviewer",
	"secret_scope_reviewer",
	"policy_reviewer",
]
SUPPORTED_STAGES = ["development", "test", "staging", "production"]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"environments": {
		"environment_owner_required": True,
		"stage_policy_required": True,
		"supported_stages": SUPPORTED_STAGES,
		"region_policy_required": True,
		"configuration_source_required": True,
		"production_locked_by_default": True,
	},
	"promotion": {
		"promotion_path_required": True,
		"approval_required": True,
		"deployment_link_required": True,
		"rollback_environment_required": True,
		"artifact_reference_required": True,
	},
	"drift": {
		"drift_detection_enabled": True,
		"drift_threshold_percent": 5,
		"configuration_source_required": True,
		"remediation_supported": True,
		"review_required_above_threshold": True,
	},
	"secrets": {
		"secret_scope_policy_required": True,
		"secret_references_required": True,
		"access_roles_required": True,
		"provider": "keym",
	},
	"envm_agents": {
		"agent_assist_enabled": True,
		"agent_registration_required": True,
		"agent_runtime_required": True,
		"agent_role_required": True,
		"agent_scope_required": True,
		"agent_contribution_disclosure_required": True,
		"supported_runtimes": SUPPORTED_ENVM_AGENT_RUNTIMES,
		"allowed_roles": SUPPORTED_ENVM_AGENT_ROLES,
	},
	"governance": {
		"require_tenant_context": True,
		"audit_environment_changes": True,
		"secret_scope_policy_required": True,
		"rbac_policy_required": True,
		"batch_event_stream": "bytewax",
	},
	"observability": {
		"audit_required": True,
		"drift_metrics_required": True,
		"promotion_metrics_required": True,
		"agent_activity_required": True,
		"event_stream": "bytewax",
	},
	"adapters": {
		"generated_app_runtime": "service.EnvmService",
		"api_helpers": "api.py",
		"view_models": "views.py",
		"event_stream": "bytewax",
		"audit_sink": "audl",
		"identity": "auth",
		"configuration": "conf",
		"deployment": "depl",
		"secrets": "keym",
		"monitoring": "moni",
	},
	"ui": {
		"enable_environment_inventory": True,
		"enable_promotion_console": True,
		"enable_drift_dashboard": True,
		"enable_secret_scope_manager": True,
		"enable_agent_panel": True,
		"enable_rules": True,
		"enable_audit": True,
		"enable_analytics": True,
	},
	"theme": {
		"default_theme": "envm_environment_ops",
		"allow_tenant_overrides": True,
	},
}


CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"environments",
		"promotion",
		"drift",
		"secrets",
		"envm_agents",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {
		key: {"type": "object"}
		for key in [
			"environments",
			"promotion",
			"drift",
			"secrets",
			"envm_agents",
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
	{"name": "tenant_context_required", "description": "All environment operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "environment_requires_owner", "description": "Environments require an accountable owner.", "condition": {"operation": "create_environment", "environment_owner_assigned": False}, "effect": {"decision": "deny", "reason": "environment_owner_required", "required_action": "assign_environment_owner"}},
	{"name": "environment_requires_region_policy", "description": "Environments require region policy.", "condition": {"operation": "create_environment", "region_policy_present": False}, "effect": {"decision": "deny", "reason": "region_policy_required", "required_action": "attach_region_policy"}},
	{"name": "environment_requires_configuration_source", "description": "Environments require declared configuration source.", "condition": {"operation": "create_environment", "configuration_source_present": False}, "effect": {"decision": "deny", "reason": "configuration_source_required", "required_action": "attach_configuration_source"}},
	{"name": "environment_requires_rbac_policy", "description": "Environments require RBAC policy.", "condition": {"operation": "create_environment", "rbac_policy_present": False}, "effect": {"decision": "deny", "reason": "rbac_policy_required", "required_action": "attach_rbac_policy"}},
	{"name": "production_change_requires_approval", "description": "Production environment changes require approval.", "condition": {"environment": "production", "approval_recorded": False}, "effect": {"decision": "deny", "reason": "production_approval_required", "required_action": "record_production_approval"}},
	{"name": "promotion_requires_path", "description": "Promotion requires a declared path.", "condition": {"operation": "promote", "promotion_path_attached": False}, "effect": {"decision": "deny", "reason": "promotion_path_required", "required_action": "attach_promotion_path"}},
	{"name": "promotion_requires_artifact_reference", "description": "Promotion requires artifact reference.", "condition": {"operation": "run_promotion", "artifact_reference_present": False}, "effect": {"decision": "deny", "reason": "artifact_reference_required", "required_action": "attach_artifact_reference"}},
	{"name": "secret_scope_requires_policy", "description": "Environment secrets require scope policy.", "condition": {"secret_scope_present": True, "secret_policy_attached": False}, "effect": {"decision": "deny", "reason": "secret_policy_required", "required_action": "attach_secret_policy"}},
	{"name": "secret_scope_requires_references", "description": "Environment secret scopes require secret references.", "condition": {"secret_scope_present": True, "secret_references_present": False}, "effect": {"decision": "deny", "reason": "secret_references_required", "required_action": "attach_secret_references"}},
	{"name": "secret_scope_requires_access_roles", "description": "Environment secret scopes require access roles.", "condition": {"secret_scope_present": True, "access_roles_present": False}, "effect": {"decision": "deny", "reason": "access_roles_required", "required_action": "attach_access_roles"}},
	{"name": "high_drift_requires_review", "description": "High configuration drift requires review.", "condition": {"drift_percent_gt": 5, "drift_review_recorded": False}, "effect": {"decision": "require_review", "reason": "drift_review_required", "required_action": "review_drift"}},
	{"name": "envm_agent_requires_registration", "description": "AI environment agents must be registered.", "condition": {"envm_agent_present": True, "agent_registered": False}, "effect": {"decision": "deny", "reason": "envm_agent_registration_required", "required_action": "register_envm_agent"}},
	{"name": "envm_agent_runtime_supported", "description": "AI environment agents must use a supported runtime.", "condition": {"envm_agent_present": True, "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "envm_agent_runtime_not_supported", "required_action": "choose_supported_envm_agent_runtime"}},
	{"name": "envm_agent_role_supported", "description": "AI environment agents must use a supported role.", "condition": {"envm_agent_present": True, "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "envm_agent_role_not_supported", "required_action": "choose_supported_envm_agent_role"}},
	{"name": "envm_agent_requires_scope", "description": "AI environment agents require explicit scope.", "condition": {"envm_agent_present": True, "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "envm_agent_scope_required", "required_action": "set_envm_agent_scope"}},
	{"name": "envm_agent_requires_disclosure", "description": "AI environment-agent contributions require disclosure.", "condition": {"envm_agent_present": True, "agent_contribution_disclosed": False}, "effect": {"decision": "deny", "reason": "envm_agent_disclosure_required", "required_action": "disclose_envm_agent"}},
	{"name": "environment_state_change_requires_audit", "description": "Environment lifecycle state changes require audit evidence.", "condition": {"state_change_requested": True, "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "environment_audit_event_required", "required_action": "record_environment_audit_event"}},
	{"name": "batch_environment_mutation_requires_bytewax", "description": "Batch environment mutations must use Bytewax event streams.", "condition": {"requested_operation": "batch_environment_mutation", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}},
]


UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/envm/dashboard", "component": "EnvmDashboard", "permission": "envm:view", "nav_group": "Overview"},
	{"name": "environments", "path": "/envm/environments", "component": "EnvironmentInventory", "permission": "envm:manage_environments", "nav_group": "Inventory"},
	{"name": "promotion", "path": "/envm/promotion", "component": "PromotionConsole", "permission": "envm:promote", "nav_group": "Promotion"},
	{"name": "drift", "path": "/envm/drift", "component": "DriftDashboard", "permission": "envm:view", "nav_group": "Governance"},
	{"name": "secrets", "path": "/envm/secrets", "component": "SecretScopes", "permission": "envm:manage_secrets", "nav_group": "Security"},
	{"name": "agents", "path": "/envm/agents", "component": "EnvmAgentPanel", "permission": "envm:govern", "nav_group": "Governance"},
	{"name": "policies", "path": "/envm/policies", "component": "EnvironmentPolicies", "permission": "envm:admin", "nav_group": "Governance"},
	{"name": "rules", "path": "/envm/rules", "component": "EnvmRules", "permission": "envm:govern", "nav_group": "Governance"},
	{"name": "analytics", "path": "/envm/analytics", "component": "EnvironmentAnalytics", "permission": "envm:view", "nav_group": "Operations"},
	{"name": "audit", "path": "/envm/audit", "component": "EnvironmentAudit", "permission": "envm:view", "nav_group": "Governance"},
	{"name": "settings", "path": "/envm/settings", "component": "EnvmSettings", "permission": "envm:admin", "nav_group": "Administration"},
]


THEME: dict[str, Any] = {
	"name": "envm_environment_ops",
	"tokens": {
		"color.primary": "#28536B",
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
		"environment_grid": {"icon": "server", "status_indicator": "stage-pill", "risk_style": "policy-band"},
		"promotion_flow": {"visual": "stage-pipeline", "highlight": "approval-chip"},
		"drift_dashboard": {"visual": "diff-summary", "status_style": "drift-chip"},
		"secret_scope": {"visual": "scope-list", "status_style": "access-chip"},
		"envm_agent_panel": {"icon": "bot", "status_indicator": "scope-chip"},
		"stream_health": {"visual": "event-lane", "status_style": "stream-chip"},
		"audit": {"visual": "event-ledger", "status_style": "decision-chip"},
	},
}


def streaming_manifest() -> dict[str, Any]:
	return {
		"processor": "bytewax",
		"topic": "apg.envm.lifecycle",
		"state": ["environments", "promotion_paths", "promotion_runs", "drift_reports", "secret_scopes", "envm_agents", "audit_events"],
		"events": [
			"environment_registered",
			"promotion_path_created",
			"environment_promoted",
			"drift_recorded",
			"secret_scope_registered",
			"envm_agent_registered",
		],
		"batch_mutation_guardrail": "batch_environment_mutation_requires_bytewax",
	}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "envm",
		"display_name": "Environment Management",
		"version": "1.0.0",
		"provides": [
			"environment_inventory",
			"environment_promotion",
			"configuration_drift",
			"secret_scopes",
			"environment_policy",
			"envm_agents",
		],
		"requires": ["auth", "conf", "audl", "depl", "keym", "moni"],
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"view_module": "views.py",
			"api_prefix": "/envm/api/v1",
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
