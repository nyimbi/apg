"""Executable capability contract for APG Environment Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"environments": {"environment_owner_required": True, "stage_policy_required": True, "region_policy_required": True, "production_locked_by_default": True},
	"promotion": {"promotion_path_required": True, "approval_required": True, "deployment_link_required": True, "rollback_environment_required": True},
	"drift": {"drift_detection_enabled": True, "drift_threshold_percent": 5, "configuration_source_required": True, "remediation_supported": True},
	"governance": {"require_tenant_context": True, "audit_environment_changes": True, "secret_scope_policy_required": True, "rbac_policy_required": True},
	"ui": {"enable_environment_inventory": True, "enable_promotion_console": True, "enable_drift_dashboard": True, "enable_secret_scope_manager": True},
	"theme": {"default_theme": "envm_environment_ops", "allow_tenant_overrides": True}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {"type": "object", "required": ["tenant_id", "environments", "promotion", "drift", "governance", "ui", "theme"], "properties": {key: {"type": "object"} for key in ["environments", "promotion", "drift", "governance", "ui", "theme"]} | {"tenant_id": {"type": "string", "minLength": 1}}}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All environment operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "environment_requires_owner", "description": "Environments require an accountable owner.", "condition": {"operation": "create_environment", "environment_owner_assigned": False}, "effect": {"decision": "deny", "reason": "environment_owner_required", "required_action": "assign_environment_owner"}},
	{"name": "production_change_requires_approval", "description": "Production environment changes require approval.", "condition": {"environment": "production", "approval_recorded": False}, "effect": {"decision": "deny", "reason": "production_approval_required", "required_action": "record_production_approval"}},
	{"name": "promotion_requires_path", "description": "Promotion requires a declared path.", "condition": {"operation": "promote", "promotion_path_attached": False}, "effect": {"decision": "deny", "reason": "promotion_path_required", "required_action": "attach_promotion_path"}},
	{"name": "secret_scope_requires_policy", "description": "Environment secrets require scope policy.", "condition": {"secret_scope_present": True, "secret_policy_attached": False}, "effect": {"decision": "deny", "reason": "secret_policy_required", "required_action": "attach_secret_policy"}},
	{"name": "high_drift_requires_review", "description": "High configuration drift requires review.", "condition": {"drift_percent_gt": 5, "drift_review_recorded": False}, "effect": {"decision": "require_review", "reason": "drift_review_required", "required_action": "review_drift"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/envm/dashboard", "component": "ENVMDashboard", "permission": "envm:view", "nav_group": "Overview"},
	{"name": "environments", "path": "/envm/environments", "component": "EnvironmentInventory", "permission": "envm:manage_environments", "nav_group": "Inventory"},
	{"name": "promotion", "path": "/envm/promotion", "component": "PromotionConsole", "permission": "envm:promote", "nav_group": "Promotion"},
	{"name": "drift", "path": "/envm/drift", "component": "DriftDashboard", "permission": "envm:view", "nav_group": "Governance"},
	{"name": "secrets", "path": "/envm/secrets", "component": "SecretScopes", "permission": "envm:manage_secrets", "nav_group": "Security"},
	{"name": "policies", "path": "/envm/policies", "component": "EnvironmentPolicies", "permission": "envm:admin", "nav_group": "Governance"},
	{"name": "analytics", "path": "/envm/analytics", "component": "EnvironmentAnalytics", "permission": "envm:view", "nav_group": "Operations"},
	{"name": "settings", "path": "/envm/settings", "component": "ENVMSettings", "permission": "envm:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {"name": "envm_environment_ops", "tokens": {"color.primary": "#28536B", "color.accent": "#805AD5", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F7F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"}, "components": {"environment_grid": {"icon": "server", "status_indicator": "stage-pill", "risk_style": "policy-band"}, "promotion_flow": {"visual": "stage-pipeline", "highlight": "approval-chip"}, "drift_dashboard": {"visual": "diff-summary", "status_style": "drift-chip"}, "secret_scope": {"visual": "scope-list", "status_style": "access-chip"}}}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {"capability": "envm", "display_name": "Environment Management", "configuration": config, "configuration_schema": CONFIGURATION_SCHEMA, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "flask_appbuilder", "view_module": "views.py", "api_prefix": "/envm/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True}, "theme": deepcopy(THEME)}


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
