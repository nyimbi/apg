"""Executable capability contract for APG Sandbox/Testing Environment."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"sandboxes": {"sandbox_owner_required": True, "template_required": True, "ttl_hours": 24, "environment_isolation_required": True},
	"isolation": {"network_policy_required": True, "secret_redaction_required": True, "data_masking_required": True, "outbound_access_denied_by_default": True},
	"datasets": {"synthetic_data_supported": True, "production_data_review_required": True, "dataset_lineage_required": True, "retention_policy_required": True},
	"governance": {"require_tenant_context": True, "audit_sandbox_runs": True, "long_lived_review_hours": 48, "plugin_test_policy_required": True},
	"ui": {"enable_sandbox_console": True, "enable_template_library": True, "enable_run_monitor": True, "enable_policy_center": True},
	"theme": {"default_theme": "sbox_safe_testing", "allow_tenant_overrides": True}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": ["tenant_id", "sandboxes", "isolation", "datasets", "governance", "ui", "theme"],
	"properties": {key: {"type": "object"} for key in ["sandboxes", "isolation", "datasets", "governance", "ui", "theme"]} | {"tenant_id": {"type": "string", "minLength": 1}}
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All sandbox operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "sandbox_requires_owner", "description": "Sandboxes require an accountable owner.", "condition": {"operation": "create_sandbox", "sandbox_owner_assigned": False}, "effect": {"decision": "deny", "reason": "sandbox_owner_required", "required_action": "assign_sandbox_owner"}},
	{"name": "sandbox_requires_isolation_profile", "description": "Sandboxes require an isolation profile.", "condition": {"isolation_profile_attached": False}, "effect": {"decision": "deny", "reason": "isolation_profile_required", "required_action": "attach_isolation_profile"}},
	{"name": "secrets_require_redaction", "description": "Sandbox secrets require redaction policy.", "condition": {"secret_access_requested": True, "secret_redaction_enabled": False}, "effect": {"decision": "deny", "reason": "secret_redaction_required", "required_action": "enable_secret_redaction"}},
	{"name": "outbound_network_requires_approval", "description": "Outbound sandbox network access requires approval.", "condition": {"outbound_network_requested": True, "network_approval_recorded": False}, "effect": {"decision": "deny", "reason": "outbound_network_approval_required", "required_action": "approve_outbound_network"}},
	{"name": "long_lived_sandbox_requires_review", "description": "Long-lived sandboxes require review.", "condition": {"ttl_hours_gt": 48, "lifecycle_review_recorded": False}, "effect": {"decision": "require_review", "reason": "long_lived_sandbox_review_required", "required_action": "review_sandbox_lifecycle"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/sbox/dashboard", "component": "SBOXDashboard", "permission": "sbox:view", "nav_group": "Overview"},
	{"name": "sandboxes", "path": "/sbox/sandboxes", "component": "SandboxConsole", "permission": "sbox:create", "nav_group": "Sandboxes"},
	{"name": "templates", "path": "/sbox/templates", "component": "TemplateLibrary", "permission": "sbox:create", "nav_group": "Templates"},
	{"name": "datasets", "path": "/sbox/datasets", "component": "DatasetManager", "permission": "sbox:manage_policy", "nav_group": "Data"},
	{"name": "runs", "path": "/sbox/runs", "component": "RunMonitor", "permission": "sbox:run_tests", "nav_group": "Runs"},
	{"name": "policies", "path": "/sbox/policies", "component": "PolicyCenter", "permission": "sbox:manage_policy", "nav_group": "Governance"},
	{"name": "logs", "path": "/sbox/logs", "component": "SandboxLogs", "permission": "sbox:view", "nav_group": "Operations"},
	{"name": "settings", "path": "/sbox/settings", "component": "SBOXSettings", "permission": "sbox:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {
	"name": "sbox_safe_testing",
	"tokens": {"color.primary": "#234E52", "color.accent": "#3182CE", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F7F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"},
	"components": {"sandbox_card": {"icon": "container", "status_indicator": "ttl-pill", "risk_style": "isolation-band"}, "run_monitor": {"visual": "test-timeline", "highlight": "result-chip"}, "dataset_manager": {"visual": "masked-data-grid", "status_style": "lineage-chip"}, "policy_center": {"visual": "guardrail-list", "status_style": "approval-chip"}}
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {"capability": "sbox", "display_name": "Sandbox/Testing Environment", "configuration": config, "configuration_schema": CONFIGURATION_SCHEMA, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "view_module": "views.py", "api_prefix": "/sbox/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True}, "theme": deepcopy(THEME)}


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
