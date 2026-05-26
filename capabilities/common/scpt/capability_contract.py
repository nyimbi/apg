"""Executable capability contract for APG Custom Scripting Engine."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"scripts": {
		"script_owner_required": True,
		"versioning_enabled": True,
		"review_required_for_publish": True,
		"allowed_languages": ["python", "javascript", "apg"]
	},
	"sandbox": {
		"sandbox_required": True,
		"network_disabled_by_default": True,
		"max_runtime_seconds": 300,
		"max_memory_mb": 512
	},
	"packages": {
		"allowlist_required": True,
		"secret_access_policy_required": True,
		"filesystem_access_policy_required": True,
		"dangerous_import_blocking": True
	},
	"governance": {
		"require_tenant_context": True,
		"audit_executions": True,
		"dangerous_permission_approval_required": True,
		"workflow_binding_policy_required": True
	},
	"ui": {
		"enable_script_workbench": True,
		"enable_execution_console": True,
		"enable_sandbox_monitor": True,
		"enable_package_policy": True
	},
	"theme": {
		"default_theme": "scpt_script_workbench",
		"allow_tenant_overrides": True
	}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": ["tenant_id", "scripts", "sandbox", "packages", "governance", "ui", "theme"],
	"properties": {key: {"type": "object"} for key in ["scripts", "sandbox", "packages", "governance", "ui", "theme"]} | {
		"tenant_id": {"type": "string", "minLength": 1}
	}
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All scripting operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "script_requires_owner", "description": "Scripts require an accountable owner.", "condition": {"operation": "create_script", "script_owner_assigned": False}, "effect": {"decision": "deny", "reason": "script_owner_required", "required_action": "assign_script_owner"}},
	{"name": "sandbox_required", "description": "Script execution requires an active sandbox.", "condition": {"operation": "execute_script", "sandbox_attached": False}, "effect": {"decision": "deny", "reason": "sandbox_required", "required_action": "attach_sandbox"}},
	{"name": "dangerous_permission_requires_approval", "description": "Dangerous permissions require approval.", "condition": {"dangerous_permission_requested": True, "approval_recorded": False}, "effect": {"decision": "deny", "reason": "dangerous_permission_approval_required", "required_action": "record_permission_approval"}},
	{"name": "external_network_requires_policy", "description": "Network access requires an explicit policy.", "condition": {"network_access_requested": True, "network_policy_attached": False}, "effect": {"decision": "deny", "reason": "network_policy_required", "required_action": "attach_network_policy"}},
	{"name": "high_resource_script_requires_review", "description": "High resource scripts require review.", "condition": {"requested_memory_mb_gt": 512, "resource_review_recorded": False}, "effect": {"decision": "require_review", "reason": "resource_review_required", "required_action": "review_script_resources"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/scpt/dashboard", "component": "SCPTDashboard", "permission": "scpt:view", "nav_group": "Overview"},
	{"name": "workbench", "path": "/scpt/workbench", "component": "ScriptWorkbench", "permission": "scpt:write", "nav_group": "Scripts"},
	{"name": "scripts", "path": "/scpt/scripts", "component": "ScriptRegistry", "permission": "scpt:view", "nav_group": "Scripts"},
	{"name": "executions", "path": "/scpt/executions", "component": "ExecutionConsole", "permission": "scpt:execute", "nav_group": "Runtime"},
	{"name": "sandboxes", "path": "/scpt/sandboxes", "component": "SandboxMonitor", "permission": "scpt:admin", "nav_group": "Runtime"},
	{"name": "packages", "path": "/scpt/packages", "component": "PackagePolicy", "permission": "scpt:approve", "nav_group": "Governance"},
	{"name": "approvals", "path": "/scpt/approvals", "component": "ScriptApprovals", "permission": "scpt:approve", "nav_group": "Governance"},
	{"name": "settings", "path": "/scpt/settings", "component": "SCPTSettings", "permission": "scpt:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {
	"name": "scpt_script_workbench",
	"tokens": {
		"color.primary": "#2A4365",
		"color.accent": "#805AD5",
		"color.success": "#2F855A",
		"color.warning": "#B7791F",
		"color.danger": "#C53030",
		"surface.canvas": "#F7F8FA",
		"surface.panel": "#FFFFFF",
		"text.primary": "#172033",
		"text.secondary": "#52606D",
		"border.radius": "8px",
		"density": "compact"
	},
	"components": {
		"script_editor": {"icon": "code-2", "status_indicator": "script-pill", "risk_style": "permission-band"},
		"execution_log": {"visual": "log-stream", "highlight": "runtime-chip"},
		"sandbox_monitor": {"visual": "resource-meter", "status_style": "isolation-chip"},
		"package_policy": {"visual": "allowlist-table", "status_style": "approval-chip"}
	}
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable SCPT capability contract."""
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "scpt",
		"display_name": "Custom Scripting Engine",
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "flask_appbuilder",
			"view_module": "views.py",
			"api_prefix": "/scpt/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True
		},
		"theme": deepcopy(THEME)
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate default SCPT governance rules."""
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
