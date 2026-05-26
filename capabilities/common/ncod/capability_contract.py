"""Executable capability contract for APG No-Code/Low-Code Builder."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"apps": {
		"app_owner_required": True,
		"versioning_enabled": True,
		"publish_approval_required": True,
		"production_change_review_required": True
	},
	"builder": {
		"component_catalog_enabled": True,
		"theme_policy_required": True,
		"accessibility_checks_required": True,
		"data_binding_validation_required": True
	},
	"extensions": {
		"workflow_binding_enabled": True,
		"script_extension_policy_required": True,
		"external_connector_policy_required": True,
		"custom_component_review_required": True
	},
	"governance": {
		"require_tenant_context": True,
		"audit_app_changes": True,
		"rbac_policy_required": True,
		"data_residency_policy_required": True
	},
	"ui": {
		"enable_app_builder": True,
		"enable_page_composer": True,
		"enable_component_catalog": True,
		"enable_publish_center": True
	},
	"theme": {
		"default_theme": "ncod_app_builder",
		"allow_tenant_overrides": True
	}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": ["tenant_id", "apps", "builder", "extensions", "governance", "ui", "theme"],
	"properties": {key: {"type": "object"} for key in ["apps", "builder", "extensions", "governance", "ui", "theme"]} | {
		"tenant_id": {"type": "string", "minLength": 1}
	}
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All no-code operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "app_requires_owner", "description": "Applications require an accountable owner.", "condition": {"operation": "create_app", "app_owner_assigned": False}, "effect": {"decision": "deny", "reason": "app_owner_required", "required_action": "assign_app_owner"}},
	{"name": "publish_requires_approval", "description": "Publishing applications requires approval.", "condition": {"operation": "publish_app", "approval_recorded": False}, "effect": {"decision": "deny", "reason": "publish_approval_required", "required_action": "record_publish_approval"}},
	{"name": "script_extension_requires_policy", "description": "Script extensions require an approved policy.", "condition": {"script_extension_present": True, "script_policy_attached": False}, "effect": {"decision": "deny", "reason": "script_policy_required", "required_action": "attach_script_policy"}},
	{"name": "external_connector_requires_policy", "description": "External connectors require a connector policy.", "condition": {"external_connector_present": True, "connector_policy_attached": False}, "effect": {"decision": "deny", "reason": "connector_policy_required", "required_action": "attach_connector_policy"}},
	{"name": "production_change_requires_review", "description": "Production app changes require review.", "condition": {"production_change": True, "change_review_recorded": False}, "effect": {"decision": "require_review", "reason": "production_change_review_required", "required_action": "review_production_change"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/ncod/dashboard", "component": "NCODDashboard", "permission": "ncod:view", "nav_group": "Overview"},
	{"name": "apps", "path": "/ncod/apps", "component": "AppLibrary", "permission": "ncod:manage_apps", "nav_group": "Apps"},
	{"name": "builder", "path": "/ncod/builder", "component": "AppBuilder", "permission": "ncod:build", "nav_group": "Build"},
	{"name": "pages", "path": "/ncod/pages", "component": "PageComposer", "permission": "ncod:build", "nav_group": "Build"},
	{"name": "components", "path": "/ncod/components", "component": "ComponentCatalog", "permission": "ncod:build", "nav_group": "Build"},
	{"name": "publishing", "path": "/ncod/publishing", "component": "PublishCenter", "permission": "ncod:publish", "nav_group": "Release"},
	{"name": "connectors", "path": "/ncod/connectors", "component": "ConnectorBindings", "permission": "ncod:build", "nav_group": "Integrations"},
	{"name": "settings", "path": "/ncod/settings", "component": "NCODSettings", "permission": "ncod:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {
	"name": "ncod_app_builder",
	"tokens": {
		"color.primary": "#2C5282",
		"color.accent": "#38A169",
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
		"app_library": {"icon": "layout-dashboard", "status_indicator": "app-pill", "risk_style": "release-band"},
		"page_composer": {"visual": "component-canvas", "highlight": "theme-chip"},
		"component_catalog": {"visual": "component-grid", "status_style": "accessibility-chip"},
		"publish_center": {"visual": "release-checklist", "status_style": "approval-chip"}
	}
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable NCOD capability contract."""
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "ncod",
		"display_name": "No-Code/Low-Code Builder",
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "flask_appbuilder",
			"view_module": "views.py",
			"api_prefix": "/ncod/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True
		},
		"theme": deepcopy(THEME)
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate default NCOD governance rules."""
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
