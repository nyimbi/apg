"""Executable capability contract for APG Plugin/Extension Framework."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"marketplace": {"curated_listing_required": True, "publisher_verification_required": True, "release_channel_policy_required": True, "tenant_install_policy_enabled": True},
	"plugins": {"plugin_owner_required": True, "manifest_schema_required": True, "signature_required": True, "dependency_validation_required": True},
	"security": {"permission_review_required": True, "sandbox_policy_required": True, "secret_access_denied_by_default": True, "supply_chain_scan_required": True},
	"governance": {"require_tenant_context": True, "audit_plugin_changes": True, "external_plugin_review_required": True, "configuration_policy_required": True},
	"ui": {"enable_marketplace": True, "enable_plugin_registry": True, "enable_permission_review": True, "enable_release_manager": True},
	"theme": {"default_theme": "plgn_extension_marketplace", "allow_tenant_overrides": True}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": ["tenant_id", "marketplace", "plugins", "security", "governance", "ui", "theme"],
	"properties": {key: {"type": "object"} for key in ["marketplace", "plugins", "security", "governance", "ui", "theme"]} | {"tenant_id": {"type": "string", "minLength": 1}}
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All plugin operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "plugin_requires_owner", "description": "Plugins require an accountable owner.", "condition": {"operation": "register_plugin", "plugin_owner_assigned": False}, "effect": {"decision": "deny", "reason": "plugin_owner_required", "required_action": "assign_plugin_owner"}},
	{"name": "plugin_requires_signature", "description": "Plugin packages require verified signatures.", "condition": {"signature_verified": False}, "effect": {"decision": "deny", "reason": "plugin_signature_required", "required_action": "verify_plugin_signature"}},
	{"name": "permissions_require_review", "description": "Requested plugin permissions require review.", "condition": {"permissions_requested": True, "permission_review_recorded": False}, "effect": {"decision": "deny", "reason": "permission_review_required", "required_action": "review_plugin_permissions"}},
	{"name": "plugin_requires_sandbox", "description": "Plugins require sandbox policy before execution.", "condition": {"operation": "enable_plugin", "sandbox_policy_attached": False}, "effect": {"decision": "deny", "reason": "plugin_sandbox_required", "required_action": "attach_sandbox_policy"}},
	{"name": "external_plugin_requires_review", "description": "External plugins require review.", "condition": {"external_plugin": True, "external_review_recorded": False}, "effect": {"decision": "require_review", "reason": "external_plugin_review_required", "required_action": "review_external_plugin"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/plgn/dashboard", "component": "PLGNDashboard", "permission": "plgn:view", "nav_group": "Overview"},
	{"name": "marketplace", "path": "/plgn/marketplace", "component": "ExtensionMarketplace", "permission": "plgn:install", "nav_group": "Marketplace"},
	{"name": "plugins", "path": "/plgn/plugins", "component": "PluginRegistry", "permission": "plgn:view", "nav_group": "Plugins"},
	{"name": "manifests", "path": "/plgn/manifests", "component": "ManifestEditor", "permission": "plgn:publish", "nav_group": "Plugins"},
	{"name": "permissions", "path": "/plgn/permissions", "component": "PermissionReview", "permission": "plgn:review", "nav_group": "Security"},
	{"name": "sandbox", "path": "/plgn/sandbox", "component": "PluginSandboxPolicy", "permission": "plgn:review", "nav_group": "Security"},
	{"name": "releases", "path": "/plgn/releases", "component": "ReleaseManager", "permission": "plgn:publish", "nav_group": "Release"},
	{"name": "settings", "path": "/plgn/settings", "component": "PLGNSettings", "permission": "plgn:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {
	"name": "plgn_extension_marketplace",
	"tokens": {"color.primary": "#2B4C7E", "color.accent": "#D69E2E", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F7F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"},
	"components": {"plugin_card": {"icon": "package-plus", "status_indicator": "trust-pill", "risk_style": "permission-band"}, "marketplace_grid": {"visual": "extension-grid", "highlight": "verified-chip"}, "permission_review": {"visual": "scope-table", "status_style": "review-chip"}, "release_manager": {"visual": "channel-lanes", "status_style": "signature-chip"}}
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {"capability": "plgn", "display_name": "Plugin/Extension Framework", "configuration": config, "configuration_schema": CONFIGURATION_SCHEMA, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "flask_appbuilder", "view_module": "views.py", "api_prefix": "/plgn/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True}, "theme": deepcopy(THEME)}


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
