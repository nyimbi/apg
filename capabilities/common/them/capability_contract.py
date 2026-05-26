"""Executable capability contract for APG UI/UX Theming and Branding."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"themes": {"theme_owner_required": True, "default_theme": "them_brand_system", "multi_brand_enabled": True, "preview_required": True},
	"tokens": {"governed_tokens": ["color", "typography", "spacing", "density"], "contrast_validation_required": True, "token_versioning_enabled": True},
	"branding": {"license_verification_required": True, "asset_approval_required": True, "brand_guidelines_required": True, "fallback_brand_enabled": True},
	"governance": {"require_tenant_context": True, "publish_approval_required": True, "large_rollout_review_threshold": 5, "audit_theme_changes": True},
	"ui": {"enable_theme_console": True, "enable_token_editor": True, "enable_brand_asset_manager": True, "enable_live_preview": True},
	"theme": {"default_theme": "them_brand_system", "allow_tenant_overrides": True}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": ["tenant_id", "themes", "tokens", "branding", "governance", "ui", "theme"],
	"properties": {key: {"type": "object"} for key in ["themes", "tokens", "branding", "governance", "ui", "theme"]} | {"tenant_id": {"type": "string", "minLength": 1}}
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All theme operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "theme_requires_owner", "description": "Themes require an accountable owner.", "condition": {"operation": "create_theme", "theme_owner_assigned": False}, "effect": {"decision": "deny", "reason": "theme_owner_required", "required_action": "assign_theme_owner"}},
	{"name": "publish_requires_approval", "description": "Theme publishing requires approval.", "condition": {"operation": "publish_theme", "approval_recorded": False}, "effect": {"decision": "deny", "reason": "theme_publish_approval_required", "required_action": "record_publish_approval"}},
	{"name": "brand_asset_requires_license", "description": "Brand assets require license verification.", "condition": {"brand_asset_present": True, "license_verified": False}, "effect": {"decision": "deny", "reason": "brand_asset_license_required", "required_action": "verify_brand_license"}},
	{"name": "accessible_contrast_required", "description": "Published themes require contrast validation.", "condition": {"operation": "publish_theme", "accessibility_contrast_passed": False}, "effect": {"decision": "deny", "reason": "contrast_validation_required", "required_action": "validate_theme_contrast"}},
	{"name": "large_rollout_requires_review", "description": "Broad theme rollouts require review.", "condition": {"target_tenant_count_gt": 5, "rollout_review_recorded": False}, "effect": {"decision": "require_review", "reason": "large_rollout_review_required", "required_action": "review_theme_rollout"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/them/dashboard", "component": "THEMDashboard", "permission": "them:view", "nav_group": "Overview"},
	{"name": "themes", "path": "/them/themes", "component": "ThemeConsole", "permission": "them:design", "nav_group": "Design"},
	{"name": "tokens", "path": "/them/tokens", "component": "TokenEditor", "permission": "them:design", "nav_group": "Design"},
	{"name": "branding", "path": "/them/branding", "component": "BrandGuidelines", "permission": "them:manage_brand", "nav_group": "Brand"},
	{"name": "assets", "path": "/them/assets", "component": "BrandAssetManager", "permission": "them:manage_brand", "nav_group": "Brand"},
	{"name": "preview", "path": "/them/preview", "component": "ThemePreview", "permission": "them:view", "nav_group": "Review"},
	{"name": "policies", "path": "/them/policies", "component": "ThemePolicies", "permission": "them:admin", "nav_group": "Governance"},
	{"name": "settings", "path": "/them/settings", "component": "THEMSettings", "permission": "them:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {
	"name": "them_brand_system",
	"tokens": {"color.primary": "#1F4E5F", "color.accent": "#D69E2E", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F7F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"},
	"components": {"theme_card": {"icon": "palette", "status_indicator": "theme-pill", "risk_style": "contrast-band"}, "token_editor": {"visual": "token-table", "highlight": "changed-token-chip"}, "asset_library": {"visual": "asset-grid", "status_style": "license-chip"}, "preview_shell": {"visual": "responsive-preview", "status_style": "approval-chip"}}
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {"capability": "them", "display_name": "UI/UX Theming and Branding", "configuration": config, "configuration_schema": CONFIGURATION_SCHEMA, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "flask_appbuilder", "view_module": "views.py", "api_prefix": "/them/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True}, "theme": deepcopy(THEME)}


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
