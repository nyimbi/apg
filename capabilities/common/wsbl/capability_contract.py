"""Executable capability contract for APG Website Builder."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"sites": {"site_owner_required": True, "domain_validation_required": True, "multi_locale_enabled": True, "environment_preview_enabled": True},
	"pages": {"structured_sections_required": True, "custom_component_review_required": True, "draft_autosave_enabled": True, "content_versioning_enabled": True},
	"publishing": {"approval_required": True, "accessibility_pass_required": True, "privacy_banner_policy_required": True, "rollback_supported": True},
	"governance": {"require_tenant_context": True, "audit_publication_changes": True, "component_policy_required": True, "public_site_controls_required": True},
	"ui": {"enable_site_console": True, "enable_page_editor": True, "enable_component_library": True, "enable_publish_queue": True},
	"theme": {"default_theme": "wsbl_site_builder", "allow_tenant_overrides": True}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": ["tenant_id", "sites", "pages", "publishing", "governance", "ui", "theme"],
	"properties": {key: {"type": "object"} for key in ["sites", "pages", "publishing", "governance", "ui", "theme"]} | {"tenant_id": {"type": "string", "minLength": 1}}
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All website-builder operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "site_requires_owner", "description": "Sites require an accountable owner.", "condition": {"operation": "create_site", "site_owner_assigned": False}, "effect": {"decision": "deny", "reason": "site_owner_required", "required_action": "assign_site_owner"}},
	{"name": "publish_requires_approval", "description": "Site publishing requires approval.", "condition": {"operation": "publish_site", "approval_recorded": False}, "effect": {"decision": "deny", "reason": "site_publish_approval_required", "required_action": "record_publish_approval"}},
	{"name": "custom_component_requires_review", "description": "Custom components require review before use.", "condition": {"custom_component_present": True, "component_review_recorded": False}, "effect": {"decision": "deny", "reason": "component_review_required", "required_action": "review_custom_component"}},
	{"name": "public_site_requires_accessibility_pass", "description": "Public sites require an accessibility pass.", "condition": {"public_site": True, "accessibility_passed": False}, "effect": {"decision": "deny", "reason": "accessibility_pass_required", "required_action": "complete_accessibility_pass"}},
	{"name": "privacy_banner_requires_consent_policy", "description": "Privacy banners require an attached consent policy.", "condition": {"privacy_banner_required": True, "consent_policy_attached": False}, "effect": {"decision": "require_review", "reason": "consent_policy_required", "required_action": "attach_consent_policy"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/wsbl/dashboard", "component": "WSBLDashboard", "permission": "wsbl:view", "nav_group": "Overview"},
	{"name": "sites", "path": "/wsbl/sites", "component": "SiteConsole", "permission": "wsbl:manage_sites", "nav_group": "Sites"},
	{"name": "pages", "path": "/wsbl/pages", "component": "PageLibrary", "permission": "wsbl:build", "nav_group": "Pages"},
	{"name": "editor", "path": "/wsbl/editor", "component": "PageEditor", "permission": "wsbl:build", "nav_group": "Build"},
	{"name": "components", "path": "/wsbl/components", "component": "ComponentLibrary", "permission": "wsbl:build", "nav_group": "Build"},
	{"name": "publishing", "path": "/wsbl/publishing", "component": "PublishQueue", "permission": "wsbl:publish", "nav_group": "Release"},
	{"name": "analytics", "path": "/wsbl/analytics", "component": "SiteAnalytics", "permission": "wsbl:view", "nav_group": "Operations"},
	{"name": "settings", "path": "/wsbl/settings", "component": "WSBLSettings", "permission": "wsbl:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {
	"name": "wsbl_site_builder",
	"tokens": {"color.primary": "#2C5282", "color.accent": "#38A169", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F7F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"},
	"components": {"site_card": {"icon": "layout-template", "status_indicator": "site-pill", "risk_style": "publish-band"}, "page_editor": {"visual": "section-builder", "highlight": "component-chip"}, "publish_queue": {"visual": "release-checklist", "status_style": "approval-chip"}, "analytics_panel": {"visual": "traffic-grid", "status_style": "trend-chip"}}
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {"capability": "wsbl", "display_name": "Website Builder", "configuration": config, "configuration_schema": CONFIGURATION_SCHEMA, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "view_module": "views.py", "api_prefix": "/wsbl/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True}, "theme": deepcopy(THEME)}


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
