"""Executable capability contract for APG Internationalization."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"locales": {"default_locale": "en-US", "locale_owner_required": True, "fallback_locale_required": True, "regional_formatting_enabled": True},
	"translations": {"translation_memory_enabled": True, "glossary_required": True, "machine_translation_review_required": True, "minimum_coverage_percent": 95},
	"publishing": {"publication_approval_required": True, "missing_key_blocking": True, "versioning_enabled": True, "rollback_supported": True},
	"governance": {"require_tenant_context": True, "audit_translation_changes": True, "restricted_content_filtering": True, "language_policy_required": True},
	"ui": {"enable_locale_console": True, "enable_translation_workbench": True, "enable_coverage_dashboard": True, "enable_glossary_manager": True},
	"theme": {"default_theme": "i18n_localization_workbench", "allow_tenant_overrides": True}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": ["tenant_id", "locales", "translations", "publishing", "governance", "ui", "theme"],
	"properties": {key: {"type": "object"} for key in ["locales", "translations", "publishing", "governance", "ui", "theme"]} | {"tenant_id": {"type": "string", "minLength": 1}}
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All localization operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "locale_requires_owner", "description": "Locales require an accountable owner.", "condition": {"operation": "create_locale", "locale_owner_assigned": False}, "effect": {"decision": "deny", "reason": "locale_owner_required", "required_action": "assign_locale_owner"}},
	{"name": "machine_translation_requires_review", "description": "Machine translations require review before publishing.", "condition": {"machine_translation_used": True, "translation_review_recorded": False}, "effect": {"decision": "deny", "reason": "translation_review_required", "required_action": "review_translation"}},
	{"name": "publish_requires_approval", "description": "Translation publication requires approval.", "condition": {"operation": "publish_translations", "approval_recorded": False}, "effect": {"decision": "deny", "reason": "publication_approval_required", "required_action": "record_publication_approval"}},
	{"name": "restricted_content_requires_filtering", "description": "Restricted content requires RBAC filtering.", "condition": {"restricted_content_present": True, "rbac_filter_applied": False}, "effect": {"decision": "deny", "reason": "rbac_filter_required", "required_action": "apply_rbac_filter"}},
	{"name": "low_coverage_requires_review", "description": "Low localization coverage requires review.", "condition": {"coverage_percent_lt": 95, "coverage_review_recorded": False}, "effect": {"decision": "require_review", "reason": "coverage_review_required", "required_action": "review_locale_coverage"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/i18n/dashboard", "component": "I18NDashboard", "permission": "i18n:view", "nav_group": "Overview"},
	{"name": "locales", "path": "/i18n/locales", "component": "LocaleConsole", "permission": "i18n:manage_locales", "nav_group": "Locales"},
	{"name": "translations", "path": "/i18n/translations", "component": "TranslationWorkbench", "permission": "i18n:translate", "nav_group": "Translations"},
	{"name": "glossaries", "path": "/i18n/glossaries", "component": "GlossaryManager", "permission": "i18n:translate", "nav_group": "Translations"},
	{"name": "coverage", "path": "/i18n/coverage", "component": "CoverageDashboard", "permission": "i18n:view", "nav_group": "Quality"},
	{"name": "publishing", "path": "/i18n/publishing", "component": "PublishQueue", "permission": "i18n:publish", "nav_group": "Release"},
	{"name": "policies", "path": "/i18n/policies", "component": "LanguagePolicies", "permission": "i18n:admin", "nav_group": "Governance"},
	{"name": "settings", "path": "/i18n/settings", "component": "I18NSettings", "permission": "i18n:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {
	"name": "i18n_localization_workbench",
	"tokens": {"color.primary": "#28536B", "color.accent": "#38A169", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F7F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"},
	"components": {
		"locale_matrix": {"icon": "languages", "status_indicator": "locale-pill", "risk_style": "coverage-band"},
		"translation_editor": {"visual": "side-by-side-editor", "highlight": "glossary-chip"},
		"coverage_dashboard": {"visual": "coverage-grid", "status_style": "gap-chip"},
		"publish_queue": {"visual": "release-checklist", "status_style": "approval-chip"}
	}
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable I18N capability contract."""
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {"capability": "i18n", "display_name": "Internationalization", "configuration": config, "configuration_schema": CONFIGURATION_SCHEMA, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "view_module": "views.py", "api_prefix": "/i18n/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True}, "theme": deepcopy(THEME)}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate default I18N governance rules."""
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
