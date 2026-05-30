"""Executable capability contract for APG Internationalization."""

from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Any


SUPPORTED_I18N_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_I18N_AGENT_ROLES = [
	"locale_planner",
	"translator",
	"translation_reviewer",
	"glossary_steward",
	"coverage_reviewer",
	"publication_reviewer",
]
AFRICAN_LANGUAGE_CODES = [
	"af",
	"ak",
	"am",
	"ar",
	"asa",
	"bem",
	"bez",
	"bm",
	"byn",
	"cgg",
	"dav",
	"dua",
	"ebu",
	"ee",
	"ewo",
	"ff",
	"fon",
	"gaa",
	"ha",
	"ig",
	"kam",
	"ki",
	"kkj",
	"kln",
	"ksb",
	"ksf",
	"lg",
	"ln",
	"lu",
	"luo",
	"luy",
	"mas",
	"mer",
	"mfe",
	"mg",
	"mgh",
	"mua",
	"naq",
	"nd",
	"nmg",
	"nnh",
	"nso",
	"ny",
	"om",
	"rn",
	"rw",
	"seh",
	"sg",
	"shi",
	"sn",
	"so",
	"ss",
	"st",
	"sw",
	"teo",
	"ti",
	"tn",
	"ts",
	"vai",
	"ve",
	"wo",
	"xh",
	"yo",
	"zu",
]
CORE_LANGUAGE_CODES = ["en", "es", "fr", "pt", "de", "it", "zh", "ja", "ko", "hi"]
SUPPORTED_LANGUAGE_CODES = sorted(set(CORE_LANGUAGE_CODES + AFRICAN_LANGUAGE_CODES))


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"locales": {
		"default_locale": "en-US",
		"locale_owner_required": True,
		"fallback_locale_required": True,
		"regional_formatting_enabled": True,
		"supported_language_codes": SUPPORTED_LANGUAGE_CODES,
		"african_language_codes": AFRICAN_LANGUAGE_CODES,
	},
	"translations": {
		"translation_memory_enabled": True,
		"glossary_required": True,
		"machine_translation_review_required": True,
		"minimum_coverage_percent": 95,
		"restricted_content_filtering_required": True,
	},
	"publishing": {
		"publication_approval_required": True,
		"missing_key_blocking": True,
		"versioning_enabled": True,
		"rollback_supported": True,
		"release_audit_required": True,
	},
	"i18n_agents": {
		"agent_assist_enabled": True,
		"agent_registration_required": True,
		"agent_runtime_required": True,
		"agent_role_required": True,
		"agent_scope_required": True,
		"agent_contribution_disclosure_required": True,
		"supported_runtimes": SUPPORTED_I18N_AGENT_RUNTIMES,
		"allowed_roles": SUPPORTED_I18N_AGENT_ROLES,
	},
	"governance": {
		"require_tenant_context": True,
		"audit_translation_changes": True,
		"restricted_content_filtering": True,
		"language_policy_required": True,
		"tenant_isolation_required": True,
		"batch_event_stream": "bytewax",
	},
	"observability": {
		"audit_required": True,
		"coverage_metrics_required": True,
		"translation_metrics_required": True,
		"agent_activity_required": True,
		"event_stream": "bytewax",
	},
	"adapters": {
		"generated_app_runtime": "service.I18nService",
		"api_helpers": "api.py",
		"view_models": "views.py",
		"event_stream": "bytewax",
		"identity": "auth",
		"configuration": "conf",
		"audit_sink": "audl",
		"natural_language": "nlpc",
		"machine_translation": "mchn",
		"help_content": "help",
		"theme": "them",
	},
	"ui": {
		"enable_locale_console": True,
		"enable_translation_workbench": True,
		"enable_coverage_dashboard": True,
		"enable_glossary_manager": True,
		"enable_agent_panel": True,
		"enable_audit": True,
		"enable_policies": True,
	},
	"theme": {
		"default_theme": "i18n_localization_workbench",
		"allow_tenant_overrides": True,
	},
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"locales",
		"translations",
		"publishing",
		"i18n_agents",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {
		key: {"type": "object"}
		for key in [
			"locales",
			"translations",
			"publishing",
			"i18n_agents",
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
	{"name": "tenant_context_required", "description": "All localization operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "locale_requires_owner", "description": "Locales require an accountable owner.", "condition": {"operation": "create_locale", "locale_owner_assigned": False}, "effect": {"decision": "deny", "reason": "locale_owner_required", "required_action": "assign_locale_owner"}},
	{"name": "locale_language_supported", "description": "Locales must use a supported language code.", "condition": {"operation": "create_locale", "language_code_supported": False}, "effect": {"decision": "deny", "reason": "language_code_not_supported", "required_action": "choose_supported_language_code"}},
	{"name": "locale_requires_fallback", "description": "Locales require fallback locale policy.", "condition": {"operation": "create_locale", "fallback_locale_present": False}, "effect": {"decision": "deny", "reason": "fallback_locale_required", "required_action": "set_fallback_locale"}},
	{"name": "locale_requires_regional_format", "description": "Locales require regional format metadata.", "condition": {"operation": "create_locale", "regional_format_present": False}, "effect": {"decision": "deny", "reason": "regional_format_required", "required_action": "set_regional_format"}},
	{"name": "glossary_requires_owner", "description": "Glossary terms require an accountable owner.", "condition": {"operation": "add_glossary_term", "glossary_owner_present": False}, "effect": {"decision": "deny", "reason": "glossary_owner_required", "required_action": "assign_glossary_owner"}},
	{"name": "translation_requires_key", "description": "Translations require a localization key.", "condition": {"operation": "upsert_translation", "translation_key_present": False}, "effect": {"decision": "deny", "reason": "translation_key_required", "required_action": "set_translation_key"}},
	{"name": "translation_requires_text", "description": "Translations require localized text.", "condition": {"operation": "upsert_translation", "translated_text_present": False}, "effect": {"decision": "deny", "reason": "translated_text_required", "required_action": "set_translated_text"}},
	{"name": "machine_translation_requires_review", "description": "Machine translations require review before publishing.", "condition": {"machine_translation_used": True, "translation_review_recorded": False}, "effect": {"decision": "deny", "reason": "translation_review_required", "required_action": "review_translation"}},
	{"name": "restricted_content_requires_filtering", "description": "Restricted content requires RBAC filtering.", "condition": {"restricted_content_present": True, "rbac_filter_applied": False}, "effect": {"decision": "deny", "reason": "rbac_filter_required", "required_action": "apply_rbac_filter"}},
	{"name": "publish_requires_approval", "description": "Translation publication requires approval.", "condition": {"operation": "publish_translations", "approval_recorded": False}, "effect": {"decision": "deny", "reason": "publication_approval_required", "required_action": "record_publication_approval"}},
	{"name": "publish_requires_approver", "description": "Translation publication requires an approver.", "condition": {"operation": "publish_translations", "approver_present": False}, "effect": {"decision": "deny", "reason": "publication_approver_required", "required_action": "set_publication_approver"}},
	{"name": "publish_blocks_missing_keys", "description": "Publication blocks missing localization keys unless reviewed.", "condition": {"operation": "publish_translations", "missing_key_count_gt": 0, "missing_key_review_recorded": False}, "effect": {"decision": "deny", "reason": "missing_key_review_required", "required_action": "review_missing_keys"}},
	{"name": "low_coverage_requires_review", "description": "Low localization coverage requires review.", "condition": {"coverage_percent_lt": 95, "coverage_review_recorded": False}, "effect": {"decision": "require_review", "reason": "coverage_review_required", "required_action": "review_locale_coverage"}},
	{"name": "i18n_agent_requires_registration", "description": "AI localization agents must be registered.", "condition": {"i18n_agent_present": True, "agent_registered": False}, "effect": {"decision": "deny", "reason": "i18n_agent_registration_required", "required_action": "register_i18n_agent"}},
	{"name": "i18n_agent_runtime_supported", "description": "AI localization agents must use a supported runtime.", "condition": {"i18n_agent_present": True, "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "i18n_agent_runtime_not_supported", "required_action": "choose_supported_i18n_agent_runtime"}},
	{"name": "i18n_agent_role_supported", "description": "AI localization agents must use a supported role.", "condition": {"i18n_agent_present": True, "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "i18n_agent_role_not_supported", "required_action": "choose_supported_i18n_agent_role"}},
	{"name": "i18n_agent_requires_scope", "description": "AI localization agents require explicit scope.", "condition": {"i18n_agent_present": True, "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "i18n_agent_scope_required", "required_action": "set_i18n_agent_scope"}},
	{"name": "i18n_agent_requires_disclosure", "description": "AI localization-agent contributions require disclosure.", "condition": {"i18n_agent_present": True, "agent_contribution_disclosed": False}, "effect": {"decision": "deny", "reason": "i18n_agent_disclosure_required", "required_action": "disclose_i18n_agent"}},
	{"name": "i18n_state_change_requires_audit", "description": "Localization lifecycle state changes require audit evidence.", "condition": {"state_change_requested": True, "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "i18n_audit_event_required", "required_action": "record_i18n_audit_event"}},
	{"name": "batch_i18n_mutation_requires_bytewax", "description": "Batch localization mutations must use Bytewax event streams.", "condition": {"requested_operation": "batch_i18n_mutation", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}},
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/i18n/dashboard", "component": "I18NDashboard", "permission": "i18n:view", "nav_group": "Overview"},
	{"name": "locales", "path": "/i18n/locales", "component": "LocaleConsole", "permission": "i18n:manage_locales", "nav_group": "Locales"},
	{"name": "translations", "path": "/i18n/translations", "component": "TranslationWorkbench", "permission": "i18n:translate", "nav_group": "Translations"},
	{"name": "glossaries", "path": "/i18n/glossaries", "component": "GlossaryManager", "permission": "i18n:translate", "nav_group": "Translations"},
	{"name": "coverage", "path": "/i18n/coverage", "component": "CoverageDashboard", "permission": "i18n:view", "nav_group": "Quality"},
	{"name": "publishing", "path": "/i18n/publishing", "component": "PublishQueue", "permission": "i18n:publish", "nav_group": "Release"},
	{"name": "agents", "path": "/i18n/agents", "component": "I18NAgentPanel", "permission": "i18n:admin", "nav_group": "Governance"},
	{"name": "audit", "path": "/i18n/audit", "component": "I18NAuditTrail", "permission": "i18n:admin", "nav_group": "Governance"},
	{"name": "policies", "path": "/i18n/policies", "component": "LanguagePolicies", "permission": "i18n:admin", "nav_group": "Governance"},
	{"name": "settings", "path": "/i18n/settings", "component": "I18NSettings", "permission": "i18n:admin", "nav_group": "Administration"},
]

THEME: dict[str, Any] = {
	"name": "i18n_localization_workbench",
	"tokens": {
		"color.primary": "#28536B",
		"color.accent": "#38A169",
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
		"locale_matrix": {"icon": "languages", "status_indicator": "locale-pill", "risk_style": "coverage-band"},
		"translation_editor": {"visual": "side-by-side-editor", "highlight": "glossary-chip"},
		"coverage_dashboard": {"visual": "coverage-grid", "status_style": "gap-chip"},
		"publish_queue": {"visual": "release-checklist", "status_style": "approval-chip"},
		"agent_panel": {"visual": "agent-roster", "status_style": "scope-chip"},
		"audit_trail": {"visual": "event-timeline", "status_style": "policy-chip"},
		"language_policy": {"visual": "policy-table", "status_style": "language-chip"},
	},
}


def streaming_manifest() -> dict[str, Any]:
	return {
		"processor": "bytewax",
		"topic": "apg.i18n.lifecycle",
		"state": ["locales", "glossary_terms", "translations", "coverage_reports", "publish_batches", "i18n_agents", "audit_events"],
		"events": [
			"i18n_locale_created",
			"i18n_glossary_term_added",
			"i18n_translation_upserted",
			"i18n_translation_published",
			"i18n_coverage_reported",
			"i18n_agent_registered",
		],
		"batch_mutation_guardrail": "batch_i18n_mutation_requires_bytewax",
	}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable I18N capability contract."""
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "i18n",
		"display_name": "Internationalization",
		"version": "1.0.0",
		"provides": [
			"locale_management",
			"translation_memory",
			"content_localization",
			"language_fallbacks",
			"regional_formatting",
			"language_policy",
			"i18n_agents",
		],
		"requires": ["conf", "nlpc", "auth", "audl"],
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"view_module": "views.py",
			"api_prefix": "/i18n/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True,
		},
		"theme": deepcopy(THEME),
		"streaming": streaming_manifest(),
	}


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


def language_code_supported(locale_code: str) -> bool:
	"""Return whether the locale language subtag is enabled for this package."""
	return language_subtag(locale_code) in SUPPORTED_LANGUAGE_CODES


def language_subtag(locale_code: str) -> str:
	"""Return the lower-case language subtag from a locale code."""
	return locale_code.replace("_", "-").split("-", 1)[0].lower()


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
