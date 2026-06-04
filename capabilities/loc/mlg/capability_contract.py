"""Executable capability contract for APG Multi-Language & Localisation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "loc_mlg"
CAPABILITY_NAME = "Multi-Language & Localisation"
CAPABILITY_VERSION = "1.0.0"
MLG_EVENT_STREAM = "apg.loc.mlg.lifecycle"

# --- Supported enum constants ---
SUPPORTED_LOCALES = [
	"en_KE", "en_US", "en_GB", "en_ZA", "en_GH", "en_NG", "en_TZ", "en_UG",
	"sw_KE", "sw_TZ", "sw_UG",
	"fr_FR", "fr_BE", "fr_CI", "fr_SN", "fr_CM",
	"ar_SA", "ar_EG", "ar_AE", "ar_MA",
	"de_DE", "de_AT", "de_CH",
	"pt_BR", "pt_PT",
	"zh_CN", "zh_TW",
	"hi_IN", "es_ES", "es_MX",
]
SUPPORTED_LANGUAGES = ["en", "sw", "fr", "ar", "de", "pt", "zh", "hi", "es", "am", "ha", "yo", "ig", "zu", "xh"]
SUPPORTED_SCRIPTS = ["latin", "arabic", "cyrillic", "cjk", "devanagari", "ethiopic"]
SUPPORTED_TEXT_DIRECTIONS = ["ltr", "rtl", "ttb"]
SUPPORTED_DATE_FORMATS = ["DD/MM/YYYY", "MM/DD/YYYY", "YYYY-MM-DD", "D MMMM YYYY", "DD.MM.YYYY", "YYYY/MM/DD"]
SUPPORTED_NUMBER_FORMATS = ["1,234.56", "1.234,56", "1 234,56", "1 234.56", "1234.56"]
SUPPORTED_CURRENCY_DISPLAY_MODES = ["symbol", "code", "name", "narrowSymbol"]
SUPPORTED_TRANSLATION_STATUSES = ["draft", "pending_review", "approved", "published", "deprecated"]
SUPPORTED_CONTENT_TYPES = ["ui_string", "document", "email_template", "report_label", "notification", "help_text", "legal_text"]
SUPPORTED_WORKFLOW_ACTIONS = ["submit_for_review", "approve", "reject", "publish", "deprecate", "request_changes"]
SUPPORTED_APPROVAL_STATUSES = ["pending", "approved", "rejected", "escalated"]
SUPPORTED_RTL_LANGUAGES = ["ar", "he", "fa", "ur", "yi", "dv"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["translation_assistant", "locale_steward", "content_reviewer", "rtl_validator", "terminology_manager"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"locales": {
		"supported_locales": SUPPORTED_LOCALES,
		"supported_languages": SUPPORTED_LANGUAGES,
		"supported_scripts": SUPPORTED_SCRIPTS,
		"default_locale": "en_KE",
		"fallback_locale": "en_US",
		"locale_code_required": True,
	},
	"translations": {
		"supported_statuses": SUPPORTED_TRANSLATION_STATUSES,
		"supported_content_types": SUPPORTED_CONTENT_TYPES,
		"source_language_required": True,
		"target_language_required": True,
		"translator_required": True,
		"reviewer_required_for_approval": True,
		"evidence_required": True,
	},
	"formatting": {
		"supported_date_formats": SUPPORTED_DATE_FORMATS,
		"supported_number_formats": SUPPORTED_NUMBER_FORMATS,
		"supported_currency_display_modes": SUPPORTED_CURRENCY_DISPLAY_MODES,
		"supported_text_directions": SUPPORTED_TEXT_DIRECTIONS,
		"locale_required": True,
	},
	"rtl": {
		"supported_rtl_languages": SUPPORTED_RTL_LANGUAGES,
		"auto_detect_direction": True,
		"bidi_algorithm": "unicode",
	},
	"content_localisation": {
		"supported_content_types": SUPPORTED_CONTENT_TYPES,
		"version_tracking": True,
		"approval_workflow_required": True,
		"publish_requires_approved_status": True,
	},
	"governance": {
		"require_tenant_context": True,
		"policy_attached_for_writes": True,
		"audit_events": True,
		"cross_tenant_translation_denied": True,
		"unapproved_publish_denied": True,
		"untranslated_legal_text_blocked": True,
		"rtl_bypass_denied": True,
	},
	"observability": {"event_stream": MLG_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {
		"auth": "auth",
		"audit": "audl",
		"notifications": "ntfy",
		"workflow": "wflo",
		"nlp": "nlpc",
		"monitoring": "moni",
		"event_stream": "bytewax",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_locales": True,
		"enable_translations": True,
		"enable_formatting": True,
		"enable_rtl": True,
		"enable_content_workflow": True,
		"enable_agents": True,
	},
	"theme": {"default_theme": "loc_mlg_global", "allow_tenant_overrides": True},
}

PROVIDES = [
	"locale_configuration",
	"translation_management",
	"rtl_support",
	"date_number_formatting",
	"content_localisation_workflow",
	"locale_registry",
	"terminology_management",
	"translation_memory",
	"locale_aware_rendering",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "nlpc", "moni", "mqeb"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/loc-mlg/dashboard", "component": "MlgDashboard", "permission": "loc_mlg:view", "nav_group": "Overview"},
	{"name": "locales", "path": "/loc-mlg/locales", "component": "MlgLocaleRegistry", "permission": "loc_mlg:locales", "nav_group": "Setup"},
	{"name": "locales_create", "path": "/loc-mlg/locales/create", "component": "MlgLocaleCreate", "permission": "loc_mlg:locales_write", "nav_group": "Setup"},
	{"name": "translations", "path": "/loc-mlg/translations", "component": "MlgTranslationLedger", "permission": "loc_mlg:translations", "nav_group": "Translations"},
	{"name": "translations_create", "path": "/loc-mlg/translations/create", "component": "MlgTranslationCreate", "permission": "loc_mlg:translations_write", "nav_group": "Translations"},
	{"name": "translation_review", "path": "/loc-mlg/translations/review", "component": "MlgTranslationReviewQueue", "permission": "loc_mlg:translations_review", "nav_group": "Translations"},
	{"name": "formatting", "path": "/loc-mlg/formatting", "component": "MlgFormattingConfig", "permission": "loc_mlg:formatting", "nav_group": "Configuration"},
	{"name": "formatting_create", "path": "/loc-mlg/formatting/create", "component": "MlgFormattingCreate", "permission": "loc_mlg:formatting_write", "nav_group": "Configuration"},
	{"name": "rtl_config", "path": "/loc-mlg/rtl", "component": "MlgRtlConsole", "permission": "loc_mlg:rtl", "nav_group": "Configuration"},
	{"name": "content_workflow", "path": "/loc-mlg/content", "component": "MlgContentWorkflow", "permission": "loc_mlg:content", "nav_group": "Content"},
	{"name": "terminology", "path": "/loc-mlg/terminology", "component": "MlgTerminologyManager", "permission": "loc_mlg:terminology", "nav_group": "Content"},
	{"name": "agents", "path": "/loc-mlg/agents", "component": "MlgAgentWorkbench", "permission": "loc_mlg:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/loc-mlg/settings", "component": "MlgSettings", "permission": "loc_mlg:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "loc_mlg_global",
	"tokens": {
		"color.primary": "#7C3AED",
		"color.accent": "#0891B2",
		"color.success": "#15803D",
		"color.warning": "#B45309",
		"color.danger": "#B91C1C",
		"surface.canvas": "#F8FAFC",
		"surface.panel": "#FFFFFF",
		"text.primary": "#111827",
		"text.secondary": "#4B5563",
		"border.radius": "6px",
		"density": "comfortable",
	},
	"components": {
		"locales": {"icon": "globe", "status_indicator": "locale-status-chip"},
		"translations": {"icon": "languages", "status_indicator": "translation-status-chip"},
		"formatting": {"icon": "sliders", "status_indicator": "format-type-chip"},
		"rtl": {"icon": "align-right", "status_indicator": "direction-chip"},
		"content_workflow": {"icon": "file-edit", "status_indicator": "content-status-chip"},
		"terminology": {"icon": "book-open", "status_indicator": "term-status-chip"},
		"agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": MLG_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"locale_configured",
		"locale_updated",
		"translation_created",
		"translation_submitted_for_review",
		"translation_approved",
		"translation_published",
		"translation_deprecated",
		"formatting_rule_configured",
		"rtl_locale_activated",
		"content_localised",
		"terminology_added",
		"agent_registered",
	],
	"guardrails": [
		"cross_tenant_translation_denied",
		"unapproved_publish_denied",
		"untranslated_legal_text_blocked",
		"rtl_bypass_denied",
		"privileged_agent_action_requires_human_approval",
	],
}

RULES: list[dict[str, Any]] = [
	# Tenant governance
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "policy_required_for_writes", "required_action": "attach_policy"}},
	{"name": "cross_tenant_translation_denied", "condition": {"cross_tenant_operation": True}, "effect": {"decision": "deny", "reason": "cross_tenant_translation_access_denied", "required_action": "use_tenant_scoped_operation"}},
	# Locale rules
	{"name": "locale_code_supported", "condition": {"operation": "configure_locale", "locale_supported": False}, "effect": {"decision": "deny", "reason": "locale_code_not_supported", "required_action": "select_supported_locale"}},
	{"name": "locale_language_supported", "condition": {"operation": "configure_locale", "language_supported": False}, "effect": {"decision": "deny", "reason": "language_code_not_supported", "required_action": "select_supported_language"}},
	{"name": "locale_script_supported", "condition": {"operation": "configure_locale", "script_supported": False}, "effect": {"decision": "deny", "reason": "script_not_supported", "required_action": "select_supported_script"}},
	{"name": "locale_direction_supported", "condition": {"operation": "configure_locale", "direction_supported": False}, "effect": {"decision": "deny", "reason": "text_direction_not_supported", "required_action": "select_supported_direction"}},
	{"name": "locale_date_format_supported", "condition": {"operation": "configure_locale", "date_format_supported": False}, "effect": {"decision": "deny", "reason": "date_format_not_supported", "required_action": "select_supported_date_format"}},
	{"name": "locale_number_format_supported", "condition": {"operation": "configure_locale", "number_format_supported": False}, "effect": {"decision": "deny", "reason": "number_format_not_supported", "required_action": "select_supported_number_format"}},
	# Translation rules
	{"name": "translation_source_language_required", "condition": {"operation": "create_translation", "source_language_present": False}, "effect": {"decision": "deny", "reason": "source_language_required", "required_action": "specify_source_language"}},
	{"name": "translation_target_language_required", "condition": {"operation": "create_translation", "target_language_present": False}, "effect": {"decision": "deny", "reason": "target_language_required", "required_action": "specify_target_language"}},
	{"name": "translation_content_type_supported", "condition": {"operation": "create_translation", "content_type_supported": False}, "effect": {"decision": "deny", "reason": "content_type_not_supported", "required_action": "select_supported_content_type"}},
	{"name": "translation_translator_required", "condition": {"operation": "create_translation", "translator_present": False}, "effect": {"decision": "deny", "reason": "translator_required", "required_action": "assign_translator"}},
	{"name": "translation_key_required", "condition": {"operation": "create_translation", "translation_key_present": False}, "effect": {"decision": "deny", "reason": "translation_key_required", "required_action": "provide_translation_key"}},
	{"name": "translation_reviewer_required_for_approval", "condition": {"operation": "approve_translation", "reviewer_present": False}, "effect": {"decision": "deny", "reason": "reviewer_required_for_approval", "required_action": "assign_reviewer"}},
	{"name": "unapproved_publish_denied", "condition": {"operation": "publish_translation", "status_is_approved": False}, "effect": {"decision": "deny", "reason": "translation_must_be_approved_before_publish", "required_action": "obtain_translation_approval"}},
	{"name": "self_review_denied", "condition": {"operation": "approve_translation", "reviewer_is_translator": True}, "effect": {"decision": "deny", "reason": "translator_cannot_self_review", "required_action": "assign_independent_reviewer"}},
	# RTL rules
	{"name": "rtl_bypass_denied", "condition": {"operation": "configure_locale", "rtl_language": True, "rtl_direction_set": False}, "effect": {"decision": "deny", "reason": "rtl_language_requires_rtl_direction", "required_action": "set_direction_to_rtl"}},
	{"name": "rtl_language_supported", "condition": {"operation": "activate_rtl", "rtl_language_supported": False}, "effect": {"decision": "deny", "reason": "rtl_language_not_in_supported_list", "required_action": "select_supported_rtl_language"}},
	# Content localisation rules
	{"name": "content_type_supported", "condition": {"operation": "localise_content", "content_type_supported": False}, "effect": {"decision": "deny", "reason": "content_type_not_supported", "required_action": "select_supported_content_type"}},
	{"name": "untranslated_legal_text_blocked", "condition": {"operation": "publish_content", "content_type": "legal_text", "translation_approved": False}, "effect": {"decision": "deny", "reason": "legal_text_must_be_translated_and_approved", "required_action": "complete_legal_text_translation"}},
	{"name": "content_locale_required", "condition": {"operation": "localise_content", "target_locale_present": False}, "effect": {"decision": "deny", "reason": "target_locale_required", "required_action": "specify_target_locale"}},
	# Formatting rules
	{"name": "formatting_locale_required", "condition": {"operation": "configure_formatting", "locale_present": False}, "effect": {"decision": "deny", "reason": "locale_required_for_formatting", "required_action": "specify_locale"}},
	# Agent rules
	{"name": "agent_runtime_supported", "condition": {"operation": "register_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "agent_role_supported", "condition": {"operation": "register_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "privileged_agent_action_requires_human_approval", "condition": {"operation": "agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required_for_privileged_action", "required_action": "record_human_approval"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	schema_props = {k: {"type": "object"} for k in configuration if k != "tenant_id"}
	schema_props["tenant_id"] = {"type": "string", "minLength": 1}
	schema_props["ui"] = {"type": "object"}
	schema_props["theme"] = {"type": "object"}
	return {
		"capability": CAPABILITY_ID,
		"name": CAPABILITY_NAME,
		"display_name": CAPABILITY_NAME,
		"version": CAPABILITY_VERSION,
		"provides": list(PROVIDES),
		"requires": list(REQUIRES),
		"configuration": configuration,
		"configuration_schema": {
			"type": "object",
			"required": ["tenant_id", "ui", "theme"],
			"properties": schema_props,
		},
		"rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"api_prefix": "/loc-mlg/api/v1",
			"requires_theme": True,
			"view_module": "views.py",
			"template_roots": ["templates/", "static/"],
			"routes": deepcopy(UI_ROUTES),
		},
		"theme": deepcopy(THEME),
		"streaming": deepcopy(STREAMING),
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	actions: list[dict[str, Any]] = []
	for rule in RULES:
		if _matches(rule["condition"], context):
			actions.append(rule["effect"] | {"rule": rule["name"]})
	if not actions:
		return {"decision": "allow", "actions": [], "context": dict(context)}
	return {"decision": "deny", "actions": actions, "context": dict(context)}


def _matches(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if key.endswith("_ne"):
			if context.get(key[:-3]) == expected:
				return False
			continue
		if context.get(key) != expected:
			return False
	return True
