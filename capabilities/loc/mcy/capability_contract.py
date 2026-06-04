"""Executable capability contract for APG Multi-Currency Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "loc_mcy"
CAPABILITY_NAME = "Multi-Currency Management"
CAPABILITY_VERSION = "1.0.0"
MCY_EVENT_STREAM = "apg.loc.mcy.lifecycle"

# --- Supported enum constants ---
SUPPORTED_CURRENCIES = [
	"KES", "USD", "EUR", "GBP", "ZAR", "NGN", "GHS", "TZS", "UGX", "RWF",
	"ETB", "XOF", "XAF", "MWK", "ZMW", "JPY", "CNY", "INR", "AED", "CAD",
	"AUD", "CHF", "SEK", "NOK", "DKK", "SGD", "HKD", "BRL", "MXN", "CLP",
]
SUPPORTED_RATE_TYPES = ["spot", "forward", "average", "closing", "opening", "budget", "custom"]
SUPPORTED_RATE_SOURCES = ["central_bank", "ecb", "bloomberg", "reuters", "xe", "manual", "api_feed"]
SUPPORTED_REVALUATION_METHODS = ["closing_rate", "average_rate", "historical_rate", "monetary_nonmonetary"]
SUPPORTED_TRANSLATION_METHODS = ["current_rate", "temporal", "current_noncurrent", "monetary_nonmonetary"]
SUPPORTED_ROUNDING_MODES = ["round_half_up", "round_half_even", "round_up", "round_down", "truncate"]
SUPPORTED_FX_ACCOUNT_TYPES = ["realised_gain", "realised_loss", "unrealised_gain", "unrealised_loss", "translation_reserve", "rounding_difference"]
SUPPORTED_REVALUATION_STATUSES = ["draft", "pending_approval", "approved", "posted", "reversed"]
SUPPORTED_TRANSLATION_STATUSES = ["draft", "pending_approval", "approved", "posted", "reversed"]
SUPPORTED_APPROVAL_STATUSES = ["pending", "approved", "rejected", "escalated"]
SUPPORTED_CURRENCY_STATUSES = ["active", "inactive", "suspended"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["rate_steward", "revaluation_reviewer", "translation_reviewer", "fx_reporter", "rate_feed_monitor"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"currencies": {
		"supported_currencies": SUPPORTED_CURRENCIES,
		"supported_statuses": SUPPORTED_CURRENCY_STATUSES,
		"functional_currency_required": True,
		"presentation_currency_configurable": True,
	},
	"exchange_rates": {
		"supported_rate_types": SUPPORTED_RATE_TYPES,
		"supported_sources": SUPPORTED_RATE_SOURCES,
		"effective_date_required": True,
		"source_required": True,
		"approval_required_for_manual": True,
	},
	"revaluation": {
		"supported_methods": SUPPORTED_REVALUATION_METHODS,
		"supported_statuses": SUPPORTED_REVALUATION_STATUSES,
		"period_required": True,
		"approval_required": True,
		"fx_account_required": True,
	},
	"translation": {
		"supported_methods": SUPPORTED_TRANSLATION_METHODS,
		"supported_statuses": SUPPORTED_TRANSLATION_STATUSES,
		"target_currency_required": True,
		"approval_required": True,
		"translation_reserve_account_required": True,
	},
	"rounding": {
		"supported_modes": SUPPORTED_ROUNDING_MODES,
		"default_mode": "round_half_even",
		"rounding_difference_account_required": True,
	},
	"fx_accounts": {
		"supported_types": SUPPORTED_FX_ACCOUNT_TYPES,
		"account_code_required": True,
	},
	"governance": {
		"require_tenant_context": True,
		"policy_attached_for_writes": True,
		"audit_events": True,
		"cross_tenant_rate_denied": True,
		"unapproved_revaluation_posting_denied": True,
		"unapproved_translation_posting_denied": True,
		"rate_backdating_restricted": True,
		"fx_gain_loss_account_bypass_denied": True,
	},
	"observability": {"event_stream": MCY_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {
		"auth": "auth",
		"audit": "audl",
		"notifications": "ntfy",
		"workflow": "wflo",
		"monitoring": "moni",
		"scheduler": "schd",
		"event_stream": "bytewax",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_currencies": True,
		"enable_exchange_rates": True,
		"enable_revaluation": True,
		"enable_translation": True,
		"enable_fx_reporting": True,
		"enable_agents": True,
	},
	"theme": {"default_theme": "loc_mcy_finance", "allow_tenant_overrides": True},
}

PROVIDES = [
	"currency_configuration",
	"exchange_rate_management",
	"fx_revaluation_workflow",
	"currency_translation_workflow",
	"fx_gain_loss_reporting",
	"multi_currency_rounding",
	"rate_feed_integration",
	"currency_exposure_dashboard",
	"fx_account_registry",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "moni", "schd", "mqeb"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/loc-mcy/dashboard", "component": "McyDashboard", "permission": "loc_mcy:view", "nav_group": "Overview"},
	{"name": "currencies", "path": "/loc-mcy/currencies", "component": "McyCurrencyRegistry", "permission": "loc_mcy:currencies", "nav_group": "Setup"},
	{"name": "currencies_create", "path": "/loc-mcy/currencies/create", "component": "McyCurrencyCreate", "permission": "loc_mcy:currencies_write", "nav_group": "Setup"},
	{"name": "exchange_rates", "path": "/loc-mcy/exchange-rates", "component": "McyRateLedger", "permission": "loc_mcy:exchange_rates", "nav_group": "Rates"},
	{"name": "exchange_rates_create", "path": "/loc-mcy/exchange-rates/create", "component": "McyRateCreate", "permission": "loc_mcy:exchange_rates_write", "nav_group": "Rates"},
	{"name": "exchange_rates_upload", "path": "/loc-mcy/exchange-rates/upload", "component": "McyRateBulkUpload", "permission": "loc_mcy:exchange_rates_write", "nav_group": "Rates"},
	{"name": "revaluation", "path": "/loc-mcy/revaluation", "component": "McyRevaluationConsole", "permission": "loc_mcy:revaluation", "nav_group": "Processing"},
	{"name": "revaluation_create", "path": "/loc-mcy/revaluation/create", "component": "McyRevaluationCreate", "permission": "loc_mcy:revaluation_write", "nav_group": "Processing"},
	{"name": "translation", "path": "/loc-mcy/translation", "component": "McyTranslationConsole", "permission": "loc_mcy:translation", "nav_group": "Processing"},
	{"name": "translation_create", "path": "/loc-mcy/translation/create", "component": "McyTranslationCreate", "permission": "loc_mcy:translation_write", "nav_group": "Processing"},
	{"name": "fx_accounts", "path": "/loc-mcy/fx-accounts", "component": "McyFxAccountRegistry", "permission": "loc_mcy:fx_accounts", "nav_group": "Setup"},
	{"name": "fx_reporting", "path": "/loc-mcy/fx-reporting", "component": "McyFxGainLossReport", "permission": "loc_mcy:fx_reporting", "nav_group": "Reporting"},
	{"name": "agents", "path": "/loc-mcy/agents", "component": "McyAgentWorkbench", "permission": "loc_mcy:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/loc-mcy/settings", "component": "McySettings", "permission": "loc_mcy:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "loc_mcy_finance",
	"tokens": {
		"color.primary": "#0F4C81",
		"color.accent": "#059669",
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
		"currencies": {"icon": "banknote", "status_indicator": "currency-status-chip"},
		"exchange_rates": {"icon": "trending-up", "status_indicator": "rate-source-chip"},
		"revaluation": {"icon": "refresh-cw", "status_indicator": "revaluation-status-chip"},
		"translation": {"icon": "globe-2", "status_indicator": "translation-status-chip"},
		"fx_accounts": {"icon": "landmark", "status_indicator": "fx-account-type-chip"},
		"fx_reporting": {"icon": "bar-chart-2", "status_indicator": "fx-impact-chip"},
		"agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": MCY_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"currency_configured",
		"currency_updated",
		"exchange_rate_recorded",
		"exchange_rate_bulk_loaded",
		"revaluation_created",
		"revaluation_approved",
		"revaluation_posted",
		"revaluation_reversed",
		"translation_created",
		"translation_approved",
		"translation_posted",
		"fx_account_registered",
		"fx_gain_loss_calculated",
		"agent_registered",
	],
	"guardrails": [
		"cross_tenant_rate_denied",
		"unapproved_revaluation_posting_denied",
		"unapproved_translation_posting_denied",
		"rate_backdating_restricted",
		"fx_gain_loss_account_bypass_denied",
		"privileged_agent_action_requires_human_approval",
	],
}

RULES: list[dict[str, Any]] = [
	# Tenant governance
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "policy_required_for_writes", "required_action": "attach_policy"}},
	{"name": "cross_tenant_rate_denied", "condition": {"cross_tenant_operation": True}, "effect": {"decision": "deny", "reason": "cross_tenant_currency_access_denied", "required_action": "use_tenant_scoped_operation"}},
	# Currency configuration rules
	{"name": "currency_code_supported", "condition": {"operation": "configure_currency", "currency_supported": False}, "effect": {"decision": "deny", "reason": "currency_code_not_supported", "required_action": "select_supported_currency"}},
	{"name": "currency_name_required", "condition": {"operation": "configure_currency", "currency_name_present": False}, "effect": {"decision": "deny", "reason": "currency_name_required", "required_action": "provide_currency_name"}},
	{"name": "currency_precision_valid", "condition": {"operation": "configure_currency", "precision_valid": False}, "effect": {"decision": "deny", "reason": "currency_decimal_precision_invalid", "required_action": "set_valid_precision_0_to_6"}},
	{"name": "rounding_mode_supported", "condition": {"operation": "configure_currency", "rounding_mode_supported": False}, "effect": {"decision": "deny", "reason": "rounding_mode_not_supported", "required_action": "select_supported_rounding_mode"}},
	# Exchange rate rules
	{"name": "rate_from_currency_supported", "condition": {"operation": "record_rate", "from_currency_supported": False}, "effect": {"decision": "deny", "reason": "from_currency_not_supported", "required_action": "select_supported_from_currency"}},
	{"name": "rate_to_currency_supported", "condition": {"operation": "record_rate", "to_currency_supported": False}, "effect": {"decision": "deny", "reason": "to_currency_not_supported", "required_action": "select_supported_to_currency"}},
	{"name": "rate_type_supported", "condition": {"operation": "record_rate", "rate_type_supported": False}, "effect": {"decision": "deny", "reason": "rate_type_not_supported", "required_action": "select_supported_rate_type"}},
	{"name": "rate_source_supported", "condition": {"operation": "record_rate", "rate_source_supported": False}, "effect": {"decision": "deny", "reason": "rate_source_not_supported", "required_action": "select_supported_rate_source"}},
	{"name": "rate_effective_date_required", "condition": {"operation": "record_rate", "effective_date_present": False}, "effect": {"decision": "deny", "reason": "effective_date_required", "required_action": "set_effective_date"}},
	{"name": "rate_value_positive", "condition": {"operation": "record_rate", "rate_positive": False}, "effect": {"decision": "deny", "reason": "exchange_rate_must_be_positive", "required_action": "set_positive_rate"}},
	{"name": "manual_rate_approval_required", "condition": {"operation": "record_rate", "rate_source": "manual", "approval_present": False}, "effect": {"decision": "deny", "reason": "manual_rate_requires_approval", "required_action": "attach_rate_approval"}},
	{"name": "rate_backdating_restricted", "condition": {"operation": "record_rate", "backdated": True, "backdating_override_present": False}, "effect": {"decision": "deny", "reason": "rate_backdating_requires_override", "required_action": "attach_backdating_override"}},
	# Revaluation rules
	{"name": "revaluation_method_supported", "condition": {"operation": "create_revaluation", "revaluation_method_supported": False}, "effect": {"decision": "deny", "reason": "revaluation_method_not_supported", "required_action": "select_supported_revaluation_method"}},
	{"name": "revaluation_period_required", "condition": {"operation": "create_revaluation", "period_present": False}, "effect": {"decision": "deny", "reason": "revaluation_period_required", "required_action": "set_revaluation_period"}},
	{"name": "revaluation_fx_account_required", "condition": {"operation": "create_revaluation", "fx_account_present": False}, "effect": {"decision": "deny", "reason": "fx_gain_loss_account_required", "required_action": "assign_fx_account"}},
	{"name": "unapproved_revaluation_posting_denied", "condition": {"operation": "post_revaluation", "approval_present": False}, "effect": {"decision": "deny", "reason": "revaluation_approval_required_before_posting", "required_action": "obtain_revaluation_approval"}},
	{"name": "revaluation_reversal_requires_posted_status", "condition": {"operation": "reverse_revaluation", "status_is_posted": False}, "effect": {"decision": "deny", "reason": "only_posted_revaluations_can_be_reversed", "required_action": "ensure_posted_status"}},
	# Translation rules
	{"name": "translation_method_supported", "condition": {"operation": "create_translation", "translation_method_supported": False}, "effect": {"decision": "deny", "reason": "translation_method_not_supported", "required_action": "select_supported_translation_method"}},
	{"name": "translation_target_currency_required", "condition": {"operation": "create_translation", "target_currency_present": False}, "effect": {"decision": "deny", "reason": "target_currency_required", "required_action": "select_target_currency"}},
	{"name": "translation_target_currency_supported", "condition": {"operation": "create_translation", "target_currency_supported": False}, "effect": {"decision": "deny", "reason": "target_currency_not_supported", "required_action": "select_supported_target_currency"}},
	{"name": "translation_reserve_account_required", "condition": {"operation": "create_translation", "reserve_account_present": False}, "effect": {"decision": "deny", "reason": "translation_reserve_account_required", "required_action": "assign_reserve_account"}},
	{"name": "unapproved_translation_posting_denied", "condition": {"operation": "post_translation", "approval_present": False}, "effect": {"decision": "deny", "reason": "translation_approval_required_before_posting", "required_action": "obtain_translation_approval"}},
	# FX account rules
	{"name": "fx_account_type_supported", "condition": {"operation": "register_fx_account", "account_type_supported": False}, "effect": {"decision": "deny", "reason": "fx_account_type_not_supported", "required_action": "select_supported_account_type"}},
	{"name": "fx_account_code_required", "condition": {"operation": "register_fx_account", "account_code_present": False}, "effect": {"decision": "deny", "reason": "fx_account_code_required", "required_action": "provide_account_code"}},
	{"name": "fx_gain_loss_account_bypass_denied", "condition": {"operation": "post_revaluation", "fx_account_bypass": True}, "effect": {"decision": "deny", "reason": "fx_gain_loss_account_bypass_denied", "required_action": "use_designated_fx_account"}},
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
			"api_prefix": "/loc-mcy/api/v1",
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
