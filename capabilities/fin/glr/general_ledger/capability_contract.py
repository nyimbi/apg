"""Executable capability contract for Financial Management General Ledger."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "glr_general_ledger"
CAPABILITY_NAME = "Financial Management General Ledger"
CAPABILITY_VERSION = "2.1.0"
GLR_EVENT_STREAM = "apg.fin.glr.lifecycle"

SUPPORTED_ACCOUNT_TYPES = ["asset", "liability", "equity", "revenue", "expense"]
SUPPORTED_CURRENCIES = ["USD", "EUR", "GBP", "KES", "ZAR", "NGN", "GHS", "UGX", "TZS"]
SUPPORTED_JOURNAL_SOURCES = ["manual", "ap", "ar", "cash", "payroll", "allocation", "reversal", "import"]
SUPPORTED_GLR_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_GLR_AGENT_ROLES = [
	"journal_reviewer",
	"posting_reviewer",
	"period_close_reviewer",
	"reconciliation_reviewer",
	"allocation_reviewer",
	"trial_balance_reviewer",
]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"chart_of_accounts": {
		"account_code_required": True,
		"account_name_required": True,
		"account_type_required": True,
		"supported_account_types": SUPPORTED_ACCOUNT_TYPES,
		"parent_cycle_check_required": True,
		"posting_account_required_for_journals": True,
	},
	"dimensions": {
		"enabled": True,
		"supported_dimensions": ["department", "cost_center", "project", "location", "product"],
		"dimension_policy_required": True,
	},
	"periods": {
		"fiscal_year_required": True,
		"period_dates_required": True,
		"open_period_required_for_posting": True,
		"closed_period_adjustment_requires_approval": True,
	},
	"journals": {
		"batch_required": True,
		"description_required": True,
		"minimum_line_count": 2,
		"balanced_entry_required": True,
		"approval_required_for_posting": True,
		"segregation_of_duties": True,
		"idempotency_key_required": True,
		"supported_sources": SUPPORTED_JOURNAL_SOURCES,
	},
	"currency": {
		"base_currency": "USD",
		"supported_currencies": SUPPORTED_CURRENCIES,
		"exchange_rate_required_for_foreign_currency": True,
		"positive_exchange_rate_required": True,
	},
	"balances": {
		"maintain_running_balances": True,
		"trial_balance_must_balance": True,
		"posting_emits_balance_events": True,
	},
	"reversals": {
		"posted_entry_required": True,
		"reason_required": True,
		"approval_required": True,
	},
	"allocations": {
		"basis_required": True,
		"source_account_required": True,
		"target_account_required": True,
		"review_required": True,
	},
	"glr_agents": {
		"enabled": True,
		"supported_runtimes": SUPPORTED_GLR_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_GLR_AGENT_ROLES,
		"max_autonomous_scope": "recommend_validate_and_prepare",
		"human_approval_required": True,
	},
	"governance": {
		"require_tenant_context": True,
		"policy_attached_for_writes": True,
		"audit_state_changes": True,
		"segregation_of_duties": True,
	},
	"observability": {
		"event_stream": GLR_EVENT_STREAM,
		"stream_processor": "bytewax",
		"emit_account_events": True,
		"emit_period_events": True,
		"emit_journal_events": True,
		"emit_posting_events": True,
		"emit_balance_events": True,
		"emit_agent_events": True,
	},
	"adapters": {
		"authorization": "adapter",
		"audit": "adapter",
		"notification": "adapter",
		"document_management": "adapter",
		"business_intelligence": "adapter",
		"composition_events": "bytewax",
		"event_stream": "bytewax",
		"theme": "adapter",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_accounts": True,
		"enable_dimensions": True,
		"enable_periods": True,
		"enable_journal_batches": True,
		"enable_journals": True,
		"enable_postings": True,
		"enable_trial_balance": True,
		"enable_allocations": True,
		"enable_reversals": True,
		"enable_agents": True,
		"enable_settings": True,
	},
	"theme": {
		"default_theme": "glr_general_ledger_control",
		"allow_tenant_overrides": True,
	},
}


PROVIDES = [
	"chart_of_accounts_lifecycle",
	"ledger_dimension_management",
	"accounting_period_lifecycle",
	"journal_batch_lifecycle",
	"journal_entry_lifecycle",
	"journal_posting_workflow",
	"ledger_balance_service",
	"trial_balance_reporting",
	"allocation_and_reversal_workflow",
	"glr_agents",
]

REQUIRES = [
	"auth",
	"audl",
	"mten",
	"conf",
	"ntfy",
	"mqeb",
	"wflo",
	"srch",
]


UI_ROUTES = [
	{"name": "dashboard", "path": "/glr-general-ledger/dashboard", "component": "GeneralLedgerDashboard", "permission": "glr_general_ledger:view", "nav_group": "Overview"},
	{"name": "accounts", "path": "/glr-general-ledger/accounts", "component": "ChartOfAccountsWorkbench", "permission": "glr_general_ledger:manage_accounts", "nav_group": "Ledger"},
	{"name": "dimensions", "path": "/glr-general-ledger/dimensions", "component": "LedgerDimensionConsole", "permission": "glr_general_ledger:manage_dimensions", "nav_group": "Ledger"},
	{"name": "periods", "path": "/glr-general-ledger/periods", "component": "AccountingPeriodConsole", "permission": "glr_general_ledger:manage_periods", "nav_group": "Close"},
	{"name": "journal_batches", "path": "/glr-general-ledger/batches", "component": "JournalBatchQueue", "permission": "glr_general_ledger:enter_journals", "nav_group": "Journals"},
	{"name": "journals", "path": "/glr-general-ledger/journals", "component": "JournalEntryWorkbench", "permission": "glr_general_ledger:enter_journals", "nav_group": "Journals"},
	{"name": "postings", "path": "/glr-general-ledger/postings", "component": "PostingControlDesk", "permission": "glr_general_ledger:post", "nav_group": "Journals"},
	{"name": "trial_balance", "path": "/glr-general-ledger/trial-balance", "component": "TrialBalanceConsole", "permission": "glr_general_ledger:report", "nav_group": "Reports"},
	{"name": "allocations", "path": "/glr-general-ledger/allocations", "component": "AllocationWorkbench", "permission": "glr_general_ledger:allocate", "nav_group": "Controls"},
	{"name": "reversals", "path": "/glr-general-ledger/reversals", "component": "ReversalWorkbench", "permission": "glr_general_ledger:reverse", "nav_group": "Controls"},
	{"name": "agents", "path": "/glr-general-ledger/agents", "component": "GLRAgentWorkbench", "permission": "glr_general_ledger:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/glr-general-ledger/settings", "component": "GeneralLedgerSettings", "permission": "glr_general_ledger:admin", "nav_group": "Administration"},
]


THEME = {
	"name": "glr_general_ledger_control",
	"tokens": {
		"color.primary": "#28536B",
		"color.accent": "#C44536",
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
		"accounts": {"icon": "book-open", "status_indicator": "account-pill", "risk_style": "ledger-band"},
		"periods": {"visual": "period-calendar", "status_style": "period-chip"},
		"journals": {"visual": "balanced-entry-grid", "status_style": "journal-chip"},
		"postings": {"visual": "posting-queue", "status_style": "posting-chip"},
		"trial_balance": {"visual": "balance-grid", "status_style": "balance-chip"},
		"allocations": {"visual": "allocation-map", "status_style": "review-chip"},
		"agents": {"visual": "review-lane", "status_style": "agent-chip"},
	},
}


STREAMING = {
	"processor": "bytewax",
	"stream": GLR_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"account_created",
		"dimension_recorded",
		"period_opened",
		"journal_batch_created",
		"journal_entry_created",
		"journal_approved",
		"journal_posted",
		"journal_reversed",
		"trial_balance_generated",
		"allocation_created",
		"glr_agent_registered",
	],
	"states": ["draft", "open", "balanced", "approved", "posted", "reversed", "closed", "blocked"],
	"guardrails": [
		"glr_batch_requires_bytewax",
		"glr_event_requires_bytewax",
		"privileged_agent_glr_action_requires_human_approval",
	],
}


RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "General ledger operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "glr_write_requires_policy", "description": "General ledger writes require policy attachment.", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "operation_policy_required", "required_action": "attach_operation_policy"}},
	{"name": "account_requires_code", "description": "Ledger accounts require an account code.", "condition": {"operation": "create_account", "account_code_present": False}, "effect": {"decision": "deny", "reason": "account_code_required", "required_action": "set_account_code"}},
	{"name": "account_requires_name", "description": "Ledger accounts require a name.", "condition": {"operation": "create_account", "account_name_present": False}, "effect": {"decision": "deny", "reason": "account_name_required", "required_action": "set_account_name"}},
	{"name": "account_type_supported", "description": "Ledger account type must be supported.", "condition": {"operation": "create_account", "account_type_supported": False}, "effect": {"decision": "deny", "reason": "account_type_not_supported", "required_action": "select_supported_account_type"}},
	{"name": "account_parent_cycle_blocked", "description": "Account hierarchy must not contain cycles.", "condition": {"operation": "create_account", "parent_cycle_detected": True}, "effect": {"decision": "deny", "reason": "account_parent_cycle_detected", "required_action": "select_valid_parent"}},
	{"name": "period_requires_name", "description": "Accounting periods require a name.", "condition": {"operation": "open_period", "period_name_present": False}, "effect": {"decision": "deny", "reason": "period_name_required", "required_action": "name_period"}},
	{"name": "period_requires_fiscal_year", "description": "Accounting periods require fiscal year.", "condition": {"operation": "open_period", "fiscal_year_present": False}, "effect": {"decision": "deny", "reason": "fiscal_year_required", "required_action": "set_fiscal_year"}},
	{"name": "period_requires_dates", "description": "Accounting periods require start and end dates.", "condition": {"operation": "open_period", "period_dates_present": False}, "effect": {"decision": "deny", "reason": "period_dates_required", "required_action": "set_period_dates"}},
	{"name": "period_end_after_start", "description": "Accounting period end must be after start.", "condition": {"operation": "open_period", "period_range_valid": False}, "effect": {"decision": "deny", "reason": "period_range_invalid", "required_action": "set_valid_period_range"}},
	{"name": "journal_batch_requires_period", "description": "Journal batches require an open period.", "condition": {"operation": "create_journal_batch", "period_open": False}, "effect": {"decision": "deny", "reason": "journal_batch_open_period_required", "required_action": "open_period"}},
	{"name": "journal_batch_source_supported", "description": "Journal batch source must be supported.", "condition": {"operation": "create_journal_batch", "journal_source_supported": False}, "effect": {"decision": "deny", "reason": "journal_source_not_supported", "required_action": "select_supported_source"}},
	{"name": "journal_batch_currency_supported", "description": "Journal batch currency must be supported.", "condition": {"operation": "create_journal_batch", "currency_supported": False}, "effect": {"decision": "deny", "reason": "journal_currency_not_supported", "required_action": "select_supported_currency"}},
	{"name": "journal_requires_batch", "description": "Journal entries require a batch.", "condition": {"operation": "create_journal_entry", "batch_present": False}, "effect": {"decision": "deny", "reason": "journal_batch_required", "required_action": "attach_batch"}},
	{"name": "journal_requires_description", "description": "Journal entries require a description.", "condition": {"operation": "create_journal_entry", "journal_description_present": False}, "effect": {"decision": "deny", "reason": "journal_description_required", "required_action": "describe_journal"}},
	{"name": "journal_requires_two_lines", "description": "Journal entries require at least two lines.", "condition": {"operation": "create_journal_entry", "journal_line_count_lt": 2}, "effect": {"decision": "deny", "reason": "journal_lines_required", "required_action": "add_journal_lines"}},
	{"name": "journal_requires_posting_accounts", "description": "Journal lines require valid posting accounts.", "condition": {"operation": "create_journal_entry", "posting_accounts_valid": False}, "effect": {"decision": "deny", "reason": "posting_accounts_required", "required_action": "select_posting_accounts"}},
	{"name": "journal_must_balance", "description": "Journal debits and credits must balance.", "condition": {"operation": "create_journal_entry", "balanced": False}, "effect": {"decision": "deny", "reason": "journal_not_balanced", "required_action": "balance_journal"}},
	{"name": "foreign_currency_requires_rate", "description": "Foreign currency journals require an exchange rate.", "condition": {"operation": "create_journal_entry", "foreign_currency": True, "exchange_rate_present": False}, "effect": {"decision": "deny", "reason": "exchange_rate_required", "required_action": "attach_exchange_rate"}},
	{"name": "exchange_rate_positive", "description": "Exchange rates must be positive.", "condition": {"operation": "record_currency_rate", "exchange_rate_lte": 0}, "effect": {"decision": "deny", "reason": "exchange_rate_positive_required", "required_action": "set_positive_exchange_rate"}},
	{"name": "post_requires_entry", "description": "Posting requires a journal entry.", "condition": {"operation": "post_journal", "journal_present": False}, "effect": {"decision": "deny", "reason": "journal_entry_required", "required_action": "select_journal"}},
	{"name": "post_requires_approval", "description": "Posting requires journal approval.", "condition": {"operation": "post_journal", "approval_recorded": False}, "effect": {"decision": "deny", "reason": "journal_approval_required", "required_action": "approve_journal"}},
	{"name": "post_requires_open_period", "description": "Posting requires an open accounting period.", "condition": {"operation": "post_journal", "period_open": False}, "effect": {"decision": "deny", "reason": "open_period_required", "required_action": "open_period"}},
	{"name": "post_requires_idempotency_key", "description": "Posting requires idempotency key.", "condition": {"operation": "post_journal", "idempotency_key_present": False}, "effect": {"decision": "deny", "reason": "idempotency_key_required", "required_action": "attach_idempotency_key"}},
	{"name": "post_requires_sod", "description": "Posting requires segregation of duties.", "condition": {"operation": "post_journal", "same_preparer_and_poster": True}, "effect": {"decision": "deny", "reason": "segregation_of_duties_required", "required_action": "assign_independent_poster"}},
	{"name": "closed_period_adjustment_requires_approval", "description": "Closed-period adjustments require review.", "condition": {"operation": "post_journal", "closed_period_adjustment": True, "approval_recorded": False}, "effect": {"decision": "deny", "reason": "closed_period_adjustment_approval_required", "required_action": "approve_adjustment"}},
	{"name": "reverse_requires_posted_entry", "description": "Reversals require a posted journal.", "condition": {"operation": "reverse_journal", "posted_entry_present": False}, "effect": {"decision": "deny", "reason": "posted_entry_required", "required_action": "select_posted_entry"}},
	{"name": "reverse_requires_reason", "description": "Reversals require a reason.", "condition": {"operation": "reverse_journal", "reversal_reason_present": False}, "effect": {"decision": "deny", "reason": "reversal_reason_required", "required_action": "record_reversal_reason"}},
	{"name": "trial_balance_must_balance", "description": "Trial balance total debits and credits must balance.", "condition": {"operation": "generate_trial_balance", "trial_balance_balanced": False}, "effect": {"decision": "deny", "reason": "trial_balance_not_balanced", "required_action": "investigate_balance"}},
	{"name": "allocation_requires_basis", "description": "Allocations require a basis.", "condition": {"operation": "create_allocation", "allocation_basis_present": False}, "effect": {"decision": "deny", "reason": "allocation_basis_required", "required_action": "define_allocation_basis"}},
	{"name": "allocation_requires_review", "description": "Allocations require review.", "condition": {"operation": "create_allocation", "allocation_review_recorded": False}, "effect": {"decision": "require_review", "reason": "allocation_review_required", "required_action": "record_allocation_review"}},
	{"name": "glr_batch_requires_bytewax", "description": "General ledger batches require Bytewax coordination.", "condition": {"operation": "glr_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_glr_batch_to_bytewax"}},
	{"name": "glr_event_requires_bytewax", "description": "General ledger lifecycle events require Bytewax.", "condition": {"operation": "glr_event", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_glr_event_to_bytewax"}},
	{"name": "glr_agent_runtime_supported", "description": "GLR agents must use an approved runtime.", "condition": {"operation": "register_glr_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "glr_agent_runtime_not_supported", "required_action": "select_supported_agent_runtime"}},
	{"name": "glr_agent_role_supported", "description": "GLR agents must use an approved role.", "condition": {"operation": "register_glr_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "glr_agent_role_not_supported", "required_action": "select_supported_agent_role"}},
	{"name": "privileged_agent_glr_action_requires_human_approval", "description": "Privileged GLR actions proposed by agents require human approval.", "condition": {"operation": "agent_glr_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
]


def _configuration_schema() -> dict[str, Any]:
	return {
		"type": "object",
		"required": list(DEFAULT_CONFIGURATION),
		"properties": {
			key: {"type": "object"} for key in DEFAULT_CONFIGURATION if key != "tenant_id"
		} | {"tenant_id": {"type": "string", "minLength": 1}},
	}


def _matches_condition(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if key.endswith("_lte"):
			if context.get(key[:-4]) is None or context[key[:-4]] > expected:
				return False
			continue
		if key.endswith("_lt"):
			if context.get(key[:-3]) is None or context[key[:-3]] >= expected:
				return False
			continue
		if key.endswith("_gte"):
			if context.get(key[:-4]) is None or context[key[:-4]] < expected:
				return False
			continue
		if key.endswith("_gt"):
			if context.get(key[:-3]) is None or context[key[:-3]] <= expected:
				return False
			continue
		if key.endswith("_ne"):
			if context.get(key[:-3]) == expected:
				return False
			continue
		if context.get(key) != expected:
			return False
	return True


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	if overrides:
		for key, value in overrides.items():
			if isinstance(value, dict) and isinstance(configuration.get(key), dict):
				configuration[key].update(value)
			else:
				configuration[key] = value

	return {
		"capability": CAPABILITY_ID,
		"display_name": CAPABILITY_NAME,
		"version": CAPABILITY_VERSION,
		"configuration": configuration,
		"configuration_schema": _configuration_schema(),
		"provides": PROVIDES,
		"requires": REQUIRES,
		"rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"api_prefix": "/glr-general-ledger/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"requires_theme": True,
			"view_module": "views.py",
			"template_roots": ["templates/", "static/"],
		},
		"theme": deepcopy(THEME),
		"streaming": deepcopy(STREAMING),
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	contract = get_capability_contract(context.get("tenant_id", "default"))
	matched: list[dict[str, Any]] = [
		rule for rule in contract["rule_engine"]["rules"]
		if _matches_condition(rule["condition"], context)
	]
	decision = "allow"
	for rule in matched:
		rule_decision = rule["effect"]["decision"]
		if rule_decision == "deny":
			decision = "deny"
			break
		if rule_decision == "require_review" and decision == "allow":
			decision = "require_review"
	return {
		"decision": decision,
		"matched_rules": [rule["name"] for rule in matched],
		"effects": [rule["effect"] for rule in matched],
	}
