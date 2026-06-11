"""Executable capability contract for Bank Account Management (ACCT)."""

from __future__ import annotations
from copy import deepcopy
from typing import Any

CAPABILITY_ID = "fin_acct"
CAPABILITY_NAME = "Bank Account Management"
CAPABILITY_VERSION = "1.0.0"
ACCT_EVENT_STREAM = "apg.fin.acct.lifecycle"

SUPPORTED_CURRENCIES = ["USD", "EUR", "GBP", "KES", "ZAR", "NGN", "GHS", "UGX", "TZS"]
SUPPORTED_ACCOUNT_TYPES = ["current", "savings", "fixed_deposit", "loan", "overdraft", "escrow"]
SUPPORTED_TRANSACTION_TYPES = [
	"deposit", "withdrawal", "transfer_in", "transfer_out",
	"fee", "interest", "reversal", "adjustment", "bulk_credit",
]
SUPPORTED_CLOSE_REASONS = ["customer_request", "dormant", "regulatory", "fraud", "deceased"]
SUPPORTED_FREEZE_REASONS = ["fraud_investigation", "legal_order", "aml", "kyc_pending", "admin"]
SUPPORTED_SIGNING_AUTHORITIES = ["single", "joint_any", "joint_all"]

DORMANCY_THRESHOLD_DAYS = 180

STREAMING = {
	"account_opened": ACCT_EVENT_STREAM,
	"account_closed": ACCT_EVENT_STREAM,
	"account_frozen": ACCT_EVENT_STREAM,
	"account_unfrozen": ACCT_EVENT_STREAM,
	"account_dormant": ACCT_EVENT_STREAM,
	"account_reactivated": ACCT_EVENT_STREAM,
	"credit_posted": ACCT_EVENT_STREAM,
	"debit_posted": ACCT_EVENT_STREAM,
	"transfer_completed": ACCT_EVENT_STREAM,
	"funds_locked": ACCT_EVENT_STREAM,
	"funds_released": ACCT_EVENT_STREAM,
	"overdraft_limit_set": ACCT_EVENT_STREAM,
	"gl_journal_requested": "apg.fin.glr.lifecycle",
}

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"dormancy_threshold_days": DORMANCY_THRESHOLD_DAYS,
	"max_overdraft_limit": 1_000_000,
	"min_opening_deposit": 0,
	"iban_country_code": "KE",
	"supported_currencies": SUPPORTED_CURRENCIES,
	"supported_account_types": SUPPORTED_ACCOUNT_TYPES,
	"gl_integration_enabled": True,
	"nats_events_enabled": True,
	"circuit_breaker_threshold": 5,
	"circuit_breaker_timeout_seconds": 60,
	"statement_formats": ["json", "pdf"],
	"bulk_credit_max_items": 5000,
	"lock_max_duration_days": 90,
}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate governance rules. Returns {decision, matched_rules, effects}."""
	matched: list[dict[str, Any]] = []
	decision = "allow"
	effects: list[dict[str, Any]] = []

	for rule in rule_engine["rules"]:
		condition = rule.get("condition", {})
		for field, expected_falsy in condition.items():
			# Condition fires when context field equals the expected value
			ctx_val = context.get(field, not expected_falsy if isinstance(expected_falsy, bool) else None)
			if ctx_val == expected_falsy:
				effect = rule.get("effect", {})
				matched.append(rule["name"])
				effects.append(effect)
				if effect.get("decision") == "deny":
					decision = "deny"
				break

	return {"decision": decision, "matched_rules": matched, "effects": effects}

THEME = {
	"name": "fin_acct_banking",
	"tokens": {
		"color.primary": "#1B4F72",
		"color.accent": "#2874A6",
		"color.success": "#1E8449",
		"color.warning": "#B7950B",
		"color.danger": "#922B21",
		"surface.canvas": "#F4F6F7",
		"surface.panel": "#FFFFFF",
		"text.primary": "#17202A",
		"text.secondary": "#5D6D7E",
		"border.radius": "6px",
		"density": "normal",
	},
	"components": {
		"accounts": {"icon": "account_balance"},
		"transactions": {"icon": "swap_horiz"},
	},
}

configuration_schema = {
	"type": "object",
	"required": ["tenant_id", "ui", "theme"],
	"properties": {
		"tenant_id": {"type": "string"},
		"ui": {"type": "object"},
		"theme": {"type": "object"},
		"default_currency": {"type": "string", "default": "KES"},
	},
}

rule_engine = {
	"type": "deterministic",
	"rules": [
		{"name": "tenant_context_required", "description": "Account ops require tenant context.",
		 "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
		{"name": "sufficient_funds_check", "description": "Debit requires available balance.",
		 "condition": {"sufficient_funds": False}, "effect": {"decision": "deny", "reason": "insufficient_funds", "required_action": "reduce_amount"}},
		{"name": "account_active_required", "description": "Cannot transact on frozen/closed account.",
		 "condition": {"account_active": False}, "effect": {"decision": "deny", "reason": "account_not_active", "required_action": "reactivate_account"}},
		{"name": "double_entry_required", "description": "Monetary transactions must post to GL.",
		 "condition": {"gl_posting_enabled": False}, "effect": {"decision": "deny", "reason": "double_entry_required", "required_action": "enable_gl_posting"}},
		{"name": "decimal_precision_enforced", "description": "Amounts must use Decimal type.",
		 "condition": {"amount_is_decimal": False}, "effect": {"decision": "deny", "reason": "invalid_precision", "required_action": "use_decimal_type"}},
		{"name": "cross_tenant_denied", "description": "Cannot access accounts across tenant boundaries.",
		 "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_denied", "required_action": "use_correct_tenant"}},
		{"name": "audit_all_mutations", "description": "All debit/credit operations must be audited.",
		 "condition": {"audit_enabled": False}, "effect": {"decision": "warn", "reason": "audit_disabled", "required_action": "enable_audit_trail"}},
		{"name": "gl_posting_required", "description": "All monetary txns must have GL entries.",
		 "condition": {"gl_linked": False}, "effect": {"decision": "deny", "reason": "gl_posting_required", "required_action": "link_gl_account"}},
		{"name": "overdraft_requires_approval", "description": "Overdraft limits require manager approval.",
		 "condition": {"overdraft_approval_present": False}, "effect": {"decision": "deny", "reason": "overdraft_not_approved", "required_action": "get_overdraft_approval"}},
		{"name": "dormant_account_restricted", "description": "Dormant accounts require reactivation.",
		 "condition": {"account_dormant": True}, "effect": {"decision": "deny", "reason": "account_dormant", "required_action": "reactivate_account"}},
		{"name": "currency_validated", "description": "Currency code must be in supported set.",
		 "condition": {"currency_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_currency", "required_action": "use_supported_currency"}},
	],
}


def get_capability_contract(tenant_id: str = "default") -> dict:
	"""Return the capability contract for the given tenant."""
	return {
		"capability": CAPABILITY_ID,
		"display_name": CAPABILITY_NAME,
		"name": CAPABILITY_ID,
		"version": CAPABILITY_VERSION,
		"provides": [
			"account_lifecycle", "debit_credit_processing",
			"fund_locking", "balance_inquiry", "statement_generation",
		],
		"requires": ["auth", "audl", "mten", "conf"],
		"configuration": {
			"tenant_id": tenant_id,
			"ui": {},
			"theme": {},
			**deepcopy(DEFAULT_CONFIGURATION),
		},
		"configuration_schema": configuration_schema,
		"rule_engine": rule_engine,
		"ui": {
			"shell": "apg_python",
			"api_prefix": "/fin/acct/api/v1",
			"requires_theme": True,
			"view_module": "views.py",
			"template_roots": ["templates/", "static/"],
			"routes": [
				{"name": "dashboard", "path": "/fin/acct/dashboard",
				 "component": "AccountDashboard",
				 "permission": "fin_acct:view", "nav_group": "Overview"},
				{"name": "accounts", "path": "/fin/acct/accounts",
				 "component": "AccountList",
				 "permission": "fin_acct:view", "nav_group": "Accounts"},
				{"name": "transactions", "path": "/fin/acct/transactions",
				 "component": "TransactionList",
				 "permission": "fin_acct:view", "nav_group": "Transactions"},
				{"name": "statements", "path": "/fin/acct/statements",
				 "component": "StatementView",
				 "permission": "fin_acct:view", "nav_group": "Reporting"},
			],
		},
		"theme": THEME,
		"streaming": {"stream": "apg.fin.acct.lifecycle", "events": []},
	}
