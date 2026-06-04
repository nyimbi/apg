"""Executable capability contract for APG Treasury Management System."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "fintech_treasury"
CAPABILITY_NAME = "Treasury Management System"
CAPABILITY_VERSION = "1.1.0"
TREASURY_EVENT_STREAM = "apg.fintech.treasury.lifecycle"

SUPPORTED_CURRENCIES = [
	"KES", "UGX", "TZS", "RWF", "GHS", "NGN", "ZAR",
	"USD", "EUR", "GBP", "JPY", "CHF", "AED", "INR",
]
SUPPORTED_INSTRUMENT_TYPES = [
	"cash", "government_bond", "corporate_bond", "treasury_bill",
	"commercial_paper", "fx_spot", "fx_forward", "fx_swap",
	"interest_rate_swap", "cds", "money_market_deposit", "repo",
	"reverse_repo", "equity", "mutual_fund",
]
SUPPORTED_ACCOUNT_TYPES = [
	"nostro", "vostro", "current", "settlement", "suspense",
	"clearing", "escrow", "reserve", "float",
]
SUPPORTED_DEAL_TYPES = [
	"fx_deal", "mm_deal", "bond_deal", "repo_deal", "swap_deal",
	"futures_deal", "options_deal",
]
SUPPORTED_LIMIT_TYPES = [
	"counterparty_credit_limit", "currency_exposure_limit",
	"maturity_limit", "value_at_risk_limit", "open_position_limit",
	"daylight_overdraft_limit", "overnight_limit", "settlement_limit",
]
SUPPORTED_PAYMENT_SYSTEMS = [
	"rtgs", "eft", "swift", "pesalink", "target2",
	"chips", "chaps", "bacs", "sepa",
]
SUPPORTED_APPROVAL_LEVELS = ["analyst", "dealer", "head_of_treasury", "cfo", "board"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = [
	"treasury_ops_reviewer", "dealing_reviewer", "limit_reviewer",
	"settlement_reviewer", "risk_reviewer", "compliance_reviewer",
	"liquidity_reviewer", "fx_reviewer",
]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"cash_management": {
		"supported_account_types": SUPPORTED_ACCOUNT_TYPES,
		"supported_currencies": SUPPORTED_CURRENCIES,
		"double_entry_required": True,
		"value_dating_required": True,
		"intraday_liquidity_tracking": True,
		"eod_position_required": True,
		"nostro_reconciliation_required": True,
	},
	"dealing": {
		"supported_deal_types": SUPPORTED_DEAL_TYPES,
		"supported_instrument_types": SUPPORTED_INSTRUMENT_TYPES,
		"counterparty_required": True,
		"dealer_required": True,
		"limit_check_required": True,
		"pre_deal_approval_threshold_usd": 1000000,
		"board_approval_threshold_usd": 50000000,
		"confirmation_required": True,
	},
	"limits": {
		"supported_limit_types": SUPPORTED_LIMIT_TYPES,
		"breach_review_required": True,
		"hard_limit_deny": True,
		"soft_limit_warn": True,
		"limit_version_tracking": True,
	},
	"settlement": {
		"supported_payment_systems": SUPPORTED_PAYMENT_SYSTEMS,
		"settlement_instruction_required": True,
		"nostro_account_required": True,
		"swift_message_validation_required": True,
		"high_value_threshold_usd": 5000000,
		"dual_authorization_required": True,
		"cutoff_time_enforcement": True,
	},
	"fx": {
		"spot_rate_source_required": True,
		"rate_tolerance_bps": 50,
		"forward_curve_required": True,
		"hedge_accounting_supported": True,
		"ndf_supported": True,
		"kes_usd_rate_required": True,
		"mpesa_fx_rate_feed_supported": True,
	},
	"liquidity": {
		"intraday_forecast_required": True,
		"lcr_calculation_supported": True,
		"nsfr_calculation_supported": True,
		"stress_testing_supported": True,
		"cbk_reserve_requirement_tracking": True,
		"statutory_liquidity_ratio_tracking": True,
	},
	"reporting": {
		"cbk_returns_supported": True,
		"basel3_reporting_supported": True,
		"ifrs9_classification_supported": True,
		"board_reporting_supported": True,
		"daily_position_report_required": True,
	},
	"compliance": {
		"cbk_prudential_guidelines_required": True,
		"kyc_required_for_counterparties": True,
		"aml_monitoring_required": True,
		"sanctions_screening_required": True,
		"dealing_mandate_required": True,
		"segregation_of_duties_required": True,
	},
	"agents": {
		"enabled": True,
		"supported_runtimes": SUPPORTED_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_AGENT_ROLES,
		"human_approval_required_for_privileged_actions": True,
	},
	"governance": {
		"require_tenant_context": True,
		"policy_attached_for_writes": True,
		"audit_treasury_events": True,
		"segregation_of_duties": True,
		"four_eyes_principle_required": True,
		"dealing_mandate_enforcement": True,
	},
	"observability": {
		"event_stream": TREASURY_EVENT_STREAM,
		"stream_processor": "bytewax",
		"emit_cash_events": True,
		"emit_deal_events": True,
		"emit_limit_events": True,
		"emit_settlement_events": True,
		"emit_fx_events": True,
		"emit_liquidity_events": True,
		"emit_compliance_events": True,
		"emit_agent_events": True,
	},
	"adapters": {
		"auth": "auth",
		"audit": "audl",
		"notifications": "ntfy",
		"keys": "keym",
		"payments": "fintech_payments",
		"wallets": "fintech_wallets",
		"kyc": "fintech_kyc",
		"aml": "fintech_aml",
		"risk": "fintech_risk",
		"event_stream": "bytewax",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_cash_management": True,
		"enable_dealing": True,
		"enable_limits": True,
		"enable_settlement": True,
		"enable_fx": True,
		"enable_liquidity": True,
		"enable_reporting": True,
		"enable_agents": True,
	},
	"theme": {
		"default_theme": "fintech_treasury_control",
		"allow_tenant_overrides": True,
	},
}

PROVIDES = [
	"cash_position_management",
	"treasury_dealing_workflow",
	"counterparty_limit_governance",
	"settlement_instruction_workflow",
	"fx_rate_management",
	"liquidity_forecasting",
	"nostro_reconciliation",
	"cbk_regulatory_reporting",
	"treasury_risk_monitoring",
	"treasury_agent_workflow",
]

REQUIRES = [
	"auth",
	"audl",
	"ntfy",
	"keym",
	"fintech_payments",
	"fintech_kyc",
	"fintech_aml",
	"fintech_risk",
]

UI_ROUTES = [
	{"name": "dashboard", "path": "/fintech-treasury/dashboard", "component": "TreasuryDashboard", "permission": "fintech_treasury:view", "nav_group": "Overview"},
	{"name": "cash_management", "path": "/fintech-treasury/cash", "component": "CashPositionWorkbench", "permission": "fintech_treasury:manage_cash", "nav_group": "Cash"},
	{"name": "dealing", "path": "/fintech-treasury/dealing", "component": "DealingConsole", "permission": "fintech_treasury:deal", "nav_group": "Dealing"},
	{"name": "limits", "path": "/fintech-treasury/limits", "component": "LimitManagementConsole", "permission": "fintech_treasury:manage_limits", "nav_group": "Risk"},
	{"name": "settlement", "path": "/fintech-treasury/settlement", "component": "SettlementWorkbench", "permission": "fintech_treasury:settle", "nav_group": "Settlement"},
	{"name": "fx", "path": "/fintech-treasury/fx", "component": "FxRateConsole", "permission": "fintech_treasury:manage_fx", "nav_group": "FX"},
	{"name": "liquidity", "path": "/fintech-treasury/liquidity", "component": "LiquidityForecastConsole", "permission": "fintech_treasury:manage_liquidity", "nav_group": "Liquidity"},
	{"name": "nostro", "path": "/fintech-treasury/nostro", "component": "NostroReconciliationWorkbench", "permission": "fintech_treasury:reconcile", "nav_group": "Reconciliation"},
	{"name": "reporting", "path": "/fintech-treasury/reporting", "component": "TreasuryReportingConsole", "permission": "fintech_treasury:report", "nav_group": "Reporting"},
	{"name": "compliance", "path": "/fintech-treasury/compliance", "component": "TreasuryComplianceConsole", "permission": "fintech_treasury:compliance", "nav_group": "Compliance"},
	{"name": "agents", "path": "/fintech-treasury/agents", "component": "TreasuryAgentWorkbench", "permission": "fintech_treasury:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/fintech-treasury/settings", "component": "TreasurySettings", "permission": "fintech_treasury:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "fintech_treasury_control",
	"tokens": {
		"color.primary": "#1E3A5F",
		"color.accent": "#0369A1",
		"color.success": "#15803D",
		"color.warning": "#B45309",
		"color.danger": "#B91C1C",
		"surface.canvas": "#F0F6FF",
		"surface.panel": "#FFFFFF",
		"text.primary": "#0F172A",
		"text.secondary": "#475569",
		"border.radius": "8px",
		"density": "compact",
	},
	"components": {
		"cash_management": {"icon": "banknote", "status_indicator": "position-pill"},
		"dealing": {"icon": "trending-up", "visual": "deal-blotter", "status_style": "deal-chip"},
		"limits": {"icon": "gauge", "status_indicator": "limit-breach-chip"},
		"settlement": {"visual": "settlement-grid", "status_style": "settlement-chip"},
		"fx": {"icon": "refresh-cw", "status_indicator": "rate-chip"},
		"liquidity": {"visual": "liquidity-waterfall", "status_style": "lcr-chip"},
		"nostro": {"visual": "reconciliation-grid", "status_style": "break-chip"},
		"reporting": {"icon": "file-text", "status_indicator": "return-chip"},
		"compliance": {"icon": "shield-check", "status_indicator": "mandate-chip"},
		"agents": {"visual": "review-lane", "status_style": "agent-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": TREASURY_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"treasury_cash_position_updated",
		"treasury_deal_booked",
		"treasury_deal_confirmed",
		"treasury_deal_settled",
		"treasury_deal_cancelled",
		"treasury_limit_checked",
		"treasury_limit_breached",
		"treasury_limit_updated",
		"treasury_settlement_instruction_sent",
		"treasury_settlement_confirmed",
		"treasury_nostro_reconciled",
		"treasury_fx_rate_updated",
		"treasury_liquidity_forecast_updated",
		"treasury_cbk_return_filed",
		"treasury_compliance_event_recorded",
		"treasury_agent_registered",
	],
	"guardrails": [
		"treasury_batch_requires_bytewax",
		"treasury_event_requires_bytewax",
		"treasury_deal_requires_four_eyes",
		"privileged_treasury_agent_action_requires_human_approval",
	],
}

RULES: list[dict[str, Any]] = [
	# Governance
	{"name": "tenant_context_required", "description": "Treasury operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "treasury_write_requires_policy", "description": "Treasury writes require policy evidence.", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "treasury_policy_required", "required_action": "attach_treasury_policy"}},
	{"name": "cross_tenant_access_denied", "description": "Treasury resources cannot be accessed across tenant boundaries.", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_access_denied", "required_action": "use_tenant_scoped_credentials"}},
	{"name": "privilege_escalation_denied", "description": "Treasury privilege escalation without approval is denied.", "condition": {"privilege_escalation_attempt": True, "approval_recorded": False}, "effect": {"decision": "deny", "reason": "privilege_escalation_denied", "required_action": "obtain_escalation_approval"}},
	{"name": "segregation_of_duties_required", "description": "Treasury deals cannot be booked and settled by the same user.", "condition": {"operation": "settle_deal", "same_user_booked_and_settling": True}, "effect": {"decision": "deny", "reason": "segregation_of_duties_violated", "required_action": "assign_separate_settling_officer"}},

	# Cash management
	{"name": "cash_account_type_supported", "description": "Treasury account type must be supported.", "condition": {"operation": "open_account", "account_type_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_account_type", "required_action": "use_supported_account_type"}},
	{"name": "cash_currency_supported", "description": "Treasury account currency must be supported.", "condition": {"operation": "open_account", "currency_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_currency", "required_action": "use_supported_currency"}},
	{"name": "cash_posting_requires_double_entry", "description": "Cash postings require double-entry ledger pairs.", "condition": {"operation": "post_cash", "double_entry_present": False}, "effect": {"decision": "deny", "reason": "double_entry_required", "required_action": "create_double_entry"}},
	{"name": "value_date_required", "description": "Treasury transactions require a value date.", "condition": {"operation": "post_cash", "value_date_present": False}, "effect": {"decision": "deny", "reason": "value_date_required", "required_action": "set_value_date"}},
	{"name": "nostro_reconciliation_required", "description": "Nostro accounts require periodic reconciliation.", "condition": {"operation": "close_period", "nostro_reconciled": False}, "effect": {"decision": "deny", "reason": "nostro_reconciliation_required", "required_action": "reconcile_nostro_accounts"}},

	# Dealing
	{"name": "deal_type_supported", "description": "Deal type must be supported.", "condition": {"operation": "book_deal", "deal_type_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_deal_type", "required_action": "use_supported_deal_type"}},
	{"name": "deal_counterparty_required", "description": "Deals require a counterparty reference.", "condition": {"operation": "book_deal", "counterparty_present": False}, "effect": {"decision": "deny", "reason": "counterparty_required", "required_action": "attach_counterparty"}},
	{"name": "deal_counterparty_kyc_required", "description": "Deal counterparties require KYC evidence.", "condition": {"operation": "book_deal", "counterparty_kyc_present": False}, "effect": {"decision": "deny", "reason": "counterparty_kyc_required", "required_action": "attach_counterparty_kyc"}},
	{"name": "deal_dealer_required", "description": "Deals require an assigned dealer.", "condition": {"operation": "book_deal", "dealer_present": False}, "effect": {"decision": "deny", "reason": "dealer_required", "required_action": "assign_dealer"}},
	{"name": "deal_confirmation_required", "description": "Booked deals require confirmation before settlement.", "condition": {"operation": "settle_deal", "deal_confirmed": False}, "effect": {"decision": "deny", "reason": "deal_confirmation_required", "required_action": "confirm_deal"}},
	{"name": "pre_deal_approval_threshold", "description": "Large deals require pre-deal approval.", "condition": {"operation": "book_deal", "exceeds_pre_deal_threshold": True, "approval_recorded": False}, "effect": {"decision": "require_review", "reason": "pre_deal_approval_required", "required_action": "record_pre_deal_approval"}},
	{"name": "board_approval_required_for_very_large_deals", "description": "Very large deals require board-level approval.", "condition": {"operation": "book_deal", "exceeds_board_threshold": True, "board_approval_recorded": False}, "effect": {"decision": "deny", "reason": "board_approval_required", "required_action": "obtain_board_approval"}},
	{"name": "four_eyes_required_for_deals", "description": "Deal booking requires four-eyes approval.", "condition": {"operation": "book_deal", "four_eyes_recorded": False}, "effect": {"decision": "deny", "reason": "four_eyes_principle_required", "required_action": "record_four_eyes_approval"}},
	{"name": "dealing_mandate_required", "description": "Dealers must operate within their mandate.", "condition": {"operation": "book_deal", "within_dealing_mandate": False}, "effect": {"decision": "deny", "reason": "dealing_mandate_exceeded", "required_action": "obtain_mandate_exception_approval"}},

	# Limits
	{"name": "limit_type_supported", "description": "Limit type must be supported.", "condition": {"operation": "set_limit", "limit_type_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_limit_type", "required_action": "use_supported_limit_type"}},
	{"name": "hard_limit_breach_denied", "description": "Hard limit breaches deny deal booking.", "condition": {"operation": "book_deal", "hard_limit_breached": True}, "effect": {"decision": "deny", "reason": "hard_limit_breached", "required_action": "reduce_deal_size_or_obtain_waiver"}},
	{"name": "soft_limit_breach_requires_review", "description": "Soft limit breaches require reviewer sign-off.", "condition": {"operation": "book_deal", "soft_limit_breached": True, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "soft_limit_breach_review_required", "required_action": "record_limit_review"}},
	{"name": "limit_update_requires_approval", "description": "Limit updates require head-of-treasury approval.", "condition": {"operation": "update_limit", "approval_recorded": False}, "effect": {"decision": "deny", "reason": "limit_update_approval_required", "required_action": "record_limit_approval"}},

	# Settlement
	{"name": "settlement_instruction_required", "description": "Deal settlement requires settlement instructions.", "condition": {"operation": "settle_deal", "settlement_instruction_present": False}, "effect": {"decision": "deny", "reason": "settlement_instruction_required", "required_action": "create_settlement_instruction"}},
	{"name": "settlement_nostro_required", "description": "Settlement instructions require a nostro account.", "condition": {"operation": "settle_deal", "nostro_account_present": False}, "effect": {"decision": "deny", "reason": "nostro_account_required", "required_action": "assign_nostro_account"}},
	{"name": "high_value_settlement_requires_dual_auth", "description": "High-value settlements require dual authorization.", "condition": {"operation": "settle_deal", "high_value": True, "dual_authorization_recorded": False}, "effect": {"decision": "deny", "reason": "dual_authorization_required", "required_action": "record_dual_authorization"}},
	{"name": "swift_message_validation_required", "description": "SWIFT settlement messages must be validated.", "condition": {"operation": "send_swift", "swift_validated": False}, "effect": {"decision": "deny", "reason": "swift_validation_required", "required_action": "validate_swift_message"}},
	{"name": "settlement_cutoff_enforced", "description": "Settlement instructions past the cutoff time are denied.", "condition": {"operation": "send_settlement", "past_cutoff": True}, "effect": {"decision": "deny", "reason": "settlement_cutoff_passed", "required_action": "process_next_business_day"}},

	# FX
	{"name": "fx_rate_source_required", "description": "FX deals require a rate source reference.", "condition": {"operation": "book_fx_deal", "rate_source_present": False}, "effect": {"decision": "deny", "reason": "fx_rate_source_required", "required_action": "attach_rate_source"}},
	{"name": "fx_rate_tolerance_enforced", "description": "FX deal rate must be within configured tolerance.", "condition": {"operation": "book_fx_deal", "rate_outside_tolerance": True, "approval_recorded": False}, "effect": {"decision": "require_review", "reason": "fx_rate_tolerance_exceeded", "required_action": "record_rate_exception_approval"}},
	{"name": "fx_forward_curve_required", "description": "FX forward deals require a yield curve reference.", "condition": {"operation": "book_fx_forward", "forward_curve_present": False}, "effect": {"decision": "deny", "reason": "forward_curve_required", "required_action": "attach_forward_curve"}},

	# Liquidity & CBK
	{"name": "cbk_reserve_requirement_check", "description": "CBK minimum reserve requirements must be maintained.", "condition": {"operation": "post_cash", "cbk_reserve_breached": True}, "effect": {"decision": "require_review", "reason": "cbk_reserve_requirement_breached", "required_action": "restore_cbk_reserve"}},
	{"name": "statutory_liquidity_ratio_check", "description": "Statutory liquidity ratio must be maintained.", "condition": {"operation": "post_cash", "slr_breached": True}, "effect": {"decision": "require_review", "reason": "slr_breached", "required_action": "restore_slr"}},
	{"name": "eod_position_required", "description": "End-of-day cash position must be recorded.", "condition": {"operation": "close_day", "eod_position_recorded": False}, "effect": {"decision": "deny", "reason": "eod_position_required", "required_action": "record_eod_position"}},

	# AML / compliance
	{"name": "aml_screening_required", "description": "Treasury counterparties require AML screening.", "condition": {"operation": "book_deal", "aml_screened": False}, "effect": {"decision": "deny", "reason": "aml_screening_required", "required_action": "screen_counterparty"}},
	{"name": "sanctions_screening_required", "description": "Treasury counterparties require sanctions screening.", "condition": {"operation": "book_deal", "sanctions_screened": False}, "effect": {"decision": "deny", "reason": "sanctions_screening_required", "required_action": "screen_counterparty_sanctions"}},
	{"name": "sanctions_hit_blocks_deal", "description": "Sanctions hits block deal booking.", "condition": {"operation": "book_deal", "sanctions_hit": True}, "effect": {"decision": "deny", "reason": "sanctions_hit", "required_action": "escalate_sanctions_hit"}},

	# Streaming
	{"name": "treasury_batch_requires_bytewax", "description": "Treasury batches require Bytewax.", "condition": {"operation": "treasury_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_required", "required_action": "route_to_bytewax"}},
	{"name": "treasury_event_requires_bytewax", "description": "Treasury events require Bytewax.", "condition": {"operation": "treasury_event", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_required", "required_action": "route_to_bytewax"}},

	# Agents
	{"name": "treasury_agent_runtime_supported", "description": "Treasury agents must use a supported runtime.", "condition": {"operation": "register_treasury_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "treasury_agent_role_supported", "description": "Treasury agents must use a supported role.", "condition": {"operation": "register_treasury_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "privileged_treasury_agent_action_requires_human_approval", "description": "Privileged treasury-agent actions require human approval.", "condition": {"operation": "treasury_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
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
	"""Return the executable APG Treasury Management System capability contract."""
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
		"name": CAPABILITY_NAME,
		"version": CAPABILITY_VERSION,
		"configuration": configuration,
		"configuration_schema": _configuration_schema(),
		"provides": PROVIDES,
		"requires": REQUIRES,
		"rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"api_prefix": "/fintech-treasury/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"requires_theme": True,
			"view_module": "views.py",
			"template_roots": ["templates/", "static/"],
		},
		"theme": deepcopy(THEME),
		"streaming": deepcopy(STREAMING),
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate deterministic treasury management guardrails."""
	contract = get_capability_contract(str(context.get("tenant_id") or "default"))
	matched = [
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
		"actions": [rule["effect"] for rule in matched],
		"effects": [rule["effect"] for rule in matched],
	}
