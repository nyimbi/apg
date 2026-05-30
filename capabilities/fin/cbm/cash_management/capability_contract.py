"""Executable capability contract for Cash Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "cbm_cash_management"
CAPABILITY_NAME = "Cash Management"
CAPABILITY_VERSION = "2.1.0"
CBM_EVENT_STREAM = "apg.fin.cbm.lifecycle"

SUPPORTED_CURRENCIES = ["USD", "EUR", "GBP", "KES", "ZAR", "NGN", "GHS", "UGX", "TZS"]
SUPPORTED_ACCOUNT_TYPES = ["operating", "savings", "money_market", "investment", "petty_cash", "lockbox"]
SUPPORTED_FLOW_TYPES = ["inflow", "outflow", "transfer"]
SUPPORTED_FORECAST_SCENARIOS = ["base", "optimistic", "pessimistic", "stress"]
SUPPORTED_INVESTMENT_TYPES = ["deposit", "treasury_bill", "money_market", "commercial_paper", "overnight_sweep"]
SUPPORTED_CBM_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_CBM_AGENT_ROLES = [
	"cash_position_reviewer",
	"forecast_reviewer",
	"liquidity_reviewer",
	"bank_reconciliation_reviewer",
	"investment_reviewer",
	"payment_run_reviewer",
]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"banks": {
		"bank_code_required": True,
		"bank_name_required": True,
		"connectivity_status_required": True,
	},
	"cash_accounts": {
		"bank_required": True,
		"account_number_required": True,
		"account_name_required": True,
		"account_type_required": True,
		"supported_account_types": SUPPORTED_ACCOUNT_TYPES,
		"currency_required": True,
		"supported_currencies": SUPPORTED_CURRENCIES,
	},
	"cash_positions": {
		"account_required": True,
		"as_of_date_required": True,
		"available_balance_required": True,
		"minimum_liquidity_review_required": True,
	},
	"cash_flows": {
		"account_required": True,
		"flow_type_required": True,
		"supported_flow_types": SUPPORTED_FLOW_TYPES,
		"amount_must_be_positive": True,
		"category_required": True,
		"expected_date_required": True,
	},
	"forecasts": {
		"horizon_required": True,
		"scenario_required": True,
		"supported_scenarios": SUPPORTED_FORECAST_SCENARIOS,
		"confidence_review_threshold": 0.75,
		"source_flow_required": True,
	},
	"liquidity": {
		"minimum_buffer_required": True,
		"stress_review_required": True,
		"deficit_requires_review": True,
	},
	"reconciliation": {
		"bank_statement_required": True,
		"ledger_balance_required": True,
		"variance_review_threshold": 100,
		"review_required_for_variance": True,
	},
	"investments": {
		"supported_types": SUPPORTED_INVESTMENT_TYPES,
		"counterparty_required": True,
		"maturity_date_required": True,
		"yield_review_required": True,
		"approval_required": True,
	},
	"payment_runs": {
		"funding_account_required": True,
		"cash_position_required": True,
		"approval_required": True,
		"deficit_blocking": True,
	},
	"cbm_agents": {
		"enabled": True,
		"supported_runtimes": SUPPORTED_CBM_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_CBM_AGENT_ROLES,
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
		"event_stream": CBM_EVENT_STREAM,
		"stream_processor": "bytewax",
		"emit_bank_events": True,
		"emit_account_events": True,
		"emit_position_events": True,
		"emit_flow_events": True,
		"emit_forecast_events": True,
		"emit_reconciliation_events": True,
		"emit_investment_events": True,
		"emit_agent_events": True,
	},
	"adapters": {
		"authorization": "adapter",
		"audit": "adapter",
		"notification": "adapter",
		"document_management": "adapter",
		"business_intelligence": "adapter",
		"general_ledger": "adapter",
		"accounts_payable": "adapter",
		"accounts_receivable": "adapter",
		"event_stream": "bytewax",
		"theme": "adapter",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_banks": True,
		"enable_accounts": True,
		"enable_positions": True,
		"enable_flows": True,
		"enable_forecasts": True,
		"enable_liquidity": True,
		"enable_reconciliation": True,
		"enable_investments": True,
		"enable_payment_runs": True,
		"enable_agents": True,
		"enable_settings": True,
	},
	"theme": {
		"default_theme": "cbm_cash_management_control",
		"allow_tenant_overrides": True,
	},
}


PROVIDES = [
	"bank_relationship_lifecycle",
	"cash_account_lifecycle",
	"cash_position_service",
	"cash_flow_lifecycle",
	"cash_forecasting_workflow",
	"liquidity_control_workflow",
	"bank_reconciliation_workflow",
	"treasury_investment_workflow",
	"payment_run_funding_control",
	"cbm_agents",
]

REQUIRES = [
	"auth",
	"audl",
	"ntfy",
	"composition_events",
	"composition_config",
	"general_ledger",
	"accounts_payable",
	"accounts_receivable",
	"document_management",
	"business_intelligence",
]


UI_ROUTES = [
	{"name": "dashboard", "path": "/cbm-cash-management/dashboard", "component": "CashManagementDashboard", "permission": "cbm_cash_management:view", "nav_group": "Overview"},
	{"name": "banks", "path": "/cbm-cash-management/banks", "component": "BankRelationshipWorkbench", "permission": "cbm_cash_management:manage_banks", "nav_group": "Banks"},
	{"name": "accounts", "path": "/cbm-cash-management/accounts", "component": "CashAccountWorkbench", "permission": "cbm_cash_management:manage_accounts", "nav_group": "Banks"},
	{"name": "positions", "path": "/cbm-cash-management/positions", "component": "CashPositionConsole", "permission": "cbm_cash_management:view_positions", "nav_group": "Cash"},
	{"name": "flows", "path": "/cbm-cash-management/flows", "component": "CashFlowConsole", "permission": "cbm_cash_management:manage_flows", "nav_group": "Cash"},
	{"name": "forecasts", "path": "/cbm-cash-management/forecasts", "component": "CashForecastWorkbench", "permission": "cbm_cash_management:forecast", "nav_group": "Forecasting"},
	{"name": "liquidity", "path": "/cbm-cash-management/liquidity", "component": "LiquidityControlDesk", "permission": "cbm_cash_management:liquidity", "nav_group": "Controls"},
	{"name": "reconciliation", "path": "/cbm-cash-management/reconciliation", "component": "BankReconciliationWorkbench", "permission": "cbm_cash_management:reconcile", "nav_group": "Controls"},
	{"name": "investments", "path": "/cbm-cash-management/investments", "component": "TreasuryInvestmentWorkbench", "permission": "cbm_cash_management:invest", "nav_group": "Treasury"},
	{"name": "payment_runs", "path": "/cbm-cash-management/payment-runs", "component": "PaymentFundingWorkbench", "permission": "cbm_cash_management:fund_payments", "nav_group": "Treasury"},
	{"name": "agents", "path": "/cbm-cash-management/agents", "component": "CBMAgentWorkbench", "permission": "cbm_cash_management:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/cbm-cash-management/settings", "component": "CashManagementSettings", "permission": "cbm_cash_management:admin", "nav_group": "Administration"},
]


THEME = {
	"name": "cbm_cash_management_control",
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
		"banks": {"icon": "landmark", "status_indicator": "connectivity-pill", "risk_style": "bank-band"},
		"accounts": {"visual": "account-list", "status_style": "account-chip"},
		"positions": {"visual": "position-grid", "status_style": "liquidity-chip"},
		"flows": {"visual": "flow-timeline", "status_style": "flow-chip"},
		"forecasts": {"visual": "forecast-fan", "status_style": "confidence-chip"},
		"reconciliation": {"visual": "match-grid", "status_style": "variance-chip"},
		"investments": {"visual": "maturity-ladder", "status_style": "yield-chip"},
		"agents": {"visual": "review-lane", "status_style": "agent-chip"},
	},
}


STREAMING = {
	"processor": "bytewax",
	"stream": CBM_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"bank_created",
		"cash_account_created",
		"cash_position_recorded",
		"cash_flow_recorded",
		"cash_forecast_created",
		"liquidity_review_recorded",
		"bank_reconciliation_recorded",
		"treasury_investment_created",
		"payment_run_validated",
		"cbm_agent_registered",
	],
	"states": ["draft", "active", "recorded", "forecasted", "reviewed", "approved", "funded", "blocked"],
	"guardrails": [
		"cbm_batch_requires_bytewax",
		"cbm_event_requires_bytewax",
		"privileged_agent_cbm_action_requires_human_approval",
	],
}


RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "Cash management operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "cbm_write_requires_policy", "description": "Cash management writes require policy attachment.", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "operation_policy_required", "required_action": "attach_operation_policy"}},
	{"name": "bank_requires_code", "description": "Banks require a code.", "condition": {"operation": "create_bank", "bank_code_present": False}, "effect": {"decision": "deny", "reason": "bank_code_required", "required_action": "set_bank_code"}},
	{"name": "bank_requires_name", "description": "Banks require a name.", "condition": {"operation": "create_bank", "bank_name_present": False}, "effect": {"decision": "deny", "reason": "bank_name_required", "required_action": "set_bank_name"}},
	{"name": "cash_account_requires_bank", "description": "Cash accounts require a bank.", "condition": {"operation": "create_cash_account", "bank_present": False}, "effect": {"decision": "deny", "reason": "bank_required", "required_action": "attach_bank"}},
	{"name": "cash_account_requires_number", "description": "Cash accounts require an account number.", "condition": {"operation": "create_cash_account", "account_number_present": False}, "effect": {"decision": "deny", "reason": "account_number_required", "required_action": "set_account_number"}},
	{"name": "cash_account_requires_name", "description": "Cash accounts require a name.", "condition": {"operation": "create_cash_account", "account_name_present": False}, "effect": {"decision": "deny", "reason": "account_name_required", "required_action": "set_account_name"}},
	{"name": "cash_account_type_supported", "description": "Cash account type must be supported.", "condition": {"operation": "create_cash_account", "account_type_supported": False}, "effect": {"decision": "deny", "reason": "account_type_not_supported", "required_action": "select_supported_account_type"}},
	{"name": "cash_account_currency_supported", "description": "Cash account currency must be supported.", "condition": {"operation": "create_cash_account", "currency_supported": False}, "effect": {"decision": "deny", "reason": "currency_not_supported", "required_action": "select_supported_currency"}},
	{"name": "position_requires_account", "description": "Cash positions require an account.", "condition": {"operation": "record_cash_position", "account_present": False}, "effect": {"decision": "deny", "reason": "cash_account_required", "required_action": "attach_cash_account"}},
	{"name": "position_requires_date", "description": "Cash positions require an as-of date.", "condition": {"operation": "record_cash_position", "as_of_date_present": False}, "effect": {"decision": "deny", "reason": "as_of_date_required", "required_action": "set_as_of_date"}},
	{"name": "position_requires_available_balance", "description": "Cash positions require available balance.", "condition": {"operation": "record_cash_position", "available_balance_present": False}, "effect": {"decision": "deny", "reason": "available_balance_required", "required_action": "record_available_balance"}},
	{"name": "position_below_buffer_requires_review", "description": "Positions below liquidity buffer require review.", "condition": {"operation": "record_cash_position", "below_minimum_buffer": True, "liquidity_review_recorded": False}, "effect": {"decision": "require_review", "reason": "liquidity_review_required", "required_action": "record_liquidity_review"}},
	{"name": "cash_flow_requires_account", "description": "Cash flows require an account.", "condition": {"operation": "record_cash_flow", "account_present": False}, "effect": {"decision": "deny", "reason": "cash_account_required", "required_action": "attach_cash_account"}},
	{"name": "cash_flow_type_supported", "description": "Cash flow type must be supported.", "condition": {"operation": "record_cash_flow", "flow_type_supported": False}, "effect": {"decision": "deny", "reason": "flow_type_not_supported", "required_action": "select_supported_flow_type"}},
	{"name": "cash_flow_amount_positive", "description": "Cash flow amount must be positive.", "condition": {"operation": "record_cash_flow", "amount_lte": 0}, "effect": {"decision": "deny", "reason": "cash_flow_amount_positive_required", "required_action": "set_positive_amount"}},
	{"name": "cash_flow_requires_category", "description": "Cash flows require category.", "condition": {"operation": "record_cash_flow", "category_present": False}, "effect": {"decision": "deny", "reason": "cash_flow_category_required", "required_action": "set_category"}},
	{"name": "cash_flow_requires_expected_date", "description": "Cash flows require expected date.", "condition": {"operation": "record_cash_flow", "expected_date_present": False}, "effect": {"decision": "deny", "reason": "cash_flow_expected_date_required", "required_action": "set_expected_date"}},
	{"name": "forecast_requires_horizon", "description": "Cash forecasts require a horizon.", "condition": {"operation": "create_cash_forecast", "horizon_days_lte": 0}, "effect": {"decision": "deny", "reason": "forecast_horizon_required", "required_action": "set_forecast_horizon"}},
	{"name": "forecast_scenario_supported", "description": "Cash forecast scenario must be supported.", "condition": {"operation": "create_cash_forecast", "scenario_supported": False}, "effect": {"decision": "deny", "reason": "forecast_scenario_not_supported", "required_action": "select_supported_scenario"}},
	{"name": "forecast_confidence_requires_review", "description": "Low forecast confidence requires review.", "condition": {"operation": "create_cash_forecast", "confidence_score_lt": 0.75, "forecast_review_recorded": False}, "effect": {"decision": "require_review", "reason": "forecast_review_required", "required_action": "record_forecast_review"}},
	{"name": "reconciliation_requires_statement", "description": "Bank reconciliation requires statement balance.", "condition": {"operation": "record_bank_reconciliation", "bank_statement_present": False}, "effect": {"decision": "deny", "reason": "bank_statement_required", "required_action": "attach_bank_statement"}},
	{"name": "reconciliation_requires_ledger_balance", "description": "Bank reconciliation requires ledger balance.", "condition": {"operation": "record_bank_reconciliation", "ledger_balance_present": False}, "effect": {"decision": "deny", "reason": "ledger_balance_required", "required_action": "attach_ledger_balance"}},
	{"name": "reconciliation_variance_requires_review", "description": "Material reconciliation variance requires review.", "condition": {"operation": "record_bank_reconciliation", "variance_abs_gt": 100, "reconciliation_review_recorded": False}, "effect": {"decision": "require_review", "reason": "reconciliation_review_required", "required_action": "record_reconciliation_review"}},
	{"name": "investment_type_supported", "description": "Treasury investment type must be supported.", "condition": {"operation": "create_treasury_investment", "investment_type_supported": False}, "effect": {"decision": "deny", "reason": "investment_type_not_supported", "required_action": "select_supported_investment_type"}},
	{"name": "investment_requires_counterparty", "description": "Treasury investments require counterparty.", "condition": {"operation": "create_treasury_investment", "counterparty_present": False}, "effect": {"decision": "deny", "reason": "investment_counterparty_required", "required_action": "set_counterparty"}},
	{"name": "investment_requires_maturity", "description": "Treasury investments require maturity date.", "condition": {"operation": "create_treasury_investment", "maturity_date_present": False}, "effect": {"decision": "deny", "reason": "investment_maturity_required", "required_action": "set_maturity_date"}},
	{"name": "investment_requires_approval", "description": "Treasury investments require approval.", "condition": {"operation": "create_treasury_investment", "approval_recorded": False}, "effect": {"decision": "deny", "reason": "investment_approval_required", "required_action": "approve_investment"}},
	{"name": "payment_run_requires_funding_account", "description": "Payment runs require funding account.", "condition": {"operation": "validate_payment_run", "funding_account_present": False}, "effect": {"decision": "deny", "reason": "funding_account_required", "required_action": "select_funding_account"}},
	{"name": "payment_run_requires_position", "description": "Payment runs require current cash position.", "condition": {"operation": "validate_payment_run", "cash_position_present": False}, "effect": {"decision": "deny", "reason": "cash_position_required", "required_action": "record_cash_position"}},
	{"name": "payment_run_blocks_deficit", "description": "Payment runs cannot create unapproved cash deficits.", "condition": {"operation": "validate_payment_run", "projected_deficit": True, "approval_recorded": False}, "effect": {"decision": "deny", "reason": "cash_deficit_approval_required", "required_action": "approve_deficit"}},
	{"name": "cbm_batch_requires_bytewax", "description": "Cash management batches require Bytewax coordination.", "condition": {"operation": "cbm_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_cbm_batch_to_bytewax"}},
	{"name": "cbm_event_requires_bytewax", "description": "Cash management lifecycle events require Bytewax.", "condition": {"operation": "cbm_event", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_cbm_event_to_bytewax"}},
	{"name": "cbm_agent_runtime_supported", "description": "CBM agents must use an approved runtime.", "condition": {"operation": "register_cbm_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "cbm_agent_runtime_not_supported", "required_action": "select_supported_agent_runtime"}},
	{"name": "cbm_agent_role_supported", "description": "CBM agents must use an approved role.", "condition": {"operation": "register_cbm_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "cbm_agent_role_not_supported", "required_action": "select_supported_agent_role"}},
	{"name": "privileged_agent_cbm_action_requires_human_approval", "description": "Privileged CBM actions proposed by agents require human approval.", "condition": {"operation": "agent_cbm_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
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
		if key.endswith("_abs_gt"):
			value = context.get(key[:-7])
			if value is None or abs(value) <= expected:
				return False
			continue
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
		"name": CAPABILITY_NAME,
		"version": CAPABILITY_VERSION,
		"configuration": configuration,
		"configuration_schema": _configuration_schema(),
		"provides": PROVIDES,
		"requires": REQUIRES,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"api_prefix": "/cbm-cash-management/api/v1",
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
		"effects": [rule["effect"] for rule in matched],
	}
