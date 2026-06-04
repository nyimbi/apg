"""Executable capability contract for APG Algorithmic Trading."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "fintech_trading"
CAPABILITY_NAME = "Algorithmic Trading"
CAPABILITY_VERSION = "1.1.0"
TRADING_EVENT_STREAM = "apg.fintech.trading.lifecycle"

SUPPORTED_STRATEGY_TYPES = ["mean_reversion", "momentum", "market_making", "pairs", "arbitrage", "hedging", "execution_algo"]
SUPPORTED_ASSET_CLASSES = ["equity", "fixed_income", "fx", "fund", "commodity", "crypto"]
SUPPORTED_ORDER_TYPES = ["market", "limit", "stop", "twap", "vwap", "iceberg"]
SUPPORTED_VENUES = ["exchange", "ats", "otc", "dark_pool", "internal_cross"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_ALERT_SEVERITIES = ["low", "medium", "high", "critical"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = [
	"strategy_reviewer",
	"signal_quality_reviewer",
	"backtest_reviewer",
	"risk_limit_reviewer",
	"order_intent_reviewer",
	"surveillance_reviewer",
	"trading_compliance_reviewer",
]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"strategies": {"supported_types": SUPPORTED_STRATEGY_TYPES, "supported_asset_classes": SUPPORTED_ASSET_CLASSES, "owner_required": True, "policy_reference_required": True},
	"signals": {"strategy_required": True, "source_required": True, "freshness_required": True, "lineage_required": True},
	"backtests": {"strategy_required": True, "period_required": True, "positive_trade_count_required": True, "data_source_required": True},
	"risk_limits": {"strategy_required": True, "metric_required": True, "positive_limit_required": True, "approval_required": True},
	"orders": {"supported_order_types": SUPPORTED_ORDER_TYPES, "strategy_required": True, "risk_limit_required": True, "positive_quantity_required": True, "instrument_required": True, "approval_required": True},
	"executions": {"supported_venues": SUPPORTED_VENUES, "order_required": True, "positive_filled_quantity_required": True, "venue_required": True, "source_required": True},
	"positions": {"strategy_required": True, "as_of_date_required": True, "source_required": True},
	"surveillance": {"supported_severities": SUPPORTED_ALERT_SEVERITIES, "evidence_required": True, "review_required": True},
	"reviews": {"supported_statuses": SUPPORTED_REVIEW_STATUSES, "reviewer_required": True, "evidence_required": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True},
	"observability": {"event_stream": TRADING_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "nlp": "nlpc", "keys": "keym", "portfolio": "fintech_portfolio", "wealth": "fintech_wealth", "robo": "fintech_robo", "payments": "fintech_payments", "wallets": "fintech_wallets", "kyc": "fintech_kyc", "aml": "fintech_aml", "fraud": "fintech_fraud", "analytics": "bia", "reporting": "fin_rpt", "market_data": "market_data", "event_stream": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_strategies": True, "enable_signals": True, "enable_backtests": True, "enable_risk": True, "enable_orders": True, "enable_executions": True, "enable_positions": True, "enable_surveillance": True, "enable_reviews": True, "enable_agents": True},
	"theme": {"default_theme": "algorithmic_trading_control", "allow_tenant_overrides": True},
}

PROVIDES = [
	"trading_strategy_workflow",
	"trading_signal_workflow",
	"trading_backtest_workflow",
	"trading_risk_limit_workflow",
	"trading_order_intent_workflow",
	"trading_execution_workflow",
	"trading_position_workflow",
	"trading_surveillance_workflow",
	"trading_review_workflow",
	"trading_agent_workflow",
]
REQUIRES = [
	"auth",
	"audl",
	"ntfy",
	"nlpc",
	"keym",
	"fintech_portfolio",
	"fintech_wealth",
	"fintech_robo",
	"fintech_payments",
	"fintech_wallets",
	"fintech_kyc",
	"fintech_aml",
	"fintech_fraud",
	"bia_anl",
	"fin_rpt",
]

UI_ROUTES = [
	{"name": "dashboard", "path": "/fintech-trading/dashboard", "component": "TradingDashboard", "permission": "fintech_trading:view", "nav_group": "Overview"},
	{"name": "strategies", "path": "/fintech-trading/strategies", "component": "StrategyConsole", "permission": "fintech_trading:strategies", "nav_group": "Strategies"},
	{"name": "signals", "path": "/fintech-trading/signals", "component": "SignalSourceConsole", "permission": "fintech_trading:signals", "nav_group": "Strategies"},
	{"name": "backtests", "path": "/fintech-trading/backtests", "component": "BacktestWorkbench", "permission": "fintech_trading:backtests", "nav_group": "Validation"},
	{"name": "risk", "path": "/fintech-trading/risk", "component": "TradingRiskConsole", "permission": "fintech_trading:risk", "nav_group": "Risk"},
	{"name": "orders", "path": "/fintech-trading/orders", "component": "OrderIntentWorkbench", "permission": "fintech_trading:orders", "nav_group": "Trading"},
	{"name": "executions", "path": "/fintech-trading/executions", "component": "ExecutionLedger", "permission": "fintech_trading:executions", "nav_group": "Trading"},
	{"name": "positions", "path": "/fintech-trading/positions", "component": "PositionSnapshotConsole", "permission": "fintech_trading:positions", "nav_group": "Risk"},
	{"name": "surveillance", "path": "/fintech-trading/surveillance", "component": "TradingSurveillanceConsole", "permission": "fintech_trading:surveillance", "nav_group": "Governance"},
	{"name": "reviews", "path": "/fintech-trading/reviews", "component": "TradingReviewConsole", "permission": "fintech_trading:reviews", "nav_group": "Governance"},
	{"name": "agents", "path": "/fintech-trading/agents", "component": "TradingAgentWorkbench", "permission": "fintech_trading:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/fintech-trading/settings", "component": "TradingSettings", "permission": "fintech_trading:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "algorithmic_trading_control",
	"tokens": {"color.primary": "#0E7490", "color.accent": "#4F46E5", "color.success": "#15803D", "color.warning": "#A16207", "color.danger": "#B91C1C", "surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF", "text.primary": "#111827", "text.secondary": "#4B5563", "border.radius": "8px", "density": "compact"},
	"components": {"strategies": {"icon": "workflow", "status_indicator": "strategy-chip"}, "signals": {"icon": "radio-tower", "status_indicator": "signal-chip"}, "backtests": {"icon": "history", "status_indicator": "backtest-chip"}, "risk": {"icon": "shield-alert", "status_indicator": "risk-chip"}, "orders": {"icon": "send-horizontal", "status_indicator": "order-chip"}, "executions": {"icon": "receipt-text", "status_indicator": "execution-chip"}, "positions": {"icon": "chart-candlestick", "status_indicator": "position-chip"}, "surveillance": {"icon": "scan-eye", "status_indicator": "alert-chip"}, "reviews": {"icon": "clipboard-check", "status_indicator": "review-chip"}, "agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"}},
}

STREAMING = {
	"processor": "bytewax",
	"stream": TRADING_EVENT_STREAM,
	"key": "tenant_id",
	"events": ["trading_strategy_registered", "signal_source_attached", "backtest_recorded", "risk_limit_set", "order_intent_staged", "execution_recorded", "position_snapshot_recorded", "surveillance_alert_recorded", "trading_review_recorded", "trading_agent_registered"],
	"guardrails": ["trading_batch_requires_bytewax", "privileged_trading_agent_action_requires_human_approval"],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "trading_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "policy_evidence_required", "required_action": "attach_policy_evidence"}},
	{"name": "strategy_owner_required", "condition": {"operation": "register_strategy", "owner_present": False}, "effect": {"decision": "deny", "reason": "strategy_owner_required", "required_action": "select_owner"}},
	{"name": "strategy_type_supported", "condition": {"operation": "register_strategy", "strategy_type_supported": False}, "effect": {"decision": "deny", "reason": "strategy_type_not_supported", "required_action": "select_supported_strategy_type"}},
	{"name": "strategy_asset_class_supported", "condition": {"operation": "register_strategy", "asset_class_supported": False}, "effect": {"decision": "deny", "reason": "asset_class_not_supported", "required_action": "select_supported_asset_class"}},
	{"name": "strategy_policy_reference_required", "condition": {"operation": "register_strategy", "policy_reference_present": False}, "effect": {"decision": "deny", "reason": "strategy_policy_reference_required", "required_action": "attach_policy_reference"}},
	{"name": "signal_strategy_required", "condition": {"operation": "attach_signal_source", "strategy_present": False}, "effect": {"decision": "deny", "reason": "signal_strategy_required", "required_action": "select_strategy"}},
	{"name": "signal_source_required", "condition": {"operation": "attach_signal_source", "source_present": False}, "effect": {"decision": "deny", "reason": "signal_source_required", "required_action": "attach_signal_source"}},
	{"name": "signal_freshness_required", "condition": {"operation": "attach_signal_source", "freshness_present": False}, "effect": {"decision": "deny", "reason": "signal_freshness_required", "required_action": "set_freshness_sla"}},
	{"name": "backtest_strategy_required", "condition": {"operation": "record_backtest", "strategy_present": False}, "effect": {"decision": "deny", "reason": "backtest_strategy_required", "required_action": "select_strategy"}},
	{"name": "backtest_period_required", "condition": {"operation": "record_backtest", "period_present": False}, "effect": {"decision": "deny", "reason": "backtest_period_required", "required_action": "set_backtest_period"}},
	{"name": "backtest_positive_trade_count", "condition": {"operation": "record_backtest", "positive_trade_count": False}, "effect": {"decision": "deny", "reason": "positive_trade_count_required", "required_action": "record_trade_count"}},
	{"name": "backtest_data_source_required", "condition": {"operation": "record_backtest", "data_source_present": False}, "effect": {"decision": "deny", "reason": "backtest_data_source_required", "required_action": "attach_data_source"}},
	{"name": "risk_strategy_required", "condition": {"operation": "set_risk_limit", "strategy_present": False}, "effect": {"decision": "deny", "reason": "risk_strategy_required", "required_action": "select_strategy"}},
	{"name": "risk_metric_required", "condition": {"operation": "set_risk_limit", "metric_present": False}, "effect": {"decision": "deny", "reason": "risk_metric_required", "required_action": "set_risk_metric"}},
	{"name": "risk_positive_limit", "condition": {"operation": "set_risk_limit", "positive_limit": False}, "effect": {"decision": "deny", "reason": "positive_risk_limit_required", "required_action": "set_positive_limit"}},
	{"name": "risk_approval_required", "condition": {"operation": "set_risk_limit", "approval_present": False}, "effect": {"decision": "deny", "reason": "risk_approval_required", "required_action": "attach_risk_approval"}},
	{"name": "order_strategy_required", "condition": {"operation": "stage_order_intent", "strategy_present": False}, "effect": {"decision": "deny", "reason": "order_strategy_required", "required_action": "select_strategy"}},
	{"name": "order_risk_limit_required", "condition": {"operation": "stage_order_intent", "risk_limit_present": False}, "effect": {"decision": "deny", "reason": "order_risk_limit_required", "required_action": "set_risk_limit"}},
	{"name": "order_type_supported", "condition": {"operation": "stage_order_intent", "order_type_supported": False}, "effect": {"decision": "deny", "reason": "order_type_not_supported", "required_action": "select_supported_order_type"}},
	{"name": "order_instrument_required", "condition": {"operation": "stage_order_intent", "instrument_present": False}, "effect": {"decision": "deny", "reason": "order_instrument_required", "required_action": "select_instrument"}},
	{"name": "order_positive_quantity", "condition": {"operation": "stage_order_intent", "positive_quantity": False}, "effect": {"decision": "deny", "reason": "positive_order_quantity_required", "required_action": "set_positive_quantity"}},
	{"name": "order_approval_required", "condition": {"operation": "stage_order_intent", "approval_present": False}, "effect": {"decision": "deny", "reason": "order_approval_required", "required_action": "attach_order_approval"}},
	{"name": "execution_order_required", "condition": {"operation": "record_execution", "order_present": False}, "effect": {"decision": "deny", "reason": "execution_order_required", "required_action": "select_order"}},
	{"name": "execution_venue_supported", "condition": {"operation": "record_execution", "venue_supported": False}, "effect": {"decision": "deny", "reason": "execution_venue_not_supported", "required_action": "select_supported_venue"}},
	{"name": "execution_positive_filled_quantity", "condition": {"operation": "record_execution", "positive_filled_quantity": False}, "effect": {"decision": "deny", "reason": "positive_filled_quantity_required", "required_action": "set_filled_quantity"}},
	{"name": "execution_source_required", "condition": {"operation": "record_execution", "source_present": False}, "effect": {"decision": "deny", "reason": "execution_source_required", "required_action": "attach_execution_source"}},
	{"name": "position_strategy_required", "condition": {"operation": "record_position_snapshot", "strategy_present": False}, "effect": {"decision": "deny", "reason": "position_strategy_required", "required_action": "select_strategy"}},
	{"name": "position_as_of_required", "condition": {"operation": "record_position_snapshot", "as_of_date_present": False}, "effect": {"decision": "deny", "reason": "position_as_of_date_required", "required_action": "set_as_of_date"}},
	{"name": "position_source_required", "condition": {"operation": "record_position_snapshot", "source_present": False}, "effect": {"decision": "deny", "reason": "position_source_required", "required_action": "attach_position_source"}},
	{"name": "surveillance_severity_supported", "condition": {"operation": "record_surveillance_alert", "severity_supported": False}, "effect": {"decision": "deny", "reason": "surveillance_severity_not_supported", "required_action": "select_supported_severity"}},
	{"name": "surveillance_evidence_required", "condition": {"operation": "record_surveillance_alert", "evidence_present": False}, "effect": {"decision": "deny", "reason": "surveillance_evidence_required", "required_action": "attach_alert_evidence"}},
	{"name": "review_status_supported", "condition": {"operation": "record_review", "status_supported": False}, "effect": {"decision": "deny", "reason": "review_status_not_supported", "required_action": "select_supported_status"}},
	{"name": "review_evidence_required", "condition": {"operation": "record_review", "evidence_present": False}, "effect": {"decision": "deny", "reason": "review_evidence_required", "required_action": "attach_review_evidence"}},
	{"name": "trading_batch_requires_bytewax", "condition": {"operation": "trading_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_trading_batch_to_bytewax"}},
	{"name": "trading_agent_runtime_supported", "condition": {"operation": "register_trading_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "trading_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "trading_agent_role_supported", "condition": {"operation": "register_trading_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "trading_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "privileged_trading_agent_action_requires_human_approval", "condition": {"operation": "trading_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},

	# Cross-tenant and privilege escalation guards
	{"name": "cross_tenant_trading_access_denied", "description": "Trading resources cannot be accessed across tenant boundaries.", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_access_denied", "required_action": "use_tenant_scoped_credentials"}},
	{"name": "privilege_escalation_denied", "description": "Trading privilege escalation without approval is denied.", "condition": {"privilege_escalation_attempt": True, "approval_recorded": False}, "effect": {"decision": "deny", "reason": "privilege_escalation_denied", "required_action": "obtain_escalation_approval"}},

	# Africa-specific trading rules
	{"name": "ke_nse_member_broker_required", "description": "NSE trading requires routing through a licensed NSE member broker.", "condition": {"operation": "place_order", "exchange": "NSE", "nse_member_broker_present": False}, "effect": {"decision": "deny", "reason": "ke_nse_member_broker_required", "required_action": "route_through_nse_member_broker"}},
	{"name": "ke_cma_trading_licence_required", "description": "Kenya CMA stockbroker or dealer licence required for trading operations.", "condition": {"operation": "execute_trade", "country": "KE", "cma_trading_licence_present": False}, "effect": {"decision": "deny", "reason": "ke_cma_trading_licence_required", "required_action": "obtain_cma_trading_licence"}},
	{"name": "mobile_money_trade_funding_kyc", "description": "Mobile money trade account funding requires investor KYC.", "condition": {"operation": "fund_trading_account", "method": "mobile_money", "investor_kyc_present": False}, "effect": {"decision": "deny", "reason": "mobile_money_trade_funding_kyc_required", "required_action": "complete_investor_kyc"}},
	{"name": "ng_sec_dealing_licence_required", "description": "Nigeria SEC dealing licence required for securities trading.", "condition": {"operation": "execute_trade", "country": "NG", "ng_sec_dealing_licence_present": False}, "effect": {"decision": "deny", "reason": "ng_sec_dealing_licence_required", "required_action": "obtain_ng_sec_dealing_licence"}},
	{"name": "ke_nse_settlement_t2_required", "description": "NSE trades must settle on T+2 schedule.", "condition": {"operation": "settle_trade", "exchange": "NSE", "t2_settlement_observed": False}, "effect": {"decision": "deny", "reason": "ke_nse_t2_settlement_required", "required_action": "settle_on_t2_schedule"}},
	{"name": "trading_aml_screening_required", "description": "Trading clients require AML screening.", "condition": {"operation": "onboard_trading_client", "aml_screened": False}, "effect": {"decision": "deny", "reason": "trading_client_aml_screening_required", "required_action": "screen_trading_client"}},
]



def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	return {
		"capability": CAPABILITY_ID,
		"name": CAPABILITY_NAME,
		"display_name": CAPABILITY_NAME,
		"version": CAPABILITY_VERSION,
		"provides": list(PROVIDES),
		"requires": list(REQUIRES),
		"configuration": configuration,
		"configuration_schema": {"type": "object", "required": list(configuration), "properties": {key: {"type": "object"} for key in configuration if key != "tenant_id"} | {"tenant_id": {"type": "string", "minLength": 1}}},
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {"shell": "apg_python", "requires_theme": True, "api_prefix": "/fintech-trading/api/v1", "template_roots": ["templates/", "static/"], "view_module": "views.py", "routes": deepcopy(UI_ROUTES)},
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
