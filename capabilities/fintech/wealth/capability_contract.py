"""Executable capability contract for APG Wealth Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "fintech_wealth"
CAPABILITY_NAME = "Wealth Management"
CAPABILITY_VERSION = "1.1.0"
WEALTH_EVENT_STREAM = "apg.fintech.wealth.lifecycle"

SUPPORTED_RISK_PROFILES = ["conservative", "balanced", "growth", "aggressive"]
SUPPORTED_TOLERANCES = ["low", "medium", "high"]
SUPPORTED_HORIZONS = ["one_year", "three_years", "five_years", "ten_years", "retirement"]
SUPPORTED_MANDATES = ["advisory", "discretionary", "model", "execution_only"]
SUPPORTED_ORDER_SIDES = ["buy", "sell", "switch"]
SUPPORTED_CURRENCIES = ["USD", "KES", "EUR", "GBP", "NGN", "GHS", "ZAR"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["advisor_reviewer", "suitability_reviewer", "portfolio_reviewer", "order_reviewer", "fee_reviewer", "wealth_compliance_reviewer"]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"clients": {"kyc_required": True, "tax_required": True, "risk_evidence_required": True},
	"suitability": {"supported_risk_profiles": SUPPORTED_RISK_PROFILES, "supported_tolerances": SUPPORTED_TOLERANCES, "supported_horizons": SUPPORTED_HORIZONS, "goals_required": True},
	"portfolios": {"supported_currencies": SUPPORTED_CURRENCIES, "advisor_required": True, "investment_policy_required": True},
	"mandates": {"supported_types": SUPPORTED_MANDATES, "policy_required": True, "suitability_required": True},
	"rebalances": {"allocation_total_percent": 100, "analysis_required": True},
	"orders": {"supported_sides": SUPPORTED_ORDER_SIDES, "risk_reference_required": True, "large_order_threshold_minor": 10000000},
	"performance": {"valuation_required": True, "benchmark_required": True},
	"fees": {"minimum_percent": 0, "maximum_percent": 100, "contract_required": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True},
	"observability": {"event_stream": WEALTH_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "nlp": "nlpc", "keys": "keym", "kyc": "fintech_kyc", "aml": "fintech_aml", "fraud": "fintech_fraud", "payments": "fintech_payments", "wallets": "fintech_wallets", "analytics": "bia", "reporting": "fin_rpt", "event_stream": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_clients": True, "enable_suitability": True, "enable_portfolios": True, "enable_mandates": True, "enable_rebalances": True, "enable_orders": True, "enable_performance": True, "enable_fees": True, "enable_agents": True},
	"theme": {"default_theme": "wealth_management_control", "allow_tenant_overrides": True},
}

PROVIDES = ["wealth_client_profile_workflow", "suitability_profile_workflow", "portfolio_management_workflow", "advisory_mandate_workflow", "portfolio_rebalance_workflow", "wealth_order_workflow", "performance_reporting_workflow", "wealth_fee_workflow", "wealth_agent_workflow"]
REQUIRES = ["auth", "audl", "ntfy", "nlpc", "keym", "fintech_kyc", "fintech_aml", "fintech_fraud", "fintech_payments", "fintech_wallets", "bia", "fin_rpt"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/fintech-wealth/dashboard", "component": "WealthDashboard", "permission": "fintech_wealth:view", "nav_group": "Overview"},
	{"name": "clients", "path": "/fintech-wealth/clients", "component": "ClientProfileConsole", "permission": "fintech_wealth:clients", "nav_group": "Clients"},
	{"name": "suitability", "path": "/fintech-wealth/suitability", "component": "SuitabilityWorkbench", "permission": "fintech_wealth:suitability", "nav_group": "Clients"},
	{"name": "portfolios", "path": "/fintech-wealth/portfolios", "component": "PortfolioConsole", "permission": "fintech_wealth:portfolios", "nav_group": "Portfolios"},
	{"name": "mandates", "path": "/fintech-wealth/mandates", "component": "MandateConsole", "permission": "fintech_wealth:mandates", "nav_group": "Portfolios"},
	{"name": "rebalances", "path": "/fintech-wealth/rebalances", "component": "RebalanceWorkbench", "permission": "fintech_wealth:rebalances", "nav_group": "Portfolios"},
	{"name": "orders", "path": "/fintech-wealth/orders", "component": "OrderBlotter", "permission": "fintech_wealth:orders", "nav_group": "Trading"},
	{"name": "performance", "path": "/fintech-wealth/performance", "component": "PerformanceConsole", "permission": "fintech_wealth:performance", "nav_group": "Reporting"},
	{"name": "fees", "path": "/fintech-wealth/fees", "component": "FeeScheduleConsole", "permission": "fintech_wealth:fees", "nav_group": "Operations"},
	{"name": "agents", "path": "/fintech-wealth/agents", "component": "WealthAgentWorkbench", "permission": "fintech_wealth:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/fintech-wealth/settings", "component": "WealthSettings", "permission": "fintech_wealth:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "wealth_management_control",
	"tokens": {"color.primary": "#166534", "color.accent": "#1D4ED8", "color.success": "#15803D", "color.warning": "#A16207", "color.danger": "#B91C1C", "surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF", "text.primary": "#111827", "text.secondary": "#4B5563", "border.radius": "8px", "density": "compact"},
	"components": {"clients": {"icon": "users", "status_indicator": "client-chip"}, "suitability": {"icon": "clipboard-check", "status_indicator": "suitability-chip"}, "portfolios": {"icon": "pie-chart", "status_indicator": "portfolio-chip"}, "mandates": {"icon": "file-signature", "status_indicator": "mandate-chip"}, "rebalances": {"icon": "scale", "status_indicator": "rebalance-chip"}, "orders": {"icon": "list-ordered", "status_indicator": "order-chip"}, "performance": {"icon": "line-chart", "status_indicator": "performance-chip"}, "fees": {"icon": "percent", "status_indicator": "fee-chip"}, "agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"}},
}

STREAMING = {
	"processor": "bytewax",
	"stream": WEALTH_EVENT_STREAM,
	"key": "tenant_id",
	"events": ["client_profile_registered", "suitability_profile_captured", "portfolio_created", "advisory_mandate_created", "rebalance_proposed", "order_staged", "performance_recorded", "fee_schedule_recorded", "wealth_agent_registered"],
	"guardrails": ["wealth_batch_requires_bytewax", "privileged_wealth_agent_action_requires_human_approval"],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "wealth_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "policy_evidence_required", "required_action": "attach_policy_evidence"}},
	{"name": "client_kyc_required", "condition": {"operation": "register_client_profile", "kyc_present": False}, "effect": {"decision": "deny", "reason": "client_kyc_required", "required_action": "attach_kyc_evidence"}},
	{"name": "client_tax_required", "condition": {"operation": "register_client_profile", "tax_present": False}, "effect": {"decision": "deny", "reason": "client_tax_profile_required", "required_action": "attach_tax_profile"}},
	{"name": "client_risk_required", "condition": {"operation": "register_client_profile", "risk_present": False}, "effect": {"decision": "deny", "reason": "client_risk_evidence_required", "required_action": "attach_risk_evidence"}},
	{"name": "suitability_client_required", "condition": {"operation": "capture_suitability_profile", "client_present": False}, "effect": {"decision": "deny", "reason": "suitability_client_required", "required_action": "select_client"}},
	{"name": "suitability_risk_supported", "condition": {"operation": "capture_suitability_profile", "risk_profile_supported": False}, "effect": {"decision": "deny", "reason": "risk_profile_not_supported", "required_action": "select_supported_risk_profile"}},
	{"name": "suitability_tolerance_supported", "condition": {"operation": "capture_suitability_profile", "tolerance_supported": False}, "effect": {"decision": "deny", "reason": "risk_tolerance_not_supported", "required_action": "select_supported_tolerance"}},
	{"name": "suitability_horizon_supported", "condition": {"operation": "capture_suitability_profile", "horizon_supported": False}, "effect": {"decision": "deny", "reason": "investment_horizon_not_supported", "required_action": "select_supported_horizon"}},
	{"name": "suitability_goals_required", "condition": {"operation": "capture_suitability_profile", "goals_present": False}, "effect": {"decision": "deny", "reason": "investment_goals_required", "required_action": "attach_goals"}},
	{"name": "portfolio_client_required", "condition": {"operation": "create_portfolio", "client_present": False}, "effect": {"decision": "deny", "reason": "portfolio_client_required", "required_action": "select_client"}},
	{"name": "portfolio_currency_supported", "condition": {"operation": "create_portfolio", "currency_supported": False}, "effect": {"decision": "deny", "reason": "portfolio_currency_not_supported", "required_action": "select_supported_currency"}},
	{"name": "portfolio_advisor_required", "condition": {"operation": "create_portfolio", "advisor_present": False}, "effect": {"decision": "deny", "reason": "portfolio_advisor_required", "required_action": "assign_advisor"}},
	{"name": "portfolio_policy_required", "condition": {"operation": "create_portfolio", "policy_present": False}, "effect": {"decision": "deny", "reason": "investment_policy_required", "required_action": "attach_policy_statement"}},
	{"name": "mandate_portfolio_required", "condition": {"operation": "create_advisory_mandate", "portfolio_present": False}, "effect": {"decision": "deny", "reason": "mandate_portfolio_required", "required_action": "select_portfolio"}},
	{"name": "mandate_suitability_required", "condition": {"operation": "create_advisory_mandate", "suitability_present": False}, "effect": {"decision": "deny", "reason": "mandate_suitability_required", "required_action": "select_suitability_profile"}},
	{"name": "mandate_type_supported", "condition": {"operation": "create_advisory_mandate", "mandate_type_supported": False}, "effect": {"decision": "deny", "reason": "mandate_type_not_supported", "required_action": "select_supported_mandate"}},
	{"name": "mandate_policy_required", "condition": {"operation": "create_advisory_mandate", "policy_present": False}, "effect": {"decision": "deny", "reason": "mandate_policy_required", "required_action": "attach_mandate_policy"}},
	{"name": "rebalance_portfolio_required", "condition": {"operation": "propose_rebalance", "portfolio_present": False}, "effect": {"decision": "deny", "reason": "rebalance_portfolio_required", "required_action": "select_portfolio"}},
	{"name": "rebalance_mandate_required", "condition": {"operation": "propose_rebalance", "mandate_present": False}, "effect": {"decision": "deny", "reason": "rebalance_mandate_required", "required_action": "select_mandate"}},
	{"name": "rebalance_mandate_matches_portfolio", "condition": {"operation": "propose_rebalance", "mandate_matches_portfolio": False}, "effect": {"decision": "deny", "reason": "rebalance_mandate_portfolio_mismatch", "required_action": "select_portfolio_mandate"}},
	{"name": "rebalance_allocation_total", "condition": {"operation": "propose_rebalance", "allocation_totals_100": False}, "effect": {"decision": "deny", "reason": "allocation_total_must_equal_100", "required_action": "rebalance_allocation"}},
	{"name": "rebalance_analysis_required", "condition": {"operation": "propose_rebalance", "analysis_present": False}, "effect": {"decision": "deny", "reason": "rebalance_analysis_required", "required_action": "attach_analysis"}},
	{"name": "order_portfolio_required", "condition": {"operation": "stage_order", "portfolio_present": False}, "effect": {"decision": "deny", "reason": "order_portfolio_required", "required_action": "select_portfolio"}},
	{"name": "order_side_supported", "condition": {"operation": "stage_order", "side_supported": False}, "effect": {"decision": "deny", "reason": "order_side_not_supported", "required_action": "select_supported_side"}},
	{"name": "order_quantity_positive", "condition": {"operation": "stage_order", "positive_quantity": False}, "effect": {"decision": "deny", "reason": "positive_order_quantity_required", "required_action": "set_positive_quantity"}},
	{"name": "order_risk_required", "condition": {"operation": "stage_order", "risk_reference_present": False}, "effect": {"decision": "deny", "reason": "order_risk_reference_required", "required_action": "attach_risk_reference"}},
	{"name": "large_order_requires_approval", "condition": {"operation": "stage_order", "large_order": True, "human_approval_recorded": False}, "effect": {"decision": "require_review", "reason": "large_order_approval_required", "required_action": "record_order_approval"}},
	{"name": "performance_portfolio_required", "condition": {"operation": "record_performance", "portfolio_present": False}, "effect": {"decision": "deny", "reason": "performance_portfolio_required", "required_action": "select_portfolio"}},
	{"name": "performance_valuation_required", "condition": {"operation": "record_performance", "valuation_present": False}, "effect": {"decision": "deny", "reason": "performance_valuation_required", "required_action": "attach_valuation"}},
	{"name": "performance_benchmark_required", "condition": {"operation": "record_performance", "benchmark_present": False}, "effect": {"decision": "deny", "reason": "performance_benchmark_required", "required_action": "attach_benchmark"}},
	{"name": "fee_portfolio_required", "condition": {"operation": "record_fee_schedule", "portfolio_present": False}, "effect": {"decision": "deny", "reason": "fee_portfolio_required", "required_action": "select_portfolio"}},
	{"name": "fee_percent_bounded", "condition": {"operation": "record_fee_schedule", "percent_bounded": False}, "effect": {"decision": "deny", "reason": "fee_percent_out_of_bounds", "required_action": "set_valid_fee_percent"}},
	{"name": "fee_contract_required", "condition": {"operation": "record_fee_schedule", "contract_present": False}, "effect": {"decision": "deny", "reason": "fee_contract_required", "required_action": "attach_fee_contract"}},
	{"name": "wealth_batch_requires_bytewax", "condition": {"operation": "wealth_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_wealth_batch_to_bytewax"}},
	{"name": "wealth_agent_runtime_supported", "condition": {"operation": "register_wealth_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "wealth_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "wealth_agent_role_supported", "condition": {"operation": "register_wealth_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "wealth_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "privileged_wealth_agent_action_requires_human_approval", "condition": {"operation": "wealth_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
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
		"ui": {"shell": "apg_python", "requires_theme": True, "api_prefix": "/fintech-wealth/api/v1", "template_roots": ["templates/", "static/"], "view_module": "views.py", "routes": deepcopy(UI_ROUTES)},
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
	decision = "require_review" if any(action["decision"] == "require_review" for action in actions) else "deny"
	return {"decision": decision, "actions": actions, "context": dict(context)}


def _matches(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if key.endswith("_ne"):
			if context.get(key[:-3]) == expected:
				return False
			continue
		if context.get(key) != expected:
			return False
	return True
