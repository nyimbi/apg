"""Executable capability contract for APG Portfolio Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "fintech_portfolio"
CAPABILITY_NAME = "Portfolio Management"
CAPABILITY_VERSION = "1.1.0"
PORTFOLIO_EVENT_STREAM = "apg.fintech.portfolio.lifecycle"

SUPPORTED_CURRENCIES = ["USD", "KES", "EUR", "GBP", "NGN", "GHS", "ZAR"]
SUPPORTED_PORTFOLIO_TYPES = ["discretionary", "advisory", "model", "execution_only", "treasury"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_BREACH_SEVERITIES = ["low", "medium", "high", "critical"]
SUPPORTED_CORPORATE_ACTIONS = ["dividend", "split", "merger", "spin_off", "rights_issue", "coupon", "redemption"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = [
	"portfolio_book_reviewer",
	"allocation_policy_reviewer",
	"valuation_reviewer",
	"risk_exposure_reviewer",
	"performance_attribution_reviewer",
	"corporate_action_reviewer",
	"portfolio_compliance_reviewer",
]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"portfolios": {"supported_types": SUPPORTED_PORTFOLIO_TYPES, "supported_currencies": SUPPORTED_CURRENCIES, "owner_required": True, "base_currency_required": True},
	"holdings": {"portfolio_required": True, "instrument_required": True, "positive_quantity_required": True, "positive_cost_required": True},
	"allocation_policies": {"allocation_total_percent": 100, "policy_reference_required": True, "review_required_before_activation": True},
	"valuations": {"positive_market_value_required": True, "source_required": True, "valuation_date_required": True},
	"benchmarks": {"portfolio_required": True, "index_required": True, "policy_reference_required": True},
	"risk": {"exposure_source_required": True, "as_of_date_required": True, "risk_limit_reference_required": True},
	"attribution": {"period_required": True, "source_required": True, "benchmark_required": True},
	"cash": {"positive_amount_required": True, "supported_currencies": SUPPORTED_CURRENCIES, "reference_required": True},
	"corporate_actions": {"supported_types": SUPPORTED_CORPORATE_ACTIONS, "instrument_required": True, "effective_date_required": True, "evidence_required": True},
	"compliance": {"supported_severities": SUPPORTED_BREACH_SEVERITIES, "evidence_required": True, "review_required": True},
	"reviews": {"supported_statuses": SUPPORTED_REVIEW_STATUSES, "reviewer_required": True, "evidence_required": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True},
	"observability": {"event_stream": PORTFOLIO_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {
		"auth": "auth",
		"audit": "audl",
		"notifications": "ntfy",
		"nlp": "nlpc",
		"keys": "keym",
		"wealth": "fintech_wealth",
		"robo": "fintech_robo",
		"payments": "fintech_payments",
		"wallets": "fintech_wallets",
		"kyc": "fintech_kyc",
		"aml": "fintech_aml",
		"fraud": "fintech_fraud",
		"analytics": "bia",
		"reporting": "fin_rpt",
		"market_data": "market_data",
		"event_stream": "bytewax",
	},
	"ui": {"enable_dashboard": True, "enable_portfolios": True, "enable_holdings": True, "enable_allocations": True, "enable_valuations": True, "enable_benchmarks": True, "enable_risk": True, "enable_attribution": True, "enable_cash": True, "enable_corporate_actions": True, "enable_compliance": True, "enable_reviews": True, "enable_agents": True},
	"theme": {"default_theme": "portfolio_management_control", "allow_tenant_overrides": True},
}

PROVIDES = [
	"portfolio_book_workflow",
	"portfolio_holding_workflow",
	"portfolio_allocation_policy_workflow",
	"portfolio_valuation_workflow",
	"portfolio_benchmark_workflow",
	"portfolio_risk_workflow",
	"portfolio_attribution_workflow",
	"portfolio_cash_workflow",
	"portfolio_corporate_action_workflow",
	"portfolio_compliance_workflow",
	"portfolio_review_workflow",
	"portfolio_agent_workflow",
]
REQUIRES = ["auth", "audl", "ntfy", "nlpc", "keym", "fintech_wealth", "fintech_robo", "fintech_payments", "fintech_wallets", "fintech_kyc", "fintech_aml", "fintech_fraud", "bia", "fin_rpt"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/fintech-portfolio/dashboard", "component": "PortfolioDashboard", "permission": "fintech_portfolio:view", "nav_group": "Overview"},
	{"name": "portfolios", "path": "/fintech-portfolio/portfolios", "component": "PortfolioBookConsole", "permission": "fintech_portfolio:portfolios", "nav_group": "Books"},
	{"name": "holdings", "path": "/fintech-portfolio/holdings", "component": "HoldingLedger", "permission": "fintech_portfolio:holdings", "nav_group": "Books"},
	{"name": "allocations", "path": "/fintech-portfolio/allocations", "component": "AllocationPolicyConsole", "permission": "fintech_portfolio:allocations", "nav_group": "Policy"},
	{"name": "valuations", "path": "/fintech-portfolio/valuations", "component": "ValuationWorkbench", "permission": "fintech_portfolio:valuations", "nav_group": "Operations"},
	{"name": "benchmarks", "path": "/fintech-portfolio/benchmarks", "component": "BenchmarkAssignmentConsole", "permission": "fintech_portfolio:benchmarks", "nav_group": "Policy"},
	{"name": "risk", "path": "/fintech-portfolio/risk", "component": "PortfolioRiskConsole", "permission": "fintech_portfolio:risk", "nav_group": "Risk"},
	{"name": "attribution", "path": "/fintech-portfolio/attribution", "component": "PerformanceAttributionWorkbench", "permission": "fintech_portfolio:attribution", "nav_group": "Performance"},
	{"name": "cash", "path": "/fintech-portfolio/cash", "component": "PortfolioCashConsole", "permission": "fintech_portfolio:cash", "nav_group": "Operations"},
	{"name": "corporate_actions", "path": "/fintech-portfolio/corporate-actions", "component": "CorporateActionConsole", "permission": "fintech_portfolio:corporate_actions", "nav_group": "Operations"},
	{"name": "compliance", "path": "/fintech-portfolio/compliance", "component": "PortfolioComplianceConsole", "permission": "fintech_portfolio:compliance", "nav_group": "Governance"},
	{"name": "reviews", "path": "/fintech-portfolio/reviews", "component": "PortfolioReviewConsole", "permission": "fintech_portfolio:reviews", "nav_group": "Governance"},
	{"name": "agents", "path": "/fintech-portfolio/agents", "component": "PortfolioAgentWorkbench", "permission": "fintech_portfolio:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/fintech-portfolio/settings", "component": "PortfolioSettings", "permission": "fintech_portfolio:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "portfolio_management_control",
	"tokens": {"color.primary": "#0F766E", "color.accent": "#2563EB", "color.success": "#15803D", "color.warning": "#A16207", "color.danger": "#B91C1C", "surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF", "text.primary": "#111827", "text.secondary": "#4B5563", "border.radius": "8px", "density": "compact"},
	"components": {"portfolios": {"icon": "briefcase-business", "status_indicator": "portfolio-chip"}, "holdings": {"icon": "layers", "status_indicator": "holding-chip"}, "allocations": {"icon": "pie-chart", "status_indicator": "allocation-chip"}, "valuations": {"icon": "line-chart", "status_indicator": "valuation-chip"}, "benchmarks": {"icon": "gauge", "status_indicator": "benchmark-chip"}, "risk": {"icon": "shield-alert", "status_indicator": "risk-chip"}, "attribution": {"icon": "chart-no-axes-combined", "status_indicator": "attribution-chip"}, "cash": {"icon": "wallet-cards", "status_indicator": "cash-chip"}, "corporate_actions": {"icon": "git-branch-plus", "status_indicator": "action-chip"}, "compliance": {"icon": "scale", "status_indicator": "breach-chip"}, "reviews": {"icon": "clipboard-check", "status_indicator": "review-chip"}, "agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"}},
}

STREAMING = {
	"processor": "bytewax",
	"stream": PORTFOLIO_EVENT_STREAM,
	"key": "tenant_id",
	"events": ["portfolio_book_created", "portfolio_holding_recorded", "allocation_policy_activated", "portfolio_valuation_recorded", "benchmark_assigned", "risk_exposure_recorded", "performance_attribution_recorded", "cash_movement_recorded", "corporate_action_recorded", "compliance_breach_recorded", "portfolio_review_recorded", "portfolio_agent_registered"],
	"guardrails": ["portfolio_batch_requires_bytewax", "privileged_portfolio_agent_action_requires_human_approval"],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "portfolio_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "policy_evidence_required", "required_action": "attach_policy_evidence"}},
	{"name": "portfolio_owner_required", "condition": {"operation": "create_portfolio_book", "owner_present": False}, "effect": {"decision": "deny", "reason": "portfolio_owner_required", "required_action": "select_owner"}},
	{"name": "portfolio_type_supported", "condition": {"operation": "create_portfolio_book", "portfolio_type_supported": False}, "effect": {"decision": "deny", "reason": "portfolio_type_not_supported", "required_action": "select_supported_portfolio_type"}},
	{"name": "portfolio_currency_supported", "condition": {"operation": "create_portfolio_book", "currency_supported": False}, "effect": {"decision": "deny", "reason": "portfolio_currency_not_supported", "required_action": "select_supported_currency"}},
	{"name": "holding_portfolio_required", "condition": {"operation": "record_holding", "portfolio_present": False}, "effect": {"decision": "deny", "reason": "holding_portfolio_required", "required_action": "select_portfolio"}},
	{"name": "holding_instrument_required", "condition": {"operation": "record_holding", "instrument_present": False}, "effect": {"decision": "deny", "reason": "holding_instrument_required", "required_action": "select_instrument"}},
	{"name": "holding_positive_quantity", "condition": {"operation": "record_holding", "positive_quantity": False}, "effect": {"decision": "deny", "reason": "positive_holding_quantity_required", "required_action": "set_positive_quantity"}},
	{"name": "holding_positive_cost", "condition": {"operation": "record_holding", "positive_cost": False}, "effect": {"decision": "deny", "reason": "positive_holding_cost_required", "required_action": "set_positive_cost"}},
	{"name": "allocation_portfolio_required", "condition": {"operation": "activate_allocation_policy", "portfolio_present": False}, "effect": {"decision": "deny", "reason": "allocation_portfolio_required", "required_action": "select_portfolio"}},
	{"name": "allocation_total_required", "condition": {"operation": "activate_allocation_policy", "allocation_totals_100": False}, "effect": {"decision": "deny", "reason": "allocation_total_must_equal_100", "required_action": "rebalance_allocation"}},
	{"name": "allocation_policy_reference_required", "condition": {"operation": "activate_allocation_policy", "policy_reference_present": False}, "effect": {"decision": "deny", "reason": "allocation_policy_reference_required", "required_action": "attach_policy_reference"}},
	{"name": "valuation_portfolio_required", "condition": {"operation": "record_valuation", "portfolio_present": False}, "effect": {"decision": "deny", "reason": "valuation_portfolio_required", "required_action": "select_portfolio"}},
	{"name": "valuation_positive_market_value", "condition": {"operation": "record_valuation", "positive_market_value": False}, "effect": {"decision": "deny", "reason": "positive_market_value_required", "required_action": "set_positive_market_value"}},
	{"name": "valuation_source_required", "condition": {"operation": "record_valuation", "source_present": False}, "effect": {"decision": "deny", "reason": "valuation_source_required", "required_action": "attach_valuation_source"}},
	{"name": "valuation_date_required", "condition": {"operation": "record_valuation", "valuation_date_present": False}, "effect": {"decision": "deny", "reason": "valuation_date_required", "required_action": "set_valuation_date"}},
	{"name": "benchmark_portfolio_required", "condition": {"operation": "assign_benchmark", "portfolio_present": False}, "effect": {"decision": "deny", "reason": "benchmark_portfolio_required", "required_action": "select_portfolio"}},
	{"name": "benchmark_index_required", "condition": {"operation": "assign_benchmark", "index_present": False}, "effect": {"decision": "deny", "reason": "benchmark_index_required", "required_action": "select_benchmark_index"}},
	{"name": "risk_portfolio_required", "condition": {"operation": "record_risk_exposure", "portfolio_present": False}, "effect": {"decision": "deny", "reason": "risk_portfolio_required", "required_action": "select_portfolio"}},
	{"name": "risk_source_required", "condition": {"operation": "record_risk_exposure", "source_present": False}, "effect": {"decision": "deny", "reason": "risk_source_required", "required_action": "attach_risk_source"}},
	{"name": "risk_as_of_date_required", "condition": {"operation": "record_risk_exposure", "as_of_date_present": False}, "effect": {"decision": "deny", "reason": "risk_as_of_date_required", "required_action": "set_as_of_date"}},
	{"name": "attribution_portfolio_required", "condition": {"operation": "record_attribution", "portfolio_present": False}, "effect": {"decision": "deny", "reason": "attribution_portfolio_required", "required_action": "select_portfolio"}},
	{"name": "attribution_period_required", "condition": {"operation": "record_attribution", "period_present": False}, "effect": {"decision": "deny", "reason": "attribution_period_required", "required_action": "set_period"}},
	{"name": "attribution_source_required", "condition": {"operation": "record_attribution", "source_present": False}, "effect": {"decision": "deny", "reason": "attribution_source_required", "required_action": "attach_attribution_source"}},
	{"name": "cash_portfolio_required", "condition": {"operation": "record_cash_movement", "portfolio_present": False}, "effect": {"decision": "deny", "reason": "cash_portfolio_required", "required_action": "select_portfolio"}},
	{"name": "cash_positive_amount", "condition": {"operation": "record_cash_movement", "positive_amount": False}, "effect": {"decision": "deny", "reason": "positive_cash_amount_required", "required_action": "set_positive_amount"}},
	{"name": "cash_currency_supported", "condition": {"operation": "record_cash_movement", "currency_supported": False}, "effect": {"decision": "deny", "reason": "cash_currency_not_supported", "required_action": "select_supported_currency"}},
	{"name": "cash_reference_required", "condition": {"operation": "record_cash_movement", "reference_present": False}, "effect": {"decision": "deny", "reason": "cash_reference_required", "required_action": "attach_reference"}},
	{"name": "corporate_action_type_supported", "condition": {"operation": "record_corporate_action", "action_type_supported": False}, "effect": {"decision": "deny", "reason": "corporate_action_type_not_supported", "required_action": "select_supported_action_type"}},
	{"name": "corporate_action_evidence_required", "condition": {"operation": "record_corporate_action", "evidence_present": False}, "effect": {"decision": "deny", "reason": "corporate_action_evidence_required", "required_action": "attach_action_evidence"}},
	{"name": "compliance_severity_supported", "condition": {"operation": "record_compliance_breach", "severity_supported": False}, "effect": {"decision": "deny", "reason": "compliance_severity_not_supported", "required_action": "select_supported_severity"}},
	{"name": "compliance_evidence_required", "condition": {"operation": "record_compliance_breach", "evidence_present": False}, "effect": {"decision": "deny", "reason": "compliance_evidence_required", "required_action": "attach_breach_evidence"}},
	{"name": "review_status_supported", "condition": {"operation": "record_review", "status_supported": False}, "effect": {"decision": "deny", "reason": "review_status_not_supported", "required_action": "select_supported_status"}},
	{"name": "review_evidence_required", "condition": {"operation": "record_review", "evidence_present": False}, "effect": {"decision": "deny", "reason": "review_evidence_required", "required_action": "attach_review_evidence"}},
	{"name": "portfolio_batch_requires_bytewax", "condition": {"operation": "portfolio_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_portfolio_batch_to_bytewax"}},
	{"name": "portfolio_agent_runtime_supported", "condition": {"operation": "register_portfolio_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "portfolio_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "portfolio_agent_role_supported", "condition": {"operation": "register_portfolio_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "portfolio_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "privileged_portfolio_agent_action_requires_human_approval", "condition": {"operation": "portfolio_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
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
		"ui": {"shell": "apg_python", "requires_theme": True, "api_prefix": "/fintech-portfolio/api/v1", "template_roots": ["templates/", "static/"], "view_module": "views.py", "routes": deepcopy(UI_ROUTES)},
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
