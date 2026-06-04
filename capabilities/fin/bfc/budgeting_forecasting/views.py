"""View models for APG budgeting and forecasting screens."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import SUPPORTED_BFC_AGENT_ROLES, SUPPORTED_BFC_AGENT_RUNTIMES, get_capability_contract
	from .context import get_current_user_id, get_tenant_id_from_request
	from .service import BudgetingForecastingService
except ImportError:
	from capability_contract import SUPPORTED_BFC_AGENT_ROLES, SUPPORTED_BFC_AGENT_RUNTIMES, get_capability_contract
	from context import get_current_user_id, get_tenant_id_from_request
	from service import BudgetingForecastingService


def navigation_model(tenant_id: str | None = None) -> dict[str, Any]:
	tenant_id = tenant_id or get_tenant_id_from_request()
	contract = get_capability_contract(tenant_id)
	return {"capability": contract["capability"], "routes": contract["ui"]["routes"], "theme": contract["theme"], "api_prefix": contract["ui"]["api_prefix"]}


def dashboard_model(service: BudgetingForecastingService, tenant_id: str | None = None) -> dict[str, Any]:
	tenant_id = tenant_id or get_tenant_id_from_request()
	user_id = get_current_user_id()
	return {
		"screen": "dashboard",
		"title": "Budgeting and Forecasting",
		"summary": service.dashboard_summary(tenant_id),
		"sections": ["budgets", "forecasts", "scenarios", "variances", "approvals"],
		"current_user": user_id,
	}


def budget_model(service: BudgetingForecastingService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"screen": "budgets",
		"records": service.list_budgets(tenant_id),
		"columns": ["budget_id", "name", "owner", "fiscal_year", "currency", "status", "total_amount"],
		"actions": ["create_budget", "add_budget_line", "submit_budget", "approve_budget"],
	}


def budget_line_model(service: BudgetingForecastingService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"screen": "budget_lines",
		"records": service.list_budget_lines(tenant_id),
		"columns": ["line_id", "budget_record_id", "account_id", "line_type", "amount", "period"],
		"actions": ["add_budget_line", "review_account_mapping"],
	}


def forecast_model(service: BudgetingForecastingService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"screen": "forecasts",
		"records": service.list_forecasts(tenant_id),
		"summary": service.forecast_summary(tenant_id),
		"columns": ["forecast_id", "name", "method", "horizon_months", "confidence", "status"],
		"actions": ["create_forecast", "record_forecast_point"],
	}


def scenario_model(service: BudgetingForecastingService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"screen": "scenarios",
		"records": service.list_scenarios(tenant_id),
		"columns": ["scenario_id", "name", "probability", "status"],
		"actions": ["create_scenario", "compare_scenarios"],
	}


def variance_model(service: BudgetingForecastingService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"screen": "variances",
		"records": service.list_variances(tenant_id),
		"summary": service.variance_summary(tenant_id),
		"columns": ["variance_id", "account_id", "budget_amount", "actual_amount", "variance_percent", "reviewed_by"],
		"actions": ["record_variance", "review_variance"],
	}


def approval_model(service: BudgetingForecastingService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"screen": "approvals",
		"records": [budget for budget in service.list_budgets(tenant_id) if budget["status"] == "submitted"],
		"columns": ["budget_id", "name", "submitted_by", "total_amount", "status"],
		"actions": ["approve_budget", "request_revision"],
	}


def collaboration_model(service: BudgetingForecastingService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"screen": "collaboration",
		"records": service.list_collaboration_sessions(tenant_id),
		"columns": ["session_id", "budget_record_id", "participants", "status"],
		"actions": ["start_collaboration_session", "add_participant"],
	}


def agent_workbench_model(service: BudgetingForecastingService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"screen": "agents",
		"records": service.list_bfc_agents(tenant_id),
		"supported_runtimes": SUPPORTED_BFC_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_BFC_AGENT_ROLES,
		"actions": ["register_agent", "validate_action", "record_human_approval"],
	}
