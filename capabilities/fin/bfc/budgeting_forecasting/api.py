"""Dependency-light API helpers for APG budgeting and forecasting."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import BudgetingForecastingService
except ImportError:
	from capability_contract import get_capability_contract
	from service import BudgetingForecastingService


_SERVICE = BudgetingForecastingService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"ok": True,
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"provides": contract["provides"],
		"requires": contract["requires"],
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		"streaming": contract["streaming"],
		"summary": _SERVICE.dashboard_summary(tenant_id),
	}


def create_budget(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.create_budget(
		payload["budget_id"],
		payload.get("tenant_id", "default"),
		payload["name"],
		payload["owner"],
		payload["fiscal_year"],
		payload["currency"],
		payload["period_start"],
		payload["period_end"],
	)


def add_budget_line(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.add_budget_line(
		payload["line_id"],
		payload.get("tenant_id", "default"),
		payload["budget_record_id"],
		payload["account_id"],
		payload["line_type"],
		payload["amount"],
		payload["period"],
		payload.get("cost_center"),
	)


def submit_budget(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.submit_budget(
		payload.get("tenant_id", "default"),
		payload["budget_record_id"],
		payload["submitted_by"],
	)


def approve_budget(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.approve_budget(
		payload.get("tenant_id", "default"),
		payload["budget_record_id"],
		payload["approved_by"],
		payload.get("approval_recorded", True),
		payload.get("high_value_reviewed_by"),
	)


def create_forecast(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.create_forecast(
		payload["forecast_id"],
		payload.get("tenant_id", "default"),
		payload["name"],
		payload["method"],
		payload["horizon_months"],
		payload.get("confidence", 80),
		payload.get("base_budget_record_id"),
	)


def record_forecast_point(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.record_forecast_point(
		payload["point_id"],
		payload.get("tenant_id", "default"),
		payload["forecast_record_id"],
		payload["period"],
		payload["value"],
	)


def create_scenario(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.create_scenario(
		payload["scenario_id"],
		payload.get("tenant_id", "default"),
		payload["name"],
		payload["probability"],
		payload["drivers"],
	)


def record_variance(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.record_variance(
		payload["variance_id"],
		payload.get("tenant_id", "default"),
		payload["budget_record_id"],
		payload["account_id"],
		payload["budget_amount"],
		payload.get("actual_amount"),
		payload.get("reviewed_by"),
	)


def register_bfc_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.register_bfc_agent(
		payload.get("tenant_id", "default"),
		payload["name"],
		payload["runtime"],
		payload["role"],
		payload.get("instructions", ""),
	)


def create_record(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.create_record(payload)


def list_records(tenant_id: str = "default") -> list[dict[str, Any]]:
	return _SERVICE.list_records(tenant_id)


def service() -> BudgetingForecastingService:
	return _SERVICE
