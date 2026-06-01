"""Dependency-light API helpers for APG Portfolio Management."""

from __future__ import annotations

from typing import Any

try:
	from .service import PortfolioManagementService
except ImportError:  # pragma: no cover
	from service import PortfolioManagementService  # type: ignore


_SERVICE = PortfolioManagementService()


def service() -> PortfolioManagementService:
	return _SERVICE


def create_portfolio_book(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.create_portfolio_book(payload["portfolio_id"], payload["tenant_id"], payload["owner_id"], payload["name"], payload["portfolio_type"], payload["base_currency"], payload["policy_reference"], payload.get("policy_attached", True))


def record_holding(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.record_holding(payload["holding_id"], payload["tenant_id"], payload["portfolio_id"], payload["instrument_id"], payload["quantity"], payload["cost_minor"], payload["currency"])


def activate_allocation_policy(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.activate_allocation_policy(payload["allocation_id"], payload["tenant_id"], payload["portfolio_id"], dict(payload["target_allocation"]), payload["policy_reference"])


def record_valuation(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.record_valuation(payload["valuation_id"], payload["tenant_id"], payload["portfolio_id"], payload["market_value_minor"], payload["currency"], payload["valuation_date"], payload["source_reference"])


def assign_benchmark(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.assign_benchmark(payload["benchmark_id"], payload["tenant_id"], payload["portfolio_id"], payload["index_id"], payload["policy_reference"])


def record_risk_exposure(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.record_risk_exposure(payload["exposure_id"], payload["tenant_id"], payload["portfolio_id"], payload["metric"], payload["value"], payload["as_of_date"], payload["source_reference"], payload["limit_reference"])


def record_attribution(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.record_attribution(payload["attribution_id"], payload["tenant_id"], payload["portfolio_id"], payload["period"], payload["benchmark_id"], payload["source_reference"], dict(payload["contributions"]))


def record_cash_movement(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.record_cash_movement(payload["movement_id"], payload["tenant_id"], payload["portfolio_id"], payload["amount_minor"], payload["currency"], payload["reference"])


def record_corporate_action(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.record_corporate_action(payload["action_id"], payload["tenant_id"], payload["instrument_id"], payload["action_type"], payload["effective_date"], payload["evidence_reference"])


def record_compliance_breach(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.record_compliance_breach(payload["breach_id"], payload["tenant_id"], payload["portfolio_id"], payload["severity"], payload["evidence_reference"])


def record_review(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.record_review(payload["review_id"], payload["tenant_id"], payload["reference_id"], payload["reviewer_id"], payload["status"], payload["evidence_reference"])


def register_portfolio_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.register_portfolio_agent(payload["agent_id"], payload["tenant_id"], payload["name"], payload["runtime"], payload["role"], payload.get("scope", "portfolio management review"))


def dashboard(tenant_id: str) -> dict[str, Any]:
	return _SERVICE.dashboard_summary(tenant_id)
