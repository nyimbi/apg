"""Dependency-light API helpers for APG Wealth Management."""

from __future__ import annotations

from typing import Any

try:
	from .service import WealthManagementService
except ImportError:  # pragma: no cover
	from service import WealthManagementService  # type: ignore


_SERVICE = WealthManagementService()


def service() -> WealthManagementService:
	return _SERVICE


def register_client_profile(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.register_client_profile(payload["client_id"], payload["tenant_id"], payload["name"], payload["kyc_reference"], payload["tax_reference"], payload["risk_reference"], payload.get("policy_attached", True))


def capture_suitability_profile(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.capture_suitability_profile(payload["suitability_id"], payload["tenant_id"], payload["client_id"], payload["risk_profile"], payload["risk_tolerance"], payload["horizon"], list(payload["goals"]), payload.get("policy_attached", True))


def create_portfolio(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.create_portfolio(payload["portfolio_id"], payload["tenant_id"], payload["client_id"], payload["name"], payload["base_currency"], payload["advisor_id"], payload["policy_reference"])


def create_advisory_mandate(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.create_advisory_mandate(payload["mandate_id"], payload["tenant_id"], payload["portfolio_id"], payload["suitability_id"], payload["mandate_type"], payload["policy_reference"])


def propose_rebalance(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.propose_rebalance(payload["rebalance_id"], payload["tenant_id"], payload["portfolio_id"], payload["mandate_id"], dict(payload["target_allocation"]), payload["analysis_reference"])


def stage_order(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.stage_order(payload["order_id"], payload["tenant_id"], payload["portfolio_id"], payload["instrument_id"], payload["side"], payload["quantity"], payload["notional_minor"], payload["risk_reference"], payload.get("human_approval", ""))


def record_performance(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.record_performance(payload["snapshot_id"], payload["tenant_id"], payload["portfolio_id"], payload["period"], payload["valuation_reference"], payload["benchmark_reference"], payload["return_percent"])


def record_fee_schedule(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.record_fee_schedule(payload["fee_id"], payload["tenant_id"], payload["portfolio_id"], payload["advisory_percent"], payload["performance_percent"], payload["platform_percent"], payload["contract_reference"])


def register_wealth_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.register_wealth_agent(payload["agent_id"], payload["tenant_id"], payload["name"], payload["runtime"], payload["role"], payload.get("scope", "wealth management review"))


def dashboard(tenant_id: str) -> dict[str, Any]:
	return _SERVICE.dashboard_summary(tenant_id)
