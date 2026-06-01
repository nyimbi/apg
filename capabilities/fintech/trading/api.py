"""Dependency-light API helpers for APG Algorithmic Trading."""

from __future__ import annotations

from typing import Any

try:
	from .service import AlgorithmicTradingService
except ImportError:  # pragma: no cover
	from service import AlgorithmicTradingService  # type: ignore


_SERVICE = AlgorithmicTradingService()


def service() -> AlgorithmicTradingService:
	return _SERVICE


def register_strategy(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.register_strategy(payload["strategy_id"], payload["tenant_id"], payload["owner_id"], payload["name"], payload["strategy_type"], payload["asset_class"], payload["policy_reference"], payload.get("policy_attached", True))


def attach_signal_source(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.attach_signal_source(payload["signal_id"], payload["tenant_id"], payload["strategy_id"], payload["source_reference"], payload["freshness_sla"], payload["lineage_reference"])


def record_backtest(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.record_backtest(payload["backtest_id"], payload["tenant_id"], payload["strategy_id"], payload["period"], payload["trade_count"], payload["data_source_reference"], dict(payload["metrics"]))


def set_risk_limit(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.set_risk_limit(payload["limit_id"], payload["tenant_id"], payload["strategy_id"], payload["metric"], payload["limit_value"], payload["approval_reference"])


def stage_order_intent(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.stage_order_intent(payload["order_id"], payload["tenant_id"], payload["strategy_id"], payload["risk_limit_id"], payload["instrument_id"], payload["order_type"], payload["quantity"], payload["approval_reference"])


def record_execution(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.record_execution(payload["execution_id"], payload["tenant_id"], payload["order_id"], payload["venue"], payload["filled_quantity"], payload["source_reference"])


def record_position_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.record_position_snapshot(payload["snapshot_id"], payload["tenant_id"], payload["strategy_id"], payload["as_of_date"], payload["gross_exposure_minor"], payload["net_exposure_minor"], payload["source_reference"])


def record_surveillance_alert(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.record_surveillance_alert(payload["alert_id"], payload["tenant_id"], payload["strategy_id"], payload["severity"], payload["evidence_reference"])


def record_review(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.record_review(payload["review_id"], payload["tenant_id"], payload["reference_id"], payload["reviewer_id"], payload["status"], payload["evidence_reference"])


def register_trading_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.register_trading_agent(payload["agent_id"], payload["tenant_id"], payload["name"], payload["runtime"], payload["role"], payload.get("scope", "algorithmic trading review"))


def dashboard(tenant_id: str) -> dict[str, Any]:
	return _SERVICE.dashboard_summary(tenant_id)
