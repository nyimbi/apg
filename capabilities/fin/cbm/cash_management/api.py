"""Dependency-light API helpers for Cash Management."""

from __future__ import annotations

from typing import Any

try:
	from .service import CashManagementService
except ImportError:  # pragma: no cover - supports direct file loading in tests
	from service import CashManagementService  # type: ignore


_SERVICE = CashManagementService()


def service() -> CashManagementService:
	"""Return the process-local CBM service."""
	return _SERVICE


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	return {"ok": True, "capability": "cbm_cash_management", "summary": _SERVICE.dashboard_summary(tenant_id)}


def create_bank(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.create_bank(payload.get("bank_id", "bank"), payload["tenant_id"], payload["code"], payload["name"], payload.get("connectivity_status", "manual"))


def create_cash_account(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.create_cash_account(
		payload.get("account_id", "account"),
		payload["tenant_id"],
		payload["bank_id"],
		payload["account_number"],
		payload["name"],
		payload.get("account_type", "operating"),
		payload.get("currency", "USD"),
		payload.get("minimum_buffer", 0),
	)


def record_cash_position(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.record_cash_position(
		payload.get("position_id", "position"),
		payload["tenant_id"],
		payload["account_id"],
		payload["as_of_date"],
		payload["available_balance"],
		payload.get("ledger_balance"),
		payload.get("liquidity_reviewed_by"),
	)


def record_cash_flow(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.record_cash_flow(
		payload.get("flow_id", "flow"),
		payload["tenant_id"],
		payload["account_id"],
		payload["flow_type"],
		payload["amount"],
		payload["category"],
		payload["expected_date"],
	)


def create_cash_forecast(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.create_cash_forecast(
		payload.get("forecast_id", "forecast"),
		payload["tenant_id"],
		payload["horizon_days"],
		payload.get("scenario", "base"),
		payload.get("confidence_score", 1.0),
		payload.get("reviewed_by"),
	)


def record_bank_reconciliation(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.record_bank_reconciliation(
		payload.get("reconciliation_id", "reconciliation"),
		payload["tenant_id"],
		payload["account_id"],
		payload["bank_statement_balance"],
		payload["ledger_balance"],
		payload.get("reviewed_by"),
	)


def create_treasury_investment(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.create_treasury_investment(
		payload.get("investment_id", "investment"),
		payload["tenant_id"],
		payload["investment_type"],
		payload["counterparty"],
		payload["principal"],
		payload["maturity_date"],
		payload["yield_rate"],
		payload["approved_by"],
	)


def validate_payment_run(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.validate_payment_run(
		payload.get("payment_run_id", "payment-run"),
		payload["tenant_id"],
		payload["funding_account_id"],
		payload["payment_total"],
		payload.get("approved_by"),
	)


def register_cbm_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.register_cbm_agent(
		payload["tenant_id"],
		payload["name"],
		payload["runtime"],
		payload["role"],
		payload.get("scope", "review cash operations"),
	)


def create_record(payload: dict[str, Any]) -> dict[str, Any]:
	"""Generic composition helper used by APG package smoke tests."""
	return create_bank({
		"tenant_id": payload["tenant_id"],
		"bank_id": payload.get("bank_id", "api-bank"),
		"code": payload.get("code", "APIBANK"),
		"name": payload.get("name", "API Bank"),
	})


def list_records(collection: str, tenant_id: str = "default") -> list[dict[str, Any]]:
	return _SERVICE.list_records(collection, tenant_id)
