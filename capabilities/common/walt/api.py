"""API helpers for the Wallet and Payment Core capability."""

from __future__ import annotations

from typing import Any

from .service import WaltService


SERVICE = WaltService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	summary = SERVICE.dashboard_summary(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		"wallet_count": summary["wallet_count"],
		"instrument_count": summary["instrument_count"],
		"transaction_count": summary["transaction_count"],
		"settlement_batch_count": summary["settlement_batch_count"],
		"reconciliation_count": summary["reconciliation_count"],
		"total_balance": summary["total_balance"],
	}


def create_wallet(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_wallet(
		tenant_id=str(payload.get("tenant_id") or "default"),
		owner_ref=str(payload.get("owner_ref") or ""),
		currency=str(payload.get("currency") or "USD"),
		ledger_ref=str(payload.get("ledger_ref") or ""),
		compliance_policy_ref=str(payload.get("compliance_policy_ref") or ""),
		initial_balance=payload.get("initial_balance", 0),
		actor=str(payload.get("actor") or "system"),
	)


def register_instrument(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_instrument(
		tenant_id=str(payload.get("tenant_id") or "default"),
		wallet_id=str(payload["wallet_id"]),
		instrument_ref=str(payload.get("instrument_ref") or ""),
		instrument_type=str(payload.get("instrument_type") or "external"),
		token_ref=str(payload.get("token_ref") or ""),
		encrypted=bool(payload.get("encrypted", False)),
		verified_by=str(payload.get("verified_by") or ""),
	)


def authorize_transaction(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.authorize_transaction(
		tenant_id=str(payload.get("tenant_id") or "default"),
		wallet_id=str(payload["wallet_id"]),
		instrument_id=str(payload["instrument_id"]),
		amount=payload["amount"],
		currency=str(payload.get("currency") or "USD"),
		direction=str(payload.get("direction") or "debit"),
		mfa_completed=bool(payload.get("mfa_completed", False)),
		risk_score=float(payload.get("risk_score", 0.0)),
		risk_review_recorded=bool(payload.get("risk_review_recorded", False)),
		idempotency_key=str(payload.get("idempotency_key") or ""),
		actor=str(payload.get("actor") or "system"),
	)


def capture_transaction(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.capture_transaction(
		tenant_id=str(payload.get("tenant_id") or "default"),
		transaction_id=str(payload["transaction_id"]),
		actor=str(payload.get("actor") or "system"),
	)


def create_settlement_batch(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_settlement_batch(
		tenant_id=str(payload.get("tenant_id") or "default"),
		transaction_ids=[str(item) for item in payload.get("transaction_ids", [])],
		settlement_account_ref=str(payload.get("settlement_account_ref") or ""),
		reconciliation_completed=bool(payload.get("reconciliation_completed", False)),
		created_by=str(payload.get("created_by") or "system"),
	)


def record_reconciliation(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.record_reconciliation(
		tenant_id=str(payload.get("tenant_id") or "default"),
		settlement_batch_id=str(payload["settlement_batch_id"]),
		reconciliation_ref=str(payload.get("reconciliation_ref") or ""),
		matched_count=int(payload.get("matched_count", 0)),
		exception_count=int(payload.get("exception_count", 0)),
		recorded_by=str(payload.get("recorded_by") or "system"),
	)


def create_record(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_record(
		record_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		metadata=dict(payload.get("metadata") or {}),
		status=str(payload.get("status") or "active"),
	)


def list_records(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_records(tenant_id)


def list_wallet_payments(tenant_id: str = "default") -> dict[str, Any]:
	return {
		"wallets": SERVICE.list_wallets(tenant_id),
		"instruments": SERVICE.list_instruments(tenant_id),
		"transactions": SERVICE.list_transactions(tenant_id),
		"settlement_batches": SERVICE.list_settlement_batches(tenant_id),
		"reconciliations": SERVICE.list_reconciliations(tenant_id),
		"audit_events": SERVICE.list_audit_events(tenant_id),
		"summary": SERVICE.dashboard_summary(tenant_id),
	}
