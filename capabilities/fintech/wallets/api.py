"""Process-local API helpers for APG Digital Wallets."""

from __future__ import annotations

from typing import Any

try:
	from .service import DigitalWalletsService
except ImportError:  # pragma: no cover
	from service import DigitalWalletsService  # type: ignore


SERVICE = DigitalWalletsService()


def service() -> DigitalWalletsService:
	return SERVICE


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	summary = SERVICE.dashboard_summary(tenant_id)
	return {"capability": contract["capability"], "display_name": contract["display_name"], "tenant_id": tenant_id, "route_count": len(contract["ui"]["routes"]), "rule_count": len(contract["rule_engine"]["rules"]), "wallet_count": summary["wallet_count"], "total_available": summary["total_available"], "streaming": summary["streaming"]}


def open_wallet(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.open_wallet(str(payload["wallet_id"]), str(payload.get("tenant_id") or "default"), str(payload["owner_reference"]), str(payload.get("wallet_type") or "consumer"), str(payload.get("currency") or "USD"), payload.get("initial_balance", 0), dict(payload.get("metadata") or {}), bool(payload.get("policy_attached", True)))


def register_instrument(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_instrument(str(payload["instrument_id"]), str(payload.get("tenant_id") or "default"), str(payload["wallet_id"]), str(payload["instrument_type"]), str(payload["token_reference"]), str(payload["verified_by"]))


def credit_wallet(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.credit_wallet(str(payload["entry_id"]), str(payload.get("tenant_id") or "default"), str(payload["wallet_id"]), payload["amount"], str(payload.get("description") or "wallet credit"), str(payload["idempotency_key"]))


def transfer(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.transfer(str(payload["transfer_id"]), str(payload.get("tenant_id") or "default"), str(payload["source_wallet_id"]), str(payload["target_wallet_id"]), payload["amount"], str(payload.get("review_id") or ""))


def register_wallet_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_wallet_agent(str(payload["agent_id"]), str(payload.get("tenant_id") or "default"), str(payload["name"]), str(payload["runtime"]), str(payload["role"]), str(payload.get("scope") or "review wallet operations"))


def list_wallets(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_wallets(tenant_id)


def list_ledger(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_ledger(tenant_id)
