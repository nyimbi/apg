"""Process-local API helpers for APG Digital Payments."""

from __future__ import annotations

from typing import Any

try:
	from .service import DigitalPaymentsService
except ImportError:  # pragma: no cover - supports direct file loading in tests
	from service import DigitalPaymentsService  # type: ignore


SERVICE = DigitalPaymentsService(tenant_id="default")


def service() -> DigitalPaymentsService:
	"""Return the shared dependency-light service instance."""
	return SERVICE


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	"""Return capability status for generated applications."""
	contract = SERVICE.describe(tenant_id)
	summary = SERVICE.dashboard_summary(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		"order_count": summary["order_count"],
		"captured_volume": summary["captured_volume"],
		"streaming": summary["streaming"],
	}


def open_payment_account(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.open_payment_account(
		account_id=str(payload["account_id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		owner_reference=str(payload["owner_reference"]),
		currency=str(payload.get("currency") or "USD"),
		metadata=dict(payload.get("metadata") or {}),
		policy_attached=bool(payload.get("policy_attached", True)),
	)


def register_instrument(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_instrument(
		instrument_id=str(payload["instrument_id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		account_id=str(payload["account_id"]),
		instrument_type=str(payload["instrument_type"]),
		token_reference=str(payload["token_reference"]),
		policy_attached=bool(payload.get("policy_attached", True)),
	)


def create_payment_order(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_payment_order(
		order_id=str(payload["order_id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		account_id=str(payload["account_id"]),
		instrument_id=str(payload["instrument_id"]),
		amount=payload["amount"],
		currency=str(payload.get("currency") or "USD"),
		counterparty_reference=str(payload["counterparty_reference"]),
		purpose=str(payload.get("purpose") or "payment"),
		policy_attached=bool(payload.get("policy_attached", True)),
	)


def register_payment_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_payment_agent(
		agent_id=str(payload["agent_id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		runtime=str(payload["runtime"]),
		role=str(payload["role"]),
		scope=str(payload.get("scope") or "review payments"),
	)


def list_orders(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_orders(tenant_id)


def list_evidence(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_evidence(tenant_id)
