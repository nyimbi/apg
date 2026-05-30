"""Dependency-light API helpers for Fintech Gateway."""

from __future__ import annotations

from typing import Any

try:
	from .service import FintechGatewayService
except ImportError:  # pragma: no cover - supports direct file loading in tests
	from service import FintechGatewayService  # type: ignore


_SERVICE = FintechGatewayService()


def service() -> FintechGatewayService:
	"""Return the process-local gateway service."""
	return _SERVICE


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	return {"ok": True, "capability": "fintech_gateway", "summary": _SERVICE.dashboard_summary(tenant_id)}


def onboard_merchant(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.onboard_merchant(
		payload.get("merchant_id", "merchant"),
		payload["tenant_id"],
		payload["merchant_code"],
		payload["legal_name"],
		payload["country"],
		payload.get("risk_level", "low"),
		payload.get("reviewed_by"),
	)


def connect_provider(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.connect_provider(
		payload.get("connection_id", "provider"),
		payload["tenant_id"],
		payload["provider"],
		payload["provider_type"],
		payload["credential_reference"],
		payload.get("priority", 100),
	)


def tokenize_payment_method(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.tokenize_payment_method(
		payload.get("method_id", "method"),
		payload["tenant_id"],
		payload["merchant_id"],
		payload["customer_reference"],
		payload["method_type"],
		payload["token_reference"],
	)


def create_payment_intent(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.create_payment_intent(
		payload.get("intent_id", "intent"),
		payload["tenant_id"],
		payload["merchant_id"],
		payload["payment_method_id"],
		payload["amount"],
		payload["currency"],
		payload.get("description", ""),
	)


def assess_payment_risk(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.assess_payment_risk(
		payload.get("review_id", "risk"),
		payload["tenant_id"],
		payload["payment_intent_id"],
		payload.get("risk_level", "low"),
		payload.get("risk_score", 0.0),
		payload.get("reviewed_by"),
	)


def authorize_payment(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.authorize_payment(
		payload.get("authorization_id", "authorization"),
		payload["tenant_id"],
		payload["payment_intent_id"],
		payload["provider_connection_id"],
		payload.get("approved_by"),
	)


def capture_payment(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.capture_payment(
		payload.get("capture_id", "capture"),
		payload["tenant_id"],
		payload["authorization_id"],
		payload["capture_amount"],
	)


def refund_payment(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.refund_payment(
		payload.get("refund_id", "refund"),
		payload["tenant_id"],
		payload["payment_intent_id"],
		payload["refund_amount"],
		payload.get("reason", "merchant_request"),
		payload.get("reviewed_by"),
	)


def ingest_webhook(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.ingest_webhook(
		payload.get("webhook_id", "webhook"),
		payload["tenant_id"],
		payload["provider_connection_id"],
		payload["event_id"],
		payload["signature"],
		payload["idempotency_key"],
		payload.get("event_type", "payment.updated"),
		payload.get("payload", {}),
	)


def record_settlement(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.record_settlement(
		payload.get("settlement_id", "settlement"),
		payload["tenant_id"],
		payload["provider_connection_id"],
		payload["settlement_reference"],
		payload["amount"],
		payload.get("expected_amount"),
		payload.get("reviewed_by"),
	)


def open_dispute(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.open_dispute(
		payload.get("dispute_id", "dispute"),
		payload["tenant_id"],
		payload["payment_intent_id"],
		payload["reason"],
		payload["owner"],
	)


def resolve_dispute(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.resolve_dispute(payload["dispute_id"], payload["tenant_id"], payload["resolution"], payload["reviewed_by"])


def register_gateway_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.register_gateway_agent(
		payload["tenant_id"],
		payload["name"],
		payload["runtime"],
		payload["role"],
		payload.get("scope", "review gateway operations"),
	)


def create_record(payload: dict[str, Any]) -> dict[str, Any]:
	"""Generic composition helper used by APG package smoke tests."""
	return onboard_merchant({
		"tenant_id": payload["tenant_id"],
		"merchant_id": payload.get("merchant_id", "api-merchant"),
		"merchant_code": payload.get("merchant_code", "APIMERCH"),
		"legal_name": payload.get("legal_name", "API Merchant"),
		"country": payload.get("country", "KE"),
	})


def list_records(collection: str, tenant_id: str = "default") -> list[dict[str, Any]]:
	return _SERVICE.list_records(collection, tenant_id)
