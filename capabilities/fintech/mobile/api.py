"""Dependency-light API helpers for APG Mobile Banking."""

from __future__ import annotations

from typing import Any

try:
	from .service import MobileBankingService
except ImportError:  # pragma: no cover
	from service import MobileBankingService  # type: ignore


_SERVICE = MobileBankingService()


def service() -> MobileBankingService:
	return _SERVICE


def register_program(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.register_program(payload["program_id"], payload["tenant_id"], payload["name"], payload["owner_id"], payload["country"], payload["currency"], list(payload["platforms"]), payload.get("policy_attached", True))


def enroll_customer(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.enroll_customer(payload["customer_id"], payload["tenant_id"], payload["customer_reference"], payload["country"], payload["kyc_reference"], payload["consent_reference"], payload["aml_reference"], payload["fraud_reference"], payload.get("policy_attached", True))


def bind_device(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.bind_device(payload["device_id"], payload["tenant_id"], payload["customer_id"], payload["platform"], payload["fingerprint"], payload["attestation_reference"], payload["risk_tier"], payload.get("policy_attached", True))


def register_auth_factor(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.register_auth_factor(payload["factor_id"], payload["tenant_id"], payload["customer_id"], payload["device_id"], payload["factor_type"], payload["strength_reference"], payload.get("policy_attached", True))


def link_account(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.link_account(payload["link_id"], payload["tenant_id"], payload["customer_id"], payload["link_type"], payload["account_reference"], payload["currency"], payload["provider_reference"], payload.get("policy_attached", True))


def initiate_payment(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.initiate_payment(payload["payment_id"], payload["tenant_id"], payload["customer_id"], payload["device_id"], payload["account_link_id"], payload["payment_type"], payload["amount"], payload["currency"], payload["recipient_reference"], payload["risk_reference"], payload.get("human_approval", ""), payload.get("policy_attached", True))


def record_bill_payment(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.record_bill_payment(payload["bill_id"], payload["tenant_id"], payload["payment_id"], payload["biller_reference"], payload["bill_account_reference"])


def purchase_airtime(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.purchase_airtime(payload["airtime_id"], payload["tenant_id"], payload["payment_id"], payload["operator_reference"], payload["phone_reference"])


def open_service_request(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.open_service_request(payload["request_id"], payload["tenant_id"], payload["customer_id"], payload["reason"], payload["reviewer_id"], list(payload["evidence_references"]))


def set_notification_preference(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.set_notification_preference(payload["preference_id"], payload["tenant_id"], payload["customer_id"], payload["channel"], payload["consent_reference"], payload.get("enabled", True))


def record_fraud_event(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.record_fraud_event(payload["event_id"], payload["tenant_id"], payload["customer_id"], payload["severity"], list(payload["evidence_references"]), payload.get("human_approval", ""))


def register_mobile_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.register_mobile_agent(payload["agent_id"], payload["tenant_id"], payload["name"], payload["runtime"], payload["role"], payload.get("scope", "mobile banking review"))


def dashboard(tenant_id: str) -> dict[str, Any]:
	return _SERVICE.dashboard_summary(tenant_id)
