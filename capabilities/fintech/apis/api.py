"""Dependency-light API helpers for APG Banking APIs."""

from __future__ import annotations

from typing import Any

try:
	from .service import BankingAPIService
except ImportError:  # pragma: no cover
	from service import BankingAPIService  # type: ignore


_SERVICE = BankingAPIService()


def service() -> BankingAPIService:
	return _SERVICE


def register_api_product(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.register_api_product(payload["product_id"], payload["tenant_id"], payload["name"], payload["owner_id"], payload["product_type"], payload["environment"], list(payload["scopes"]), payload.get("policy_attached", True))


def onboard_developer(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.onboard_developer(payload["developer_id"], payload["tenant_id"], payload["name"], payload["kyb_reference"], payload["security_review_reference"], payload["risk_clearance_reference"], payload.get("policy_attached", True))


def register_application(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.register_application(payload["application_id"], payload["tenant_id"], payload["developer_id"], payload["name"], payload["environment"], payload["redirect_uri"], payload["terms_reference"], payload.get("policy_attached", True))


def create_consent_grant(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.create_consent_grant(payload["consent_id"], payload["tenant_id"], payload["application_id"], payload["customer_reference"], list(payload["scopes"]), payload["expiry_date"], payload.get("policy_attached", True))


def issue_api_client(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.issue_api_client(payload["client_id"], payload["tenant_id"], payload["application_id"], payload["auth_flow"], payload["key_reference"], list(payload["scopes"]), payload.get("policy_attached", True))


def publish_endpoint_policy(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.publish_endpoint_policy(payload["endpoint_id"], payload["tenant_id"], payload["product_id"], payload["route"], payload["required_scope"], payload["throttle_policy_reference"], payload["risk_policy_reference"])


def subscribe_webhook(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.subscribe_webhook(payload["webhook_id"], payload["tenant_id"], payload["application_id"], payload["event_type"], payload["endpoint"], payload["signing_secret_reference"])


def record_api_call(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.record_api_call(payload["call_id"], payload["tenant_id"], payload["client_id"], payload["product_id"], payload["endpoint_id"], payload["status_code"], payload.get("call_count", 1), payload["risk_reference"], payload.get("human_approval", ""))


def update_rate_limit(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.update_rate_limit(payload["bucket_id"], payload["tenant_id"], payload["client_id"], payload["limit"], payload.get("window_seconds", 60))


def open_sla_incident(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.open_sla_incident(payload["incident_id"], payload["tenant_id"], payload["severity"], payload["owner_id"], list(payload["evidence_references"]), payload.get("human_approval", ""))


def register_api_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.register_api_agent(payload["agent_id"], payload["tenant_id"], payload["name"], payload["runtime"], payload["role"], payload.get("scope", "banking API review"))


def dashboard(tenant_id: str) -> dict[str, Any]:
	return _SERVICE.dashboard_summary(tenant_id)
