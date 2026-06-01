"""Dependency-light API helpers for APG Embedded Finance."""

from __future__ import annotations

from typing import Any

try:
	from .service import EmbeddedFinanceService
except ImportError:  # pragma: no cover
	from service import EmbeddedFinanceService  # type: ignore


_SERVICE = EmbeddedFinanceService()


def service() -> EmbeddedFinanceService:
	return _SERVICE


def register_partner_program(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.register_partner_program(payload["program_id"], payload["tenant_id"], payload["name"], payload["kyb_reference"], payload["contract_reference"], payload["risk_reference"], payload.get("policy_attached", True))


def register_host_application(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.register_host_application(payload["application_id"], payload["tenant_id"], payload["program_id"], payload["name"], payload["environment"], payload["domain"], payload["terms_reference"], payload.get("policy_attached", True))


def publish_product_placement(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.publish_product_placement(payload["placement_id"], payload["tenant_id"], payload["application_id"], payload["product_type"], payload["channel"], list(payload["scopes"]), payload["risk_policy_reference"])


def capture_customer_consent(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.capture_customer_consent(payload["consent_id"], payload["tenant_id"], payload["application_id"], payload["customer_reference"], list(payload["scopes"]), payload["expiry_date"], payload.get("policy_attached", True))


def open_embedded_account(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.open_embedded_account(payload["account_id"], payload["tenant_id"], payload["application_id"], payload["customer_reference"], payload["wallet_reference"], payload["kyc_reference"])


def initiate_embedded_payment(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.initiate_embedded_payment(payload["payment_id"], payload["tenant_id"], payload["application_id"], payload["placement_id"], payload["consent_id"], payload["source_reference"], payload["destination_reference"], payload["amount_minor"], payload["currency"], payload["risk_reference"])


def offer_embedded_card(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.offer_embedded_card(payload["card_id"], payload["tenant_id"], payload["application_id"], payload["customer_reference"], payload["limit_minor"], payload["risk_reference"])


def create_lending_offer(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.create_lending_offer(payload["offer_id"], payload["tenant_id"], payload["application_id"], payload["customer_reference"], payload["amount_minor"], payload["affordability_reference"], payload["underwriting_reference"])


def close_settlement_batch(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.close_settlement_batch(payload["batch_id"], payload["tenant_id"], payload["program_id"], payload["amount_minor"], payload["currency"], payload["reconciliation_reference"])


def record_revenue_share(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.record_revenue_share(payload["share_id"], payload["tenant_id"], payload["program_id"], payload["percent"], payload["contract_reference"])


def register_embedded_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.register_embedded_agent(payload["agent_id"], payload["tenant_id"], payload["name"], payload["runtime"], payload["role"], payload.get("scope", "embedded finance review"))


def dashboard(tenant_id: str) -> dict[str, Any]:
	return _SERVICE.dashboard_summary(tenant_id)
