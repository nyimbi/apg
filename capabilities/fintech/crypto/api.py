"""Process-local API helpers for APG Cryptocurrency Services."""

from __future__ import annotations

try:
	from .service import CryptocurrencyServicesService
except ImportError:  # pragma: no cover
	from service import CryptocurrencyServicesService  # type: ignore


_SERVICE = CryptocurrencyServicesService()


def service() -> CryptocurrencyServicesService:
	return _SERVICE


def register_asset(payload: dict):
	return _SERVICE.register_asset(payload["asset_id"], payload.get("tenant_id", "default"), payload["symbol"], payload["asset_type"], payload["network_reference"], payload.get("contract_reference", ""), payload["precision"], payload["owner_id"], payload["evidence_reference"], payload.get("policy_attached", True))


def open_custody_account(payload: dict):
	return _SERVICE.open_custody_account(payload["account_id"], payload.get("tenant_id", "default"), payload["provider_reference"], payload["custody_model"], payload["policy_reference"], payload["owner_id"], payload["evidence_reference"])


def record_balance(payload: dict):
	return _SERVICE.record_balance(payload["balance_id"], payload.get("tenant_id", "default"), payload["account_id"], payload["asset_id"], payload["amount_minor"], payload["valuation_minor"], payload["valuation_currency"], payload["evidence_reference"])


def create_order(payload: dict):
	return _SERVICE.create_order(payload["order_id"], payload.get("tenant_id", "default"), payload["account_id"], payload["asset_id"], payload["side"], payload["order_type"], payload["quantity_minor"], payload.get("limit_price_minor", 0), payload["policy_reference"], payload["requester_id"], payload["evidence_reference"])


def record_trade(payload: dict):
	return _SERVICE.record_trade(payload["trade_id"], payload.get("tenant_id", "default"), payload["order_id"], payload["venue_reference"], payload["execution_price_minor"], payload["quantity_minor"], payload["fee_minor"], payload["status"], payload["settlement_reference"])


def request_transfer(payload: dict):
	return _SERVICE.request_transfer(payload["transfer_id"], payload.get("tenant_id", "default"), payload["account_id"], payload["asset_id"], payload["transfer_type"], payload["destination_reference"], payload["amount_minor"], payload["approval_reference"], payload["evidence_reference"], payload.get("status", "requested"))


def record_screening(payload: dict):
	return _SERVICE.record_screening(payload["screening_id"], payload.get("tenant_id", "default"), payload["reference_id"], payload["screening_type"], payload["status"], payload["evidence_reference"], payload.get("reviewer_id", ""))


def record_price(payload: dict):
	return _SERVICE.record_price(payload["price_id"], payload.get("tenant_id", "default"), payload["asset_id"], payload["source"], payload["price_minor"], payload["currency"], payload["observed_at"], payload["evidence_reference"])


def record_review(payload: dict):
	return _SERVICE.record_review(payload["review_id"], payload.get("tenant_id", "default"), payload["reference_id"], payload["reviewer_id"], payload["status"], payload["evidence_reference"])


def register_crypto_agent(payload: dict):
	return _SERVICE.register_crypto_agent(payload["agent_id"], payload.get("tenant_id", "default"), payload["name"], payload["runtime"], payload["role"], payload.get("scope", "crypto operations"))


def dashboard(payload: dict):
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))
