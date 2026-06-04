"""Process-local API helpers for APG Telecom Billing."""

from __future__ import annotations

from .service import TelecomBilService

_SERVICE = TelecomBilService()


def service() -> TelecomBilService:
	return _SERVICE


def record_cdr(payload: dict) -> dict:
	return _SERVICE.record_cdr(payload["cdr_id"], payload.get("tenant_id", "default"), payload["source"], payload.get("mediation_status", "raw"), payload["msisdn"], payload.get("duration_seconds", 0), payload.get("data_volume_bytes", 0), payload.get("recorded_at", ""), payload.get("policy_attached", True))


def record_charge(payload: dict) -> dict:
	return _SERVICE.record_charge(payload["charge_id"], payload.get("tenant_id", "default"), payload["customer_id"], payload["charge_type"], payload["rating_type"], payload["amount"], payload.get("currency", "KES"), payload.get("tax_amount", 0.0), payload.get("cdr_id"))


def create_bill_cycle(payload: dict) -> dict:
	return _SERVICE.create_bill_cycle(payload["cycle_id"], payload.get("tenant_id", "default"), payload["cycle_type"], payload["cutoff_date"], payload["start_date"], payload["end_date"])


def generate_invoice(payload: dict) -> dict:
	return _SERVICE.generate_invoice(payload["invoice_id"], payload.get("tenant_id", "default"), payload["customer_id"], payload["cycle_id"], payload["total_amount"], payload.get("currency", "KES"), payload["due_date"])


def approve_invoice(payload: dict) -> dict:
	return _SERVICE.approve_invoice(payload["invoice_id"], payload.get("tenant_id", "default"), payload["approval_reference"])


def trigger_dunning(payload: dict) -> dict:
	return _SERVICE.trigger_dunning(payload["dunning_id"], payload.get("tenant_id", "default"), payload["invoice_id"], payload["step"], payload.get("triggered_at", ""), payload.get("next_step_date"))


def record_payment(payload: dict) -> dict:
	return _SERVICE.record_payment(payload["payment_id"], payload.get("tenant_id", "default"), payload["invoice_id"], payload["payment_method"], payload["amount"], payload.get("currency", "KES"), payload.get("reference", ""), payload.get("paid_at", ""))


def apply_discount(payload: dict) -> dict:
	return _SERVICE.apply_discount(payload["discount_id"], payload.get("tenant_id", "default"), payload["customer_id"], payload["discount_type"], payload["discount_pct"], payload["approval_reference"], payload["valid_from"], payload["valid_to"])


def register_agent(payload: dict) -> dict:
	return _SERVICE.register_agent(payload["agent_id"], payload.get("tenant_id", "default"), payload["name"], payload["runtime"], payload["role"], payload.get("scope", "billing operations"))


def validate_batch(payload: dict) -> dict:
	return _SERVICE.validate_batch(payload.get("tenant_id", "default"), payload["item_count"], payload.get("event_stream", "bytewax"))


def dashboard(payload: dict) -> dict:
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))
