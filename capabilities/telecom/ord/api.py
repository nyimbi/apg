"""Process-local API helpers for APG Order Management."""

from __future__ import annotations

from .service import TelecomOrdService

_SERVICE = TelecomOrdService()


def service() -> TelecomOrdService:
	return _SERVICE


def submit_order(payload: dict) -> dict:
	return _SERVICE.submit_order(payload["order_id"], payload.get("tenant_id", "default"), payload["order_type"], payload["customer_id"], payload.get("channel", "web_self_service"), payload.get("priority", "normal"), payload.get("submitted_at", ""), payload.get("policy_attached", True), payload.get("is_duplicate", False))


def validate_order(payload: dict) -> dict:
	return _SERVICE.validate_order(payload["order_id"], payload.get("tenant_id", "default"))


def decompose_order(payload: dict) -> dict:
	return _SERVICE.decompose_order(payload["order_id"], payload.get("tenant_id", "default"))


def create_task(payload: dict) -> dict:
	return _SERVICE.create_task(payload["task_id"], payload.get("tenant_id", "default"), payload["order_id"], payload["task_type"], payload.get("depends_on"))


def complete_task(payload: dict) -> dict:
	return _SERVICE.complete_task(payload["task_id"], payload.get("tenant_id", "default"), payload.get("completed_at", ""))


def record_fallout(payload: dict) -> dict:
	return _SERVICE.record_fallout(payload["fallout_id"], payload.get("tenant_id", "default"), payload["order_id"], payload["fallout_category"], payload["description"])


def resolve_fallout(payload: dict) -> dict:
	return _SERVICE.resolve_fallout(payload["fallout_id"], payload.get("tenant_id", "default"), payload["resolution"], payload.get("resolved_at", ""))


def complete_order(payload: dict) -> dict:
	return _SERVICE.complete_order(payload["order_id"], payload.get("tenant_id", "default"), payload.get("completed_at", ""))


def submit_portability_request(payload: dict) -> dict:
	return _SERVICE.submit_portability_request(payload["request_id"], payload.get("tenant_id", "default"), payload["order_id"], payload["msisdn"], payload["donor_operator"], payload["recipient_operator"], payload.get("submitted_at", ""))


def submit_bulk_order(payload: dict) -> dict:
	return _SERVICE.submit_bulk_order(payload["bulk_id"], payload.get("tenant_id", "default"), payload["order_type"], payload["item_count"], payload["approval_reference"], payload.get("submitted_by", ""), payload.get("submitted_at", ""))


def register_agent(payload: dict) -> dict:
	return _SERVICE.register_agent(payload["agent_id"], payload.get("tenant_id", "default"), payload["name"], payload["runtime"], payload["role"], payload.get("scope", "order management operations"))


def validate_batch(payload: dict) -> dict:
	return _SERVICE.validate_batch(payload.get("tenant_id", "default"), payload["item_count"], payload.get("event_stream", "bytewax"))


def dashboard(payload: dict) -> dict:
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))
