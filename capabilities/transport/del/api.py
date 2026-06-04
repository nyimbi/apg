"""Process-local API helpers for APG Delivery Management."""

from __future__ import annotations

try:
	from .service import DeliveryManagementService
except ImportError:
	from service import DeliveryManagementService  # type: ignore

_SERVICE = DeliveryManagementService()


def service() -> DeliveryManagementService:
	return _SERVICE


def create_delivery(payload: dict):
	return _SERVICE.create_delivery(payload["delivery_id"], payload.get("tenant_id", "default"), payload["delivery_type"], payload["recipient_name"], payload["delivery_address"], payload["time_window_start"], payload["time_window_end"], payload.get("sla_tier", "silver"), payload.get("policy_attached", True))


def record_pod(payload: dict):
	return _SERVICE.record_pod(payload["pod_id"], payload.get("tenant_id", "default"), payload["delivery_id"], payload["pod_type"], payload["geo_stamp"], payload["captured_at"], payload.get("signatory_name", ""))


def record_failed_delivery(payload: dict):
	return _SERVICE.record_failed_delivery(payload["failed_id"], payload.get("tenant_id", "default"), payload["delivery_id"], payload["failure_reason"], payload["failed_at"], payload.get("notes", ""))


def reschedule_delivery(payload: dict):
	return _SERVICE.reschedule_delivery(payload["reschedule_id"], payload.get("tenant_id", "default"), payload["delivery_id"], payload["source"], payload["new_time_window_start"], payload["new_time_window_end"])


def set_sla(payload: dict):
	return _SERVICE.set_sla(payload["sla_id"], payload.get("tenant_id", "default"), payload["delivery_id"], payload["sla_tier"], payload["committed_at"])


def send_notification(payload: dict):
	return _SERVICE.send_notification(payload["notification_id"], payload.get("tenant_id", "default"), payload["delivery_id"], payload["channel"], payload["recipient_contact"], payload["notification_type"], payload["sent_at"])


def create_return(payload: dict):
	return _SERVICE.create_return(payload["return_id"], payload.get("tenant_id", "default"), payload["delivery_id"], payload["return_reason"], payload["rma_number"], payload["initiated_at"])


def register_delivery_agent(payload: dict):
	return _SERVICE.register_delivery_agent(payload["agent_id"], payload.get("tenant_id", "default"), payload["name"], payload["runtime"], payload["role"], payload.get("scope", "delivery management operations"))


def validate_batch(payload: dict):
	return _SERVICE.validate_batch(payload.get("tenant_id", "default"), payload["item_count"], payload.get("event_stream", "bytewax"))


def dashboard(payload: dict):
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))
