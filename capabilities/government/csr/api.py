"""Process-local API helpers for APG Citizen Services Portal."""

from __future__ import annotations

try:
	from .service import CitizenServicesService
except ImportError:  # pragma: no cover
	from service import CitizenServicesService  # type: ignore


_SERVICE = CitizenServicesService()


def service() -> CitizenServicesService:
	return _SERVICE


def register_service(payload: dict):
	return _SERVICE.register_service(payload["service_id"], payload.get("tenant_id", "default"), payload["service_type"], payload["name"], payload["description"], payload["fee_amount"], payload.get("fee_currency", "KES"), payload.get("sla_days", 5), payload.get("evidence_required", True))


def submit_application(payload: dict):
	return _SERVICE.submit_application(payload["application_id"], payload.get("tenant_id", "default"), payload["service_id"], payload["citizen_id"], payload["channel"], payload["reference_number"], payload["evidence_reference"], payload.get("policy_attached", True))


def record_payment(payload: dict):
	return _SERVICE.record_payment(payload["payment_id"], payload.get("tenant_id", "default"), payload["application_id"], payload["payment_method"], payload["amount"], payload.get("currency", "KES"), payload["receipt_number"], payload["transaction_reference"])


def verify_document(payload: dict):
	return _SERVICE.verify_document(payload["verification_id"], payload.get("tenant_id", "default"), payload["application_id"], payload["verification_type"], payload["document_reference"], payload["evidence_reference"])


def update_application_status(payload: dict):
	return _SERVICE.update_application_status(payload["application_id"], payload.get("tenant_id", "default"), payload["new_status"])


def send_notification(payload: dict):
	return _SERVICE.send_notification(payload["notification_id"], payload.get("tenant_id", "default"), payload["application_id"], payload["citizen_id"], payload["notification_type"], payload["message"])


def record_delivery(payload: dict):
	return _SERVICE.record_delivery(payload["delivery_id"], payload.get("tenant_id", "default"), payload["application_id"], payload["delivery_method"], payload["certificate_reference"])


def record_review(payload: dict):
	return _SERVICE.record_review(payload["review_id"], payload.get("tenant_id", "default"), payload["reference_id"], payload["reviewer_id"], payload["status"], payload["evidence_reference"])


def register_agent(payload: dict):
	return _SERVICE.register_agent(payload["agent_id"], payload.get("tenant_id", "default"), payload["name"], payload["runtime"], payload["role"], payload.get("scope", "citizen services operations"))


def validate_batch(payload: dict):
	return _SERVICE.validate_batch(payload.get("tenant_id", "default"), payload["item_count"], payload.get("event_stream", "bytewax"))


def dashboard(payload: dict):
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))
