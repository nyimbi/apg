"""Process-local API helpers for APG Customer Management."""

from __future__ import annotations

from .service import TelecomCusService

_SERVICE = TelecomCusService()


def service() -> TelecomCusService:
	return _SERVICE


def create_customer(payload: dict) -> dict:
	return _SERVICE.create_customer(payload["customer_id"], payload.get("tenant_id", "default"), payload["customer_type"], payload["msisdn"], payload["name"], payload.get("created_by", ""), payload.get("policy_attached", True))


def submit_kyc_document(payload: dict) -> dict:
	return _SERVICE.submit_kyc_document(payload["doc_id"], payload.get("tenant_id", "default"), payload["customer_id"], payload["document_type"], payload["document_reference"], payload.get("expires_at"))


def verify_kyc(payload: dict) -> dict:
	return _SERVICE.verify_kyc(payload["doc_id"], payload.get("tenant_id", "default"), payload["verified_by"])


def activate_plan(payload: dict) -> dict:
	return _SERVICE.activate_plan(payload["plan_id"], payload.get("tenant_id", "default"), payload["customer_id"], payload["plan_type"], payload["plan_name"], payload["plan_reference"], payload.get("activated_at", ""), payload.get("credit_check_completed", True))


def provision_sim(payload: dict) -> dict:
	return _SERVICE.provision_sim(payload["sim_id"], payload.get("tenant_id", "default"), payload["customer_id"], payload["iccid"], payload["imsi"], payload["msisdn"], payload.get("provisioned_at", ""))


def register_device(payload: dict) -> dict:
	return _SERVICE.register_device(payload["device_id"], payload.get("tenant_id", "default"), payload["customer_id"], payload["device_type"], payload["imei"], payload.get("model", ""), payload.get("registered_at", ""))


def open_case(payload: dict) -> dict:
	return _SERVICE.open_case(payload["case_id"], payload.get("tenant_id", "default"), payload["customer_id"], payload["case_type"], payload["description"], payload.get("opened_at", ""))


def update_case_status(payload: dict) -> dict:
	return _SERVICE.update_case_status(payload["case_id"], payload.get("tenant_id", "default"), payload["new_status"], payload.get("resolved_at"))


def record_lifecycle_event(payload: dict) -> dict:
	return _SERVICE.record_lifecycle_event(payload["event_id"], payload.get("tenant_id", "default"), payload["customer_id"], payload["event_type"], payload.get("event_reference", ""), payload.get("occurred_at", ""), payload.get("recorded_by", ""))


def register_agent(payload: dict) -> dict:
	return _SERVICE.register_agent(payload["agent_id"], payload.get("tenant_id", "default"), payload["name"], payload["runtime"], payload["role"], payload.get("scope", "customer management operations"))


def validate_batch(payload: dict) -> dict:
	return _SERVICE.validate_batch(payload.get("tenant_id", "default"), payload["item_count"], payload.get("event_stream", "bytewax"))


def dashboard(payload: dict) -> dict:
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))
