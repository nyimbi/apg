"""Process-local API helpers for APG Telecom Security."""

from __future__ import annotations

from .service import TelecomSecService

_SERVICE = TelecomSecService()


def service() -> TelecomSecService:
	return _SERVICE


def raise_fraud_case(payload: dict) -> dict:
	return _SERVICE.raise_fraud_case(payload["case_id"], payload.get("tenant_id", "default"), payload["fraud_type"], payload["msisdn"], payload["confidence_score"], payload["evidence_reference"], payload.get("detected_at", ""), payload.get("policy_attached", True))


def apply_fraud_block(payload: dict) -> dict:
	return _SERVICE.apply_fraud_block(payload["case_id"], payload.get("tenant_id", "default"), payload["evidence_reference"])


def record_ss7_attack(payload: dict) -> dict:
	return _SERVICE.record_ss7_attack(payload["attack_id"], payload.get("tenant_id", "default"), payload["attack_type"], payload["source_reference"], payload["target_reference"], payload["evidence_reference"], payload.get("detected_at", ""))


def record_diameter_attack(payload: dict) -> dict:
	return _SERVICE.record_diameter_attack(payload["attack_id"], payload.get("tenant_id", "default"), payload["attack_type"], payload["source_realm"], payload["target_realm"], payload["evidence_reference"], payload.get("detected_at", ""))


def activate_intercept(payload: dict) -> dict:
	return _SERVICE.activate_intercept(payload["intercept_id"], payload.get("tenant_id", "default"), payload["intercept_type"], payload["target_msisdn"], payload["warrant_reference"], payload["regulatory_authority"], payload.get("activated_at", ""), payload["expires_at"])


def update_intercept_status(payload: dict) -> dict:
	return _SERVICE.update_intercept_status(payload["intercept_id"], payload.get("tenant_id", "default"), payload["new_status"])


def open_incident(payload: dict) -> dict:
	return _SERVICE.open_incident(payload["incident_id"], payload.get("tenant_id", "default"), payload["incident_type"], payload["severity"], payload["description"], payload["evidence_reference"], payload.get("opened_at", ""))


def update_incident_status(payload: dict) -> dict:
	return _SERVICE.update_incident_status(payload["incident_id"], payload.get("tenant_id", "default"), payload["new_status"], payload.get("resolved_at"))


def record_threat_intel(payload: dict) -> dict:
	return _SERVICE.record_threat_intel(payload["intel_id"], payload.get("tenant_id", "default"), payload["source"], payload["ioc_type"], payload["ioc_value"], payload.get("tlp_level", "amber"), payload["valid_from"], payload.get("valid_to"), payload.get("shared", False))


def register_agent(payload: dict) -> dict:
	return _SERVICE.register_agent(payload["agent_id"], payload.get("tenant_id", "default"), payload["name"], payload["runtime"], payload["role"], payload.get("scope", "security operations"))


def validate_batch(payload: dict) -> dict:
	return _SERVICE.validate_batch(payload.get("tenant_id", "default"), payload["item_count"], payload.get("event_stream", "bytewax"))


def dashboard(payload: dict) -> dict:
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))
