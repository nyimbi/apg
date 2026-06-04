"""Process-local API helpers for APG Quality of Service."""

from __future__ import annotations

from .service import TelecomQosService

_SERVICE = TelecomQosService()


def service() -> TelecomQosService:
	return _SERVICE


def create_qos_policy(payload: dict) -> dict:
	return _SERVICE.create_qos_policy(payload["policy_id"], payload.get("tenant_id", "default"), payload["policy_type"], payload["qos_class"], payload["name"], payload.get("parameters", "{}"), payload["approval_reference"], payload.get("created_by", ""), payload.get("policy_attached", True))


def change_qos_policy(payload: dict) -> dict:
	return _SERVICE.change_qos_policy(payload["policy_id"], payload.get("tenant_id", "default"), payload["new_parameters"], payload.get("is_downgrade", False), payload["approval_reference"])


def classify_traffic(payload: dict) -> dict:
	return _SERVICE.classify_traffic(payload["classification_id"], payload.get("tenant_id", "default"), payload["traffic_type"], payload["classification"], payload["policy_id"], payload.get("flow_reference", ""), payload.get("classified_at", ""))


def update_enforcement_status(payload: dict) -> dict:
	return _SERVICE.update_enforcement_status(payload["enforcement_id"], payload.get("tenant_id", "default"), payload["policy_id"], payload["ne_reference"], payload["status"], payload.get("enforced_at", ""), payload.get("last_updated", ""))


def record_sla_measurement(payload: dict) -> dict:
	return _SERVICE.record_sla_measurement(payload["measurement_id"], payload.get("tenant_id", "default"), payload["sla_parameter"], payload["measured_value"], payload["target_value"], payload.get("customer_id"), payload.get("measured_at", ""))


def record_degradation(payload: dict) -> dict:
	return _SERVICE.record_degradation(payload["degradation_id"], payload.get("tenant_id", "default"), payload["cause"], payload["confidence_score"], payload["description"], payload["affected_resource"], payload["evidence_reference"], payload.get("detected_at", ""))


def record_root_cause(payload: dict) -> dict:
	return _SERVICE.record_root_cause(payload["rca_id"], payload.get("tenant_id", "default"), payload["degradation_id"], payload["root_cause_description"], payload["confidence_score"], payload["evidence_reference"], payload.get("identified_at", ""))


def trigger_remediation(payload: dict) -> dict:
	return _SERVICE.trigger_remediation(payload["remediation_id"], payload.get("tenant_id", "default"), payload["degradation_id"], payload["remediation_type"], payload.get("is_disruptive", False), payload.get("approval_reference"), payload.get("triggered_at", ""))


def complete_remediation(payload: dict) -> dict:
	return _SERVICE.complete_remediation(payload["remediation_id"], payload.get("tenant_id", "default"), payload.get("completed_at", ""))


def register_agent(payload: dict) -> dict:
	return _SERVICE.register_agent(payload["agent_id"], payload.get("tenant_id", "default"), payload["name"], payload["runtime"], payload["role"], payload.get("scope", "qos management operations"))


def validate_batch(payload: dict) -> dict:
	return _SERVICE.validate_batch(payload.get("tenant_id", "default"), payload["item_count"], payload.get("event_stream", "bytewax"))


def dashboard(payload: dict) -> dict:
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))
