"""Process-local API helpers for APG Emergency Management."""

from __future__ import annotations

try:
	from .service import EmergencyManagementService
except ImportError:  # pragma: no cover
	from service import EmergencyManagementService  # type: ignore


_SERVICE = EmergencyManagementService()


def service() -> EmergencyManagementService:
	return _SERVICE


def declare_incident(payload: dict):
	return _SERVICE.declare_incident(payload["incident_id"], payload.get("tenant_id", "default"), payload["incident_type"], payload["severity"], payload["location_reference"], payload["commander_id"], payload["description"], payload["evidence_reference"], payload.get("policy_attached", True))


def transition_phase(payload: dict):
	return _SERVICE.transition_phase(payload["incident_id"], payload.get("tenant_id", "default"), payload["new_phase"])


def mobilise_resource(payload: dict):
	return _SERVICE.mobilise_resource(payload["resource_id"], payload.get("tenant_id", "default"), payload["incident_id"], payload["resource_type"], payload["quantity"], payload["unit"], payload["responsible_agency"], payload["evidence_reference"], payload.get("status", "mobilised"))


def activate_agency(payload: dict):
	return _SERVICE.activate_agency(payload["activation_id"], payload.get("tenant_id", "default"), payload["incident_id"], payload["agency_type"], payload["agency_name"], payload["contact_reference"], payload["role"])


def update_eoc(payload: dict):
	return _SERVICE.update_eoc(payload["eoc_id"], payload.get("tenant_id", "default"), payload["incident_id"], payload["eoc_status"], payload["command_structure"], payload["activation_authority"], payload["evidence_reference"], payload.get("authorised", True))


def file_sitrep(payload: dict):
	return _SERVICE.file_sitrep(payload["sitrep_id"], payload.get("tenant_id", "default"), payload["incident_id"], payload["period"], payload["author_id"], payload["summary"], payload["evidence_reference"])


def record_aar(payload: dict):
	return _SERVICE.record_aar(payload["aar_id"], payload.get("tenant_id", "default"), payload["incident_id"], payload["reviewer_id"], payload["lessons_learned"], payload["recommendations"], payload["evidence_reference"], payload.get("status", "draft"))


def record_review(payload: dict):
	return _SERVICE.record_review(payload["review_id"], payload.get("tenant_id", "default"), payload["reference_id"], payload["reviewer_id"], payload["status"], payload["evidence_reference"])


def register_agent(payload: dict):
	return _SERVICE.register_agent(payload["agent_id"], payload.get("tenant_id", "default"), payload["name"], payload["runtime"], payload["role"], payload.get("scope", "emergency management operations"))


def validate_batch(payload: dict):
	return _SERVICE.validate_batch(payload.get("tenant_id", "default"), payload["item_count"], payload.get("event_stream", "bytewax"))


def dashboard(payload: dict):
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))
