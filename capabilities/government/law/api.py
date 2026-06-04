"""Process-local API helpers for APG Law Enforcement & Justice."""

from __future__ import annotations

try:
	from .service import LawEnforcementService
except ImportError:  # pragma: no cover
	from service import LawEnforcementService  # type: ignore


_SERVICE = LawEnforcementService()


def service() -> LawEnforcementService:
	return _SERVICE


def report_incident(payload: dict):
	return _SERVICE.report_incident(payload["incident_id"], payload.get("tenant_id", "default"), payload["incident_type"], payload["ob_number"], payload["reporting_officer_id"], payload["location_reference"], payload["complainant_id"], payload["description"], payload["evidence_reference"], payload.get("policy_attached", True))


def open_docket(payload: dict):
	return _SERVICE.open_docket(payload["docket_id"], payload.get("tenant_id", "default"), payload["incident_id"], payload["investigating_officer_id"], payload["docket_number"], payload["opened_date"], payload["evidence_reference"])


def update_docket_status(payload: dict):
	return _SERVICE.update_docket_status(payload["docket_id"], payload.get("tenant_id", "default"), payload["new_status"])


def log_evidence(payload: dict):
	return _SERVICE.log_evidence(payload["evidence_id"], payload.get("tenant_id", "default"), payload["docket_id"], payload["evidence_type"], payload["description"], payload["custodian_id"], payload["evidence_reference"], payload["current_location"])


def record_custody_action(payload: dict):
	return _SERVICE.record_custody_action(payload["action_id"], payload.get("tenant_id", "default"), payload["evidence_id"], payload["custody_action"], payload["actor_id"], payload["from_location"], payload["to_location"], payload["evidence_reference"])


def schedule_hearing(payload: dict):
	return _SERVICE.schedule_hearing(payload["hearing_id"], payload.get("tenant_id", "default"), payload["docket_id"], payload["court_type"], payload["hearing_type"], payload["court_reference"], payload["hearing_date"], payload["presiding_judge"])


def record_prosecution(payload: dict):
	return _SERVICE.record_prosecution(payload["prosecution_id"], payload.get("tenant_id", "default"), payload["docket_id"], payload["dpp_reference"], payload["prosecution_status"], payload["charges"], payload["prosecutor_id"], payload["evidence_reference"])


def record_review(payload: dict):
	return _SERVICE.record_review(payload["review_id"], payload.get("tenant_id", "default"), payload["reference_id"], payload["reviewer_id"], payload["status"], payload["evidence_reference"])


def register_agent(payload: dict):
	return _SERVICE.register_agent(payload["agent_id"], payload.get("tenant_id", "default"), payload["name"], payload["runtime"], payload["role"], payload.get("scope", "law enforcement operations"))


def validate_batch(payload: dict):
	return _SERVICE.validate_batch(payload.get("tenant_id", "default"), payload["item_count"], payload.get("event_stream", "bytewax"))


def dashboard(payload: dict):
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))
