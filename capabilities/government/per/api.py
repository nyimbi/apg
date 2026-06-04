"""Process-local API helpers for APG Permits Management."""

from __future__ import annotations

try:
	from .service import PermitsManagementService
except ImportError:  # pragma: no cover
	from service import PermitsManagementService  # type: ignore


_SERVICE = PermitsManagementService()


def service() -> PermitsManagementService:
	return _SERVICE


def submit_application(payload: dict):
	return _SERVICE.submit_application(payload["application_id"], payload.get("tenant_id", "default"), payload["permit_type"], payload["applicant_id"], payload["site_reference"], payload["evidence_reference"], payload.get("fee_paid", False), payload.get("policy_attached", True))


def issue_permit(payload: dict):
	return _SERVICE.issue_permit(payload["permit_id"], payload.get("tenant_id", "default"), payload["application_id"], payload["permit_type"], payload["permit_number"], payload["holder_id"], payload["site_reference"], payload["issued_date"], payload["expiry_date"], payload["evidence_reference"])


def record_condition(payload: dict):
	return _SERVICE.record_condition(payload["condition_id"], payload.get("tenant_id", "default"), payload["permit_id"], payload["condition_type"], payload["description"], payload["due_date"], payload["responsible_party"], payload["evidence_reference"])


def schedule_inspection(payload: dict):
	return _SERVICE.schedule_inspection(payload["inspection_id"], payload.get("tenant_id", "default"), payload["permit_id"], payload["inspection_type"], payload["inspector_id"], payload["scheduled_date"], payload["evidence_reference"])


def record_inspection_outcome(payload: dict):
	return _SERVICE.record_inspection_outcome(payload["inspection_id"], payload.get("tenant_id", "default"), payload["outcome"], payload["findings"])


def record_compliance(payload: dict):
	return _SERVICE.record_compliance(payload["compliance_id"], payload.get("tenant_id", "default"), payload["permit_id"], payload["compliance_status"], payload["officer_id"], payload["assessment_date"], payload["narrative"], payload["evidence_reference"])


def initiate_enforcement(payload: dict):
	return _SERVICE.initiate_enforcement(payload["enforcement_id"], payload.get("tenant_id", "default"), payload["permit_id"], payload["compliance_id"], payload["action_type"], payload["officer_id"], payload["description"], payload["evidence_reference"])


def record_review(payload: dict):
	return _SERVICE.record_review(payload["review_id"], payload.get("tenant_id", "default"), payload["reference_id"], payload["reviewer_id"], payload["status"], payload["evidence_reference"])


def register_agent(payload: dict):
	return _SERVICE.register_agent(payload["agent_id"], payload.get("tenant_id", "default"), payload["name"], payload["runtime"], payload["role"], payload.get("scope", "permits management operations"))


def validate_batch(payload: dict):
	return _SERVICE.validate_batch(payload.get("tenant_id", "default"), payload["item_count"], payload.get("event_stream", "bytewax"))


def dashboard(payload: dict):
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))
