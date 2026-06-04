"""Process-local API helpers for APG Case Management."""

from __future__ import annotations

try:
	from .service import CaseManagementService
except ImportError:  # pragma: no cover
	from service import CaseManagementService  # type: ignore


_SERVICE = CaseManagementService()


def service() -> CaseManagementService:
	return _SERVICE


def open_case(payload: dict):
	return _SERVICE.open_case(payload["case_id"], payload.get("tenant_id", "default"), payload["case_type"], payload["intake_channel"], payload["citizen_id"], payload["priority"], payload["subject"], payload["description"], payload["evidence_reference"], payload.get("policy_attached", True))


def assign_case(payload: dict):
	return _SERVICE.assign_case(payload["assignment_id"], payload.get("tenant_id", "default"), payload["case_id"], payload["assignment_type"], payload["assignee_id"], payload["assigned_by"], payload["evidence_reference"])


def escalate_case(payload: dict):
	return _SERVICE.escalate_case(payload["escalation_id"], payload.get("tenant_id", "default"), payload["case_id"], payload["escalation_reason"], payload["escalated_to"], payload["supervisor_id"], payload["evidence_reference"])


def set_sla(payload: dict):
	return _SERVICE.set_sla(payload["sla_id"], payload.get("tenant_id", "default"), payload["case_id"], payload["sla_category"], payload["due_date"])


def record_outcome(payload: dict):
	return _SERVICE.record_outcome(payload["outcome_id"], payload.get("tenant_id", "default"), payload["case_id"], payload["outcome_type"], payload["description"], payload["approval_reference"], payload["evidence_reference"])


def send_notification(payload: dict):
	return _SERVICE.send_notification(payload["notification_id"], payload.get("tenant_id", "default"), payload["case_id"], payload["notification_type"], payload["recipient_id"], payload["message"])


def record_review(payload: dict):
	return _SERVICE.record_review(payload["review_id"], payload.get("tenant_id", "default"), payload["reference_id"], payload["reviewer_id"], payload["status"], payload["evidence_reference"])


def register_agent(payload: dict):
	return _SERVICE.register_agent(payload["agent_id"], payload.get("tenant_id", "default"), payload["name"], payload["runtime"], payload["role"], payload.get("scope", "case management operations"))


def validate_agent_action(payload: dict):
	return _SERVICE.validate_agent_action(payload.get("tenant_id", "default"), payload.get("privileged_scope", False), payload.get("human_approval_recorded", False), payload.get("evidence_fabrication_scope", False))


def validate_batch(payload: dict):
	return _SERVICE.validate_batch(payload.get("tenant_id", "default"), payload["item_count"], payload.get("event_stream", "bytewax"))


def dashboard(payload: dict):
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))
