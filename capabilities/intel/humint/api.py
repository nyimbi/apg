"""Process-local API helpers for APG Human Intelligence."""

from __future__ import annotations

try:
	from .service import HumanIntelligenceService
except ImportError:  # pragma: no cover
	from service import HumanIntelligenceService  # type: ignore


_SERVICE = HumanIntelligenceService()


def service() -> HumanIntelligenceService:
	return _SERVICE


def record_authority(payload: dict):
	return _SERVICE.record_authority(payload["authority_id"], payload.get("tenant_id", "default"), payload["authority_type"], payload["scope_reference"], payload["classification"], payload["approver_id"], payload["expires_at"], payload["evidence_reference"], payload.get("policy_attached", True))


def register_source(payload: dict):
	return _SERVICE.register_source(payload["source_id"], payload.get("tenant_id", "default"), payload["source_type"], payload["handling_status"], payload["risk_level"], payload["owner_id"], payload["authority_id"], payload["protection_reference"], payload["evidence_reference"])


def record_contact_plan(payload: dict):
	return _SERVICE.record_contact_plan(payload["plan_id"], payload.get("tenant_id", "default"), payload["authority_id"], payload["source_id"], payload["contact_method"], payload["objective_reference"], payload["safety_plan_reference"], payload["approval_reference"], payload["evidence_reference"])


def record_contact_report(payload: dict):
	return _SERVICE.record_contact_report(payload["report_id"], payload.get("tenant_id", "default"), payload["plan_id"], payload["report_reference"], payload["handler_id"], payload["source_welfare_score"], payload["evidence_reference"])


def record_debriefing(payload: dict):
	return _SERVICE.record_debriefing(payload["debriefing_id"], payload.get("tenant_id", "default"), payload["report_id"], payload["topic"], payload["classification"], payload["credibility_score"], payload["analyst_id"], payload["evidence_reference"])


def record_reliability(payload: dict):
	return _SERVICE.record_reliability(payload["assessment_id"], payload.get("tenant_id", "default"), payload["source_id"], payload["reliability_grade"], payload["confidence_score"], payload["analyst_id"], payload["evidence_reference"])


def record_lead(payload: dict):
	return _SERVICE.record_lead(payload["lead_id"], payload.get("tenant_id", "default"), payload["debriefing_id"], payload["lead_type"], payload["priority"], payload["analyst_id"], payload["evidence_reference"])


def record_dissemination(payload: dict):
	return _SERVICE.record_dissemination(payload["dissemination_id"], payload.get("tenant_id", "default"), payload["lead_id"], payload["audience"], payload["release_marking"], payload["approval_reference"], payload["evidence_reference"])


def record_review(payload: dict):
	return _SERVICE.record_review(payload["review_id"], payload.get("tenant_id", "default"), payload["reference_id"], payload["reviewer_id"], payload["status"], payload["evidence_reference"])


def register_humint_agent(payload: dict):
	return _SERVICE.register_humint_agent(payload["agent_id"], payload.get("tenant_id", "default"), payload["name"], payload["runtime"], payload["role"], payload.get("scope", "humint operations"))


def dashboard(payload: dict):
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))
