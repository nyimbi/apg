"""Process-local API helpers for APG Regulatory Technology."""

from __future__ import annotations

try:
	from .service import RegTechService
except ImportError:  # pragma: no cover
	from service import RegTechService  # type: ignore


_SERVICE = RegTechService()


def service() -> RegTechService:
	return _SERVICE


def register_source(payload: dict):
	return _SERVICE.register_source(payload["source_id"], payload.get("tenant_id", "default"), payload["regulator"], payload["jurisdiction"], payload["source_reference"], payload["owner_id"], payload["evidence_reference"], payload.get("policy_attached", True))


def record_change(payload: dict):
	return _SERVICE.record_change(payload["change_id"], payload.get("tenant_id", "default"), payload["source_id"], payload["framework"], payload["change_type"], payload["title"], payload["effective_date"], payload["severity"], payload["evidence_reference"])


def map_obligation(payload: dict):
	return _SERVICE.map_obligation(payload["mapping_id"], payload.get("tenant_id", "default"), payload["change_id"], payload["obligation_reference"], payload["policy_reference"], payload["owner_id"], payload["due_date"])


def assess_impact(payload: dict):
	return _SERVICE.assess_impact(payload["assessment_id"], payload.get("tenant_id", "default"), payload["change_id"], payload["impacted_capability"], payload["risk_rating"], payload["evidence_reference"], payload["reviewer_id"])


def prepare_filing(payload: dict):
	return _SERVICE.prepare_filing(payload["filing_id"], payload.get("tenant_id", "default"), payload["framework"], payload["filing_type"], payload["period"], payload["evidence_reference"], payload["owner_id"])


def record_submission(payload: dict):
	return _SERVICE.record_submission(payload["submission_id"], payload.get("tenant_id", "default"), payload["filing_id"], payload["channel"], payload["submitted_by"], payload["submitted_at"], payload["acknowledgment_reference"])


def open_inquiry(payload: dict):
	return _SERVICE.open_inquiry(payload["inquiry_id"], payload.get("tenant_id", "default"), payload["regulator"], payload["reference_id"], payload["severity"], payload["due_date"], payload["evidence_reference"])


def record_response(payload: dict):
	return _SERVICE.record_response(payload["response_id"], payload.get("tenant_id", "default"), payload["inquiry_id"], payload["responder_id"], payload["response_reference"], payload["approval_reference"])


def record_review(payload: dict):
	return _SERVICE.record_review(payload["review_id"], payload.get("tenant_id", "default"), payload["reference_id"], payload["reviewer_id"], payload["status"], payload["evidence_reference"])


def register_regtech_agent(payload: dict):
	return _SERVICE.register_regtech_agent(payload["agent_id"], payload.get("tenant_id", "default"), payload["name"], payload["runtime"], payload["role"], payload.get("scope", "regulatory review"))


def dashboard(payload: dict):
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))
