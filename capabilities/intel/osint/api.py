"""Process-local API helpers for APG Open Source Intelligence."""

from __future__ import annotations

try:
	from .service import OpenSourceIntelligenceService
except ImportError:  # pragma: no cover
	from service import OpenSourceIntelligenceService  # type: ignore


_SERVICE = OpenSourceIntelligenceService()


def service() -> OpenSourceIntelligenceService:
	return _SERVICE


def register_requirement(payload: dict):
	return _SERVICE.register_requirement(payload["requirement_id"], payload.get("tenant_id", "default"), payload["topic"], payload["priority"], payload["requester_id"], payload["classification"], payload["evidence_reference"], payload.get("policy_attached", True))


def register_source(payload: dict):
	return _SERVICE.register_source(payload["source_id"], payload.get("tenant_id", "default"), payload["source_type"], payload["source_reference"], payload["owner_id"], payload["terms_review_reference"], payload["risk_tier"], payload["evidence_reference"])


def record_collection_plan(payload: dict):
	return _SERVICE.record_collection_plan(payload["plan_id"], payload.get("tenant_id", "default"), payload["requirement_id"], payload["source_id"], payload["method"], payload["cadence"], payload.get("approval_reference", ""), payload["evidence_reference"])


def record_evidence(payload: dict):
	return _SERVICE.record_evidence(payload["evidence_id"], payload.get("tenant_id", "default"), payload["plan_id"], payload["content_reference"], payload["fingerprint"], payload["confidence_score"], payload["evidence_reference"])


def record_triage(payload: dict):
	return _SERVICE.record_triage(payload["triage_id"], payload.get("tenant_id", "default"), payload["evidence_id"], payload["decision"], payload["analyst_id"], payload["evidence_reference"])


def record_assessment(payload: dict):
	return _SERVICE.record_assessment(payload["assessment_id"], payload.get("tenant_id", "default"), payload["requirement_id"], payload["assessment_type"], payload["confidence_score"], payload["analyst_id"], payload["evidence_reference"])


def record_dissemination(payload: dict):
	return _SERVICE.record_dissemination(payload["package_id"], payload.get("tenant_id", "default"), payload["assessment_id"], payload["audience"], payload["release_marking"], payload["approval_reference"], payload["evidence_reference"])


def record_review(payload: dict):
	return _SERVICE.record_review(payload["review_id"], payload.get("tenant_id", "default"), payload["reference_id"], payload["reviewer_id"], payload["status"], payload["evidence_reference"])


def register_osint_agent(payload: dict):
	return _SERVICE.register_osint_agent(payload["agent_id"], payload.get("tenant_id", "default"), payload["name"], payload["runtime"], payload["role"], payload.get("scope", "osint operations"))


def dashboard(payload: dict):
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))
