"""Process-local API helpers for APG Signals Intelligence."""

from __future__ import annotations

try:
	from .service import SignalsIntelligenceService
except ImportError:  # pragma: no cover
	from service import SignalsIntelligenceService  # type: ignore


_SERVICE = SignalsIntelligenceService()


def service() -> SignalsIntelligenceService:
	return _SERVICE


def record_authority(payload: dict):
	return _SERVICE.record_authority(payload["authority_id"], payload.get("tenant_id", "default"), payload["authority_type"], payload["scope_reference"], payload["classification"], payload["approver_id"], payload["expires_at"], payload["evidence_reference"], payload.get("policy_attached", True))


def register_source(payload: dict):
	return _SERVICE.register_source(payload["source_id"], payload.get("tenant_id", "default"), payload["source_type"], payload["band"], payload["source_reference"], payload["owner_id"], payload["authority_id"], payload["evidence_reference"])


def record_collection_task(payload: dict):
	return _SERVICE.record_collection_task(payload["task_id"], payload.get("tenant_id", "default"), payload["authority_id"], payload["source_id"], payload["collection_mode"], payload["retention_days"], payload["minimization_reference"], payload["approval_reference"], payload["evidence_reference"])


def record_observation(payload: dict):
	return _SERVICE.record_observation(payload["observation_id"], payload.get("tenant_id", "default"), payload["task_id"], payload["observation_reference"], payload["fingerprint"], payload["confidence_score"], payload["evidence_reference"])


def record_processing_batch(payload: dict):
	return _SERVICE.record_processing_batch(payload["batch_id"], payload.get("tenant_id", "default"), payload["observation_id"], payload["processing_type"], payload["quality_score"], payload["analyst_id"], payload["evidence_reference"])


def record_pattern(payload: dict):
	return _SERVICE.record_pattern(payload["pattern_id"], payload.get("tenant_id", "default"), payload["batch_id"], payload["pattern_type"], payload["confidence_score"], payload["analyst_id"], payload["evidence_reference"])


def record_assessment(payload: dict):
	return _SERVICE.record_assessment(payload["assessment_id"], payload.get("tenant_id", "default"), payload["pattern_id"], payload["assessment_type"], payload["classification"], payload["analyst_id"], payload["evidence_reference"])


def record_review(payload: dict):
	return _SERVICE.record_review(payload["review_id"], payload.get("tenant_id", "default"), payload["reference_id"], payload["reviewer_id"], payload["status"], payload["evidence_reference"])


def register_sigint_agent(payload: dict):
	return _SERVICE.register_sigint_agent(payload["agent_id"], payload.get("tenant_id", "default"), payload["name"], payload["runtime"], payload["role"], payload.get("scope", "sigint operations"))


def dashboard(payload: dict):
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))
