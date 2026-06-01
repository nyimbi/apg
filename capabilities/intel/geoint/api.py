"""Process-local API helpers for APG Geospatial Intelligence."""

from __future__ import annotations

try:
	from .service import GeospatialIntelligenceService
except ImportError:  # pragma: no cover
	from service import GeospatialIntelligenceService  # type: ignore


_SERVICE = GeospatialIntelligenceService()


def service() -> GeospatialIntelligenceService:
	return _SERVICE


def record_authority(payload: dict):
	return _SERVICE.record_authority(payload["authority_id"], payload.get("tenant_id", "default"), payload["authority_type"], payload["scope_reference"], payload["classification"], payload["approver_id"], payload["expires_at"], payload["evidence_reference"], payload.get("policy_attached", True))


def record_area(payload: dict):
	return _SERVICE.record_area(payload["area_id"], payload.get("tenant_id", "default"), payload["name"], payload["geometry_reference"], payload["classification"], payload["owner_id"], payload["authority_id"], payload["evidence_reference"])


def register_source(payload: dict):
	return _SERVICE.register_source(payload["source_id"], payload.get("tenant_id", "default"), payload["source_type"], payload["sensor_type"], payload["resolution_class"], payload["owner_id"], payload["authority_id"], payload["evidence_reference"])


def record_collection_plan(payload: dict):
	return _SERVICE.record_collection_plan(payload["plan_id"], payload.get("tenant_id", "default"), payload["authority_id"], payload["area_id"], payload["source_id"], payload["collection_mode"], payload["retention_days"], payload["approval_reference"], payload["evidence_reference"])


def record_observation(payload: dict):
	return _SERVICE.record_observation(payload["observation_id"], payload.get("tenant_id", "default"), payload["plan_id"], payload["observation_reference"], payload["captured_at"], payload["geospatial_accuracy_score"], payload["evidence_reference"])


def record_feature(payload: dict):
	return _SERVICE.record_feature(payload["feature_id"], payload.get("tenant_id", "default"), payload["observation_id"], payload["feature_type"], payload["geometry_reference"], payload["confidence_score"], payload["analyst_id"], payload["evidence_reference"])


def record_change(payload: dict):
	return _SERVICE.record_change(payload["change_id"], payload.get("tenant_id", "default"), payload["feature_id"], payload["change_type"], payload["severity"], payload["confidence_score"], payload["analyst_id"], payload["evidence_reference"])


def record_assessment(payload: dict):
	return _SERVICE.record_assessment(payload["assessment_id"], payload.get("tenant_id", "default"), payload["change_id"], payload["assessment_type"], payload["classification"], payload["analyst_id"], payload["evidence_reference"])


def record_dissemination(payload: dict):
	return _SERVICE.record_dissemination(payload["dissemination_id"], payload.get("tenant_id", "default"), payload["assessment_id"], payload["audience"], payload["release_marking"], payload["approval_reference"], payload["evidence_reference"])


def record_review(payload: dict):
	return _SERVICE.record_review(payload["review_id"], payload.get("tenant_id", "default"), payload["reference_id"], payload["reviewer_id"], payload["status"], payload["evidence_reference"])


def register_geoint_agent(payload: dict):
	return _SERVICE.register_geoint_agent(payload["agent_id"], payload.get("tenant_id", "default"), payload["name"], payload["runtime"], payload["role"], payload.get("scope", "geoint operations"))


def dashboard(payload: dict):
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))
