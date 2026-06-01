"""Process-local API helpers for APG Digital Surveillance."""

from __future__ import annotations

try:
	from .service import DigitalSurveillanceService
except ImportError:  # pragma: no cover
	from service import DigitalSurveillanceService  # type: ignore


_SERVICE = DigitalSurveillanceService()


def service() -> DigitalSurveillanceService:
	return _SERVICE


def record_authority(payload: dict):
	return _SERVICE.record_authority(payload["authority_id"], payload.get("tenant_id", "default"), payload["authority_type"], payload["scope_reference"], payload["classification"], payload["approver_id"], payload["expires_at"], payload["evidence_reference"], payload.get("policy_attached", True))


def record_program(payload: dict):
	return _SERVICE.record_program(payload["program_id"], payload.get("tenant_id", "default"), payload["program_type"], payload["name"], payload["priority"], payload["authority_id"], payload["evidence_reference"])


def record_asset(payload: dict):
	return _SERVICE.record_asset(payload["asset_id"], payload.get("tenant_id", "default"), payload["asset_type"], payload["asset_reference"], payload["owner_id"], payload["authority_id"], payload["privacy_review_reference"], payload["evidence_reference"])


def register_sensor(payload: dict):
	return _SERVICE.register_sensor(payload["sensor_id"], payload.get("tenant_id", "default"), payload["sensor_type"], payload["asset_id"], payload["sensor_reference"], payload["custodian_id"], payload["calibration_reference"], payload["evidence_reference"])


def record_observation(payload: dict):
	return _SERVICE.record_observation(payload["observation_id"], payload.get("tenant_id", "default"), payload["program_id"], payload["sensor_id"], payload["observation_type"], payload["observation_reference"], payload["content_fingerprint"], payload["observed_at"], payload["confidence_score"], payload["evidence_reference"])


def record_alert(payload: dict):
	return _SERVICE.record_alert(payload["alert_id"], payload.get("tenant_id", "default"), payload["observation_id"], payload["alert_type"], payload["risk_level"], payload["confidence_score"], payload["analyst_id"], payload["evidence_reference"])


def record_risk(payload: dict):
	return _SERVICE.record_risk(payload["assessment_id"], payload.get("tenant_id", "default"), payload["alert_id"], payload["assessment_type"], payload["risk_level"], payload["confidence_score"], payload["analyst_id"], payload["evidence_reference"])


def record_referral(payload: dict):
	return _SERVICE.record_referral(payload["referral_id"], payload.get("tenant_id", "default"), payload["assessment_id"], payload["referral_type"], payload["recipient"], payload["approval_reference"], payload["evidence_reference"])


def record_dissemination(payload: dict):
	return _SERVICE.record_dissemination(payload["dissemination_id"], payload.get("tenant_id", "default"), payload["assessment_id"], payload["audience"], payload["release_marking"], payload["approval_reference"], payload["evidence_reference"])


def record_review(payload: dict):
	return _SERVICE.record_review(payload["review_id"], payload.get("tenant_id", "default"), payload["reference_id"], payload["reviewer_id"], payload["status"], payload["evidence_reference"])


def register_surveillance_agent(payload: dict):
	return _SERVICE.register_surveillance_agent(payload["agent_id"], payload.get("tenant_id", "default"), payload["name"], payload["runtime"], payload["role"], payload.get("scope", "surveillance operations"))


def dashboard(payload: dict):
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))
