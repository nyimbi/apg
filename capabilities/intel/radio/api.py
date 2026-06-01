"""Process-local API helpers for APG Radio Intelligence Listener."""

from __future__ import annotations

try:
	from .service import RadioIntelligenceListenerService
except ImportError:  # pragma: no cover
	from service import RadioIntelligenceListenerService  # type: ignore


_SERVICE = RadioIntelligenceListenerService()


def service() -> RadioIntelligenceListenerService:
	return _SERVICE


def record_authority(payload: dict):
	return _SERVICE.record_authority(payload["authority_id"], payload.get("tenant_id", "default"), payload["authority_type"], payload["scope_reference"], payload["classification"], payload["approver_id"], payload["expires_at"], payload["evidence_reference"], payload.get("policy_attached", True))


def record_band_plan(payload: dict):
	return _SERVICE.record_band_plan(payload["band_id"], payload.get("tenant_id", "default"), payload["band_type"], payload["name"], payload["frequency_min_mhz"], payload["frequency_max_mhz"], payload["authority_id"], payload["evidence_reference"])


def register_receiver(payload: dict):
	return _SERVICE.register_receiver(payload["receiver_id"], payload.get("tenant_id", "default"), payload["receiver_type"], payload["site_reference"], payload["custodian_id"], payload["authority_id"], payload["calibration_reference"], payload["evidence_reference"])


def record_session(payload: dict):
	return _SERVICE.record_session(payload["session_id"], payload.get("tenant_id", "default"), payload["band_id"], payload["receiver_id"], payload["session_type"], payload["started_at"], payload.get("ended_at", ""), payload["collection_plan_reference"], payload["evidence_reference"])


def record_observation(payload: dict):
	return _SERVICE.record_observation(payload["observation_id"], payload.get("tenant_id", "default"), payload["session_id"], payload["frequency_mhz"], payload["signal_type"], payload["signal_fingerprint"], payload["observed_at"], payload["confidence_score"], payload["evidence_reference"])


def record_classification(payload: dict):
	return _SERVICE.record_classification(payload["classification_id"], payload.get("tenant_id", "default"), payload["observation_id"], payload["classification_type"], payload["risk_level"], payload["confidence_score"], payload["analyst_id"], payload["evidence_reference"])


def record_event(payload: dict):
	return _SERVICE.record_event(payload["assessment_id"], payload.get("tenant_id", "default"), payload["classification_id"], payload["event_type"], payload["risk_level"], payload["confidence_score"], payload["analyst_id"], payload["evidence_reference"])


def record_referral(payload: dict):
	return _SERVICE.record_referral(payload["referral_id"], payload.get("tenant_id", "default"), payload["assessment_id"], payload["referral_type"], payload["recipient"], payload["approval_reference"], payload["evidence_reference"])


def record_dissemination(payload: dict):
	return _SERVICE.record_dissemination(payload["dissemination_id"], payload.get("tenant_id", "default"), payload["assessment_id"], payload["audience"], payload["release_marking"], payload["approval_reference"], payload["evidence_reference"])


def record_review(payload: dict):
	return _SERVICE.record_review(payload["review_id"], payload.get("tenant_id", "default"), payload["reference_id"], payload["reviewer_id"], payload["status"], payload["evidence_reference"])


def register_radio_agent(payload: dict):
	return _SERVICE.register_radio_agent(payload["agent_id"], payload.get("tenant_id", "default"), payload["name"], payload["runtime"], payload["role"], payload.get("scope", "radio monitoring operations"))


def dashboard(payload: dict):
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))
