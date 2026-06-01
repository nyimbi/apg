"""Process-local API helpers for APG Real-Time Monitoring."""

from __future__ import annotations

try:
	from .service import RealTimeMonitoringService
except ImportError:  # pragma: no cover
	from service import RealTimeMonitoringService  # type: ignore


_SERVICE = RealTimeMonitoringService()


def service() -> RealTimeMonitoringService:
	return _SERVICE


def record_authority(payload: dict):
	return _SERVICE.record_authority(payload["authority_id"], payload.get("tenant_id", "default"), payload["authority_type"], payload["scope_reference"], payload["classification"], payload["approver_id"], payload["expires_at"], payload["evidence_reference"], payload.get("policy_attached", True))


def record_policy(payload: dict):
	return _SERVICE.record_policy(payload["policy_id"], payload.get("tenant_id", "default"), payload["policy_type"], payload["name"], payload["severity_floor"], payload["authority_id"], payload["evidence_reference"])


def register_source(payload: dict):
	return _SERVICE.register_source(payload["source_id"], payload.get("tenant_id", "default"), payload["source_type"], payload["source_reference"], payload["owner_id"], payload["authority_id"], payload["access_review_reference"], payload["evidence_reference"])


def record_watch(payload: dict):
	return _SERVICE.record_watch(payload["watch_id"], payload.get("tenant_id", "default"), payload["policy_id"], payload["source_id"], payload["watch_type"], payload["watch_expression"], payload["retention_class"], payload["evidence_reference"])


def record_event(payload: dict):
	return _SERVICE.record_event(payload["event_id"], payload.get("tenant_id", "default"), payload["watch_id"], payload["event_type"], payload["event_reference"], payload["event_fingerprint"], payload["observed_at"], payload["confidence_score"], payload["evidence_reference"])


def record_signal(payload: dict):
	return _SERVICE.record_signal(payload["signal_id"], payload.get("tenant_id", "default"), payload["event_id"], payload["signal_type"], payload["severity"], payload["confidence_score"], payload["analyst_id"], payload["evidence_reference"])


def record_incident(payload: dict):
	return _SERVICE.record_incident(payload["incident_id"], payload.get("tenant_id", "default"), payload["signal_id"], payload["incident_type"], payload["severity"], payload["confidence_score"], payload["analyst_id"], payload["evidence_reference"])


def record_referral(payload: dict):
	return _SERVICE.record_referral(payload["referral_id"], payload.get("tenant_id", "default"), payload["incident_id"], payload["referral_type"], payload["recipient"], payload["approval_reference"], payload["evidence_reference"])


def record_dissemination(payload: dict):
	return _SERVICE.record_dissemination(payload["dissemination_id"], payload.get("tenant_id", "default"), payload["incident_id"], payload["audience"], payload["release_marking"], payload["approval_reference"], payload["evidence_reference"])


def record_review(payload: dict):
	return _SERVICE.record_review(payload["review_id"], payload.get("tenant_id", "default"), payload["reference_id"], payload["reviewer_id"], payload["status"], payload["evidence_reference"])


def register_monitoring_agent(payload: dict):
	return _SERVICE.register_monitoring_agent(payload["agent_id"], payload.get("tenant_id", "default"), payload["name"], payload["runtime"], payload["role"], payload.get("scope", "real-time monitoring operations"))


def dashboard(payload: dict):
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))
