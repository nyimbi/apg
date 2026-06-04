"""Process-local API helpers for APG Electoral & Civil Registration."""

from __future__ import annotations

try:
	from .service import ElectoralService
except ImportError:  # pragma: no cover
	from service import ElectoralService  # type: ignore


_SERVICE = ElectoralService()


def service() -> ElectoralService:
	return _SERVICE


def register_voter(payload: dict):
	return _SERVICE.register_voter(payload["registration_id"], payload.get("tenant_id", "default"), payload["registration_type"], payload["national_id"], payload["biometric_reference"], payload["constituency"], payload["polling_station_id"], payload["evidence_reference"], payload.get("policy_attached", True))


def run_deduplication(payload: dict):
	return _SERVICE.run_deduplication(payload["dedup_id"], payload.get("tenant_id", "default"), payload["registration_id"], payload["method"], payload["match_score"], payload["duplicate_detected"], payload.get("resolution", ""), payload.get("resolved_by", ""))


def assign_polling_station(payload: dict):
	return _SERVICE.assign_polling_station(payload["station_id"], payload.get("tenant_id", "default"), payload["station_type"], payload["name"], payload["constituency"], payload["location_reference"], payload["capacity"], payload["presiding_officer_id"], payload["evidence_reference"])


def create_election(payload: dict):
	return _SERVICE.create_election(payload["election_id"], payload.get("tenant_id", "default"), payload["election_type"], payload["name"], payload["polling_date"], payload["nomination_deadline"], payload["constituency"])


def collate_result(payload: dict):
	return _SERVICE.collate_result(payload["result_id"], payload.get("tenant_id", "default"), payload["election_id"], payload["polling_station_id"], payload["candidate_id"], payload["votes_cast"], payload["rejected_votes"], payload["presiding_officer_id"], payload["evidence_reference"], payload.get("status", "provisional"))


def register_civil_event(payload: dict):
	return _SERVICE.register_civil_event(payload["event_id"], payload.get("tenant_id", "default"), payload["registration_type"], payload["subject_id"], payload["registrar_id"], payload["witness_id"], payload["event_date"], payload["evidence_reference"], payload.get("status", "registered"))


def record_verification(payload: dict):
	return _SERVICE.record_verification(payload["verification_id"], payload.get("tenant_id", "default"), payload["registration_id"], payload["status"], payload["biometric_match_score"], payload["evidence_reference"])


def record_review(payload: dict):
	return _SERVICE.record_review(payload["review_id"], payload.get("tenant_id", "default"), payload["reference_id"], payload["reviewer_id"], payload["status"], payload["evidence_reference"])


def register_agent(payload: dict):
	return _SERVICE.register_agent(payload["agent_id"], payload.get("tenant_id", "default"), payload["name"], payload["runtime"], payload["role"], payload.get("scope", "electoral operations"))


def validate_batch(payload: dict):
	return _SERVICE.validate_batch(payload.get("tenant_id", "default"), payload["item_count"], payload.get("event_stream", "bytewax"))


def dashboard(payload: dict):
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))
