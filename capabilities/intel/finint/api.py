"""Process-local API helpers for APG Financial Intelligence."""

from __future__ import annotations

try:
	from .service import FinancialIntelligenceService
except ImportError:  # pragma: no cover
	from service import FinancialIntelligenceService  # type: ignore


_SERVICE = FinancialIntelligenceService()


def service() -> FinancialIntelligenceService:
	return _SERVICE


def record_authority(payload: dict):
	return _SERVICE.record_authority(payload["authority_id"], payload.get("tenant_id", "default"), payload["authority_type"], payload["scope_reference"], payload["classification"], payload["approver_id"], payload["expires_at"], payload["evidence_reference"], payload.get("policy_attached", True))


def register_source(payload: dict):
	return _SERVICE.register_source(payload["source_id"], payload.get("tenant_id", "default"), payload["source_type"], payload["jurisdiction"], payload["owner_id"], payload["authority_id"], payload["evidence_reference"])


def record_subject(payload: dict):
	return _SERVICE.record_subject(payload["subject_id"], payload.get("tenant_id", "default"), payload["subject_type"], payload["subject_reference"], payload["risk_tier"], payload["authority_id"], payload["evidence_reference"])


def record_transaction(payload: dict):
	return _SERVICE.record_transaction(payload["transaction_id"], payload.get("tenant_id", "default"), payload["source_id"], payload["subject_id"], payload["transaction_reference"], payload["amount"], payload["currency"], payload["transaction_type"], payload["occurred_at"], payload["evidence_reference"])


def record_pattern(payload: dict):
	return _SERVICE.record_pattern(payload["pattern_id"], payload.get("tenant_id", "default"), payload["transaction_id"], payload["pattern_type"], payload["confidence_score"], payload["analyst_id"], payload["evidence_reference"])


def record_risk(payload: dict):
	return _SERVICE.record_risk(payload["assessment_id"], payload.get("tenant_id", "default"), payload["pattern_id"], payload["risk_type"], payload["risk_level"], payload["confidence_score"], payload["analyst_id"], payload["evidence_reference"])


def record_referral(payload: dict):
	return _SERVICE.record_referral(payload["referral_id"], payload.get("tenant_id", "default"), payload["assessment_id"], payload["referral_type"], payload["recipient"], payload["approval_reference"], payload["evidence_reference"])


def record_dissemination(payload: dict):
	return _SERVICE.record_dissemination(payload["dissemination_id"], payload.get("tenant_id", "default"), payload["assessment_id"], payload["audience"], payload["release_marking"], payload["approval_reference"], payload["evidence_reference"])


def record_review(payload: dict):
	return _SERVICE.record_review(payload["review_id"], payload.get("tenant_id", "default"), payload["reference_id"], payload["reviewer_id"], payload["status"], payload["evidence_reference"])


def register_finint_agent(payload: dict):
	return _SERVICE.register_finint_agent(payload["agent_id"], payload.get("tenant_id", "default"), payload["name"], payload["runtime"], payload["role"], payload.get("scope", "finint operations"))


def dashboard(payload: dict):
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))
