"""Process-local API helpers for APG Licensing & Permits."""

from __future__ import annotations

try:
	from .service import LicensingService
except ImportError:  # pragma: no cover
	from service import LicensingService  # type: ignore


_SERVICE = LicensingService()


def service() -> LicensingService:
	return _SERVICE


def submit_application(payload: dict):
	return _SERVICE.submit_application(payload["application_id"], payload.get("tenant_id", "default"), payload["licence_type"], payload["applicant_id"], payload["business_registration"], payload["evidence_reference"], payload.get("fee_paid", False), payload.get("policy_attached", True))


def issue_licence(payload: dict):
	return _SERVICE.issue_licence(payload["licence_id"], payload.get("tenant_id", "default"), payload["application_id"], payload["licence_type"], payload["licence_number"], payload["holder_id"], payload["issued_date"], payload["expiry_date"], payload["evidence_reference"])


def schedule_inspection(payload: dict):
	return _SERVICE.schedule_inspection(payload["inspection_id"], payload.get("tenant_id", "default"), payload["licence_id"], payload["inspection_type"], payload["inspector_id"], payload["scheduled_date"], payload["evidence_reference"])


def record_inspection_outcome(payload: dict):
	return _SERVICE.record_inspection_outcome(payload["inspection_id"], payload.get("tenant_id", "default"), payload["outcome"], payload["findings"])


def renew_licence(payload: dict):
	return _SERVICE.renew_licence(payload["renewal_id"], payload.get("tenant_id", "default"), payload["licence_id"], payload["renewal_type"], payload["new_expiry_date"], payload["evidence_reference"], payload.get("renewal_fee_paid", False))


def collect_fee(payload: dict):
	return _SERVICE.collect_fee(payload["fee_id"], payload.get("tenant_id", "default"), payload["application_id"], payload["fee_type"], payload["amount"], payload.get("currency", "KES"), payload["receipt_number"])


def revoke_licence(payload: dict):
	return _SERVICE.revoke_licence(payload["revocation_id"], payload.get("tenant_id", "default"), payload["licence_id"], payload["reason"], payload["approval_reference"], payload["evidence_reference"], payload.get("notice_served", False))


def record_review(payload: dict):
	return _SERVICE.record_review(payload["review_id"], payload.get("tenant_id", "default"), payload["reference_id"], payload["reviewer_id"], payload["status"], payload["evidence_reference"])


def register_agent(payload: dict):
	return _SERVICE.register_agent(payload["agent_id"], payload.get("tenant_id", "default"), payload["name"], payload["runtime"], payload["role"], payload.get("scope", "licensing operations"))


def validate_batch(payload: dict):
	return _SERVICE.validate_batch(payload.get("tenant_id", "default"), payload["item_count"], payload.get("event_stream", "bytewax"))


def dashboard(payload: dict):
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))
