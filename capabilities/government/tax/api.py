"""Process-local API helpers for APG Tax Administration."""

from __future__ import annotations

try:
	from .service import TaxAdministrationService
except ImportError:  # pragma: no cover
	from service import TaxAdministrationService  # type: ignore


_SERVICE = TaxAdministrationService()


def service() -> TaxAdministrationService:
	return _SERVICE


def register_taxpayer(payload: dict):
	return _SERVICE.register_taxpayer(payload["registration_id"], payload.get("tenant_id", "default"), payload["tax_type"], payload["tax_pin"], payload["national_id"], payload["taxpayer_name"], payload["evidence_reference"], payload.get("status", "active"), payload.get("policy_attached", True))


def file_return(payload: dict):
	return _SERVICE.file_return(payload["return_id"], payload.get("tenant_id", "default"), payload["return_type"], payload["taxpayer_pin"], payload["period"], payload["gross_income"], payload["tax_liability"], payload["tax_paid"], payload["evidence_reference"], payload.get("status", "filed"))


def raise_assessment(payload: dict):
	return _SERVICE.raise_assessment(payload["assessment_id"], payload.get("tenant_id", "default"), payload["return_id"], payload["assessment_type"], payload["assessed_amount"], payload["assessor_id"], payload["assessment_date"], payload["evidence_reference"], payload.get("status", "draft"))


def file_objection(payload: dict):
	return _SERVICE.file_objection(payload["objection_id"], payload.get("tenant_id", "default"), payload["assessment_id"], payload["taxpayer_pin"], payload["grounds"], payload["amount_disputed"], payload["evidence_reference"], payload.get("filed_date", ""), payload.get("within_deadline", True))


def initiate_collection(payload: dict):
	return _SERVICE.initiate_collection(payload["collection_id"], payload.get("tenant_id", "default"), payload["taxpayer_pin"], payload["assessment_id"], payload["collection_method"], payload["amount_owed"], payload["demand_notice_reference"], payload["approval_reference"], payload["evidence_reference"])


def open_audit(payload: dict):
	return _SERVICE.open_audit(payload["audit_id"], payload.get("tenant_id", "default"), payload["taxpayer_pin"], payload["audit_type"], payload["auditor_id"], payload["period_under_review"], payload["evidence_reference"])


def complete_audit(payload: dict):
	return _SERVICE.complete_audit(payload["audit_id"], payload.get("tenant_id", "default"), payload["findings"])


def record_review(payload: dict):
	return _SERVICE.record_review(payload["review_id"], payload.get("tenant_id", "default"), payload["reference_id"], payload["reviewer_id"], payload["status"], payload["evidence_reference"])


def register_agent(payload: dict):
	return _SERVICE.register_agent(payload["agent_id"], payload.get("tenant_id", "default"), payload["name"], payload["runtime"], payload["role"], payload.get("scope", "tax administration operations"))


def validate_batch(payload: dict):
	return _SERVICE.validate_batch(payload.get("tenant_id", "default"), payload["item_count"], payload.get("event_stream", "bytewax"))


def dashboard(payload: dict):
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))
