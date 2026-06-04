"""Process-local API helpers for APG Government Contracts & Procurement."""

from __future__ import annotations

try:
	from .service import ProcurementService
except ImportError:  # pragma: no cover
	from service import ProcurementService  # type: ignore


_SERVICE = ProcurementService()


def service() -> ProcurementService:
	return _SERVICE


def publish_tender(payload: dict):
	return _SERVICE.publish_tender(payload["tender_id"], payload.get("tenant_id", "default"), payload["procurement_method"], payload["ppda_threshold"], payload["title"], payload["description"], payload["approver_id"], payload["evidence_reference"], payload.get("justification", ""), payload.get("status", "draft"), payload.get("policy_attached", True))


def record_evaluation(payload: dict):
	return _SERVICE.record_evaluation(payload["evaluation_id"], payload.get("tenant_id", "default"), payload["tender_id"], payload["bidder_id"], payload["criteria"], payload["score"], payload["evaluator_id"], payload["evidence_reference"])


def record_award(payload: dict):
	return _SERVICE.record_award(payload["award_id"], payload.get("tenant_id", "default"), payload["tender_id"], payload["awarded_to"], payload["awarded_amount"], payload["ppda_notification_reference"], payload["evidence_reference"])


def record_contract(payload: dict):
	return _SERVICE.record_contract(payload["contract_id"], payload.get("tenant_id", "default"), payload["award_id"], payload["contract_type"], payload["contract_value"], payload["start_date"], payload["end_date"], payload["signed_by"], payload["contractor_reference"], payload["evidence_reference"], payload.get("status", "draft"))


def record_variation(payload: dict):
	return _SERVICE.record_variation(payload["variation_id"], payload.get("tenant_id", "default"), payload["contract_id"], payload["variation_type"], payload["description"], payload["value_change"], payload["approval_reference"], payload["ppda_notification_reference"], payload["evidence_reference"])


def record_performance(payload: dict):
	return _SERVICE.record_performance(payload["performance_id"], payload.get("tenant_id", "default"), payload["contract_id"], payload["performance_status"], payload["reviewer_id"], payload["period"], payload["narrative"], payload["evidence_reference"])


def debar_bidder(payload: dict):
	return _SERVICE.debar_bidder(payload["debarment_id"], payload.get("tenant_id", "default"), payload["bidder_id"], payload["reason"], payload["debarred_until"], payload["evidence_reference"])


def record_review(payload: dict):
	return _SERVICE.record_review(payload["review_id"], payload.get("tenant_id", "default"), payload["reference_id"], payload["reviewer_id"], payload["status"], payload["evidence_reference"])


def register_agent(payload: dict):
	return _SERVICE.register_agent(payload["agent_id"], payload.get("tenant_id", "default"), payload["name"], payload["runtime"], payload["role"], payload.get("scope", "procurement operations"))


def validate_batch(payload: dict):
	return _SERVICE.validate_batch(payload.get("tenant_id", "default"), payload["item_count"], payload.get("event_stream", "bytewax"))


def dashboard(payload: dict):
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))
