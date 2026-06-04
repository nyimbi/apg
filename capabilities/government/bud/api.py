"""Process-local API helpers for APG Budget Management."""

from __future__ import annotations

try:
	from .service import BudgetManagementService
except ImportError:  # pragma: no cover
	from service import BudgetManagementService  # type: ignore


_SERVICE = BudgetManagementService()


def service() -> BudgetManagementService:
	return _SERVICE


def record_budget(payload: dict):
	return _SERVICE.record_budget(payload["budget_id"], payload.get("tenant_id", "default"), payload["budget_type"], payload["fund_source"], payload["vote_id"], payload["total_amount"], payload["fiscal_year"], payload["approver_id"], payload["evidence_reference"], payload.get("status", "draft"), payload.get("policy_attached", True))


def record_vote(payload: dict):
	return _SERVICE.record_vote(payload["vote_id"], payload.get("tenant_id", "default"), payload["vote_code"], payload["vote_type"], payload["budget_id"], payload["allocated_amount"], payload["evidence_reference"])


def record_revision(payload: dict):
	return _SERVICE.record_revision(payload["revision_id"], payload.get("tenant_id", "default"), payload["budget_id"], payload["revision_type"], payload["amount_change"], payload["approval_reference"], payload["treasury_notification_reference"], payload["evidence_reference"], payload.get("status", "draft"))


def record_commitment(payload: dict):
	return _SERVICE.record_commitment(payload["commitment_id"], payload.get("tenant_id", "default"), payload["vote_id"], payload["commitment_type"], payload["amount"], payload["approval_reference"], payload["supplier_reference"], payload["evidence_reference"], payload.get("status", "open"))


def record_expenditure(payload: dict):
	return _SERVICE.record_expenditure(payload["expenditure_id"], payload.get("tenant_id", "default"), payload["commitment_id"], payload["expenditure_type"], payload["amount"], payload["approval_reference"], payload["payee_reference"], payload["evidence_reference"])


def generate_report(payload: dict):
	return _SERVICE.generate_report(payload["report_id"], payload.get("tenant_id", "default"), payload["budget_id"], payload["report_type"], payload["fiscal_period"], payload["author_id"], payload["evidence_reference"], payload.get("status", "draft"))


def record_approval(payload: dict):
	return _SERVICE.record_approval(payload["approval_id"], payload.get("tenant_id", "default"), payload["reference_id"], payload["approver_id"], payload["status"], payload["evidence_reference"])


def record_review(payload: dict):
	return _SERVICE.record_review(payload["review_id"], payload.get("tenant_id", "default"), payload["reference_id"], payload["reviewer_id"], payload["status"], payload["evidence_reference"])


def register_agent(payload: dict):
	return _SERVICE.register_agent(payload["agent_id"], payload.get("tenant_id", "default"), payload["name"], payload["runtime"], payload["role"], payload.get("scope", "budget management operations"))


def validate_agent_action(payload: dict):
	return _SERVICE.validate_agent_action(payload.get("tenant_id", "default"), payload.get("privileged_scope", False), payload.get("human_approval_recorded", False), payload.get("evidence_fabrication_scope", False))


def validate_batch(payload: dict):
	return _SERVICE.validate_batch(payload.get("tenant_id", "default"), payload["item_count"], payload.get("event_stream", "bytewax"))


def dashboard(payload: dict):
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))
