"""Process-local API helpers for APG Intelligence Reporting."""

from __future__ import annotations

try:
	from .service import IntelligenceReportingService
except ImportError:  # pragma: no cover
	from service import IntelligenceReportingService  # type: ignore


_SERVICE = IntelligenceReportingService()


def service() -> IntelligenceReportingService:
	return _SERVICE


def record_authority(payload: dict):
	return _SERVICE.record_authority(payload["authority_id"], payload.get("tenant_id", "default"), payload["authority_type"], payload["scope_reference"], payload["classification"], payload["approver_id"], payload["expires_at"], payload["evidence_reference"], payload.get("policy_attached", True))


def record_workspace(payload: dict):
	return _SERVICE.record_workspace(payload["workspace_id"], payload.get("tenant_id", "default"), payload["workspace_type"], payload["name"], payload["classification"], payload["authority_id"], payload["evidence_reference"])


def record_template(payload: dict):
	return _SERVICE.record_template(payload["template_id"], payload.get("tenant_id", "default"), payload["workspace_id"], payload["template_type"], payload["template_reference"], payload["classification"], payload["evidence_reference"])


def record_product(payload: dict):
	return _SERVICE.record_product(payload["product_id"], payload.get("tenant_id", "default"), payload["template_id"], payload["product_type"], payload["title"], payload["author_id"], payload["classification"], payload["evidence_reference"])


def record_section(payload: dict):
	return _SERVICE.record_section(payload["section_id"], payload.get("tenant_id", "default"), payload["product_id"], payload["section_type"], payload["section_reference"], payload["confidence_score"], payload["evidence_reference"])


def record_citation(payload: dict):
	return _SERVICE.record_citation(payload["citation_id"], payload.get("tenant_id", "default"), payload["section_id"], payload["citation_type"], payload["source_reference"], payload["evidence_reference"])


def record_approval(payload: dict):
	return _SERVICE.record_approval(payload["approval_id"], payload.get("tenant_id", "default"), payload["product_id"], payload["approval_type"], payload["approver_id"], payload["status"], payload["evidence_reference"])


def record_distribution(payload: dict):
	return _SERVICE.record_distribution(payload["distribution_id"], payload.get("tenant_id", "default"), payload["product_id"], payload["distribution_type"], payload["recipient_reference"], payload["approval_reference"], payload["evidence_reference"])


def record_publication(payload: dict):
	return _SERVICE.record_publication(payload["publication_id"], payload.get("tenant_id", "default"), payload["distribution_id"], payload["publication_type"], payload["publication_reference"], payload["approval_reference"], payload["evidence_reference"])


def record_review(payload: dict):
	return _SERVICE.record_review(payload["review_id"], payload.get("tenant_id", "default"), payload["reference_id"], payload["reviewer_id"], payload["status"], payload["evidence_reference"])


def register_reporting_agent(payload: dict):
	return _SERVICE.register_reporting_agent(payload["agent_id"], payload.get("tenant_id", "default"), payload["name"], payload["runtime"], payload["role"], payload.get("scope", "intelligence reporting operations"))


def validate_agent_action(payload: dict):
	return _SERVICE.validate_agent_action(payload.get("tenant_id", "default"), payload.get("privileged_scope", False), payload.get("human_approval_recorded", False), payload.get("uncited_claim_scope", False), payload.get("classification_downgrade_scope", False), payload.get("source_fabrication_scope", False), payload.get("privacy_bypass_scope", False), payload.get("autonomous_publication_scope", False), payload.get("unapproved_distribution_scope", False))


def validate_batch(payload: dict):
	return _SERVICE.validate_batch(payload.get("tenant_id", "default"), payload["item_count"], payload.get("event_stream", "bytewax"))


def dashboard(payload: dict):
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))

