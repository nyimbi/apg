"""Process-local API helpers for APG FinTech Compliance Automation."""

from __future__ import annotations

try:
	from .service import ComplianceAutomationService
except ImportError:  # pragma: no cover
	from service import ComplianceAutomationService  # type: ignore


_SERVICE = ComplianceAutomationService()


def service() -> ComplianceAutomationService:
	return _SERVICE


def register_obligation(payload: dict):
	return _SERVICE.register_obligation(payload["obligation_id"], payload.get("tenant_id", "default"), payload["framework"], payload["obligation_type"], payload["title"], payload["owner_id"], payload["evidence_reference"], payload["effective_date"])


def map_control(payload: dict):
	return _SERVICE.map_control(payload["control_id"], payload.get("tenant_id", "default"), payload["obligation_id"], payload["control_type"], payload["owner_id"], payload["evidence_reference"], payload["frequency"])


def record_check(payload: dict):
	return _SERVICE.record_check(payload["check_id"], payload.get("tenant_id", "default"), payload["obligation_id"], payload["control_id"], payload["check_type"], payload["subject_reference"], payload["result"], payload.get("evidence_reference", ""))


def attach_evidence(payload: dict):
	return _SERVICE.attach_evidence(payload["evidence_id"], payload.get("tenant_id", "default"), payload["reference_id"], payload["evidence_type"], payload["source_reference"], payload["retention_days"])


def record_attestation(payload: dict):
	return _SERVICE.record_attestation(payload["attestation_id"], payload.get("tenant_id", "default"), payload["obligation_id"], payload["attestor_id"], payload["status"], payload["evidence_reference"])


def open_issue(payload: dict):
	return _SERVICE.open_issue(payload["issue_id"], payload.get("tenant_id", "default"), payload["obligation_id"], payload["severity"], payload["owner_id"], payload["evidence_reference"], payload["due_date"])


def record_remediation(payload: dict):
	return _SERVICE.record_remediation(payload["remediation_id"], payload.get("tenant_id", "default"), payload["issue_id"], payload["owner_id"], payload["plan_reference"], payload.get("high_impact", False), payload.get("approval_reference", ""))


def publish_report(payload: dict):
	return _SERVICE.publish_report(payload["report_id"], payload.get("tenant_id", "default"), payload["report_type"], payload["framework"], payload["period"], payload["evidence_reference"], payload["approver_id"])


def record_review(payload: dict):
	return _SERVICE.record_review(payload["review_id"], payload.get("tenant_id", "default"), payload["reference_id"], payload["reviewer_id"], payload["status"], payload["evidence_reference"])


def register_compliance_agent(payload: dict):
	return _SERVICE.register_compliance_agent(payload["agent_id"], payload.get("tenant_id", "default"), payload["name"], payload["runtime"], payload["role"], payload.get("scope", "compliance review"))


def dashboard(payload: dict):
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))
