"""Dependency-light API helpers for APG InsurTech."""

from __future__ import annotations

from typing import Any

try:
	from .service import InsurTechService
except ImportError:  # pragma: no cover
	from service import InsurTechService  # type: ignore


_SERVICE = InsurTechService()


def service() -> InsurTechService: return _SERVICE
def onboard_policyholder(payload: dict[str, Any]) -> dict[str, Any]: return _SERVICE.onboard_policyholder(payload["policyholder_id"], payload["tenant_id"], payload["name"], payload["kyc_reference"], payload["contact_reference"], payload["risk_profile_reference"], payload.get("policy_attached", True))
def publish_product(payload: dict[str, Any]) -> dict[str, Any]: return _SERVICE.publish_product(payload["product_id"], payload["tenant_id"], payload["name"], payload["product_line"], payload["coverage_terms_reference"], payload["pricing_reference"])
def generate_quote(payload: dict[str, Any]) -> dict[str, Any]: return _SERVICE.generate_quote(payload["quote_id"], payload["tenant_id"], payload["policyholder_id"], payload["product_id"], payload["premium_minor"], payload["currency"], payload["underwriting_reference"])
def bind_policy(payload: dict[str, Any]) -> dict[str, Any]: return _SERVICE.bind_policy(payload["policy_id"], payload["tenant_id"], payload["quote_id"], payload["effective_date"], payload["payment_reference"])
def record_premium(payload: dict[str, Any]) -> dict[str, Any]: return _SERVICE.record_premium(payload["premium_id"], payload["tenant_id"], payload["policy_id"], payload["amount_minor"], payload["currency"], payload["payment_reference"])
def open_claim(payload: dict[str, Any]) -> dict[str, Any]: return _SERVICE.open_claim(payload["claim_id"], payload["tenant_id"], payload["policy_id"], payload["claim_type"], payload["amount_minor"], payload["loss_date"], payload["evidence_reference"])
def record_document(payload: dict[str, Any]) -> dict[str, Any]: return _SERVICE.record_document(payload["document_id"], payload["tenant_id"], payload["reference_id"], payload["document_type"], payload["evidence_reference"])
def record_risk_assessment(payload: dict[str, Any]) -> dict[str, Any]: return _SERVICE.record_risk_assessment(payload["assessment_id"], payload["tenant_id"], payload["policyholder_id"], payload["score"], payload["source_reference"])
def record_reinsurance_attachment(payload: dict[str, Any]) -> dict[str, Any]: return _SERVICE.record_reinsurance_attachment(payload["attachment_id"], payload["tenant_id"], payload["policy_id"], payload["treaty_reference"], payload["share_percent"])
def record_compliance_alert(payload: dict[str, Any]) -> dict[str, Any]: return _SERVICE.record_compliance_alert(payload["alert_id"], payload["tenant_id"], payload["reference_id"], payload["severity"], payload["evidence_reference"])
def record_review(payload: dict[str, Any]) -> dict[str, Any]: return _SERVICE.record_review(payload["review_id"], payload["tenant_id"], payload["reference_id"], payload["reviewer_id"], payload["status"], payload["evidence_reference"])
def register_insurance_agent(payload: dict[str, Any]) -> dict[str, Any]: return _SERVICE.register_insurance_agent(payload["agent_id"], payload["tenant_id"], payload["name"], payload["runtime"], payload["role"], payload.get("scope", "insurance review"))
def dashboard(tenant_id: str) -> dict[str, Any]: return _SERVICE.dashboard_summary(tenant_id)
