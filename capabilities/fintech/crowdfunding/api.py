"""Dependency-light API helpers for APG Crowdfunding Platform."""

from __future__ import annotations

from typing import Any

try:
	from .service import CrowdfundingPlatformService
except ImportError:  # pragma: no cover
	from service import CrowdfundingPlatformService  # type: ignore


_SERVICE = CrowdfundingPlatformService()


def service() -> CrowdfundingPlatformService:
	return _SERVICE


def onboard_issuer(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.onboard_issuer(payload["issuer_id"], payload["tenant_id"], payload["name"], payload["kyc_reference"], payload["beneficial_owner_reference"], payload["risk_rating_reference"], payload.get("policy_attached", True))


def publish_campaign(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.publish_campaign(payload["campaign_id"], payload["tenant_id"], payload["issuer_id"], payload["name"], payload["campaign_type"], payload["target_amount_minor"], payload["currency"], payload["disclosure_reference"])


def record_disclosure(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.record_disclosure(payload["disclosure_id"], payload["tenant_id"], payload["campaign_id"], payload["disclosure_type"], payload["evidence_reference"])


def record_commitment(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.record_commitment(payload["commitment_id"], payload["tenant_id"], payload["campaign_id"], payload["investor_id"], payload["amount_minor"], payload["currency"], payload["investor_kyc_reference"], payload["risk_ack_reference"])


def record_escrow_funding(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.record_escrow_funding(payload["funding_id"], payload["tenant_id"], payload["commitment_id"], payload["wallet_reference"], payload["amount_minor"])


def record_milestone(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.record_milestone(payload["milestone_id"], payload["tenant_id"], payload["campaign_id"], payload["name"], payload["evidence_reference"])


def authorize_payout(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.authorize_payout(payload["payout_id"], payload["tenant_id"], payload["campaign_id"], payload["milestone_id"], payload["amount_minor"], payload["approval_reference"])


def publish_investor_update(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.publish_investor_update(payload["update_id"], payload["tenant_id"], payload["campaign_id"], payload["disclosure_reference"], payload["recipient_scope"])


def record_compliance_alert(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.record_compliance_alert(payload["alert_id"], payload["tenant_id"], payload["campaign_id"], payload["severity"], payload["evidence_reference"])


def record_review(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.record_review(payload["review_id"], payload["tenant_id"], payload["reference_id"], payload["reviewer_id"], payload["status"], payload["evidence_reference"])


def register_crowdfunding_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.register_crowdfunding_agent(payload["agent_id"], payload["tenant_id"], payload["name"], payload["runtime"], payload["role"], payload.get("scope", "crowdfunding platform review"))


def dashboard(tenant_id: str) -> dict[str, Any]:
	return _SERVICE.dashboard_summary(tenant_id)
