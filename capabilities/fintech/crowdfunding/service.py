"""Executable service layer for APG Crowdfunding Platform."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_ALERT_SEVERITIES, SUPPORTED_CAMPAIGN_TYPES, SUPPORTED_CURRENCIES, SUPPORTED_DISCLOSURE_TYPES, SUPPORTED_REVIEW_STATUSES, evaluate_capability_rules, get_capability_contract
	from .crowdfunding_runtime import normalize_code, normalize_currency, positive_minor
	from .models import Campaign, ComplianceAlert, CrowdfundingEvidence, CrowdfundingReview, DisclosureRecord, EscrowFunding, InvestorCommitment, InvestorUpdate, IssuerProfile, MilestoneRecord, PayoutAuthorization
except ImportError:  # pragma: no cover
	from capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_ALERT_SEVERITIES, SUPPORTED_CAMPAIGN_TYPES, SUPPORTED_CURRENCIES, SUPPORTED_DISCLOSURE_TYPES, SUPPORTED_REVIEW_STATUSES, evaluate_capability_rules, get_capability_contract  # type: ignore
	from crowdfunding_runtime import normalize_code, normalize_currency, positive_minor  # type: ignore
	from models import Campaign, ComplianceAlert, CrowdfundingEvidence, CrowdfundingReview, DisclosureRecord, EscrowFunding, InvestorCommitment, InvestorUpdate, IssuerProfile, MilestoneRecord, PayoutAuthorization  # type: ignore


class CrowdfundingPlatformService:
	"""In-memory Crowdfunding Platform runtime for generated APG applications."""

	def __init__(self) -> None:
		self.issuers: dict[str, IssuerProfile] = {}
		self.campaigns: dict[str, Campaign] = {}
		self.disclosures: dict[str, DisclosureRecord] = {}
		self.commitments: dict[str, InvestorCommitment] = {}
		self.escrow: dict[str, EscrowFunding] = {}
		self.milestones: dict[str, MilestoneRecord] = {}
		self.payouts: dict[str, PayoutAuthorization] = {}
		self.updates: dict[str, InvestorUpdate] = {}
		self.compliance: dict[str, ComplianceAlert] = {}
		self.reviews: dict[str, CrowdfundingReview] = {}
		self.evidence: dict[str, CrowdfundingEvidence] = {}
		self.audit_events: list[dict[str, Any]] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def onboard_issuer(self, issuer_id: str, tenant_id: str, name: str, kyc_reference: str, beneficial_owner_reference: str, risk_rating_reference: str, policy_attached: bool = True) -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "onboard_issuer", "kyc_present": bool(kyc_reference), "beneficial_owner_present": bool(beneficial_owner_reference), "risk_rating_present": bool(risk_rating_reference)})
		issuer = IssuerProfile(issuer_id, tenant_id, name, kyc_reference, beneficial_owner_reference, risk_rating_reference)
		self.issuers[issuer_id] = issuer
		self._audit(tenant_id, "issuer_onboarded", issuer_id)
		return issuer.to_dict()

	def publish_campaign(self, campaign_id: str, tenant_id: str, issuer_id: str, name: str, campaign_type: str, target_amount_minor: int, currency: str, disclosure_reference: str) -> dict[str, Any]:
		issuer = self._tenant_issuer_or_none(issuer_id, tenant_id)
		campaign_type = normalize_code(campaign_type)
		currency = normalize_currency(currency)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "publish_campaign", "issuer_present": issuer is not None, "campaign_type_supported": campaign_type in SUPPORTED_CAMPAIGN_TYPES, "currency_supported": currency in SUPPORTED_CURRENCIES, "positive_target": positive_minor(target_amount_minor), "disclosure_present": bool(disclosure_reference)})
		campaign = Campaign(campaign_id, tenant_id, issuer_id, name, campaign_type, int(target_amount_minor), currency, disclosure_reference)
		self.campaigns[campaign_id] = campaign
		self._audit(tenant_id, "campaign_published", campaign_id)
		return campaign.to_dict()

	def record_disclosure(self, disclosure_id: str, tenant_id: str, campaign_id: str, disclosure_type: str, evidence_reference: str) -> dict[str, Any]:
		campaign = self._tenant_campaign_or_none(campaign_id, tenant_id)
		disclosure_type = normalize_code(disclosure_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_disclosure", "campaign_present": campaign is not None, "disclosure_type_supported": disclosure_type in SUPPORTED_DISCLOSURE_TYPES, "evidence_present": bool(evidence_reference)})
		disclosure = DisclosureRecord(disclosure_id, tenant_id, campaign_id, disclosure_type, evidence_reference)
		self.disclosures[disclosure_id] = disclosure
		self._audit(tenant_id, "disclosure_recorded", disclosure_id)
		return disclosure.to_dict()

	def record_commitment(self, commitment_id: str, tenant_id: str, campaign_id: str, investor_id: str, amount_minor: int, currency: str, investor_kyc_reference: str, risk_ack_reference: str) -> dict[str, Any]:
		campaign = self._tenant_campaign_or_none(campaign_id, tenant_id)
		currency = normalize_currency(currency)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_commitment", "campaign_present": campaign is not None, "investor_kyc_present": bool(investor_kyc_reference), "positive_amount": positive_minor(amount_minor), "risk_ack_present": bool(risk_ack_reference)})
		commitment = InvestorCommitment(commitment_id, tenant_id, campaign_id, investor_id, int(amount_minor), currency, investor_kyc_reference, risk_ack_reference)
		self.commitments[commitment_id] = commitment
		self._audit(tenant_id, "investor_commitment_recorded", commitment_id)
		return commitment.to_dict()

	def record_escrow_funding(self, funding_id: str, tenant_id: str, commitment_id: str, wallet_reference: str, amount_minor: int) -> dict[str, Any]:
		commitment = self._tenant_commitment_or_none(commitment_id, tenant_id)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_escrow_funding", "funded_commitment_present": commitment is not None, "wallet_reference_present": bool(wallet_reference), "positive_amount": positive_minor(amount_minor)})
		assert commitment is not None
		commitment.status = "funded"
		funding = EscrowFunding(funding_id, tenant_id, commitment_id, wallet_reference, int(amount_minor))
		self.escrow[funding_id] = funding
		self._audit(tenant_id, "escrow_funding_recorded", funding_id)
		return funding.to_dict()

	def record_milestone(self, milestone_id: str, tenant_id: str, campaign_id: str, name: str, evidence_reference: str) -> dict[str, Any]:
		campaign = self._tenant_campaign_or_none(campaign_id, tenant_id)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_milestone", "campaign_present": campaign is not None, "evidence_present": bool(evidence_reference)})
		milestone = MilestoneRecord(milestone_id, tenant_id, campaign_id, name, evidence_reference)
		self.milestones[milestone_id] = milestone
		self._audit(tenant_id, "milestone_recorded", milestone_id)
		return milestone.to_dict()

	def authorize_payout(self, payout_id: str, tenant_id: str, campaign_id: str, milestone_id: str, amount_minor: int, approval_reference: str) -> dict[str, Any]:
		campaign = self._tenant_campaign_or_none(campaign_id, tenant_id)
		milestone = self._tenant_milestone_or_none(milestone_id, tenant_id)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "authorize_payout", "campaign_present": campaign is not None, "milestone_present": milestone is not None, "positive_amount": positive_minor(amount_minor), "approval_present": bool(approval_reference)})
		payout = PayoutAuthorization(payout_id, tenant_id, campaign_id, milestone_id, int(amount_minor), approval_reference)
		self.payouts[payout_id] = payout
		self._audit(tenant_id, "payout_authorized", payout_id)
		return payout.to_dict()

	def publish_investor_update(self, update_id: str, tenant_id: str, campaign_id: str, disclosure_reference: str, recipient_scope: str) -> dict[str, Any]:
		campaign = self._tenant_campaign_or_none(campaign_id, tenant_id)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "publish_investor_update", "campaign_present": campaign is not None, "disclosure_reference_present": bool(disclosure_reference)})
		update = InvestorUpdate(update_id, tenant_id, campaign_id, disclosure_reference, recipient_scope)
		self.updates[update_id] = update
		self._audit(tenant_id, "investor_update_published", update_id)
		return update.to_dict()

	def record_compliance_alert(self, alert_id: str, tenant_id: str, campaign_id: str, severity: str, evidence_reference: str) -> dict[str, Any]:
		campaign = self._tenant_campaign_or_none(campaign_id, tenant_id)
		severity = normalize_code(severity)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_compliance_alert", "campaign_present": campaign is not None, "severity_supported": severity in SUPPORTED_ALERT_SEVERITIES, "evidence_present": bool(evidence_reference)})
		alert = ComplianceAlert(alert_id, tenant_id, campaign_id, severity, evidence_reference)
		self.compliance[alert_id] = alert
		self._audit(tenant_id, "crowdfunding_compliance_alert_recorded", alert_id)
		return alert.to_dict()

	def record_review(self, review_id: str, tenant_id: str, reference_id: str, reviewer_id: str, status: str, evidence_reference: str) -> dict[str, Any]:
		status = normalize_code(status)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_review", "status_supported": status in SUPPORTED_REVIEW_STATUSES, "evidence_present": bool(evidence_reference) and bool(reviewer_id)})
		review = CrowdfundingReview(review_id, tenant_id, reference_id, reviewer_id, status, evidence_reference)
		self.reviews[review_id] = review
		self._audit(tenant_id, "crowdfunding_review_recorded", review_id)
		return review.to_dict()

	def register_crowdfunding_agent(self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		runtime = normalize_code(runtime)
		role = normalize_code(role)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_crowdfunding_agent", "agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES, "agent_role_supported": role in SUPPORTED_AGENT_ROLES})
		evidence = CrowdfundingEvidence(agent_id, tenant_id, "agent", agent_id, "registered", {"name": name, "runtime": runtime, "role": role, "scope": scope})
		self.evidence[agent_id] = evidence
		self._audit(tenant_id, "crowdfunding_agent_registered", agent_id)
		return evidence.to_dict()

	def validate_agent_action(self, tenant_id: str, privileged_scope: bool, human_approval_recorded: bool) -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "crowdfunding_agent_action", "privileged_scope": privileged_scope, "human_approval_recorded": human_approval_recorded})
		return {"tenant_id": tenant_id, "accepted": True, "privileged_scope": privileged_scope}

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "crowdfunding_batch", "event_stream": event_stream})
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.fintech.crowdfunding.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {"tenant_id": tenant_id, "issuer_count": self._count(self.issuers, tenant_id), "campaign_count": self._count(self.campaigns, tenant_id), "disclosure_count": self._count(self.disclosures, tenant_id), "commitment_count": self._count(self.commitments, tenant_id), "escrow_count": self._count(self.escrow, tenant_id), "milestone_count": self._count(self.milestones, tenant_id), "payout_count": self._count(self.payouts, tenant_id), "update_count": self._count(self.updates, tenant_id), "compliance_count": self._count(self.compliance, tenant_id), "review_count": self._count(self.reviews, tenant_id), "audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id), "streaming": get_capability_contract(tenant_id)["streaming"]}

	def _tenant_issuer_or_none(self, item_id: str, tenant_id: str) -> IssuerProfile | None:
		item = self.issuers.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _tenant_campaign_or_none(self, item_id: str, tenant_id: str) -> Campaign | None:
		item = self.campaigns.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _tenant_commitment_or_none(self, item_id: str, tenant_id: str) -> InvestorCommitment | None:
		item = self.commitments.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _tenant_milestone_or_none(self, item_id: str, tenant_id: str) -> MilestoneRecord | None:
		item = self.milestones.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({"tenant_id": tenant_id, "event_type": event_type, "reference_id": reference_id})

	def _count(self, items: dict[str, Any], tenant_id: str) -> int:
		return sum(1 for item in items.values() if item.tenant_id == tenant_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", "crowdfunding_policy_denied") for action in result["actions"])
		raise PermissionError(reasons or "crowdfunding_policy_denied")


CrowdfundingService = CrowdfundingPlatformService
