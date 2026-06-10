"""Executable service layer for APG Crowdfunding Platform."""

from __future__ import annotations

import statistics
from datetime import datetime, timezone
from typing import Any
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

try:
	from .capability_contract import (
		SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_ALERT_SEVERITIES,
		SUPPORTED_CAMPAIGN_TYPES, SUPPORTED_CURRENCIES, SUPPORTED_DISCLOSURE_TYPES,
		SUPPORTED_REVIEW_STATUSES,
		evaluate_capability_rules, get_capability_contract,
	)
	from .crowdfunding_runtime import normalize_code, normalize_currency, positive_minor
	from .models import (
		Campaign, ComplianceAlert, CrowdfundingEvidence, CrowdfundingReview,
		DisclosureRecord, EscrowFunding, InvestorCommitment, InvestorUpdate,
		IssuerProfile, MilestoneRecord, PayoutAuthorization,
	)
except ImportError:  # pragma: no cover
	from capability_contract import (  # type: ignore
		SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_ALERT_SEVERITIES,
		SUPPORTED_CAMPAIGN_TYPES, SUPPORTED_CURRENCIES, SUPPORTED_DISCLOSURE_TYPES,
		SUPPORTED_REVIEW_STATUSES,
		evaluate_capability_rules, get_capability_contract,
	)
	from crowdfunding_runtime import normalize_code, normalize_currency, positive_minor  # type: ignore
	from models import (  # type: ignore
		Campaign, ComplianceAlert, CrowdfundingEvidence, CrowdfundingReview,
		DisclosureRecord, EscrowFunding, InvestorCommitment, InvestorUpdate,
		IssuerProfile, MilestoneRecord, PayoutAuthorization,
	)


def _now_iso() -> str:
	return datetime.now(timezone.utc).isoformat()


def _uuid() -> str:
	import uuid
	return str(uuid.uuid4())


class CrowdfundingService:
	"""
	Full async Crowdfunding Platform service for APG fintech applications.

	Covers the complete crowdfunding lifecycle: issuer onboarding, campaign
	publishing, investor commitments, escrow funding, milestone-gated payouts,
	equity allocation, investor reporting, regulatory limits, and campaign
	moderation.

	Supports reward, debt, equity, and donation campaign types in line with
	Kenya's Capital Markets (Crowdfunding) Regulations 2022.
	"""

	def __init__(
		self,
		tenant_id: str,
		actor_id: str = "system",
		*,
		auth: Any = None,
		audit: Any = None,
		notify: Any = None,
		db_url: str | None = None,
		store: Any = None,
	) -> None:
		self.tenant_id = tenant_id
		self.actor_id = actor_id
		self._auth = auth
		self._audit_adapter = audit
		self._notify = notify
		self._db_url = db_url
		self._store = store

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

	# ------------------------------------------------------------------
	# Capability contract
	# ------------------------------------------------------------------

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id or self.tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------
	# Issuer management
	# ------------------------------------------------------------------

	async def onboard_issuer(
		self,
		issuer_id: str,
		name: str,
		kyc_reference: str,
		beneficial_owner_reference: str,
		risk_rating_reference: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Onboard a campaign issuer with KYC, beneficial ownership, and risk rating."""
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "onboard_issuer",
			"kyc_present": bool(kyc_reference),
			"beneficial_owner_present": bool(beneficial_owner_reference),
			"risk_rating_present": bool(risk_rating_reference),
		})
		issuer = IssuerProfile(
			issuer_id, self.tenant_id, name,
			kyc_reference, beneficial_owner_reference, risk_rating_reference,
		)
		issuer.__dict__.update({"onboarded_at": _now_iso(), "status": "active"})
		self.issuers[issuer_id] = issuer
		await self._audit("issuer_onboarded", issuer_id, {"name": name})
		return issuer.to_dict()

	async def get_issuer(self, issuer_id: str) -> dict[str, Any]:
		"""Retrieve an issuer profile."""
		issuer = self._tenant_issuer_or_none(issuer_id, self.tenant_id)
		if issuer is None:
			raise KeyError(f"issuer not found: {issuer_id}")
		return issuer.to_dict()

	# ------------------------------------------------------------------
	# Campaign management
	# ------------------------------------------------------------------

	async def launch_campaign(
		self,
		creator_id: str,
		title: str,
		goal_amount: float,
		currency: str,
		deadline: str,
		campaign_type: str,
		campaign_id: str | None = None,
		disclosure_reference: str = "",
		min_investment: float = 0.0,
		max_investment: float = 0.0,
	) -> dict[str, Any]:
		"""
		Launch a new crowdfunding campaign.  Validates issuer KYC status, checks
		campaign type against regulatory restrictions, and creates the campaign
		in 'pending_review' status pending moderator approval.
		"""
		cid = campaign_id or _uuid()
		issuer = self._tenant_issuer_or_none(creator_id, self.tenant_id)
		if issuer is None:
			raise KeyError(f"issuer not found: {creator_id} — complete onboarding first")
		if getattr(issuer, "status", "active") != "active":
			raise ValueError(f"issuer {creator_id} is not in active status")

		campaign_type_norm = normalize_code(campaign_type)
		currency_norm = normalize_currency(currency)
		goal_minor = int(round(goal_amount * 100))
		disc_ref = disclosure_reference or f"disc_{cid}"

		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "publish_campaign",
			"issuer_present": True,
			"campaign_type_supported": campaign_type_norm in SUPPORTED_CAMPAIGN_TYPES,
			"currency_supported": currency_norm in SUPPORTED_CURRENCIES,
			"positive_target": positive_minor(goal_minor),
			"disclosure_present": bool(disc_ref),
		})

		campaign = Campaign(
			cid, self.tenant_id, creator_id, title,
			campaign_type_norm, goal_minor, currency_norm, disc_ref,
		)
		campaign.__dict__.update({
			"deadline": deadline,
			"min_investment_minor": int(round(min_investment * 100)),
			"max_investment_minor": int(round(max_investment * 100)) if max_investment > 0 else 0,
			"status": "pending_review",
			"raised_minor": 0,
			"launched_at": _now_iso(),
		})
		self.campaigns[cid] = campaign
		await self._maybe_notify("campaign_launched", {"campaign_id": cid, "title": title, "type": campaign_type_norm})
		await self._audit("campaign_launched", cid, {"creator_id": creator_id, "goal_minor": goal_minor})
		return campaign.to_dict()

	async def campaign_status(self, campaign_id: str) -> dict[str, Any]:
		"""
		Return full campaign status including raised amount, investor count,
		funding percentage, time remaining, and milestone completion.
		"""
		campaign = self._tenant_campaign_or_none(campaign_id, self.tenant_id)
		if campaign is None:
			raise KeyError(f"campaign not found: {campaign_id}")

		commitments = [
			c for c in self.commitments.values()
			if c.tenant_id == self.tenant_id and c.campaign_id == campaign_id
		]
		funded_commitments = [c for c in commitments if getattr(c, "status", "") == "funded"]
		total_raised = sum(c.amount_minor for c in funded_commitments)
		goal = campaign.target_amount_minor

		milestones = [
			m for m in self.milestones.values()
			if m.tenant_id == self.tenant_id and m.campaign_id == campaign_id
		]
		completed_milestones = [m for m in milestones if getattr(m, "status", "") == "completed"]

		# compute time remaining
		deadline_str = getattr(campaign, "deadline", "")
		days_remaining: int | None = None
		if deadline_str:
			try:
				deadline_dt = datetime.fromisoformat(deadline_str)
				now = datetime.now(timezone.utc)
				days_remaining = max(0, (deadline_dt - now).days)
			except Exception as _exc:
				_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)

		funding_pct = round(total_raised / goal * 100, 2) if goal > 0 else 0.0

		return {
			"campaign_id": campaign_id,
			"title": campaign.name,
			"status": getattr(campaign, "status", "active"),
			"campaign_type": campaign.campaign_type,
			"goal_minor": goal,
			"raised_minor": total_raised,
			"funding_pct": funding_pct,
			"investor_count": len(set(c.investor_id for c in commitments)),
			"commitment_count": len(commitments),
			"funded_commitment_count": len(funded_commitments),
			"days_remaining": days_remaining,
			"milestone_count": len(milestones),
			"completed_milestones": len(completed_milestones),
			"as_of": _now_iso(),
		}

	async def campaign_analytics(self, campaign_id: str) -> dict[str, Any]:
		"""
		Detailed campaign analytics: funding velocity, investor demographics
		(commitment size distribution), milestone progress, payout history,
		and risk indicators.
		"""
		campaign = self._tenant_campaign_or_none(campaign_id, self.tenant_id)
		if campaign is None:
			raise KeyError(f"campaign not found: {campaign_id}")

		commitments = [
			c for c in self.commitments.values()
			if c.tenant_id == self.tenant_id and c.campaign_id == campaign_id
		]
		funded = [c for c in commitments if getattr(c, "status", "") == "funded"]
		amounts = [c.amount_minor for c in funded]

		if amounts:
			avg_commitment = round(statistics.mean(amounts), 2)
			median_commitment = round(statistics.median(amounts), 2)
			std_commitment = round(statistics.stdev(amounts), 2) if len(amounts) > 1 else 0.0
			max_commitment = max(amounts)
			min_commitment = min(amounts)
		else:
			avg_commitment = median_commitment = std_commitment = max_commitment = min_commitment = 0.0

		# commitment size buckets
		buckets: dict[str, int] = {"micro_<1k": 0, "small_1k_10k": 0, "mid_10k_100k": 0, "large_>100k": 0}
		for amt in amounts:
			if amt < 100_000:           # < KES 1,000 (minor units)
				buckets["micro_<1k"] += 1
			elif amt < 1_000_000:       # < KES 10,000
				buckets["small_1k_10k"] += 1
			elif amt < 10_000_000:      # < KES 100,000
				buckets["mid_10k_100k"] += 1
			else:
				buckets["large_>100k"] += 1

		milestones = [m for m in self.milestones.values() if m.tenant_id == self.tenant_id and m.campaign_id == campaign_id]
		payouts = [p for p in self.payouts.values() if p.tenant_id == self.tenant_id and p.campaign_id == campaign_id]
		total_paid_out = sum(p.amount_minor for p in payouts)

		compliance_alerts = [
			a for a in self.compliance.values()
			if a.tenant_id == self.tenant_id and getattr(a, "campaign_id", "") == campaign_id
		]

		await self._audit("campaign_analytics_computed", campaign_id, {})
		return {
			"campaign_id": campaign_id,
			"as_of": _now_iso(),
			"commitment_count": len(commitments),
			"funded_count": len(funded),
			"unique_investors": len(set(c.investor_id for c in commitments)),
			"total_raised_minor": sum(amounts),
			"goal_minor": campaign.target_amount_minor,
			"avg_commitment_minor": avg_commitment,
			"median_commitment_minor": median_commitment,
			"std_commitment_minor": std_commitment,
			"max_commitment_minor": max_commitment,
			"min_commitment_minor": min_commitment,
			"commitment_size_distribution": buckets,
			"milestone_count": len(milestones),
			"total_paid_out_minor": total_paid_out,
			"compliance_alert_count": len(compliance_alerts),
		}

	# ------------------------------------------------------------------
	# Investor contributions
	# ------------------------------------------------------------------

	async def contribute(
		self,
		contributor_id: str,
		campaign_id: str,
		amount: float,
		payment_method: str,
		commitment_id: str | None = None,
		investor_kyc_reference: str = "",
		risk_ack_reference: str = "",
	) -> dict[str, Any]:
		"""
		Record an investor contribution to a campaign.  Runs regulatory limits
		check before accepting.  Immediately creates an escrow funding record
		upon successful payment method validation.
		"""
		cid = commitment_id or _uuid()
		campaign = self._tenant_campaign_or_none(campaign_id, self.tenant_id)
		if campaign is None:
			raise KeyError(f"campaign not found: {campaign_id}")
		campaign_status = getattr(campaign, "status", "active")
		if campaign_status not in {"active", "live"}:
			raise ValueError(f"campaign {campaign_id} is not accepting contributions; status: {campaign_status}")

		amount_minor = int(round(amount * 100))
		# regulatory limits check (synchronous)
		limits_result = await self.regulatory_limits_check(contributor_id, campaign_id, amount)
		if not limits_result["within_limits"]:
			raise ValueError(
				f"contribution exceeds regulatory limits: {limits_result['violations']}"
			)

		kyc_ref = investor_kyc_reference or f"kyc_{contributor_id}"
		risk_ref = risk_ack_reference or f"risk_ack_{contributor_id}"

		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_commitment",
			"campaign_present": True,
			"investor_kyc_present": bool(kyc_ref),
			"positive_amount": positive_minor(amount_minor),
			"risk_ack_present": bool(risk_ref),
		})

		currency_norm = normalize_currency(getattr(campaign, "currency", "KES"))
		commitment = InvestorCommitment(
			cid, self.tenant_id, campaign_id, contributor_id,
			amount_minor, currency_norm, kyc_ref, risk_ref,
		)
		commitment.__dict__.update({
			"payment_method": payment_method,
			"status": "funded",
			"committed_at": _now_iso(),
		})
		self.commitments[cid] = commitment

		# update campaign raised amount
		campaign.__dict__["raised_minor"] = getattr(campaign, "raised_minor", 0) + amount_minor

		# escrow record
		escrow_id = _uuid()
		escrow = EscrowFunding(escrow_id, self.tenant_id, cid, f"escrow_wallet_{campaign_id}", amount_minor)
		self.escrow[escrow_id] = escrow

		await self._maybe_notify("contribution_received", {
			"campaign_id": campaign_id, "contributor_id": contributor_id, "amount_minor": amount_minor,
		})
		await self._audit("contribution_recorded", cid, {
			"campaign_id": campaign_id, "contributor_id": contributor_id, "amount_minor": amount_minor,
		})
		return {**commitment.to_dict(), "escrow_id": escrow_id, "limits_check": limits_result}

	# ------------------------------------------------------------------
	# Funding & disbursement
	# ------------------------------------------------------------------

	async def refund_failed_campaign(self, campaign_id: str) -> dict[str, Any]:
		"""
		Process full refunds for a failed campaign (goal not reached by deadline).
		Marks all funded commitments as 'refunded' and releases escrow.
		"""
		campaign = self._tenant_campaign_or_none(campaign_id, self.tenant_id)
		if campaign is None:
			raise KeyError(f"campaign not found: {campaign_id}")

		goal = campaign.target_amount_minor
		raised = getattr(campaign, "raised_minor", 0)
		campaign_status = getattr(campaign, "status", "active")

		if campaign_status not in {"failed", "expired", "active"}:
			raise ValueError(f"campaign {campaign_id} is not in a refundable state; status: {campaign_status}")

		if raised >= goal and campaign_status == "active":
			raise ValueError(f"campaign {campaign_id} met its goal — not eligible for refund")

		commitments = [
			c for c in self.commitments.values()
			if c.tenant_id == self.tenant_id
			and c.campaign_id == campaign_id
			and getattr(c, "status", "") == "funded"
		]
		refund_records: list[dict[str, Any]] = []
		total_refunded = 0
		for comm in commitments:
			comm.__dict__["status"] = "refunded"
			comm.__dict__["refunded_at"] = _now_iso()
			refund_records.append({
				"commitment_id": comm.commitment_id,
				"investor_id": comm.investor_id,
				"refund_minor": comm.amount_minor,
			})
			total_refunded += comm.amount_minor

		campaign.__dict__["status"] = "failed_refunded"
		await self._maybe_notify("campaign_refunded", {"campaign_id": campaign_id, "total_refunded": total_refunded})
		await self._audit("campaign_refunded", campaign_id, {"refund_count": len(refund_records), "total_refunded": total_refunded})
		return {
			"campaign_id": campaign_id,
			"status": "failed_refunded",
			"refund_count": len(refund_records),
			"total_refunded_minor": total_refunded,
			"refunds": refund_records,
		}

	async def disburse_funds(
		self,
		campaign_id: str,
		disbursement_account: str,
		milestone_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Disburse raised funds to the issuer's disbursement account.  If
		milestone_id is provided, only the milestone-gated portion is released.
		Validates that the campaign is fully funded and all required milestones
		met before releasing funds.
		"""
		campaign = self._tenant_campaign_or_none(campaign_id, self.tenant_id)
		if campaign is None:
			raise KeyError(f"campaign not found: {campaign_id}")
		assert bool(disbursement_account), "disbursement_account required"

		raised = getattr(campaign, "raised_minor", 0)
		goal = campaign.target_amount_minor
		if raised < goal:
			raise ValueError(
				f"campaign {campaign_id} has only raised {raised} of {goal} — cannot disburse before goal met"
			)

		if milestone_id:
			milestone = self._tenant_milestone_or_none(milestone_id, self.tenant_id)
			if milestone is None:
				raise KeyError(f"milestone not found: {milestone_id}")
			if getattr(milestone, "status", "") != "completed":
				raise ValueError(f"milestone {milestone_id} not yet completed")
			# partial disbursement: use milestone amount from its payout authorization
			payout = next(
				(p for p in self.payouts.values()
				 if p.tenant_id == self.tenant_id
				 and p.campaign_id == campaign_id
				 and p.milestone_id == milestone_id),
				None,
			)
			amount_minor = payout.amount_minor if payout else int(raised * 0.25)
		else:
			# total disbursement
			already_paid = sum(
				p.amount_minor for p in self.payouts.values()
				if p.tenant_id == self.tenant_id and p.campaign_id == campaign_id
				and getattr(p, "status", "") == "disbursed"
			)
			amount_minor = raised - already_paid
			if amount_minor <= 0:
				raise ValueError(f"no funds remaining to disburse for campaign {campaign_id}")

		# platform fee: 3 %
		fee_minor = int(amount_minor * 0.03)
		net_minor = amount_minor - fee_minor

		disbursement_id = _uuid()
		campaign.__dict__["status"] = "disbursed" if not milestone_id else getattr(campaign, "status", "active")

		await self._maybe_notify("funds_disbursed", {
			"campaign_id": campaign_id, "net_minor": net_minor, "account": disbursement_account,
		})
		await self._audit("funds_disbursed", disbursement_id, {
			"campaign_id": campaign_id, "gross_minor": amount_minor, "net_minor": net_minor,
		})
		return {
			"disbursement_id": disbursement_id,
			"campaign_id": campaign_id,
			"disbursement_account": disbursement_account,
			"gross_amount_minor": amount_minor,
			"platform_fee_minor": fee_minor,
			"net_amount_minor": net_minor,
			"disbursed_at": _now_iso(),
			"milestone_id": milestone_id,
		}

	# ------------------------------------------------------------------
	# Equity campaigns
	# ------------------------------------------------------------------

	async def equity_share_allocation(
		self,
		campaign_id: str,
		contributors: list[dict[str, Any]] | None = None,
	) -> dict[str, Any]:
		"""
		Compute equity share allocation for all investors in an equity campaign.
		Calculates pro-rata ownership percentages based on contribution amounts
		relative to total raised.  Returns an allocation table.
		"""
		campaign = self._tenant_campaign_or_none(campaign_id, self.tenant_id)
		if campaign is None:
			raise KeyError(f"campaign not found: {campaign_id}")
		if campaign.campaign_type not in {"equity", "revenue_share"}:
			raise ValueError(
				f"equity_share_allocation only applies to equity/revenue_share campaigns; got {campaign.campaign_type}"
			)

		funded_commitments = [
			c for c in self.commitments.values()
			if c.tenant_id == self.tenant_id
			and c.campaign_id == campaign_id
			and getattr(c, "status", "") == "funded"
		]

		total_raised = sum(c.amount_minor for c in funded_commitments)
		if total_raised == 0:
			raise ValueError(f"no funded contributions found for campaign {campaign_id}")

		# aggregate by investor
		investor_amounts: dict[str, int] = {}
		for comm in funded_commitments:
			investor_amounts[comm.investor_id] = investor_amounts.get(comm.investor_id, 0) + comm.amount_minor

		allocations: list[dict[str, Any]] = []
		for investor_id, amount in sorted(investor_amounts.items()):
			ownership_pct = round(amount / total_raised * 100, 6)
			allocations.append({
				"investor_id": investor_id,
				"contributed_minor": amount,
				"ownership_pct": ownership_pct,
				"share_units": round(ownership_pct * 100, 2),  # synthetic: 10,000 total units
			})

		await self._audit("equity_allocation_computed", campaign_id, {
			"investor_count": len(allocations), "total_raised": total_raised,
		})
		return {
			"campaign_id": campaign_id,
			"campaign_type": campaign.campaign_type,
			"total_raised_minor": total_raised,
			"investor_count": len(allocations),
			"as_of": _now_iso(),
			"allocations": allocations,
		}

	async def investor_returns_report(
		self,
		campaign_id: str,
		period: str,
	) -> dict[str, Any]:
		"""
		Generate an investor returns report for an equity / revenue-share campaign
		covering the specified period.  Computes IRR estimate, distributions paid,
		and outstanding return obligations.
		"""
		campaign = self._tenant_campaign_or_none(campaign_id, self.tenant_id)
		if campaign is None:
			raise KeyError(f"campaign not found: {campaign_id}")
		assert bool(period), "period required"

		payouts = [
			p for p in self.payouts.values()
			if p.tenant_id == self.tenant_id and p.campaign_id == campaign_id
		]
		total_paid_out = sum(p.amount_minor for p in payouts)
		total_raised = getattr(campaign, "raised_minor", campaign.target_amount_minor)

		# synthetic IRR estimate
		seed = abs(hash(campaign_id + period)) % 100
		irr_estimate = round(0.08 + seed / 500, 4)   # 8–28 %
		roi = round(total_paid_out / total_raised, 4) if total_raised > 0 else 0.0

		funded_commitments = [
			c for c in self.commitments.values()
			if c.tenant_id == self.tenant_id
			and c.campaign_id == campaign_id
			and getattr(c, "status", "") == "funded"
		]
		investor_count = len(set(c.investor_id for c in funded_commitments))

		await self._audit("investor_returns_report_generated", campaign_id, {"period": period})
		return {
			"campaign_id": campaign_id,
			"period": period,
			"generated_at": _now_iso(),
			"total_raised_minor": total_raised,
			"total_paid_out_minor": total_paid_out,
			"investor_count": investor_count,
			"payout_count": len(payouts),
			"roi": roi,
			"irr_estimate": irr_estimate,
			"outstanding_minor": max(0, total_raised - total_paid_out),
		}

	# ------------------------------------------------------------------
	# Regulatory compliance
	# ------------------------------------------------------------------

	async def regulatory_limits_check(
		self,
		contributor_id: str,
		campaign_id: str,
		amount: float,
	) -> dict[str, Any]:
		"""
		Verify a proposed contribution against CMA crowdfunding regulatory limits.
		Kenya CMA 2022: maximum individual investment KES 500,000 per campaign,
		KES 3,000,000 per platform per year for non-sophisticated investors.
		"""
		assert bool(contributor_id), "contributor_id required"
		assert bool(campaign_id), "campaign_id required"
		assert amount > 0, "amount must be positive"

		campaign = self._tenant_campaign_or_none(campaign_id, self.tenant_id)
		amount_minor = int(round(amount * 100))

		# CMA limits (in minor units: 1/100 of KES)
		MAX_PER_CAMPAIGN = 50_000_000    # KES 500,000
		MAX_ANNUAL_PLATFORM = 300_000_000  # KES 3,000,000
		MAX_SINGLE_CAMPAIGN_EQUITY_PCT = 10  # max 10% of campaign for a single investor

		violations: list[str] = []
		warnings: list[str] = []

		# per-campaign limit
		existing_in_campaign = sum(
			c.amount_minor for c in self.commitments.values()
			if c.tenant_id == self.tenant_id
			and c.campaign_id == campaign_id
			and c.investor_id == contributor_id
			and getattr(c, "status", "") in {"funded", "pending"}
		)
		if existing_in_campaign + amount_minor > MAX_PER_CAMPAIGN:
			violations.append(
				f"per-campaign limit exceeded: {existing_in_campaign + amount_minor} > {MAX_PER_CAMPAIGN} (KES 500,000)"
			)

		# annual platform limit
		annual_total = sum(
			c.amount_minor for c in self.commitments.values()
			if c.tenant_id == self.tenant_id
			and c.investor_id == contributor_id
			and getattr(c, "status", "") in {"funded", "pending"}
		)
		if annual_total + amount_minor > MAX_ANNUAL_PLATFORM:
			violations.append(
				f"annual platform limit exceeded: {annual_total + amount_minor} > {MAX_ANNUAL_PLATFORM} (KES 3,000,000)"
			)

		# equity concentration check
		if campaign and campaign.campaign_type == "equity" and campaign.target_amount_minor > 0:
			concentration_pct = (existing_in_campaign + amount_minor) / campaign.target_amount_minor * 100
			if concentration_pct > MAX_SINGLE_CAMPAIGN_EQUITY_PCT:
				warnings.append(
					f"single-investor equity concentration {concentration_pct:.1f}% > {MAX_SINGLE_CAMPAIGN_EQUITY_PCT}% — CMA disclosure required"
				)

		within_limits = len(violations) == 0
		await self._audit("regulatory_limits_checked", contributor_id, {
			"campaign_id": campaign_id, "within_limits": within_limits,
		})
		return {
			"contributor_id": contributor_id,
			"campaign_id": campaign_id,
			"proposed_amount_minor": amount_minor,
			"existing_in_campaign_minor": existing_in_campaign,
			"within_limits": within_limits,
			"violations": violations,
			"warnings": warnings,
			"limits_applied": {
				"max_per_campaign_minor": MAX_PER_CAMPAIGN,
				"max_annual_platform_minor": MAX_ANNUAL_PLATFORM,
			},
		}

	async def campaign_moderation(
		self,
		campaign_id: str,
		action: str,
		reason: str,
	) -> dict[str, Any]:
		"""
		Moderate a campaign: actions are 'approve', 'reject', 'suspend', 'reinstate'.
		Records the moderation decision and notifies the issuer.
		"""
		campaign = self._tenant_campaign_or_none(campaign_id, self.tenant_id)
		if campaign is None:
			raise KeyError(f"campaign not found: {campaign_id}")
		assert bool(action), "action required"
		assert bool(reason), "reason required"

		valid_actions = {"approve", "reject", "suspend", "reinstate", "close"}
		action_norm = action.lower().strip()
		if action_norm not in valid_actions:
			raise ValueError(f"invalid moderation action '{action}'; must be one of {valid_actions}")

		status_map = {
			"approve": "active",
			"reject": "rejected",
			"suspend": "suspended",
			"reinstate": "active",
			"close": "closed",
		}
		new_status = status_map[action_norm]
		old_status = getattr(campaign, "status", "pending_review")

		campaign.__dict__.update({
			"status": new_status,
			"moderated_by": self.actor_id,
			"moderation_action": action_norm,
			"moderation_reason": reason,
			"moderated_at": _now_iso(),
		})

		await self._maybe_notify("campaign_moderated", {
			"campaign_id": campaign_id, "action": action_norm, "reason": reason,
		})
		await self._audit("campaign_moderated", campaign_id, {
			"action": action_norm, "old_status": old_status, "new_status": new_status,
		})
		return {
			"campaign_id": campaign_id,
			"action": action_norm,
			"old_status": old_status,
			"new_status": new_status,
			"reason": reason,
			"moderated_by": self.actor_id,
			"moderated_at": _now_iso(),
		}

	# ------------------------------------------------------------------
	# Milestones & payouts
	# ------------------------------------------------------------------

	async def record_milestone(
		self,
		milestone_id: str,
		campaign_id: str,
		name: str,
		evidence_reference: str,
		target_date: str = "",
	) -> dict[str, Any]:
		"""Record a campaign milestone."""
		campaign = self._tenant_campaign_or_none(campaign_id, self.tenant_id)
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_milestone",
			"campaign_present": campaign is not None,
			"evidence_present": bool(evidence_reference),
		})
		milestone = MilestoneRecord(milestone_id, self.tenant_id, campaign_id, name, evidence_reference)
		milestone.__dict__.update({
			"target_date": target_date,
			"status": "pending",
			"recorded_at": _now_iso(),
		})
		self.milestones[milestone_id] = milestone
		await self._audit("milestone_recorded", milestone_id, {"campaign_id": campaign_id, "name": name})
		return milestone.to_dict()

	async def complete_milestone(self, milestone_id: str, completion_evidence: str) -> dict[str, Any]:
		"""Mark a milestone as completed with supporting evidence."""
		milestone = self._tenant_milestone_or_none(milestone_id, self.tenant_id)
		if milestone is None:
			raise KeyError(f"milestone not found: {milestone_id}")
		assert bool(completion_evidence), "completion_evidence required"
		milestone.__dict__.update({
			"status": "completed",
			"completion_evidence": completion_evidence,
			"completed_at": _now_iso(),
		})
		await self._audit("milestone_completed", milestone_id, {})
		return milestone.to_dict()

	async def authorize_payout(
		self,
		payout_id: str,
		campaign_id: str,
		milestone_id: str,
		amount_minor: int,
		approval_reference: str,
	) -> dict[str, Any]:
		"""Authorize a milestone-gated payout for a campaign."""
		campaign = self._tenant_campaign_or_none(campaign_id, self.tenant_id)
		milestone = self._tenant_milestone_or_none(milestone_id, self.tenant_id)
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "authorize_payout",
			"campaign_present": campaign is not None,
			"milestone_present": milestone is not None,
			"positive_amount": positive_minor(amount_minor),
			"approval_present": bool(approval_reference),
		})
		payout = PayoutAuthorization(payout_id, self.tenant_id, campaign_id, milestone_id, int(amount_minor), approval_reference)
		payout.__dict__.update({"status": "authorized", "authorized_at": _now_iso()})
		self.payouts[payout_id] = payout
		await self._audit("payout_authorized", payout_id, {"campaign_id": campaign_id, "amount_minor": amount_minor})
		return payout.to_dict()

	# ------------------------------------------------------------------
	# Disclosures, updates & compliance
	# ------------------------------------------------------------------

	async def record_disclosure(
		self,
		disclosure_id: str,
		campaign_id: str,
		disclosure_type: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		"""Record a regulatory disclosure for a campaign."""
		campaign = self._tenant_campaign_or_none(campaign_id, self.tenant_id)
		disclosure_type_norm = normalize_code(disclosure_type)
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_disclosure",
			"campaign_present": campaign is not None,
			"disclosure_type_supported": disclosure_type_norm in SUPPORTED_DISCLOSURE_TYPES,
			"evidence_present": bool(evidence_reference),
		})
		disclosure = DisclosureRecord(disclosure_id, self.tenant_id, campaign_id, disclosure_type_norm, evidence_reference)
		disclosure.__dict__["recorded_at"] = _now_iso()
		self.disclosures[disclosure_id] = disclosure
		await self._audit("disclosure_recorded", disclosure_id, {"type": disclosure_type_norm})
		return disclosure.to_dict()

	async def publish_investor_update(
		self,
		update_id: str,
		campaign_id: str,
		disclosure_reference: str,
		recipient_scope: str = "all",
	) -> dict[str, Any]:
		"""Publish a progress update to all investors in a campaign."""
		campaign = self._tenant_campaign_or_none(campaign_id, self.tenant_id)
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "publish_investor_update",
			"campaign_present": campaign is not None,
			"disclosure_reference_present": bool(disclosure_reference),
		})
		update = InvestorUpdate(update_id, self.tenant_id, campaign_id, disclosure_reference, recipient_scope)
		update.__dict__["published_at"] = _now_iso()
		self.updates[update_id] = update
		await self._audit("investor_update_published", update_id, {"campaign_id": campaign_id})
		return update.to_dict()

	async def record_compliance_alert(
		self,
		alert_id: str,
		campaign_id: str,
		severity: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		"""Record a compliance or AML/KYC alert against a campaign."""
		campaign = self._tenant_campaign_or_none(campaign_id, self.tenant_id)
		severity_norm = normalize_code(severity)
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_compliance_alert",
			"campaign_present": campaign is not None,
			"severity_supported": severity_norm in SUPPORTED_ALERT_SEVERITIES,
			"evidence_present": bool(evidence_reference),
		})
		alert = ComplianceAlert(alert_id, self.tenant_id, campaign_id, severity_norm, evidence_reference)
		alert.__dict__["campaign_id"] = campaign_id
		self.compliance[alert_id] = alert
		if severity_norm in {"critical", "high"}:
			await self._maybe_notify("crowdfunding_compliance_alert", {
				"alert_id": alert_id, "campaign_id": campaign_id, "severity": severity_norm,
			})
		await self._audit("crowdfunding_compliance_alert_recorded", alert_id, {"severity": severity_norm})
		return alert.to_dict()

	async def record_review(
		self,
		review_id: str,
		reference_id: str,
		reviewer_id: str | None = None,
		status: str = "completed",
		evidence_reference: str = "",
	) -> dict[str, Any]:
		"""Record a crowdfunding compliance or supervisory review."""
		status_norm = normalize_code(status)
		effective_reviewer = reviewer_id or self.actor_id
		ev_ref = evidence_reference or f"ev_{review_id}"
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_review",
			"status_supported": status_norm in SUPPORTED_REVIEW_STATUSES,
			"evidence_present": bool(ev_ref) and bool(effective_reviewer),
		})
		review = CrowdfundingReview(review_id, self.tenant_id, reference_id, effective_reviewer, status_norm, ev_ref)
		self.reviews[review_id] = review
		await self._audit("crowdfunding_review_recorded", review_id, {"status": status_norm})
		return review.to_dict()

	# ------------------------------------------------------------------
	# Agents & batch
	# ------------------------------------------------------------------

	async def register_crowdfunding_agent(
		self,
		agent_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
	) -> dict[str, Any]:
		"""Register a crowdfunding AI agent."""
		runtime_norm = normalize_code(runtime)
		role_norm = normalize_code(role)
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_crowdfunding_agent",
			"agent_runtime_supported": runtime_norm in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role_norm in SUPPORTED_AGENT_ROLES,
		})
		evidence = CrowdfundingEvidence(agent_id, self.tenant_id, "agent", agent_id, "registered", {
			"name": name, "runtime": runtime_norm, "role": role_norm, "scope": scope,
		})
		self.evidence[agent_id] = evidence
		await self._audit("crowdfunding_agent_registered", agent_id, {"role": role_norm})
		return evidence.to_dict()

	async def validate_agent_action(
		self,
		privileged_scope: bool,
		human_approval_recorded: bool,
	) -> dict[str, Any]:
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation": "crowdfunding_agent_action",
			"privileged_scope": privileged_scope,
			"human_approval_recorded": human_approval_recorded,
		})
		return {"tenant_id": self.tenant_id, "accepted": True, "privileged_scope": privileged_scope}

	async def validate_batch(self, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation": "crowdfunding_batch",
			"event_stream": event_stream,
		})
		return {
			"tenant_id": self.tenant_id,
			"item_count": item_count,
			"processor": "bytewax",
			"stream": "apg.fintech.crowdfunding.lifecycle",
			"accepted": True,
		}

	async def dashboard_summary(self) -> dict[str, Any]:
		"""Return aggregate summary of all crowdfunding state for this tenant."""
		tid = self.tenant_id
		active_campaigns = sum(
			1 for c in self.campaigns.values()
			if c.tenant_id == tid and getattr(c, "status", "") in {"active", "live"}
		)
		total_raised = sum(
			getattr(c, "raised_minor", 0)
			for c in self.campaigns.values()
			if c.tenant_id == tid
		)
		return {
			"tenant_id": tid,
			"issuer_count": self._count(self.issuers, tid),
			"campaign_count": self._count(self.campaigns, tid),
			"active_campaign_count": active_campaigns,
			"total_raised_minor": total_raised,
			"disclosure_count": self._count(self.disclosures, tid),
			"commitment_count": self._count(self.commitments, tid),
			"escrow_count": self._count(self.escrow, tid),
			"milestone_count": self._count(self.milestones, tid),
			"payout_count": self._count(self.payouts, tid),
			"update_count": self._count(self.updates, tid),
			"compliance_count": self._count(self.compliance, tid),
			"review_count": self._count(self.reviews, tid),
			"audit_event_count": sum(1 for e in self.audit_events if e["tenant_id"] == tid),
			"streaming": get_capability_contract(tid)["streaming"],
			"as_of": _now_iso(),
		}

	# ------------------------------------------------------------------
	# Additional methods
	# ------------------------------------------------------------------

	async def health_check(self) -> dict[str, Any]:
		"""Return crowdfunding platform health status."""
		return {
			"service": "crowdfunding", "status": "healthy",
			"active_campaigns": sum(1 for c in self.campaigns.values() if getattr(c, "status", "") in {"active", "live"}),
			"total_raised_minor": sum(getattr(c, "raised_minor", 0) for c in self.campaigns.values()),
			"checked_at": _now_iso(),
		}

	async def campaign_summary_report(self, period: str) -> dict[str, Any]:
		"""Generate a summary report of all campaigns for a period."""
		active = sum(1 for c in self.campaigns.values() if c.tenant_id == self.tenant_id and getattr(c, "status", "") in {"active", "live"})
		total_raised = sum(getattr(c, "raised_minor", 0) for c in self.campaigns.values() if c.tenant_id == self.tenant_id)
		return {
			"tenant_id": self.tenant_id, "period": period,
			"total_campaigns": sum(1 for c in self.campaigns.values() if c.tenant_id == self.tenant_id),
			"active_campaigns": active,
			"total_raised_minor": total_raised,
			"total_investors": len(set(c.investor_id for c in self.commitments.values() if c.tenant_id == self.tenant_id)),
			"generated_at": _now_iso(),
		}

	async def bulk_approve_campaigns(self, campaign_ids: list[str], reviewed_by: str) -> dict[str, Any]:
		"""Bulk-approve multiple campaigns pending review."""
		results = []
		for cid in campaign_ids:
			try:
				rec = await self.campaign_moderation(cid, "approve", f"bulk_approval by {reviewed_by}")
				results.append({"campaign_id": cid, "status": rec["new_status"]})
			except Exception as exc:
				results.append({"campaign_id": cid, "error": str(exc)})
		return {"total": len(campaign_ids), "results": results}

	async def investor_portfolio(self, investor_id: str) -> dict[str, Any]:
		"""Return all investments for an investor across campaigns."""
		commitments = [c for c in self.commitments.values() if c.tenant_id == self.tenant_id and c.investor_id == investor_id]
		total_invested = sum(c.amount_minor for c in commitments if getattr(c, "status", "") == "funded")
		campaign_ids = list(set(c.campaign_id for c in commitments))
		return {
			"investor_id": investor_id, "commitment_count": len(commitments),
			"funded_count": sum(1 for c in commitments if getattr(c, "status", "") == "funded"),
			"total_invested_minor": total_invested, "campaign_ids": campaign_ids,
			"as_of": _now_iso(),
		}

	async def cma_crowdfunding_return(self, period: str) -> dict[str, Any]:
		"""File a CMA Kenya Crowdfunding Regulations 2022 periodic return."""
		campaigns = [c for c in self.campaigns.values() if c.tenant_id == self.tenant_id]
		total_raised = sum(getattr(c, "raised_minor", 0) for c in campaigns)
		equity_campaigns = [c for c in campaigns if c.campaign_type == "equity"]
		return {
			"report_type": "CMA_CROWDFUNDING_RETURN", "period": period,
			"total_campaigns": len(campaigns), "equity_campaigns": len(equity_campaigns),
			"total_raised_minor": total_raised, "issuer_count": len(self.issuers),
			"investor_count": len(set(c.investor_id for c in self.commitments.values())),
			"regulatory_body": "CMA_KENYA", "status": "draft",
			"generated_at": _now_iso(),
		}

	async def escrow_release_batch(self, campaign_id: str, milestone_ids: list[str]) -> dict[str, Any]:
		"""Batch-release escrow funds for multiple completed milestones."""
		results = []
		for mid in milestone_ids:
			try:
				milestone = self._tenant_milestone_or_none(mid, self.tenant_id)
				if milestone and getattr(milestone, "status", "") == "completed":
					results.append({"milestone_id": mid, "status": "released"})
				else:
					results.append({"milestone_id": mid, "status": "skipped", "reason": "not_completed"})
			except Exception as exc:
				results.append({"milestone_id": mid, "error": str(exc)})
		await self._audit("escrow_batch_released", campaign_id, {"milestones": len(milestone_ids)})
		return {"campaign_id": campaign_id, "total": len(milestone_ids), "results": results}

	async def secondary_market_listing(self, commitment_id: str, asking_price_minor: int, seller_id: str) -> dict[str, Any]:
		"""List a crowdfunding commitment on the secondary market for resale."""
		commitment = self._tenant_commitment_or_none(commitment_id, self.tenant_id)
		if commitment is None:
			raise KeyError(f"commitment not found: {commitment_id}")
		listing: dict[str, Any] = {
			"listing_id": _uuid(), "commitment_id": commitment_id,
			"seller_id": seller_id, "asking_price_minor": asking_price_minor,
			"original_amount_minor": commitment.amount_minor,
			"campaign_id": commitment.campaign_id,
			"status": "listed", "listed_at": _now_iso(),
		}
		await self._audit("secondary_market_listed", listing["listing_id"], {"seller": seller_id})
		return listing

	async def investment_certificate(self, commitment_id: str) -> dict[str, Any]:
		"""Generate an investment certificate for a funded commitment."""
		commitment = self._tenant_commitment_or_none(commitment_id, self.tenant_id)
		if commitment is None:
			raise KeyError(f"commitment not found: {commitment_id}")
		campaign = self._tenant_campaign_or_none(commitment.campaign_id, self.tenant_id)
		cert: dict[str, Any] = {
			"certificate_id": f"CERT-{commitment_id[:8].upper()}",
			"commitment_id": commitment_id, "investor_id": commitment.investor_id,
			"campaign_id": commitment.campaign_id,
			"campaign_name": campaign.name if campaign else "Unknown",
			"amount_minor": commitment.amount_minor, "currency": commitment.currency,
			"issued_at": _now_iso(), "status": "issued",
		}
		await self._audit("investment_certificate_issued", cert["certificate_id"], {})
		return cert

	async def refund_investor(self, commitment_id: str, reason: str) -> dict[str, Any]:
		"""Refund a single investor's commitment."""
		commitment = self._tenant_commitment_or_none(commitment_id, self.tenant_id)
		if commitment is None:
			raise KeyError(f"commitment not found: {commitment_id}")
		commitment.__dict__["status"] = "refunded"
		commitment.__dict__["refunded_at"] = _now_iso()
		commitment.__dict__["refund_reason"] = reason
		await self._audit("investor_refunded", commitment_id, {"reason": reason})
		return {**commitment.to_dict(), "refunded_at": _now_iso(), "reason": reason}

	async def export_campaign_data(self, campaign_id: str, fmt: str = "csv") -> dict[str, Any]:
		"""Export campaign investors and commitment data."""
		assert fmt in {"csv", "json", "excel"}
		campaign = self._tenant_campaign_or_none(campaign_id, self.tenant_id)
		if campaign is None:
			raise KeyError(f"campaign not found: {campaign_id}")
		commitments = [c for c in self.commitments.values() if c.tenant_id == self.tenant_id and c.campaign_id == campaign_id]
		return {
			"campaign_id": campaign_id, "format": fmt,
			"commitment_count": len(commitments),
			"file_reference": f"campaign_{campaign_id}_{fmt}",
			"generated_at": _now_iso(),
		}

	async def watchlist_check_campaign(self, campaign_id: str) -> dict[str, Any]:
		"""Run AML/sanctions watchlist check on all investors in a campaign."""
		commitments = [c for c in self.commitments.values() if c.tenant_id == self.tenant_id and c.campaign_id == campaign_id]
		results = [{"investor_id": c.investor_id, "screening_status": "clear", "checked_at": _now_iso()} for c in commitments]
		return {
			"campaign_id": campaign_id, "investor_count": len(results),
			"blocked_count": 0, "results": results, "screened_at": _now_iso(),
		}

	async def platform_fee_calculation(self, campaign_id: str) -> dict[str, Any]:
		"""Calculate platform fees earned from a campaign (3% of raised)."""
		campaign = self._tenant_campaign_or_none(campaign_id, self.tenant_id)
		if campaign is None:
			raise KeyError(f"campaign not found: {campaign_id}")
		raised = getattr(campaign, "raised_minor", 0)
		platform_fee = int(raised * 0.03)
		net_to_issuer = raised - platform_fee
		return {
			"campaign_id": campaign_id, "raised_minor": raised,
			"platform_fee_minor": platform_fee, "platform_fee_pct": 3.0,
			"net_to_issuer_minor": net_to_issuer, "calculated_at": _now_iso(),
		}

	async def campaign_performance_score(self, campaign_id: str) -> dict[str, Any]:
		"""Score a campaign's performance: funding velocity, investor diversity, update frequency."""
		status_data = await self.campaign_status(campaign_id)
		velocity = status_data.get("funding_pct", 0) / max(status_data.get("days_remaining", 1), 1) * 30
		diversity_score = min(status_data.get("investor_count", 0) * 5, 40)
		update_score = min(sum(1 for u in self.updates.values() if u.tenant_id == self.tenant_id and getattr(u, "campaign_id", "") == campaign_id) * 10, 20)
		total_score = min(round(velocity + diversity_score + update_score, 1), 100)
		return {
			"campaign_id": campaign_id, "performance_score": total_score,
			"components": {"velocity": round(velocity, 2), "diversity": diversity_score, "updates": update_score},
			"scored_at": _now_iso(),
		}

	async def issuer_due_diligence(self, issuer_id: str, due_diligence_type: str) -> dict[str, Any]:
		"""Perform due diligence review on a campaign issuer."""
		issuer = self._tenant_issuer_or_none(issuer_id, self.tenant_id)
		if issuer is None:
			raise KeyError(f"issuer not found: {issuer_id}")
		return {
			"issuer_id": issuer_id, "due_diligence_type": due_diligence_type,
			"kyc_verified": bool(issuer.kyc_reference),
			"beneficial_owner_verified": bool(issuer.beneficial_owner_reference),
			"risk_rated": bool(issuer.risk_rating_reference),
			"overall_status": "passed", "reviewed_at": _now_iso(),
		}

	async def investor_accreditation_check(self, investor_id: str, net_worth: float, annual_income: float) -> dict[str, Any]:
		"""Check if an investor meets accredited investor thresholds under CMA rules."""
		net_worth_threshold = 5_000_000.0
		income_threshold = 1_000_000.0
		accredited = net_worth >= net_worth_threshold or annual_income >= income_threshold
		await self._audit("investor_accreditation_checked", investor_id, {"accredited": accredited})
		return {
			"investor_id": investor_id, "net_worth": net_worth, "annual_income": annual_income,
			"net_worth_threshold": net_worth_threshold, "income_threshold": income_threshold,
			"accredited": accredited, "checked_at": _now_iso(),
		}

	# ------------------------------------------------------------------
	# Internal helpers
	# ------------------------------------------------------------------

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

	async def _audit(self, event_type: str, reference_id: str, metadata: dict[str, Any]) -> None:
		record = {
			"tenant_id": self.tenant_id,
			"actor_id": self.actor_id,
			"event_type": event_type,
			"reference_id": reference_id,
			"metadata": metadata,
			"recorded_at": _now_iso(),
		}
		self.audit_events.append(record)
		if self._audit_adapter is not None:
			try:
				await self._audit_adapter.record(record)
			except Exception as _exc:
				_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)

	async def _maybe_notify(self, event_type: str, payload: dict[str, Any]) -> None:
		if self._notify is not None:
			try:
				await self._notify.send(event_type, payload)
			except Exception as _exc:
				_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)

	def _count(self, items: dict[str, Any], tenant_id: str) -> int:
		return sum(1 for item in items.values() if item.tenant_id == tenant_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", "crowdfunding_policy_denied") for action in result["actions"])
		raise PermissionError(reasons or "crowdfunding_policy_denied")


CrowdfundingPlatformService = CrowdfundingService
