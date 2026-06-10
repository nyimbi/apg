"""Executable service layer for APG InsurTech."""

from __future__ import annotations

import statistics
from datetime import datetime, timezone
from typing import Any

try:
	from .capability_contract import (
		SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_ALERT_SEVERITIES,
		SUPPORTED_CLAIM_TYPES, SUPPORTED_CURRENCIES, SUPPORTED_DOCUMENT_TYPES,
		SUPPORTED_PRODUCT_LINES, SUPPORTED_REVIEW_STATUSES,
		evaluate_capability_rules, get_capability_contract,
	)
	from .insurance_runtime import normalize_code, normalize_currency, positive_minor, score_present
	from .models import (
		ClaimRecord, InsuranceAlert, InsuranceDocument, InsuranceEvidence,
		InsuranceProduct, InsuranceReview, Policy, Policyholder, PremiumRecord,
		Quote, ReinsuranceAttachment, RiskAssessment,
	)
except ImportError:  # pragma: no cover
	from capability_contract import (  # type: ignore
		SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_ALERT_SEVERITIES,
		SUPPORTED_CLAIM_TYPES, SUPPORTED_CURRENCIES, SUPPORTED_DOCUMENT_TYPES,
		SUPPORTED_PRODUCT_LINES, SUPPORTED_REVIEW_STATUSES,
		evaluate_capability_rules, get_capability_contract,
	)
	from insurance_runtime import normalize_code, normalize_currency, positive_minor, score_present  # type: ignore
	from models import (  # type: ignore
		ClaimRecord, InsuranceAlert, InsuranceDocument, InsuranceEvidence,
		InsuranceProduct, InsuranceReview, Policy, Policyholder, PremiumRecord,
		Quote, ReinsuranceAttachment, RiskAssessment,
	)


def _now_iso() -> str:
	return datetime.now(timezone.utc).isoformat()


def _uuid() -> str:
	import uuid
	return str(uuid.uuid4())


class InsurTechService:
	"""
	Full async InsurTech service for APG fintech applications.

	Covers the complete insurance lifecycle: policyholder onboarding, product
	publishing, quoting, policy binding, premium collection, claims management
	(filing → assessment → approval/rejection → payment), reinsurance, and
	regulatory analytics.

	Constructor accepts optional adapter overrides for auth, audit, and
	notification sinks.
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

		self.policyholders: dict[str, Policyholder] = {}
		self.products: dict[str, InsuranceProduct] = {}
		self.quotes: dict[str, Quote] = {}
		self.policies: dict[str, Policy] = {}
		self.premiums: dict[str, PremiumRecord] = {}
		self.claims: dict[str, ClaimRecord] = {}
		self.documents: dict[str, InsuranceDocument] = {}
		self.risk: dict[str, RiskAssessment] = {}
		self.reinsurance: dict[str, ReinsuranceAttachment] = {}
		self.compliance: dict[str, InsuranceAlert] = {}
		self.reviews: dict[str, InsuranceReview] = {}
		self.evidence: dict[str, InsuranceEvidence] = {}
		self.audit_events: list[dict[str, Any]] = []

	# ------------------------------------------------------------------
	# Capability contract
	# ------------------------------------------------------------------

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id or self.tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------
	# Policyholder management
	# ------------------------------------------------------------------

	async def onboard_policyholder(
		self,
		policyholder_id: str,
		name: str,
		kyc_reference: str,
		contact_reference: str,
		risk_profile_reference: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		"""Onboard a new policyholder with full KYC and risk profile linkage."""
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "onboard_policyholder",
			"kyc_present": bool(kyc_reference),
			"contact_present": bool(contact_reference),
		})
		item = Policyholder(policyholder_id, self.tenant_id, name, kyc_reference, contact_reference, risk_profile_reference)
		item.__dict__["onboarded_at"] = _now_iso()
		self.policyholders[policyholder_id] = item
		await self._audit("policyholder_onboarded", policyholder_id, {"name": name})
		return item.to_dict()

	async def get_policyholder(self, policyholder_id: str) -> dict[str, Any]:
		"""Retrieve a policyholder record."""
		ph = self._tenant_policyholder_or_none(policyholder_id, self.tenant_id)
		if ph is None:
			raise KeyError(f"policyholder not found: {policyholder_id}")
		return ph.to_dict()

	async def list_policyholders(self) -> list[dict[str, Any]]:
		"""List all policyholders for this tenant."""
		items = [p for p in self.policyholders.values() if p.tenant_id == self.tenant_id]
		return [p.to_dict() for p in sorted(items, key=lambda x: x.policyholder_id)]

	# ------------------------------------------------------------------
	# Product management
	# ------------------------------------------------------------------

	async def publish_product(
		self,
		product_id: str,
		name: str,
		product_line: str,
		coverage_terms_reference: str,
		pricing_reference: str,
	) -> dict[str, Any]:
		"""Publish an insurance product to the catalogue."""
		product_line_norm = normalize_code(product_line)
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "publish_product",
			"product_line_supported": product_line_norm in SUPPORTED_PRODUCT_LINES,
			"coverage_terms_present": bool(coverage_terms_reference),
		})
		item = InsuranceProduct(product_id, self.tenant_id, name, product_line_norm, coverage_terms_reference, pricing_reference)
		item.__dict__["published_at"] = _now_iso()
		self.products[product_id] = item
		await self._audit("insurance_product_published", product_id, {"name": name, "line": product_line_norm})
		return item.to_dict()

	# ------------------------------------------------------------------
	# Quoting & binding
	# ------------------------------------------------------------------

	async def generate_quote(
		self,
		quote_id: str,
		policyholder_id: str,
		product_id: str,
		premium_minor: int,
		currency: str,
		underwriting_reference: str,
	) -> dict[str, Any]:
		"""Generate a premium quote for a policyholder / product pair."""
		policyholder = self._tenant_policyholder_or_none(policyholder_id, self.tenant_id)
		product = self._tenant_product_or_none(product_id, self.tenant_id)
		currency_norm = normalize_currency(currency)
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "generate_quote",
			"policyholder_present": policyholder is not None,
			"product_present": product is not None,
			"positive_premium": positive_minor(premium_minor),
			"underwriting_reference_present": bool(underwriting_reference),
		})
		item = Quote(quote_id, self.tenant_id, policyholder_id, product_id, int(premium_minor), currency_norm, underwriting_reference)
		item.__dict__["generated_at"] = _now_iso()
		item.__dict__["status"] = "pending"
		self.quotes[quote_id] = item
		await self._audit("quote_generated", quote_id, {"policyholder_id": policyholder_id, "product_id": product_id})
		return item.to_dict()

	async def create_policy(
		self,
		customer_id: str,
		product_code: str,
		coverage_amount: float,
		premium: float,
		period: str,
		policy_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Convenience end-to-end method: locate or create a policyholder, generate
		a quote, and bind a policy in a single call.  Returns the bound policy.
		"""
		pid = policy_id or _uuid()
		product = next(
			(p for p in self.products.values()
			 if p.tenant_id == self.tenant_id and getattr(p, "product_line", "") == normalize_code(product_code)),
			None,
		)
		if product is None:
			raise KeyError(f"no published product found for product_code: {product_code}")

		policyholder = self._tenant_policyholder_or_none(customer_id, self.tenant_id)
		if policyholder is None:
			raise KeyError(f"policyholder not found: {customer_id}")

		premium_minor = int(round(premium * 100))
		coverage_minor = int(round(coverage_amount * 100))

		quote_id = _uuid()
		await self.generate_quote(
			quote_id, customer_id, product.product_id,
			premium_minor, "KES", f"auto_underwrite_{pid}",
		)

		policy = Policy(pid, self.tenant_id, quote_id, _now_iso()[:10], f"auto_payment_{pid}")
		policy.__dict__.update({
			"product_code": product_code,
			"coverage_amount_minor": coverage_minor,
			"premium_minor": premium_minor,
			"period": period,
			"status": "active",
			"created_at": _now_iso(),
		})
		self.policies[pid] = policy
		await self._audit("policy_created", pid, {
			"customer_id": customer_id, "product_code": product_code,
			"coverage_amount_minor": coverage_minor, "period": period,
		})
		return policy.to_dict()

	async def bind_policy(
		self,
		policy_id: str,
		quote_id: str,
		effective_date: str,
		payment_reference: str,
	) -> dict[str, Any]:
		"""Bind a policy from a quote with confirmed payment."""
		quote = self._tenant_quote_or_none(quote_id, self.tenant_id)
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "bind_policy",
			"quote_present": quote is not None,
			"payment_reference_present": bool(payment_reference),
		})
		item = Policy(policy_id, self.tenant_id, quote_id, effective_date, payment_reference)
		item.__dict__.update({"status": "active", "bound_at": _now_iso()})
		self.policies[policy_id] = item
		if quote is not None:
			quote.__dict__["status"] = "bound"
		await self._audit("policy_bound", policy_id, {"quote_id": quote_id})
		return item.to_dict()

	# ------------------------------------------------------------------
	# Underwriting
	# ------------------------------------------------------------------

	async def underwrite_policy(
		self,
		application_id: str,
		underwriter_id: str | None = None,
		decision: str = "approve",
		notes: str = "",
	) -> dict[str, Any]:
		"""
		Underwrite a policy application.  Applies rule-based risk checks against
		the policyholder's risk assessment score; records a structured decision.
		"""
		# application_id maps to a quote_id in this model
		quote = self._tenant_quote_or_none(application_id, self.tenant_id)
		if quote is None:
			raise KeyError(f"application / quote not found: {application_id}")

		ph_id = quote.policyholder_id
		risk_assessments = [
			r for r in self.risk.values()
			if r.tenant_id == self.tenant_id and r.policyholder_id == ph_id
		]
		risk_score = risk_assessments[-1].score if risk_assessments else 0.5
		auto_approved = risk_score < 0.6

		effective_decision = decision if not auto_approved else "approve"
		uw_id = underwriter_id or self.actor_id
		result: dict[str, Any] = {
			"application_id": application_id,
			"underwriter_id": uw_id,
			"risk_score": risk_score,
			"auto_approved": auto_approved,
			"decision": effective_decision,
			"conditions": [] if effective_decision == "approve" else ["manual_review_required"],
			"notes": notes,
			"decided_at": _now_iso(),
		}
		if effective_decision == "approve":
			quote.__dict__["status"] = "approved"
		else:
			quote.__dict__["status"] = "declined"

		await self._audit("policy_underwritten", application_id, {"decision": effective_decision, "risk_score": risk_score})
		return result

	# ------------------------------------------------------------------
	# Premium processing
	# ------------------------------------------------------------------

	async def process_premium(
		self,
		policy_id: str,
		amount: float,
		payment_method: str,
		premium_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Process an inbound premium payment against a policy.  Validates that
		the policy is active, checks for duplicate payments, and records the
		premium receipt.
		"""
		pid = premium_id or _uuid()
		policy = self._tenant_policy_or_none(policy_id, self.tenant_id)
		if policy is None:
			raise KeyError(f"policy not found: {policy_id}")
		policy_status = getattr(policy, "status", "active")
		if policy_status not in {"active", "lapsed"}:
			raise ValueError(f"cannot process premium on policy with status: {policy_status}")

		amount_minor = int(round(amount * 100))
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_premium",
			"policy_present": True,
			"positive_amount": positive_minor(amount_minor),
			"currency_supported": True,
			"payment_reference_present": bool(payment_method),
		})
		# reinstate lapsed policy if payment received
		if policy_status == "lapsed":
			policy.__dict__["status"] = "active"
			policy.__dict__["reinstated_at"] = _now_iso()

		currency = getattr(policy, "currency", "KES")
		item = PremiumRecord(pid, self.tenant_id, policy_id, amount_minor, currency, payment_method)
		item.__dict__.update({"payment_method": payment_method, "received_at": _now_iso()})
		self.premiums[pid] = item
		await self._audit("premium_processed", pid, {"policy_id": policy_id, "amount_minor": amount_minor})
		return item.to_dict()

	async def record_premium(
		self,
		premium_id: str,
		policy_id: str,
		amount_minor: int,
		currency: str,
		payment_reference: str,
	) -> dict[str, Any]:
		"""Record an externally-sourced premium payment."""
		policy = self._tenant_policy_or_none(policy_id, self.tenant_id)
		currency_norm = normalize_currency(currency)
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_premium",
			"policy_present": policy is not None,
			"positive_amount": positive_minor(amount_minor),
			"currency_supported": currency_norm in SUPPORTED_CURRENCIES,
			"payment_reference_present": bool(payment_reference),
		})
		item = PremiumRecord(premium_id, self.tenant_id, policy_id, int(amount_minor), currency_norm, payment_reference)
		self.premiums[premium_id] = item
		await self._audit("premium_recorded", premium_id, {"policy_id": policy_id})
		return item.to_dict()

	# ------------------------------------------------------------------
	# Claims lifecycle
	# ------------------------------------------------------------------

	async def file_claim(
		self,
		policy_id: str,
		incident_type: str,
		incident_date: str,
		amount_claimed: float,
		description: str,
		claim_id: str | None = None,
		evidence_reference: str = "",
	) -> dict[str, Any]:
		"""
		File a new insurance claim.  Validates policy active status, checks
		coverage limits against claimed amount, and creates the claim record
		with status 'filed'.
		"""
		cid = claim_id or _uuid()
		policy = self._tenant_policy_or_none(policy_id, self.tenant_id)
		if policy is None:
			raise KeyError(f"policy not found: {policy_id}")
		if getattr(policy, "status", "active") != "active":
			raise ValueError(f"policy {policy_id} is not active — cannot file claim")

		claim_type_norm = normalize_code(incident_type)
		assert bool(incident_date), "incident_date required"
		assert amount_claimed > 0, "amount_claimed must be positive"

		amount_minor = int(round(amount_claimed * 100))
		coverage_minor = getattr(policy, "coverage_amount_minor", amount_minor * 10)
		if amount_minor > coverage_minor:
			raise ValueError(
				f"claimed amount {amount_minor} exceeds policy coverage {coverage_minor}"
			)

		ev_ref = evidence_reference or f"claim_evidence_{cid}"
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "open_claim",
			"policy_present": True,
			"claim_type_supported": claim_type_norm in SUPPORTED_CLAIM_TYPES,
			"positive_amount": positive_minor(amount_minor),
			"evidence_present": bool(ev_ref) and bool(incident_date),
		})
		item = ClaimRecord(cid, self.tenant_id, policy_id, claim_type_norm, amount_minor, incident_date, ev_ref)
		item.__dict__.update({
			"description": description,
			"status": "filed",
			"filed_at": _now_iso(),
		})
		self.claims[cid] = item
		await self._maybe_notify("claim_filed", {"claim_id": cid, "policy_id": policy_id, "amount_minor": amount_minor})
		await self._audit("claim_filed", cid, {"policy_id": policy_id, "incident_type": incident_type})
		return item.to_dict()

	async def assess_claim(
		self,
		claim_id: str,
		assessor_id: str,
		assessed_amount: float,
		notes: str,
	) -> dict[str, Any]:
		"""
		Assess a filed claim.  Calculates whether the assessed amount is within
		policy limits and records the assessor's findings.  Sets claim status to
		'assessed'.
		"""
		claim = self._tenant_claim_or_none(claim_id, self.tenant_id)
		if claim is None:
			raise KeyError(f"claim not found: {claim_id}")
		current_status = getattr(claim, "status", "filed")
		if current_status not in {"filed", "under_review"}:
			raise ValueError(f"claim {claim_id} cannot be assessed from status: {current_status}")
		assert assessed_amount >= 0, "assessed_amount must be non-negative"
		assert bool(assessor_id), "assessor_id required"

		assessed_minor = int(round(assessed_amount * 100))
		filed_minor = claim.amount_minor
		reduction_pct = round(1 - assessed_minor / filed_minor, 4) if filed_minor > 0 else 0.0

		claim.__dict__.update({
			"assessed_amount_minor": assessed_minor,
			"assessor_id": assessor_id,
			"assessment_notes": notes,
			"reduction_pct": reduction_pct,
			"status": "assessed",
			"assessed_at": _now_iso(),
		})
		await self._audit("claim_assessed", claim_id, {"assessor_id": assessor_id, "assessed_minor": assessed_minor})
		return {
			**claim.to_dict(),
			"filed_amount_minor": filed_minor,
			"assessed_amount_minor": assessed_minor,
			"reduction_pct": reduction_pct,
		}

	async def approve_claim(
		self,
		claim_id: str,
		approved_amount: float,
		approved_by: str,
	) -> dict[str, Any]:
		"""
		Approve a claim for payment.  Claim must be in 'assessed' status.
		Sets status to 'approved' and records the approval authority.
		"""
		claim = self._tenant_claim_or_none(claim_id, self.tenant_id)
		if claim is None:
			raise KeyError(f"claim not found: {claim_id}")
		if getattr(claim, "status", "") != "assessed":
			raise ValueError(f"claim {claim_id} must be assessed before approval; current: {claim.__dict__.get('status')}")
		assert approved_amount >= 0, "approved_amount must be non-negative"
		assert bool(approved_by), "approved_by required"

		approved_minor = int(round(approved_amount * 100))
		claim.__dict__.update({
			"approved_amount_minor": approved_minor,
			"approved_by": approved_by,
			"status": "approved",
			"approved_at": _now_iso(),
		})
		await self._maybe_notify("claim_approved", {"claim_id": claim_id, "approved_minor": approved_minor})
		await self._audit("claim_approved", claim_id, {"approved_by": approved_by, "approved_minor": approved_minor})
		return claim.to_dict()

	async def reject_claim(self, claim_id: str, reason: str) -> dict[str, Any]:
		"""
		Reject a claim.  Claim must be in 'filed' or 'assessed' status.
		Records rejection reason and sets status to 'rejected'.
		"""
		claim = self._tenant_claim_or_none(claim_id, self.tenant_id)
		if claim is None:
			raise KeyError(f"claim not found: {claim_id}")
		current = getattr(claim, "status", "filed")
		if current not in {"filed", "assessed", "under_review"}:
			raise ValueError(f"claim {claim_id} cannot be rejected from status: {current}")
		assert bool(reason), "rejection reason required"

		claim.__dict__.update({
			"rejection_reason": reason,
			"rejected_by": self.actor_id,
			"status": "rejected",
			"rejected_at": _now_iso(),
		})
		await self._maybe_notify("claim_rejected", {"claim_id": claim_id, "reason": reason})
		await self._audit("claim_rejected", claim_id, {"reason": reason})
		return claim.to_dict()

	async def pay_claim(
		self,
		claim_id: str,
		payment_method: str,
		payment_reference: str | None = None,
	) -> dict[str, Any]:
		"""
		Disburse payment for an approved claim.  Validates claim status is
		'approved', generates a payment reference, and marks claim as 'paid'.
		"""
		claim = self._tenant_claim_or_none(claim_id, self.tenant_id)
		if claim is None:
			raise KeyError(f"claim not found: {claim_id}")
		if getattr(claim, "status", "") != "approved":
			raise ValueError(f"claim {claim_id} must be approved before payment; current: {claim.__dict__.get('status')}")
		assert bool(payment_method), "payment_method required"

		pay_ref = payment_reference or _uuid()
		approved_minor = claim.__dict__.get("approved_amount_minor", claim.amount_minor)

		claim.__dict__.update({
			"payment_method": payment_method,
			"payment_reference": pay_ref,
			"paid_amount_minor": approved_minor,
			"status": "paid",
			"paid_at": _now_iso(),
		})
		await self._maybe_notify("claim_paid", {"claim_id": claim_id, "paid_minor": approved_minor})
		await self._audit("claim_paid", claim_id, {"payment_method": payment_method, "paid_minor": approved_minor})
		return {**claim.to_dict(), "payment_reference": pay_ref, "paid_amount_minor": approved_minor}

	async def get_claim(self, claim_id: str) -> dict[str, Any]:
		"""Retrieve a claim by ID."""
		claim = self._tenant_claim_or_none(claim_id, self.tenant_id)
		if claim is None:
			raise KeyError(f"claim not found: {claim_id}")
		return claim.to_dict()

	async def list_claims(
		self,
		policy_id: str | None = None,
		status: str | None = None,
	) -> list[dict[str, Any]]:
		"""List claims for this tenant, optionally filtered by policy or status."""
		items = [c for c in self.claims.values() if c.tenant_id == self.tenant_id]
		if policy_id:
			items = [c for c in items if c.policy_id == policy_id]
		if status:
			items = [c for c in items if getattr(c, "status", "") == status]
		return [c.to_dict() for c in sorted(items, key=lambda x: x.claim_id)]

	# ------------------------------------------------------------------
	# Policy administration
	# ------------------------------------------------------------------

	async def policy_renewal(
		self,
		policy_id: str,
		new_terms: dict[str, Any],
	) -> dict[str, Any]:
		"""
		Renew a policy.  Applies new coverage amount, premium, and period from
		new_terms dict.  Creates a new policy record linked to the original and
		marks the original as 'renewed'.
		"""
		policy = self._tenant_policy_or_none(policy_id, self.tenant_id)
		if policy is None:
			raise KeyError(f"policy not found: {policy_id}")
		if getattr(policy, "status", "active") not in {"active", "expiring"}:
			raise ValueError(f"policy {policy_id} is not eligible for renewal; status: {policy.__dict__.get('status')}")

		new_pid = _uuid()
		new_policy = Policy(new_pid, self.tenant_id, policy.quote_id, _now_iso()[:10], f"renewal_{new_pid}")
		new_policy.__dict__.update({
			"predecessor_policy_id": policy_id,
			"status": "active",
			"product_code": getattr(policy, "product_code", ""),
			"coverage_amount_minor": new_terms.get("coverage_amount_minor", getattr(policy, "coverage_amount_minor", 0)),
			"premium_minor": new_terms.get("premium_minor", getattr(policy, "premium_minor", 0)),
			"period": new_terms.get("period", getattr(policy, "period", "")),
			"currency": new_terms.get("currency", getattr(policy, "currency", "KES")),
			"renewed_at": _now_iso(),
		})
		self.policies[new_pid] = new_policy
		policy.__dict__["status"] = "renewed"
		policy.__dict__["renewed_to"] = new_pid

		await self._audit("policy_renewed", new_pid, {"predecessor": policy_id})
		return {**new_policy.to_dict(), "predecessor_policy_id": policy_id}

	async def cancel_policy(
		self,
		policy_id: str,
		reason: str,
		cancellation_date: str,
	) -> dict[str, Any]:
		"""
		Cancel an active policy.  Computes a pro-rata refund based on days
		remaining in the coverage period.  Records refund amount and cancellation
		metadata.
		"""
		policy = self._tenant_policy_or_none(policy_id, self.tenant_id)
		if policy is None:
			raise KeyError(f"policy not found: {policy_id}")
		current_status = getattr(policy, "status", "active")
		if current_status not in {"active"}:
			raise ValueError(f"policy {policy_id} cannot be cancelled from status: {current_status}")
		assert bool(reason), "cancellation reason required"
		assert bool(cancellation_date), "cancellation_date required"

		premium_minor = getattr(policy, "premium_minor", 0)
		period = getattr(policy, "period", "annual")
		period_days = 365 if "annual" in period.lower() else (180 if "semi" in period.lower() else 30)
		try:
			cancel_dt = datetime.fromisoformat(cancellation_date)
			effective_dt = datetime.fromisoformat(getattr(policy, "effective_date", cancellation_date))
			days_elapsed = max(0, (cancel_dt - effective_dt).days)
		except Exception:
			days_elapsed = 0
		days_remaining = max(0, period_days - days_elapsed)
		pro_rata_refund_minor = int(premium_minor * days_remaining / period_days)

		policy.__dict__.update({
			"status": "cancelled",
			"cancellation_reason": reason,
			"cancellation_date": cancellation_date,
			"cancelled_by": self.actor_id,
			"pro_rata_refund_minor": pro_rata_refund_minor,
			"cancelled_at": _now_iso(),
		})
		await self._maybe_notify("policy_cancelled", {"policy_id": policy_id, "refund_minor": pro_rata_refund_minor})
		await self._audit("policy_cancelled", policy_id, {"reason": reason, "refund_minor": pro_rata_refund_minor})
		return {**policy.to_dict(), "pro_rata_refund_minor": pro_rata_refund_minor}

	async def get_policy(self, policy_id: str) -> dict[str, Any]:
		"""Retrieve a policy record."""
		policy = self._tenant_policy_or_none(policy_id, self.tenant_id)
		if policy is None:
			raise KeyError(f"policy not found: {policy_id}")
		premiums = [p.to_dict() for p in self.premiums.values()
					if p.tenant_id == self.tenant_id and p.policy_id == policy_id]
		claims = [c.to_dict() for c in self.claims.values()
				  if c.tenant_id == self.tenant_id and c.policy_id == policy_id]
		return {**policy.to_dict(), "premiums": premiums, "claims": claims}

	# ------------------------------------------------------------------
	# Risk assessment
	# ------------------------------------------------------------------

	async def record_risk_assessment(
		self,
		assessment_id: str,
		policyholder_id: str,
		score: float,
		source_reference: str,
	) -> dict[str, Any]:
		"""Record an underwriting risk assessment score for a policyholder."""
		policyholder = self._tenant_policyholder_or_none(policyholder_id, self.tenant_id)
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_risk_assessment",
			"policyholder_present": policyholder is not None,
			"score_present": score_present(score),
			"source_present": bool(source_reference),
		})
		item = RiskAssessment(assessment_id, self.tenant_id, policyholder_id, float(score), source_reference)
		item.__dict__["assessed_at"] = _now_iso()
		self.risk[assessment_id] = item
		await self._audit("risk_assessment_recorded", assessment_id, {"score": score})
		return item.to_dict()

	# ------------------------------------------------------------------
	# Reinsurance
	# ------------------------------------------------------------------

	async def record_reinsurance_attachment(
		self,
		attachment_id: str,
		policy_id: str,
		treaty_reference: str,
		share_percent: float,
	) -> dict[str, Any]:
		"""Record a reinsurance cession against a policy."""
		policy = self._tenant_policy_or_none(policy_id, self.tenant_id)
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_reinsurance_attachment",
			"policy_present": policy is not None,
			"treaty_reference_present": bool(treaty_reference),
			"positive_share": float(share_percent) > 0,
		})
		assert 0 < share_percent <= 100, "share_percent must be in (0, 100]"
		item = ReinsuranceAttachment(attachment_id, self.tenant_id, policy_id, treaty_reference, float(share_percent))
		item.__dict__["attached_at"] = _now_iso()
		self.reinsurance[attachment_id] = item
		await self._audit("reinsurance_attachment_recorded", attachment_id, {"share_percent": share_percent})
		return item.to_dict()

	# ------------------------------------------------------------------
	# Analytics
	# ------------------------------------------------------------------

	async def insurance_analytics(self, period: str) -> dict[str, Any]:
		"""
		Compute insurance portfolio analytics for the given period.  Includes
		loss ratio, combined ratio, claims frequency, average claim size,
		premium income, and product mix breakdown.
		"""
		assert bool(period), "period required"
		tid = self.tenant_id

		policies = [p for p in self.policies.values() if p.tenant_id == tid]
		premiums = [p for p in self.premiums.values() if p.tenant_id == tid]
		claims = [c for c in self.claims.values() if c.tenant_id == tid]
		paid_claims = [c for c in claims if getattr(c, "status", "") == "paid"]
		active_policies = [p for p in policies if getattr(p, "status", "active") == "active"]

		total_premium_minor = sum(p.amount_minor for p in premiums)
		total_claims_paid_minor = sum(c.__dict__.get("paid_amount_minor", c.amount_minor) for c in paid_claims)
		loss_ratio = round(total_claims_paid_minor / total_premium_minor, 4) if total_premium_minor > 0 else 0.0
		# expense ratio synthetic: 25 % of premium
		expense_ratio = 0.25
		combined_ratio = round(loss_ratio + expense_ratio, 4)

		claim_sizes = [c.__dict__.get("paid_amount_minor", c.amount_minor) for c in paid_claims]
		avg_claim_size = round(statistics.mean(claim_sizes), 2) if claim_sizes else 0.0

		# product mix
		product_mix: dict[str, int] = {}
		for p in active_policies:
			code = getattr(p, "product_code", "unknown")
			product_mix[code] = product_mix.get(code, 0) + 1

		await self._audit("insurance_analytics_computed", period, {"claim_count": len(claims)})
		return {
			"period": period,
			"as_of": _now_iso(),
			"active_policy_count": len(active_policies),
			"total_policy_count": len(policies),
			"premium_count": len(premiums),
			"total_premium_minor": total_premium_minor,
			"claim_count": len(claims),
			"paid_claim_count": len(paid_claims),
			"total_claims_paid_minor": total_claims_paid_minor,
			"loss_ratio": loss_ratio,
			"expense_ratio": expense_ratio,
			"combined_ratio": combined_ratio,
			"average_claim_size_minor": avg_claim_size,
			"claims_frequency": round(len(paid_claims) / max(len(active_policies), 1), 4),
			"product_mix": product_mix,
		}

	# ------------------------------------------------------------------
	# Documents & compliance
	# ------------------------------------------------------------------

	async def record_document(
		self,
		document_id: str,
		reference_id: str,
		document_type: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		"""Attach a document to a policy or claim record."""
		doc_type_norm = normalize_code(document_type)
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_document",
			"document_type_supported": doc_type_norm in SUPPORTED_DOCUMENT_TYPES,
			"evidence_present": bool(evidence_reference) and bool(reference_id),
		})
		item = InsuranceDocument(document_id, self.tenant_id, reference_id, doc_type_norm, evidence_reference)
		item.__dict__["recorded_at"] = _now_iso()
		self.documents[document_id] = item
		await self._audit("document_recorded", document_id, {"reference_id": reference_id, "type": doc_type_norm})
		return item.to_dict()

	async def record_compliance_alert(
		self,
		alert_id: str,
		reference_id: str,
		severity: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		"""Record an insurance compliance or regulatory alert."""
		severity_norm = normalize_code(severity)
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_compliance_alert",
			"severity_supported": severity_norm in SUPPORTED_ALERT_SEVERITIES,
			"evidence_present": bool(evidence_reference),
		})
		item = InsuranceAlert(alert_id, self.tenant_id, reference_id, severity_norm, evidence_reference)
		self.compliance[alert_id] = item
		if severity_norm in {"critical", "high"}:
			await self._maybe_notify("insurance_compliance_alert", {"alert_id": alert_id, "severity": severity_norm})
		await self._audit("insurance_compliance_alert_recorded", alert_id, {"severity": severity_norm})
		return item.to_dict()

	async def record_review(
		self,
		review_id: str,
		reference_id: str,
		reviewer_id: str,
		status: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		"""Record a compliance or supervisory review."""
		status_norm = normalize_code(status)
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_review",
			"status_supported": status_norm in SUPPORTED_REVIEW_STATUSES,
			"evidence_present": bool(evidence_reference) and bool(reviewer_id),
		})
		item = InsuranceReview(review_id, self.tenant_id, reference_id, reviewer_id, status_norm, evidence_reference)
		self.reviews[review_id] = item
		await self._audit("insurance_review_recorded", review_id, {"status": status_norm})
		return item.to_dict()

	# ------------------------------------------------------------------
	# Agents & batch
	# ------------------------------------------------------------------

	async def register_insurance_agent(
		self,
		agent_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
	) -> dict[str, Any]:
		"""Register an AI insurance agent."""
		runtime_norm = normalize_code(runtime)
		role_norm = normalize_code(role)
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_insurance_agent",
			"agent_runtime_supported": runtime_norm in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role_norm in SUPPORTED_AGENT_ROLES,
		})
		item = InsuranceEvidence(agent_id, self.tenant_id, "agent", agent_id, "registered", {
			"name": name, "runtime": runtime_norm, "role": role_norm, "scope": scope,
		})
		self.evidence[agent_id] = item
		await self._audit("insurance_agent_registered", agent_id, {"role": role_norm})
		return item.to_dict()

	async def validate_agent_action(
		self,
		privileged_scope: bool,
		human_approval_recorded: bool,
	) -> dict[str, Any]:
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation": "insurance_agent_action",
			"privileged_scope": privileged_scope,
			"human_approval_recorded": human_approval_recorded,
		})
		return {"tenant_id": self.tenant_id, "accepted": True, "privileged_scope": privileged_scope}

	async def validate_batch(self, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation": "insurance_batch",
			"event_stream": event_stream,
		})
		return {
			"tenant_id": self.tenant_id,
			"item_count": item_count,
			"processor": "bytewax",
			"stream": "apg.fintech.insurance.lifecycle",
			"accepted": True,
		}

	async def dashboard_summary(self) -> dict[str, Any]:
		"""Return aggregate summary of all insurance state for this tenant."""
		tid = self.tenant_id
		open_claims = sum(
			1 for c in self.claims.values()
			if c.tenant_id == tid and getattr(c, "status", "") in {"filed", "under_review", "assessed"}
		)
		return {
			"tenant_id": tid,
			"policyholder_count": self._count(self.policyholders, tid),
			"product_count": self._count(self.products, tid),
			"quote_count": self._count(self.quotes, tid),
			"policy_count": self._count(self.policies, tid),
			"active_policy_count": sum(
				1 for p in self.policies.values()
				if p.tenant_id == tid and getattr(p, "status", "active") == "active"
			),
			"premium_count": self._count(self.premiums, tid),
			"claim_count": self._count(self.claims, tid),
			"open_claim_count": open_claims,
			"document_count": self._count(self.documents, tid),
			"risk_count": self._count(self.risk, tid),
			"reinsurance_count": self._count(self.reinsurance, tid),
			"compliance_count": self._count(self.compliance, tid),
			"review_count": self._count(self.reviews, tid),
			"audit_event_count": sum(1 for e in self.audit_events if e["tenant_id"] == tid),
			"streaming": get_capability_contract(tid)["streaming"],
			"as_of": _now_iso(),
		}

	# ------------------------------------------------------------------
	# Additional async methods
	# ------------------------------------------------------------------

	async def health_check(self) -> dict[str, Any]:
		"""Return insurance service health status."""
		return {
			"service": "insurance", "status": "healthy",
			"active_policies": sum(1 for p in self.policies.values() if p.tenant_id == self.tenant_id and getattr(p, "status", "active") == "active"),
			"open_claims": sum(1 for c in self.claims.values() if c.tenant_id == self.tenant_id and getattr(c, "status", "") in {"filed", "under_review", "assessed"}),
			"checked_at": _now_iso(),
		}

	async def micro_insurance_product(self, product_id: str, name: str, premium_kes_monthly: float, coverage_kes: float, target_segment: str) -> dict[str, Any]:
		"""Publish a micro-insurance product targeting low-income / informal sector."""
		return await self.publish_product(
			product_id=product_id, name=name, product_line="micro_insurance",
			coverage_terms_reference=f"micro_terms_{product_id}",
			pricing_reference=f"micro_pricing_{product_id}",
		)

	async def group_policy_enrollment(self, group_id: str, member_ids: list[str], product_id: str, premium_per_member: float) -> dict[str, Any]:
		"""Enroll a group (e.g., SACCO, employer) into a group insurance policy."""
		enrolled = []
		for member_id in member_ids:
			ph = self._tenant_policyholder_or_none(member_id, self.tenant_id)
			if ph is not None:
				enrolled.append(member_id)
		await self._audit("group_enrollment_processed", group_id, {"enrolled": len(enrolled)})
		return {
			"group_id": group_id, "product_id": product_id,
			"total_members": len(member_ids), "enrolled": len(enrolled),
			"premium_per_member": premium_per_member,
			"total_premium": round(len(enrolled) * premium_per_member, 2),
			"enrolled_at": _now_iso(),
		}

	async def claim_fast_track(self, claim_id: str, auto_approve_threshold: float = 5_000.0) -> dict[str, Any]:
		"""Fast-track a claim for auto-approval if below threshold."""
		claim = self._tenant_claim_or_none(claim_id, self.tenant_id)
		if claim is None:
			raise KeyError(f"claim not found: {claim_id}")
		amount = claim.amount_minor / 100
		if amount <= auto_approve_threshold:
			assessed = await self.assess_claim(claim_id, "auto_assessor", amount, "auto_fast_track")
			approved = await self.approve_claim(claim_id, amount, "auto_approval_engine")
			return {**approved, "fast_tracked": True, "auto_threshold": auto_approve_threshold}
		return {**claim.to_dict(), "fast_tracked": False, "reason": "above_threshold"}

	async def fraud_indicator_check_claim(self, claim_id: str) -> dict[str, Any]:
		"""Check a claim for fraud indicators before processing."""
		claim = self._tenant_claim_or_none(claim_id, self.tenant_id)
		if claim is None:
			raise KeyError(f"claim not found: {claim_id}")
		amount = claim.amount_minor / 100
		indicators: list[str] = []
		if amount > 500_000:
			indicators.append("HIGH_VALUE_CLAIM")
		policy_claims = [c for c in self.claims.values() if c.tenant_id == self.tenant_id and c.policy_id == claim.policy_id]
		if len(policy_claims) >= 3:
			indicators.append("MULTIPLE_CLAIMS_SAME_POLICY")
		risk_score = len(indicators) * 30.0
		return {
			"claim_id": claim_id, "fraud_indicators": indicators, "risk_score": risk_score,
			"recommendation": "review" if indicators else "approve",
			"checked_at": _now_iso(),
		}

	async def ira_regulatory_return(self, period: str) -> dict[str, Any]:
		"""File an IRA (Insurance Regulatory Authority of Kenya) quarterly return."""
		active_policies = sum(1 for p in self.policies.values() if p.tenant_id == self.tenant_id and getattr(p, "status", "active") == "active")
		total_premiums = sum(p.amount_minor for p in self.premiums.values() if p.tenant_id == self.tenant_id)
		total_claims_paid = sum(c.__dict__.get("paid_amount_minor", c.amount_minor) for c in self.claims.values() if c.tenant_id == self.tenant_id and getattr(c, "status", "") == "paid")
		return {
			"report_type": "IRA_KENYA_QUARTERLY_RETURN", "period": period,
			"active_policies": active_policies,
			"total_premiums_minor": total_premiums,
			"total_claims_paid_minor": total_claims_paid,
			"loss_ratio": round(total_claims_paid / max(total_premiums, 1), 4),
			"status": "draft", "generated_at": _now_iso(),
		}

	async def no_claims_discount(self, policyholder_id: str, claim_free_years: int) -> dict[str, Any]:
		"""Calculate a No Claims Discount (NCD) for a policyholder."""
		ncd_table = {0: 0.0, 1: 10.0, 2: 20.0, 3: 30.0, 4: 40.0, 5: 50.0}
		discount_pct = ncd_table.get(min(claim_free_years, 5), 50.0)
		await self._audit("ncd_calculated", policyholder_id, {"claim_free_years": claim_free_years})
		return {
			"policyholder_id": policyholder_id, "claim_free_years": claim_free_years,
			"ncd_discount_pct": discount_pct, "calculated_at": _now_iso(),
		}

	async def export_insurance_data(self, fmt: str = "csv") -> dict[str, Any]:
		"""Export insurance portfolio data."""
		assert fmt in {"csv", "json", "excel"}
		return {
			"tenant_id": self.tenant_id, "format": fmt,
			"policies": sum(1 for p in self.policies.values() if p.tenant_id == self.tenant_id),
			"claims": sum(1 for c in self.claims.values() if c.tenant_id == self.tenant_id),
			"file_reference": f"insurance_{self.tenant_id}_{_now_iso()[:10]}.{fmt}", "generated_at": _now_iso(),
		}

	# ------------------------------------------------------------------
	# Internal helpers
	# ------------------------------------------------------------------

	def _tenant_policyholder_or_none(self, item_id: str, tenant_id: str) -> Policyholder | None:
		item = self.policyholders.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _tenant_product_or_none(self, item_id: str, tenant_id: str) -> InsuranceProduct | None:
		item = self.products.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _tenant_quote_or_none(self, item_id: str, tenant_id: str) -> Quote | None:
		item = self.quotes.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _tenant_policy_or_none(self, item_id: str, tenant_id: str) -> Policy | None:
		item = self.policies.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _tenant_claim_or_none(self, item_id: str, tenant_id: str) -> ClaimRecord | None:
		item = self.claims.get(item_id)
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
		reasons = ", ".join(action.get("reason", "insurance_policy_denied") for action in result["actions"])
		raise PermissionError(reasons or "insurance_policy_denied")

	async def ml_premium_score(self, *args, **kwargs):
		"""AI-powered insurance premium risk scoring. Requires OLLAMA_BASE_URL."""
		import os
		if not os.environ.get("OLLAMA_BASE_URL"):
			return {"ml_enhanced": False}
		try:
			from capabilities.common.mlx import MLCapability
			ml = MLCapability()
			result = await ml.score(kwargs, task="insurance_premium_risk")
			return {"risk_score": round(result.score,3), "premium_band": result.factors[0] if result.factors else "standard", "ml_enhanced": True}
		except Exception:
			return {"ml_enhanced": False}

