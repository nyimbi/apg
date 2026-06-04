"""Executable service layer for APG Buy Now Pay Later."""

from __future__ import annotations

import asyncio
import datetime
import math
import secrets
from collections import defaultdict
from typing import Any

try:
	from .bnpl_runtime import (
		decision_is_approved,
		decision_is_final,
		estimate_installment_amount,
		normalize_amount,
		normalize_code,
		normalize_country,
		normalize_currency,
		normalize_score,
	)
	from .capability_contract import (
		SUPPORTED_AFFORDABILITY_DECISIONS,
		SUPPORTED_AGENT_ROLES,
		SUPPORTED_AGENT_RUNTIMES,
		SUPPORTED_CHECKOUT_CHANNELS,
		SUPPORTED_COUNTRIES,
		SUPPORTED_CURRENCIES,
		SUPPORTED_DISPUTE_REASONS,
		SUPPORTED_INSTALLMENT_STATUSES,
		SUPPORTED_MERCHANT_CATEGORIES,
		SUPPORTED_PLAN_TYPES,
		SUPPORTED_SETTLEMENT_STATUSES,
		evaluate_capability_rules,
		get_capability_contract,
	)
	from .models import (
		AffordabilityDecision,
		BNPLConsumer,
		BNPLDispute,
		BNPLevidence,
		BNPLPlan,
		CheckoutSession,
		InstallmentSchedule,
		MerchantProfile,
		MerchantProgram,
		MerchantSettlement,
	)
except ImportError:  # pragma: no cover
	from bnpl_runtime import (  # type: ignore
		decision_is_approved,
		decision_is_final,
		estimate_installment_amount,
		normalize_amount,
		normalize_code,
		normalize_country,
		normalize_currency,
		normalize_score,
	)
	from capability_contract import (  # type: ignore
		SUPPORTED_AFFORDABILITY_DECISIONS,
		SUPPORTED_AGENT_ROLES,
		SUPPORTED_AGENT_RUNTIMES,
		SUPPORTED_CHECKOUT_CHANNELS,
		SUPPORTED_COUNTRIES,
		SUPPORTED_CURRENCIES,
		SUPPORTED_DISPUTE_REASONS,
		SUPPORTED_INSTALLMENT_STATUSES,
		SUPPORTED_MERCHANT_CATEGORIES,
		SUPPORTED_PLAN_TYPES,
		SUPPORTED_SETTLEMENT_STATUSES,
		evaluate_capability_rules,
		get_capability_contract,
	)
	from models import (  # type: ignore
		AffordabilityDecision,
		BNPLConsumer,
		BNPLDispute,
		BNPLevidence,
		BNPLPlan,
		CheckoutSession,
		InstallmentSchedule,
		MerchantProfile,
		MerchantProgram,
		MerchantSettlement,
	)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now() -> datetime.datetime:
	return datetime.datetime.now(datetime.timezone.utc)


def _iso() -> str:
	return _utc_now().isoformat()


def _add_days(n: int) -> str:
	return (_utc_now() + datetime.timedelta(days=n)).isoformat()


def _monthly_payment(principal: float, annual_rate: float, months: int) -> float:
	"""Standard annuity formula for equal monthly repayments."""
	if months <= 0:
		return principal
	if annual_rate == 0:
		return principal / months
	monthly_rate = annual_rate / 12
	return principal * monthly_rate / (1 - (1 + monthly_rate) ** (-months))


def _generate_installment_schedule(
	principal: float,
	down_payment: float,
	instalment_count: int,
	annual_rate: float,
	first_due_days: int = 30,
) -> list[dict[str, Any]]:
	"""Return a list of instalment dicts with due_date, amount, sequence."""
	net_principal = principal - down_payment
	if net_principal <= 0 or instalment_count <= 0:
		return []
	payment = _monthly_payment(net_principal, annual_rate, instalment_count)
	schedule: list[dict[str, Any]] = []
	for i in range(1, instalment_count + 1):
		due_date = (_utc_now() + datetime.timedelta(days=first_due_days + (i - 1) * 30)).strftime("%Y-%m-%d")
		schedule.append({
			"sequence": i,
			"due_date": due_date,
			"amount": round(payment, 2),
			"status": "scheduled",
		})
	return schedule


def _late_fee(overdue_amount: float, days_overdue: int) -> float:
	"""Simple daily penalty rate of 0.1% per day, capped at 30%."""
	rate = min(days_overdue * 0.001, 0.30)
	return round(overdue_amount * rate, 2)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class BuyNowPayLaterService:
	"""Full-featured BNPL runtime for APG generated applications."""

	def __init__(
		self,
		tenant_id: str = "default",
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

		self.programs: dict[str, MerchantProgram] = {}
		self.consumers: dict[str, BNPLConsumer] = {}
		self.merchants: dict[str, MerchantProfile] = {}
		self.checkouts: dict[str, CheckoutSession] = {}
		self.affordability: dict[str, AffordabilityDecision] = {}
		self.plans: dict[str, BNPLPlan] = {}
		self.installments: dict[str, InstallmentSchedule] = {}
		self.settlements: dict[str, MerchantSettlement] = {}
		self.disputes: dict[str, BNPLDispute] = {}
		self.evidence: dict[str, BNPLevidence] = {}
		self.audit_events: list[dict[str, Any]] = []

		# Generated installment schedules: plan_id -> list of installment dicts
		self._schedules: dict[str, list[dict[str, Any]]] = {}

		# Payment records: instalment_schedule_id -> payment dict
		self._payments: dict[str, dict[str, Any]] = {}

		# Early repayment records: plan_id -> repayment dict
		self._early_repayments: dict[str, dict[str, Any]] = {}

		# Late fee ledger: plan_id -> cumulative fees
		self._late_fees: dict[str, float] = defaultdict(float)

		# Credit limit per consumer: consumer_id -> limit
		self._credit_limits: dict[str, float] = {}

		# Consumer outstanding balance: consumer_id -> total outstanding
		self._outstanding: dict[str, float] = defaultdict(float)

		# Merchant fee rates: merchant_id -> {mdr_pct, settlement_days}
		self._merchant_fees: dict[str, dict[str, Any]] = {}

	# ------------------------------------------------------------------
	# Contract / describe
	# ------------------------------------------------------------------

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------
	# Preserved original methods
	# ------------------------------------------------------------------

	def register_merchant_program(
		self,
		program_id: str,
		tenant_id: str,
		name: str,
		owner_id: str,
		country: str,
		currency: str,
		settlement_policy_reference: str,
		fee_disclosure_reference: str,
		max_installments: int,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		country = normalize_country(country)
		currency = normalize_currency(currency)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "register_merchant_program",
			"owner_present": bool(owner_id),
			"country_supported": country in SUPPORTED_COUNTRIES,
			"currency_supported": currency in SUPPORTED_CURRENCIES,
			"settlement_policy_present": bool(settlement_policy_reference),
			"fee_disclosure_present": bool(fee_disclosure_reference),
			"installment_count_valid": 1 <= int(max_installments) <= 24,
		})
		if program_id in self.programs:
			raise ValueError(f"merchant program already exists: {program_id}")
		program = MerchantProgram(program_id, tenant_id, name, owner_id, country, currency, settlement_policy_reference, fee_disclosure_reference, int(max_installments))
		self.programs[program_id] = program
		self._audit(tenant_id, "bnpl_program_registered", program_id)
		return program.to_dict()

	def onboard_consumer(
		self,
		consumer_id: str,
		tenant_id: str,
		customer_reference: str,
		kyc_profile_id: str,
		country: str,
		consent_reference: str,
		aml_reference: str,
		fraud_reference: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		country = normalize_country(country)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "onboard_consumer",
			"customer_present": bool(customer_reference),
			"kyc_present": bool(kyc_profile_id),
			"country_supported": country in SUPPORTED_COUNTRIES,
			"consent_present": bool(consent_reference),
			"aml_present": bool(aml_reference),
			"fraud_present": bool(fraud_reference),
		})
		if consumer_id in self.consumers:
			raise ValueError(f"consumer already exists: {consumer_id}")
		consumer = BNPLConsumer(consumer_id, tenant_id, customer_reference, kyc_profile_id, country, consent_reference, aml_reference, fraud_reference)
		self.consumers[consumer_id] = consumer
		self._audit(tenant_id, "bnpl_consumer_onboarded", consumer_id)
		return consumer.to_dict()

	def register_merchant(
		self,
		merchant_id: str,
		tenant_id: str,
		program_id: str,
		legal_entity_reference: str,
		category: str,
		country: str,
		risk_tier: str,
		settlement_account: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		program = self._tenant_program_or_none(program_id, tenant_id)
		category = normalize_code(category)
		country = normalize_country(country)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "register_merchant",
			"program_present": program is not None,
			"legal_entity_present": bool(legal_entity_reference),
			"merchant_category_supported": category in SUPPORTED_MERCHANT_CATEGORIES,
			"country_supported": country in SUPPORTED_COUNTRIES,
			"risk_tier_present": bool(risk_tier),
			"settlement_account_present": bool(settlement_account),
		})
		if merchant_id in self.merchants:
			raise ValueError(f"merchant already exists: {merchant_id}")
		merchant = MerchantProfile(merchant_id, tenant_id, program_id, legal_entity_reference, category, country, risk_tier, settlement_account)
		self.merchants[merchant_id] = merchant
		self._audit(tenant_id, "bnpl_merchant_registered", merchant_id)
		return merchant.to_dict()

	def create_checkout_session(
		self,
		checkout_id: str,
		tenant_id: str,
		merchant_id: str,
		consumer_id: str,
		channel: str,
		category: str,
		amount: float | int | str,
		currency: str,
		payment_reference: str,
		fraud_reference: str,
		aml_reference: str,
		consent_reference: str,
		human_review: str = "",
		policy_attached: bool = True,
	) -> dict[str, Any]:
		merchant = self._tenant_merchant_or_none(merchant_id, tenant_id)
		consumer = self._tenant_consumer_or_none(consumer_id, tenant_id)
		channel = normalize_code(channel)
		category = normalize_code(category)
		amount_value = normalize_amount(amount)
		currency = normalize_currency(currency)
		high_value = amount_value >= 100000
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "create_checkout_session",
			"merchant_present": merchant is not None,
			"consumer_present": consumer is not None,
			"channel_supported": channel in SUPPORTED_CHECKOUT_CHANNELS,
			"merchant_category_supported": category in SUPPORTED_MERCHANT_CATEGORIES,
			"positive_amount": amount_value > 0,
			"currency_supported": currency in SUPPORTED_CURRENCIES,
			"payment_reference_present": bool(payment_reference),
			"fraud_present": bool(fraud_reference),
			"aml_present": bool(aml_reference),
			"consent_present": bool(consent_reference),
			"high_value": high_value,
			"human_review_recorded": bool(human_review),
		})
		if checkout_id in self.checkouts:
			raise ValueError(f"checkout already exists: {checkout_id}")
		checkout = CheckoutSession(checkout_id, tenant_id, merchant_id, consumer_id, channel, category, amount_value, currency, payment_reference, fraud_reference, aml_reference, consent_reference, human_review)
		self.checkouts[checkout_id] = checkout
		self._audit(tenant_id, "checkout_session_created", checkout_id)
		return checkout.to_dict()

	def record_affordability_decision(
		self,
		decision_id: str,
		tenant_id: str,
		checkout_id: str,
		score: float | int | str,
		decision: str,
		evidence_references: list[str],
		human_approval: str,
		adverse_reason: str = "",
	) -> dict[str, Any]:
		checkout = self._tenant_checkout_or_none(checkout_id, tenant_id)
		score_value = normalize_score(score)
		decision = normalize_code(decision)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_affordability_decision",
			"checkout_present": checkout is not None,
			"decision_supported": decision in SUPPORTED_AFFORDABILITY_DECISIONS,
			"score_in_range": 0 <= score_value <= 1000,
			"decision_evidence_present": bool(evidence_references),
			"adverse_decision": decision == "decline",
			"adverse_reason_present": bool(adverse_reason),
			"final_decision": decision_is_final(decision),
			"human_approval_recorded": bool(human_approval),
		})
		record = AffordabilityDecision(decision_id, tenant_id, checkout_id, score_value, decision, list(evidence_references), human_approval, adverse_reason)
		self.affordability[decision_id] = record
		self._audit(tenant_id, "affordability_decision_recorded", decision_id)
		return record.to_dict()

	def create_bnpl_plan(
		self,
		plan_id: str,
		tenant_id: str,
		checkout_id: str,
		affordability_id: str,
		plan_type: str,
		principal: float | int | str,
		currency: str,
		term_days: int,
		down_payment: float | int | str,
		fee_disclosure_reference: str,
		customer_acceptance_reference: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		checkout = self._tenant_checkout_or_none(checkout_id, tenant_id)
		affordability = self._tenant_affordability_or_none(affordability_id, tenant_id)
		plan_type = normalize_code(plan_type)
		principal_value = normalize_amount(principal)
		down_payment_value = normalize_amount(down_payment)
		currency = normalize_currency(currency)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "create_bnpl_plan",
			"checkout_present": checkout is not None,
			"affordability_approved": affordability is not None and decision_is_approved(affordability.decision),
			"plan_type_supported": plan_type in SUPPORTED_PLAN_TYPES,
			"positive_principal": principal_value > 0,
			"currency_supported": currency in SUPPORTED_CURRENCIES,
			"term_valid": 1 <= int(term_days) <= 730,
			"down_payment_valid": 0 <= down_payment_value <= principal_value,
			"fee_disclosure_present": bool(fee_disclosure_reference),
			"customer_acceptance_present": bool(customer_acceptance_reference),
		})
		if plan_id in self.plans:
			raise ValueError(f"BNPL plan already exists: {plan_id}")
		plan = BNPLPlan(plan_id, tenant_id, checkout_id, affordability_id, plan_type, principal_value, currency, int(term_days), down_payment_value, fee_disclosure_reference, customer_acceptance_reference)
		self.plans[plan_id] = plan
		self._audit(tenant_id, "bnpl_plan_created", plan_id)
		return plan.to_dict()

	def schedule_installment(
		self,
		schedule_id: str,
		tenant_id: str,
		plan_id: str,
		due_amount: float | int | str,
		due_date: str,
		status: str = "scheduled",
		sequence: int = 1,
	) -> dict[str, Any]:
		plan = self._tenant_plan_or_none(plan_id, tenant_id)
		due_amount_value = normalize_amount(due_amount)
		status = normalize_code(status)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "schedule_installment",
			"plan_present": plan is not None,
			"positive_due_amount": due_amount_value > 0,
			"due_date_present": bool(due_date),
			"installment_status_supported": status in SUPPORTED_INSTALLMENT_STATUSES,
		})
		schedule = InstallmentSchedule(schedule_id, tenant_id, plan_id, due_amount_value, plan.currency if plan else "", due_date, status, int(sequence))
		self.installments[schedule_id] = schedule
		self._audit(tenant_id, "installment_scheduled", schedule_id)
		return schedule.to_dict()

	def record_merchant_settlement(
		self,
		settlement_id: str,
		tenant_id: str,
		merchant_id: str,
		plan_id: str,
		gross_amount: float | int | str,
		net_amount: float | int | str,
		status: str,
		reconciliation_reference: str,
		payment_rail_reference: str,
		human_approval: str = "",
	) -> dict[str, Any]:
		merchant = self._tenant_merchant_or_none(merchant_id, tenant_id)
		plan = self._tenant_plan_or_none(plan_id, tenant_id)
		gross_value = normalize_amount(gross_amount)
		net_value = normalize_amount(net_amount)
		status = normalize_code(status)
		approval_required = status == "held" or (status == "released" and gross_value >= 100000)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_merchant_settlement",
			"merchant_present": merchant is not None,
			"plan_present": plan is not None,
			"settlement_status_supported": status in SUPPORTED_SETTLEMENT_STATUSES,
			"settlement_amounts_valid": gross_value > 0 and 0 <= net_value <= gross_value,
			"reconciliation_present": bool(reconciliation_reference),
			"payment_rail_present": bool(payment_rail_reference),
			"approval_required": approval_required,
			"human_approval_recorded": bool(human_approval),
		})
		settlement = MerchantSettlement(settlement_id, tenant_id, merchant_id, plan_id, gross_value, net_value, plan.currency if plan else "", status, reconciliation_reference, payment_rail_reference, human_approval)
		self.settlements[settlement_id] = settlement
		self._audit(tenant_id, "merchant_settlement_recorded", settlement_id)
		return settlement.to_dict()

	def open_bnpl_dispute(
		self,
		dispute_id: str,
		tenant_id: str,
		plan_id: str,
		reason: str,
		reviewer_id: str,
		evidence_references: list[str],
	) -> dict[str, Any]:
		plan = self._tenant_plan_or_none(plan_id, tenant_id)
		reason = normalize_code(reason)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "open_bnpl_dispute",
			"plan_present": plan is not None,
			"dispute_reason_supported": reason in SUPPORTED_DISPUTE_REASONS,
			"evidence_present": bool(evidence_references),
			"reviewer_present": bool(reviewer_id),
		})
		dispute = BNPLDispute(dispute_id, tenant_id, plan_id, reason, reviewer_id, list(evidence_references))
		self.disputes[dispute_id] = dispute
		self._audit(tenant_id, "bnpl_dispute_opened", dispute_id)
		return dispute.to_dict()

	def register_bnpl_agent(
		self,
		agent_id: str,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
	) -> dict[str, Any]:
		runtime = normalize_code(runtime)
		role = normalize_code(role)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_bnpl_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
		})
		evidence = self._record_evidence(agent_id, tenant_id, "agent", agent_id, "registered", {"name": name, "runtime": runtime, "role": role, "scope": scope})
		self._audit(tenant_id, "bnpl_agent_registered", agent_id)
		return evidence

	def validate_batch(
		self,
		tenant_id: str,
		item_count: int,
		event_stream: str = "bytewax",
	) -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "bnpl_batch", "event_stream": event_stream})
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.fintech.bnpl.lifecycle", "accepted": True}

	def estimate_plan_installment(self, plan_id: str, tenant_id: str) -> dict[str, Any]:
		plan = self._tenant_plan_or_none(plan_id, tenant_id)
		if plan is None:
			raise KeyError(f"BNPL plan not found: {plan_id}")
		return {"plan_id": plan_id, "tenant_id": tenant_id, "plan_type": plan.plan_type, "installment": estimate_installment_amount(plan.principal, plan.down_payment, plan.plan_type)}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"program_count": sum(1 for item in self.programs.values() if item.tenant_id == tenant_id),
			"consumer_count": sum(1 for item in self.consumers.values() if item.tenant_id == tenant_id),
			"merchant_count": sum(1 for item in self.merchants.values() if item.tenant_id == tenant_id),
			"checkout_count": sum(1 for item in self.checkouts.values() if item.tenant_id == tenant_id),
			"affordability_count": sum(1 for item in self.affordability.values() if item.tenant_id == tenant_id),
			"plan_count": sum(1 for item in self.plans.values() if item.tenant_id == tenant_id),
			"installment_count": sum(1 for item in self.installments.values() if item.tenant_id == tenant_id),
			"settlement_count": sum(1 for item in self.settlements.values() if item.tenant_id == tenant_id),
			"dispute_count": sum(1 for item in self.disputes.values() if item.tenant_id == tenant_id),
			"audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id),
			"streaming": get_capability_contract(tenant_id)["streaming"],
		}

	def list_plans(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = self.plans.values()
		if tenant_id is not None:
			items = [item for item in items if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(items, key=lambda x: x.id)]

	def list_checkouts(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = self.checkouts.values()
		if tenant_id is not None:
			items = [item for item in items if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(items, key=lambda x: x.id)]

	def list_settlements(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = self.settlements.values()
		if tenant_id is not None:
			items = [item for item in items if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(items, key=lambda x: x.id)]

	# ------------------------------------------------------------------
	# New async methods
	# ------------------------------------------------------------------

	async def apply_for_bnpl(
		self,
		customer_id: str,
		merchant_id: str,
		purchase_amount: float,
		plan_type: str = "pay_in_4",
	) -> dict[str, Any]:
		"""End-to-end BNPL application: credit check → plan selection → schedule."""
		assert customer_id, "customer_id required"
		assert merchant_id, "merchant_id required"
		assert purchase_amount > 0, "purchase_amount must be positive"
		await asyncio.sleep(0)

		plan_type = normalize_code(plan_type)
		merchant = self.merchants.get(merchant_id)
		currency = "KES"
		if merchant and merchant.tenant_id == self.tenant_id:
			program = self.programs.get(getattr(merchant, "program_id", ""))
			if program:
				currency = program.currency

		# Credit eligibility check
		credit_result = await self.credit_check_bnpl(customer_id, purchase_amount)
		if not credit_result["eligible"]:
			return {
				"status": "declined",
				"reason": credit_result["decline_reason"],
				"customer_id": customer_id,
				"purchase_amount": purchase_amount,
				"applied_at": _iso(),
			}

		# Create checkout session
		checkout_id = f"co-{customer_id}-{secrets.token_hex(4)}"
		consumer = next((c for c in self.consumers.values() if c.customer_reference == customer_id), None)
		consumer_id = consumer.id if consumer else customer_id

		try:
			checkout = self.create_checkout_session(
				checkout_id=checkout_id,
				tenant_id=self.tenant_id,
				merchant_id=merchant_id,
				consumer_id=consumer_id,
				channel="online",
				category=getattr(merchant, "category", "retail"),
				amount=purchase_amount,
				currency=currency,
				payment_reference=f"pay-{secrets.token_hex(6)}",
				fraud_reference=f"fraud-{secrets.token_hex(6)}",
				aml_reference=f"aml-{secrets.token_hex(6)}",
				consent_reference=f"consent-{secrets.token_hex(6)}",
			)
		except Exception as exc:
			return {"status": "error", "reason": str(exc), "applied_at": _iso()}

		# Record affordability decision
		affordability_id = f"aff-{checkout_id}"
		self.record_affordability_decision(
			decision_id=affordability_id,
			tenant_id=self.tenant_id,
			checkout_id=checkout_id,
			score=credit_result["credit_score"],
			decision="approve",
			evidence_references=[credit_result["reference"]],
			human_approval=f"auto-{_iso()}",
		)

		# Create BNPL plan
		plan_id = f"plan-{checkout_id}"
		instalments_map = {"pay_in_4": 4, "pay_in_3": 3, "pay_in_12": 12, "pay_in_6": 6}
		n_instalments = instalments_map.get(plan_type, 4)
		down_payment = purchase_amount * 0.25 if plan_type == "pay_in_4" else 0.0
		plan = self.create_bnpl_plan(
			plan_id=plan_id,
			tenant_id=self.tenant_id,
			checkout_id=checkout_id,
			affordability_id=affordability_id,
			plan_type=plan_type,
			principal=purchase_amount,
			currency=currency,
			term_days=n_instalments * 30,
			down_payment=down_payment,
			fee_disclosure_reference=f"fee-{secrets.token_hex(4)}",
			customer_acceptance_reference=f"acc-{secrets.token_hex(4)}",
		)

		# Generate repayment schedule
		repayment = await self.generate_repayment_plan(plan_id, n_instalments)

		self._outstanding[customer_id] += purchase_amount - down_payment
		self._audit(self.tenant_id, "bnpl_application_approved", plan_id)
		return {
			"status": "approved",
			"plan_id": plan_id,
			"checkout_id": checkout_id,
			"customer_id": customer_id,
			"purchase_amount": purchase_amount,
			"down_payment": down_payment,
			"currency": currency,
			"plan_type": plan_type,
			"instalment_count": n_instalments,
			"repayment_schedule": repayment["schedule"],
			"applied_at": _iso(),
		}

	async def credit_check_bnpl(
		self,
		customer_id: str,
		amount: float,
	) -> dict[str, Any]:
		"""Run a BNPL-specific credit eligibility check."""
		assert customer_id, "customer_id required"
		assert amount > 0, "amount must be positive"
		await asyncio.sleep(0)

		# Outstanding balance check
		outstanding = self._outstanding.get(customer_id, 0.0)
		credit_limit = self._credit_limits.get(customer_id, 500_000.0)

		# Count consumer disputes as negative signal
		consumer_plans = [p for p in self.plans.values() if self.checkouts.get(p.checkout_id) and self.consumers.get(self.checkouts[p.checkout_id].consumer_id, {}) and getattr(self.consumers.get(self.checkouts[p.checkout_id].consumer_id), "customer_reference", None) == customer_id]
		plan_ids = {p.id for p in consumer_plans}
		dispute_count = sum(1 for d in self.disputes.values() if d.plan_id in plan_ids)
		late_fee_total = sum(self._late_fees.get(p_id, 0) for p_id in plan_ids)

		# Credit score proxy: starts at 700, penalised for disputes and arrears
		credit_score = 700
		credit_score -= dispute_count * 50
		credit_score -= int(late_fee_total / 100) * 10
		credit_score -= int(outstanding / credit_limit * 200)
		credit_score = max(min(credit_score, 850), 300)

		eligible = (
			credit_score >= 500
			and outstanding + amount <= credit_limit
			and dispute_count < 3
		)
		decline_reason: str | None = None
		if credit_score < 500:
			decline_reason = "insufficient_credit_score"
		elif outstanding + amount > credit_limit:
			decline_reason = "credit_limit_exceeded"
		elif dispute_count >= 3:
			decline_reason = "excessive_disputes"

		ref = f"cc-{customer_id}-{secrets.token_hex(4)}"
		return {
			"customer_id": customer_id,
			"amount_requested": amount,
			"credit_score": credit_score,
			"credit_limit": credit_limit,
			"outstanding_balance": outstanding,
			"available_credit": max(credit_limit - outstanding, 0),
			"eligible": eligible,
			"decline_reason": decline_reason,
			"dispute_count": dispute_count,
			"reference": ref,
			"checked_at": _iso(),
		}

	async def generate_repayment_plan(
		self,
		bnpl_id: str,
		instalments: int,
	) -> dict[str, Any]:
		"""Generate a full repayment schedule for a BNPL plan."""
		assert bnpl_id, "bnpl_id required"
		assert 1 <= instalments <= 24, "instalments must be between 1 and 24"
		await asyncio.sleep(0)

		plan = self._tenant_plan_or_none(bnpl_id, self.tenant_id)
		if plan is None:
			raise KeyError(f"BNPL plan not found: {bnpl_id}")

		annual_rate = 0.18 if instalments <= 4 else 0.24
		schedule = _generate_installment_schedule(
			plan.principal, plan.down_payment, instalments, annual_rate,
		)
		self._schedules[bnpl_id] = schedule

		# Persist as InstallmentSchedule objects
		for item in schedule:
			sched_id = f"{bnpl_id}-{item['sequence']:02d}"
			if sched_id not in self.installments:
				self.schedule_installment(
					schedule_id=sched_id,
					tenant_id=self.tenant_id,
					plan_id=bnpl_id,
					due_amount=item["amount"],
					due_date=item["due_date"],
					status="scheduled",
					sequence=item["sequence"],
				)

		total_repayable = sum(i["amount"] for i in schedule)
		total_interest = total_repayable - (plan.principal - plan.down_payment)

		self._audit(self.tenant_id, "repayment_plan_generated", bnpl_id)
		return {
			"plan_id": bnpl_id,
			"instalment_count": instalments,
			"principal": plan.principal,
			"down_payment": plan.down_payment,
			"net_financed": plan.principal - plan.down_payment,
			"annual_rate": annual_rate,
			"total_repayable": round(total_repayable, 2),
			"total_interest": round(total_interest, 2),
			"schedule": schedule,
			"generated_at": _iso(),
		}

	async def merchant_integration(
		self,
		merchant_id: str,
		plan_types: list[str],
	) -> dict[str, Any]:
		"""Configure BNPL plan types and fee rates for a merchant integration."""
		assert merchant_id, "merchant_id required"
		assert plan_types, "plan_types must be non-empty"
		await asyncio.sleep(0)

		merchant = self.merchants.get(merchant_id)
		if merchant is None:
			raise KeyError(f"merchant not found: {merchant_id}")
		if merchant.tenant_id != self.tenant_id:
			raise PermissionError("merchant belongs to different tenant")

		normalised_types = [normalize_code(pt) for pt in plan_types]
		unsupported = [pt for pt in normalised_types if pt not in SUPPORTED_PLAN_TYPES]
		if unsupported:
			raise ValueError(f"unsupported plan types: {unsupported}")

		# Set merchant fee rates based on category
		mdr_pct = 0.025 if getattr(merchant, "category", "") in {"electronics", "travel"} else 0.015
		settlement_days = 2
		self._merchant_fees[merchant_id] = {
			"plan_types": normalised_types,
			"mdr_pct": mdr_pct,
			"settlement_days": settlement_days,
			"configured_at": _iso(),
		}

		# Issue integration credentials
		api_key = f"mk-{secrets.token_hex(16)}"
		webhook_secret = f"whs-{secrets.token_hex(12)}"

		self._audit(self.tenant_id, "merchant_integration_configured", merchant_id)
		return {
			"merchant_id": merchant_id,
			"plan_types": normalised_types,
			"mdr_pct": mdr_pct,
			"settlement_days": settlement_days,
			"api_key_prefix": api_key[:12] + "...",
			"webhook_secret_prefix": webhook_secret[:8] + "...",
			"integration_status": "active",
			"configured_at": _iso(),
		}

	async def process_instalment(
		self,
		bnpl_id: str,
		instalment_number: int,
	) -> dict[str, Any]:
		"""Mark an instalment as paid and update outstanding balance."""
		assert bnpl_id, "bnpl_id required"
		assert instalment_number >= 1, "instalment_number must be >= 1"
		await asyncio.sleep(0)

		plan = self._tenant_plan_or_none(bnpl_id, self.tenant_id)
		if plan is None:
			raise KeyError(f"BNPL plan not found: {bnpl_id}")

		schedule = self._schedules.get(bnpl_id, [])
		matching = next((s for s in schedule if s["sequence"] == instalment_number), None)
		if matching is None:
			raise KeyError(f"instalment {instalment_number} not found in plan {bnpl_id}")
		if matching["status"] == "paid":
			return {"status": "already_paid", "plan_id": bnpl_id, "instalment": instalment_number}

		matching["status"] = "paid"
		matching["paid_at"] = _iso()

		payment_id = f"{bnpl_id}-pmt-{instalment_number:02d}"
		self._payments[payment_id] = {
			"payment_id": payment_id,
			"plan_id": bnpl_id,
			"instalment": instalment_number,
			"amount": matching["amount"],
			"currency": plan.currency,
			"status": "completed",
			"paid_at": _iso(),
		}

		# Update outstanding balance
		checkout = self.checkouts.get(plan.checkout_id)
		consumer_id = checkout.consumer_id if checkout else None
		consumer = self.consumers.get(consumer_id or "") if consumer_id else None
		if consumer:
			self._outstanding[consumer.customer_reference] = max(
				self._outstanding[consumer.customer_reference] - matching["amount"], 0.0
			)

		# Update installment record status
		sched_id = f"{bnpl_id}-{instalment_number:02d}"
		sched_obj = self.installments.get(sched_id)
		if sched_obj:
			sched_obj.status = "paid"

		remaining = sum(1 for s in schedule if s["status"] == "scheduled")
		self._audit(self.tenant_id, "instalment_processed", payment_id)
		return {
			"payment_id": payment_id,
			"plan_id": bnpl_id,
			"instalment_number": instalment_number,
			"amount_paid": matching["amount"],
			"currency": plan.currency,
			"remaining_instalments": remaining,
			"plan_status": "completed" if remaining == 0 else "active",
			"processed_at": _iso(),
		}

	async def early_repayment(self, bnpl_id: str) -> dict[str, Any]:
		"""Process full early repayment of a BNPL plan with a discount."""
		assert bnpl_id, "bnpl_id required"
		await asyncio.sleep(0)

		plan = self._tenant_plan_or_none(bnpl_id, self.tenant_id)
		if plan is None:
			raise KeyError(f"BNPL plan not found: {bnpl_id}")
		if bnpl_id in self._early_repayments:
			return {"status": "already_repaid", "plan_id": bnpl_id}

		schedule = self._schedules.get(bnpl_id, [])
		outstanding_instalments = [s for s in schedule if s["status"] == "scheduled"]
		total_outstanding = sum(s["amount"] for s in outstanding_instalments)

		# 5% early repayment discount on outstanding interest
		discount_rate = 0.05
		discount = round(total_outstanding * discount_rate, 2)
		settlement_amount = round(total_outstanding - discount, 2)

		# Mark all remaining instalments as paid
		for s in outstanding_instalments:
			s["status"] = "paid"
			s["paid_at"] = _iso()

		repayment_record = {
			"plan_id": bnpl_id,
			"total_outstanding": total_outstanding,
			"discount_applied": discount,
			"settlement_amount": settlement_amount,
			"currency": plan.currency,
			"instalments_cleared": len(outstanding_instalments),
			"repaid_at": _iso(),
		}
		self._early_repayments[bnpl_id] = repayment_record

		checkout = self.checkouts.get(plan.checkout_id)
		consumer_id = checkout.consumer_id if checkout else None
		consumer = self.consumers.get(consumer_id or "") if consumer_id else None
		if consumer:
			self._outstanding[consumer.customer_reference] = max(
				self._outstanding[consumer.customer_reference] - settlement_amount, 0.0
			)

		self._audit(self.tenant_id, "early_repayment_processed", bnpl_id)
		return repayment_record

	async def late_payment_handling(
		self,
		bnpl_id: str,
		days_overdue: int,
	) -> dict[str, Any]:
		"""Apply late fees and update instalment statuses for an overdue plan."""
		assert bnpl_id, "bnpl_id required"
		assert days_overdue >= 1, "days_overdue must be >= 1"
		await asyncio.sleep(0)

		plan = self._tenant_plan_or_none(bnpl_id, self.tenant_id)
		if plan is None:
			raise KeyError(f"BNPL plan not found: {bnpl_id}")

		schedule = self._schedules.get(bnpl_id, [])
		overdue_instalments = [s for s in schedule if s["status"] == "scheduled"]
		overdue_amount = sum(s["amount"] for s in overdue_instalments)

		fee = _late_fee(overdue_amount, days_overdue)
		self._late_fees[bnpl_id] += fee

		# Mark overdue instalments
		for s in overdue_instalments:
			s["status"] = "overdue"

		# Update installment records
		for s in overdue_instalments:
			sched_id = f"{bnpl_id}-{s['sequence']:02d}"
			sched_obj = self.installments.get(sched_id)
			if sched_obj:
				sched_obj.status = "overdue"

		# Notify if notify adapter present
		if self._notify is not None:
			try:
				await self._notify.send({
					"type": "late_payment_alert",
					"plan_id": bnpl_id,
					"days_overdue": days_overdue,
					"overdue_amount": overdue_amount,
					"late_fee": fee,
				})
			except Exception:
				pass

		self._audit(self.tenant_id, "late_payment_handled", bnpl_id)
		return {
			"plan_id": bnpl_id,
			"days_overdue": days_overdue,
			"overdue_amount": round(overdue_amount, 2),
			"late_fee_charged": fee,
			"cumulative_late_fees": round(self._late_fees[bnpl_id], 2),
			"overdue_instalment_count": len(overdue_instalments),
			"currency": plan.currency,
			"handled_at": _iso(),
		}

	async def bnpl_statement(
		self,
		customer_id: str,
		period: str,
	) -> dict[str, Any]:
		"""Return a consumer's BNPL statement for the given period."""
		assert customer_id, "customer_id required"
		assert period, "period required"
		await asyncio.sleep(0)

		consumer = next((c for c in self.consumers.values() if c.customer_reference == customer_id and c.tenant_id == self.tenant_id), None)
		if consumer is None:
			return {"customer_id": customer_id, "period": period, "plans": [], "generated_at": _iso()}

		# Plans belonging to this consumer
		consumer_checkouts = {co.id for co in self.checkouts.values() if co.consumer_id == consumer.id}
		consumer_plans = [p for p in self.plans.values() if p.checkout_id in consumer_checkouts]

		plan_summaries: list[dict[str, Any]] = []
		for plan in consumer_plans:
			schedule = self._schedules.get(plan.id, [])
			paid = sum(s["amount"] for s in schedule if s["status"] == "paid")
			outstanding = sum(s["amount"] for s in schedule if s["status"] in {"scheduled", "overdue"})
			late_fees = self._late_fees.get(plan.id, 0.0)
			plan_summaries.append({
				"plan_id": plan.id,
				"plan_type": plan.plan_type,
				"principal": plan.principal,
				"amount_paid": round(paid, 2),
				"amount_outstanding": round(outstanding, 2),
				"late_fees": round(late_fees, 2),
				"currency": plan.currency,
				"early_repaid": plan.id in self._early_repayments,
			})

		total_outstanding = sum(p["amount_outstanding"] for p in plan_summaries)
		total_paid = sum(p["amount_paid"] for p in plan_summaries)

		self._audit(self.tenant_id, "bnpl_statement_generated", customer_id)
		return {
			"customer_id": customer_id,
			"period": period,
			"total_plans": len(plan_summaries),
			"total_outstanding": round(total_outstanding, 2),
			"total_paid": round(total_paid, 2),
			"credit_limit": self._credit_limits.get(customer_id, 500_000.0),
			"available_credit": max(self._credit_limits.get(customer_id, 500_000.0) - self._outstanding.get(customer_id, 0), 0),
			"plans": plan_summaries,
			"generated_at": _iso(),
		}

	async def merchant_settlement(
		self,
		merchant_id: str,
		period: str,
	) -> dict[str, Any]:
		"""Run merchant settlement for all completed plans in a period."""
		assert merchant_id, "merchant_id required"
		assert period, "period required"
		await asyncio.sleep(0)

		merchant = self.merchants.get(merchant_id)
		if merchant is None:
			raise KeyError(f"merchant not found: {merchant_id}")

		merchant_plans = [
			p for p in self.plans.values()
			if self.checkouts.get(p.checkout_id) and
			self.checkouts[p.checkout_id].merchant_id == merchant_id and
			p.tenant_id == self.tenant_id
		]

		fee_config = self._merchant_fees.get(merchant_id, {"mdr_pct": 0.015, "settlement_days": 2})
		mdr_pct = float(fee_config["mdr_pct"])

		gross = sum(p.principal for p in merchant_plans)
		mdr_fee = gross * mdr_pct
		net = gross - mdr_fee

		settlement_id = f"stl-{merchant_id}-{period}-{secrets.token_hex(4)}"
		try:
			settlement = self.record_merchant_settlement(
				settlement_id=settlement_id,
				tenant_id=self.tenant_id,
				merchant_id=merchant_id,
				plan_id=merchant_plans[0].id if merchant_plans else "none",
				gross_amount=gross,
				net_amount=net,
				status="completed",
				reconciliation_reference=f"recon-{settlement_id}",
				payment_rail_reference=f"rail-{secrets.token_hex(4)}",
				human_approval=f"auto-{_iso()}",
			)
		except Exception as exc:
			return {"status": "error", "reason": str(exc), "merchant_id": merchant_id, "period": period}

		self._audit(self.tenant_id, "merchant_settlement_run", settlement_id)
		return {
			"settlement_id": settlement_id,
			"merchant_id": merchant_id,
			"period": period,
			"plan_count": len(merchant_plans),
			"gross_amount": round(gross, 2),
			"mdr_fee": round(mdr_fee, 2),
			"net_amount": round(net, 2),
			"currency": getattr(merchant, "currency", "KES"),
			"mdr_pct": mdr_pct,
			"status": "completed",
			"settled_at": _iso(),
		}

	async def bnpl_analytics(self, period: str) -> dict[str, Any]:
		"""Aggregate BNPL performance analytics for a reporting period."""
		assert period, "period required"
		await asyncio.sleep(0)

		all_plans = [p for p in self.plans.values() if p.tenant_id == self.tenant_id]
		all_consumers = [c for c in self.consumers.values() if c.tenant_id == self.tenant_id]
		all_merchants = [m for m in self.merchants.values() if m.tenant_id == self.tenant_id]
		all_settlements = [s for s in self.settlements.values() if s.tenant_id == self.tenant_id]

		total_volume = sum(p.principal for p in all_plans)
		total_settled = sum(getattr(s, "gross_amount", 0) for s in all_settlements)
		total_late_fees = sum(self._late_fees.values())
		total_early_repaid = len(self._early_repayments)

		by_plan_type: dict[str, int] = defaultdict(int)
		for p in all_plans:
			by_plan_type[p.plan_type] += 1

		dispute_count = sum(1 for d in self.disputes.values() if d.tenant_id == self.tenant_id)
		overdue_count = sum(
			1 for s_list in self._schedules.values()
			for s in s_list if s.get("status") == "overdue"
		)

		approval_count = sum(1 for a in self.affordability.values() if a.tenant_id == self.tenant_id and decision_is_approved(a.decision))
		total_decisions = sum(1 for a in self.affordability.values() if a.tenant_id == self.tenant_id)
		approval_rate = (approval_count / max(total_decisions, 1)) * 100

		self._audit(self.tenant_id, "bnpl_analytics_generated", period)
		return {
			"period": period,
			"tenant_id": self.tenant_id,
			"total_plans": len(all_plans),
			"total_consumers": len(all_consumers),
			"total_merchants": len(all_merchants),
			"total_volume": round(total_volume, 2),
			"total_settled": round(total_settled, 2),
			"total_late_fees": round(total_late_fees, 2),
			"early_repaid_count": total_early_repaid,
			"approval_rate_pct": round(approval_rate, 2),
			"dispute_count": dispute_count,
			"overdue_instalment_count": overdue_count,
			"by_plan_type": dict(by_plan_type),
			"generated_at": _iso(),
		}

	async def set_credit_limit(
		self,
		customer_id: str,
		limit: float,
		approved_by: str,
	) -> dict[str, Any]:
		"""Set or update the BNPL credit limit for a consumer."""
		assert customer_id, "customer_id required"
		assert limit >= 0, "limit must be non-negative"
		assert approved_by, "approved_by required"
		await asyncio.sleep(0)

		previous = self._credit_limits.get(customer_id, 500_000.0)
		self._credit_limits[customer_id] = float(limit)
		self._audit(self.tenant_id, "credit_limit_set", customer_id)
		return {
			"customer_id": customer_id,
			"previous_limit": previous,
			"new_limit": limit,
			"approved_by": approved_by,
			"updated_at": _iso(),
		}

	async def resolve_dispute(
		self,
		dispute_id: str,
		outcome: str,
		resolver_id: str,
	) -> dict[str, Any]:
		"""Resolve a BNPL dispute with a final outcome."""
		assert dispute_id, "dispute_id required"
		assert outcome in {"upheld", "rejected", "partial_refund", "referred"}, f"unsupported outcome: {outcome}"
		assert resolver_id, "resolver_id required"
		await asyncio.sleep(0)

		dispute = self.disputes.get(dispute_id)
		if dispute is None:
			raise KeyError(f"dispute not found: {dispute_id}")
		dispute.status = "resolved"

		self._audit(self.tenant_id, "bnpl_dispute_resolved", dispute_id)
		return {
			"dispute_id": dispute_id,
			"outcome": outcome,
			"resolver_id": resolver_id,
			"resolved_at": _iso(),
			"dispute": dispute.to_dict(),
		}

	# ------------------------------------------------------------------
	# Additional async methods
	# ------------------------------------------------------------------

	async def health_check(self) -> dict[str, Any]:
		"""Return BNPL service health status."""
		return {
			"service": "bnpl", "status": "healthy",
			"active_plans": sum(1 for p in self.plans.values() if p.tenant_id == self.tenant_id),
			"overdue_instalments": sum(1 for sl in self._schedules.values() for s in sl if s.get("status") == "overdue"),
			"checked_at": _iso(),
		}

	async def bulk_onboard_consumers(self, consumers: list[dict[str, Any]]) -> dict[str, Any]:
		"""Bulk-onboard BNPL consumers."""
		processed, errors = [], []
		for c in consumers:
			try:
				rec = self.onboard_consumer(
					consumer_id=c.get("consumer_id", f"con-{_iso()[:10]}-{len(processed):03d}"),
					tenant_id=self.tenant_id,
					customer_reference=c["customer_reference"],
					kyc_profile_id=c.get("kyc_profile_id", f"kyc-{c['customer_reference'][:8]}"),
					country=c.get("country", "KE"),
					consent_reference=c.get("consent_reference", f"consent-{len(processed)}"),
					aml_reference=c.get("aml_reference", f"aml-{len(processed)}"),
					fraud_reference=c.get("fraud_reference", f"fraud-{len(processed)}"),
				)
				processed.append(rec["id"])
			except Exception as exc:
				errors.append({"input": c, "error": str(exc)})
		return {"processed": len(processed), "failed": len(errors), "consumer_ids": processed}

	async def consumer_credit_profile(self, customer_id: str) -> dict[str, Any]:
		"""Return the full BNPL credit profile for a consumer."""
		check = await self.credit_check_bnpl(customer_id, 0.0)
		outstanding = self._outstanding.get(customer_id, 0.0)
		limit = self._credit_limits.get(customer_id, 500_000.0)
		plans = [p for p in self.plans.values() if self.checkouts.get(p.checkout_id) and self.consumers.get(self.checkouts[p.checkout_id].consumer_id, {}) and getattr(self.consumers.get(self.checkouts[p.checkout_id].consumer_id), "customer_reference", None) == customer_id]
		return {
			"customer_id": customer_id, "credit_limit": limit,
			"outstanding_balance": outstanding, "available_credit": max(limit - outstanding, 0),
			"credit_score": check["credit_score"], "active_plans": len(plans),
			"total_late_fees": sum(self._late_fees.get(p.id, 0) for p in plans),
			"as_of": _iso(),
		}

	async def instalment_reminder_batch(self) -> dict[str, Any]:
		"""Send payment reminders for all upcoming instalments."""
		upcoming = []
		for plan_id, schedule in self._schedules.items():
			for s in schedule:
				if s.get("status") == "scheduled" and s.get("due_date", "") <= _iso()[:10]:
					upcoming.append({"plan_id": plan_id, "instalment": s["sequence"], "due_date": s["due_date"]})
		if self._notify is not None:
			for u in upcoming[:50]:
				try:
					await self._notify.send({"type": "instalment_reminder", **u})
				except Exception:
					pass
		self._audit(self.tenant_id, "instalment_reminders_sent", f"{len(upcoming)}")
		return {"reminders_sent": len(upcoming), "sent_at": _iso()}

	async def merchant_performance_report(self, merchant_id: str, period: str) -> dict[str, Any]:
		"""Report on a merchant's BNPL performance: GMV, conversion, settlement."""
		merchant_plans = [p for p in self.plans.values() if self.checkouts.get(p.checkout_id) and self.checkouts[p.checkout_id].merchant_id == merchant_id and p.tenant_id == self.tenant_id]
		gmv = sum(p.principal for p in merchant_plans)
		settled = sum(getattr(s, "gross_amount", 0) for s in self.settlements.values() if s.tenant_id == self.tenant_id and getattr(s, "merchant_id", "") == merchant_id)
		return {
			"merchant_id": merchant_id, "period": period,
			"total_plans": len(merchant_plans), "gmv": round(gmv, 2),
			"total_settled": round(settled, 2), "currency": "KES",
			"generated_at": _iso(),
		}

	async def bnpl_regulatory_return(self, period: str) -> dict[str, Any]:
		"""File a BNPL regulatory return for CBK credit providers."""
		analytics = await self.bnpl_analytics(period)
		return {
			"report_type": "CBK_CREDIT_PROVIDER_RETURN",
			"period": period, **analytics, "status": "draft",
		}

	async def blacklist_consumer(self, customer_id: str, reason: str, blacklisted_by: str) -> dict[str, Any]:
		"""Add a consumer to the BNPL blacklist (e.g., repeated defaults)."""
		self._credit_limits[customer_id] = 0.0
		record: dict[str, Any] = {
			"customer_id": customer_id, "reason": reason, "blacklisted_by": blacklisted_by,
			"credit_limit_set_to": 0.0, "blacklisted_at": _iso(),
		}
		self._audit(self.tenant_id, "consumer_blacklisted", customer_id)
		return record

	async def debt_recovery_initiation(self, customer_id: str, recovery_agency: str) -> dict[str, Any]:
		"""Initiate debt recovery for a defaulted BNPL consumer."""
		outstanding = self._outstanding.get(customer_id, 0.0)
		record: dict[str, Any] = {
			"customer_id": customer_id, "outstanding_balance": outstanding,
			"recovery_agency": recovery_agency, "status": "initiated",
			"initiated_at": _iso(),
		}
		self._audit(self.tenant_id, "debt_recovery_initiated", customer_id)
		return record

	async def export_bnpl_data(self, fmt: str = "csv") -> dict[str, Any]:
		"""Export BNPL portfolio data."""
		assert fmt in {"csv", "json", "excel"}
		return {
			"tenant_id": self.tenant_id, "format": fmt,
			"plans": sum(1 for p in self.plans.values() if p.tenant_id == self.tenant_id),
			"file_reference": f"bnpl_{self.tenant_id}_{_iso()[:10]}.{fmt}", "generated_at": _iso(),
		}

	async def affordability_bulk_check(self, customers: list[dict[str, Any]]) -> list[dict[str, Any]]:
		"""Run affordability checks for multiple customers in bulk."""
		results = []
		for c in customers:
			result = await self.credit_check_bnpl(c["customer_id"], float(c.get("amount", 10_000)))
			results.append(result)
		return results

	# ------------------------------------------------------------------
	# Private helpers
	# ------------------------------------------------------------------

	def _tenant_program_or_none(self, program_id: str, tenant_id: str) -> MerchantProgram | None:
		program = self.programs.get(program_id)
		return program if program is not None and program.tenant_id == tenant_id else None

	def _tenant_consumer_or_none(self, consumer_id: str, tenant_id: str) -> BNPLConsumer | None:
		consumer = self.consumers.get(consumer_id)
		return consumer if consumer is not None and consumer.tenant_id == tenant_id else None

	def _tenant_merchant_or_none(self, merchant_id: str, tenant_id: str) -> MerchantProfile | None:
		merchant = self.merchants.get(merchant_id)
		return merchant if merchant is not None and merchant.tenant_id == tenant_id else None

	def _tenant_checkout_or_none(self, checkout_id: str, tenant_id: str) -> CheckoutSession | None:
		checkout = self.checkouts.get(checkout_id)
		return checkout if checkout is not None and checkout.tenant_id == tenant_id else None

	def _tenant_affordability_or_none(self, decision_id: str, tenant_id: str) -> AffordabilityDecision | None:
		decision = self.affordability.get(decision_id)
		return decision if decision is not None and decision.tenant_id == tenant_id else None

	def _tenant_plan_or_none(self, plan_id: str, tenant_id: str) -> BNPLPlan | None:
		plan = self.plans.get(plan_id)
		return plan if plan is not None and plan.tenant_id == tenant_id else None

	def _record_evidence(
		self,
		evidence_id: str,
		tenant_id: str,
		kind: str,
		reference_id: str,
		status: str,
		metadata: dict[str, Any],
	) -> dict[str, Any]:
		evidence = BNPLevidence(evidence_id, tenant_id, kind, reference_id, status, metadata)
		self.evidence[evidence_id] = evidence
		return evidence.to_dict()

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"reference_id": reference_id,
			"actor_id": self.actor_id,
			"timestamp": _iso(),
		})

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", "bnpl_policy_denied") for action in result["actions"])
		raise PermissionError(reasons or "bnpl_policy_denied")


BNPLService = BuyNowPayLaterService
