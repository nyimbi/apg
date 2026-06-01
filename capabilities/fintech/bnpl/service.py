"""Executable service layer for APG Buy Now Pay Later."""

from __future__ import annotations

from typing import Any

try:
	from .bnpl_runtime import decision_is_approved, decision_is_final, estimate_installment_amount, normalize_amount, normalize_code, normalize_country, normalize_currency, normalize_score
	from .capability_contract import SUPPORTED_AFFORDABILITY_DECISIONS, SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_CHECKOUT_CHANNELS, SUPPORTED_COUNTRIES, SUPPORTED_CURRENCIES, SUPPORTED_DISPUTE_REASONS, SUPPORTED_INSTALLMENT_STATUSES, SUPPORTED_MERCHANT_CATEGORIES, SUPPORTED_PLAN_TYPES, SUPPORTED_SETTLEMENT_STATUSES, evaluate_capability_rules, get_capability_contract
	from .models import AffordabilityDecision, BNPLConsumer, BNPLDispute, BNPLevidence, BNPLPlan, CheckoutSession, InstallmentSchedule, MerchantProfile, MerchantProgram, MerchantSettlement
except ImportError:  # pragma: no cover - supports direct file loading in tests
	from bnpl_runtime import decision_is_approved, decision_is_final, estimate_installment_amount, normalize_amount, normalize_code, normalize_country, normalize_currency, normalize_score  # type: ignore
	from capability_contract import SUPPORTED_AFFORDABILITY_DECISIONS, SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_CHECKOUT_CHANNELS, SUPPORTED_COUNTRIES, SUPPORTED_CURRENCIES, SUPPORTED_DISPUTE_REASONS, SUPPORTED_INSTALLMENT_STATUSES, SUPPORTED_MERCHANT_CATEGORIES, SUPPORTED_PLAN_TYPES, SUPPORTED_SETTLEMENT_STATUSES, evaluate_capability_rules, get_capability_contract  # type: ignore
	from models import AffordabilityDecision, BNPLConsumer, BNPLDispute, BNPLevidence, BNPLPlan, CheckoutSession, InstallmentSchedule, MerchantProfile, MerchantProgram, MerchantSettlement  # type: ignore


class BNPLService:
	"""Dependency-light BNPL runtime for generated applications."""

	def __init__(self) -> None:
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

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def register_merchant_program(self, program_id: str, tenant_id: str, name: str, owner_id: str, country: str, currency: str, settlement_policy_reference: str, fee_disclosure_reference: str, max_installments: int, policy_attached: bool = True) -> dict[str, Any]:
		country = normalize_country(country)
		currency = normalize_currency(currency)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "register_merchant_program", "owner_present": bool(owner_id), "country_supported": country in SUPPORTED_COUNTRIES, "currency_supported": currency in SUPPORTED_CURRENCIES, "settlement_policy_present": bool(settlement_policy_reference), "fee_disclosure_present": bool(fee_disclosure_reference), "installment_count_valid": 1 <= int(max_installments) <= 24})
		if program_id in self.programs:
			raise ValueError(f"merchant program already exists: {program_id}")
		program = MerchantProgram(program_id, tenant_id, name, owner_id, country, currency, settlement_policy_reference, fee_disclosure_reference, int(max_installments))
		self.programs[program_id] = program
		self._audit(tenant_id, "bnpl_program_registered", program_id)
		return program.to_dict()

	def onboard_consumer(self, consumer_id: str, tenant_id: str, customer_reference: str, kyc_profile_id: str, country: str, consent_reference: str, aml_reference: str, fraud_reference: str, policy_attached: bool = True) -> dict[str, Any]:
		country = normalize_country(country)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "onboard_consumer", "customer_present": bool(customer_reference), "kyc_present": bool(kyc_profile_id), "country_supported": country in SUPPORTED_COUNTRIES, "consent_present": bool(consent_reference), "aml_present": bool(aml_reference), "fraud_present": bool(fraud_reference)})
		if consumer_id in self.consumers:
			raise ValueError(f"consumer already exists: {consumer_id}")
		consumer = BNPLConsumer(consumer_id, tenant_id, customer_reference, kyc_profile_id, country, consent_reference, aml_reference, fraud_reference)
		self.consumers[consumer_id] = consumer
		self._audit(tenant_id, "bnpl_consumer_onboarded", consumer_id)
		return consumer.to_dict()

	def register_merchant(self, merchant_id: str, tenant_id: str, program_id: str, legal_entity_reference: str, category: str, country: str, risk_tier: str, settlement_account: str, policy_attached: bool = True) -> dict[str, Any]:
		program = self._tenant_program_or_none(program_id, tenant_id)
		category = normalize_code(category)
		country = normalize_country(country)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "register_merchant", "program_present": program is not None, "legal_entity_present": bool(legal_entity_reference), "merchant_category_supported": category in SUPPORTED_MERCHANT_CATEGORIES, "country_supported": country in SUPPORTED_COUNTRIES, "risk_tier_present": bool(risk_tier), "settlement_account_present": bool(settlement_account)})
		if merchant_id in self.merchants:
			raise ValueError(f"merchant already exists: {merchant_id}")
		merchant = MerchantProfile(merchant_id, tenant_id, program_id, legal_entity_reference, category, country, risk_tier, settlement_account)
		self.merchants[merchant_id] = merchant
		self._audit(tenant_id, "bnpl_merchant_registered", merchant_id)
		return merchant.to_dict()

	def create_checkout_session(self, checkout_id: str, tenant_id: str, merchant_id: str, consumer_id: str, channel: str, category: str, amount: float | int | str, currency: str, payment_reference: str, fraud_reference: str, aml_reference: str, consent_reference: str, human_review: str = "", policy_attached: bool = True) -> dict[str, Any]:
		merchant = self._tenant_merchant_or_none(merchant_id, tenant_id)
		consumer = self._tenant_consumer_or_none(consumer_id, tenant_id)
		channel = normalize_code(channel)
		category = normalize_code(category)
		amount_value = normalize_amount(amount)
		currency = normalize_currency(currency)
		high_value = amount_value >= 100000
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "create_checkout_session", "merchant_present": merchant is not None, "consumer_present": consumer is not None, "channel_supported": channel in SUPPORTED_CHECKOUT_CHANNELS, "merchant_category_supported": category in SUPPORTED_MERCHANT_CATEGORIES, "positive_amount": amount_value > 0, "currency_supported": currency in SUPPORTED_CURRENCIES, "payment_reference_present": bool(payment_reference), "fraud_present": bool(fraud_reference), "aml_present": bool(aml_reference), "consent_present": bool(consent_reference), "high_value": high_value, "human_review_recorded": bool(human_review)})
		if checkout_id in self.checkouts:
			raise ValueError(f"checkout already exists: {checkout_id}")
		checkout = CheckoutSession(checkout_id, tenant_id, merchant_id, consumer_id, channel, category, amount_value, currency, payment_reference, fraud_reference, aml_reference, consent_reference, human_review)
		self.checkouts[checkout_id] = checkout
		self._audit(tenant_id, "checkout_session_created", checkout_id)
		return checkout.to_dict()

	def record_affordability_decision(self, decision_id: str, tenant_id: str, checkout_id: str, score: float | int | str, decision: str, evidence_references: list[str], human_approval: str, adverse_reason: str = "") -> dict[str, Any]:
		checkout = self._tenant_checkout_or_none(checkout_id, tenant_id)
		score_value = normalize_score(score)
		decision = normalize_code(decision)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_affordability_decision", "checkout_present": checkout is not None, "decision_supported": decision in SUPPORTED_AFFORDABILITY_DECISIONS, "score_in_range": 0 <= score_value <= 1000, "decision_evidence_present": bool(evidence_references), "adverse_decision": decision == "decline", "adverse_reason_present": bool(adverse_reason), "final_decision": decision_is_final(decision), "human_approval_recorded": bool(human_approval)})
		record = AffordabilityDecision(decision_id, tenant_id, checkout_id, score_value, decision, list(evidence_references), human_approval, adverse_reason)
		self.affordability[decision_id] = record
		self._audit(tenant_id, "affordability_decision_recorded", decision_id)
		return record.to_dict()

	def create_bnpl_plan(self, plan_id: str, tenant_id: str, checkout_id: str, affordability_id: str, plan_type: str, principal: float | int | str, currency: str, term_days: int, down_payment: float | int | str, fee_disclosure_reference: str, customer_acceptance_reference: str, policy_attached: bool = True) -> dict[str, Any]:
		checkout = self._tenant_checkout_or_none(checkout_id, tenant_id)
		affordability = self._tenant_affordability_or_none(affordability_id, tenant_id)
		plan_type = normalize_code(plan_type)
		principal_value = normalize_amount(principal)
		down_payment_value = normalize_amount(down_payment)
		currency = normalize_currency(currency)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "create_bnpl_plan", "checkout_present": checkout is not None, "affordability_approved": affordability is not None and decision_is_approved(affordability.decision), "plan_type_supported": plan_type in SUPPORTED_PLAN_TYPES, "positive_principal": principal_value > 0, "currency_supported": currency in SUPPORTED_CURRENCIES, "term_valid": 1 <= int(term_days) <= 730, "down_payment_valid": 0 <= down_payment_value <= principal_value, "fee_disclosure_present": bool(fee_disclosure_reference), "customer_acceptance_present": bool(customer_acceptance_reference)})
		if plan_id in self.plans:
			raise ValueError(f"BNPL plan already exists: {plan_id}")
		plan = BNPLPlan(plan_id, tenant_id, checkout_id, affordability_id, plan_type, principal_value, currency, int(term_days), down_payment_value, fee_disclosure_reference, customer_acceptance_reference)
		self.plans[plan_id] = plan
		self._audit(tenant_id, "bnpl_plan_created", plan_id)
		return plan.to_dict()

	def schedule_installment(self, schedule_id: str, tenant_id: str, plan_id: str, due_amount: float | int | str, due_date: str, status: str = "scheduled", sequence: int = 1) -> dict[str, Any]:
		plan = self._tenant_plan_or_none(plan_id, tenant_id)
		due_amount_value = normalize_amount(due_amount)
		status = normalize_code(status)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "schedule_installment", "plan_present": plan is not None, "positive_due_amount": due_amount_value > 0, "due_date_present": bool(due_date), "installment_status_supported": status in SUPPORTED_INSTALLMENT_STATUSES})
		schedule = InstallmentSchedule(schedule_id, tenant_id, plan_id, due_amount_value, plan.currency if plan else "", due_date, status, int(sequence))
		self.installments[schedule_id] = schedule
		self._audit(tenant_id, "installment_scheduled", schedule_id)
		return schedule.to_dict()

	def record_merchant_settlement(self, settlement_id: str, tenant_id: str, merchant_id: str, plan_id: str, gross_amount: float | int | str, net_amount: float | int | str, status: str, reconciliation_reference: str, payment_rail_reference: str, human_approval: str = "") -> dict[str, Any]:
		merchant = self._tenant_merchant_or_none(merchant_id, tenant_id)
		plan = self._tenant_plan_or_none(plan_id, tenant_id)
		gross_value = normalize_amount(gross_amount)
		net_value = normalize_amount(net_amount)
		status = normalize_code(status)
		approval_required = status == "held" or (status == "released" and gross_value >= 100000)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_merchant_settlement", "merchant_present": merchant is not None, "plan_present": plan is not None, "settlement_status_supported": status in SUPPORTED_SETTLEMENT_STATUSES, "settlement_amounts_valid": gross_value > 0 and 0 <= net_value <= gross_value, "reconciliation_present": bool(reconciliation_reference), "payment_rail_present": bool(payment_rail_reference), "approval_required": approval_required, "human_approval_recorded": bool(human_approval)})
		settlement = MerchantSettlement(settlement_id, tenant_id, merchant_id, plan_id, gross_value, net_value, plan.currency if plan else "", status, reconciliation_reference, payment_rail_reference, human_approval)
		self.settlements[settlement_id] = settlement
		self._audit(tenant_id, "merchant_settlement_recorded", settlement_id)
		return settlement.to_dict()

	def open_bnpl_dispute(self, dispute_id: str, tenant_id: str, plan_id: str, reason: str, reviewer_id: str, evidence_references: list[str]) -> dict[str, Any]:
		plan = self._tenant_plan_or_none(plan_id, tenant_id)
		reason = normalize_code(reason)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "open_bnpl_dispute", "plan_present": plan is not None, "dispute_reason_supported": reason in SUPPORTED_DISPUTE_REASONS, "evidence_present": bool(evidence_references), "reviewer_present": bool(reviewer_id)})
		dispute = BNPLDispute(dispute_id, tenant_id, plan_id, reason, reviewer_id, list(evidence_references))
		self.disputes[dispute_id] = dispute
		self._audit(tenant_id, "bnpl_dispute_opened", dispute_id)
		return dispute.to_dict()

	def register_bnpl_agent(self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		runtime = normalize_code(runtime)
		role = normalize_code(role)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_bnpl_agent", "agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES, "agent_role_supported": role in SUPPORTED_AGENT_ROLES})
		evidence = self._record_evidence(agent_id, tenant_id, "agent", agent_id, "registered", {"name": name, "runtime": runtime, "role": role, "scope": scope})
		self._audit(tenant_id, "bnpl_agent_registered", agent_id)
		return evidence

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "bnpl_batch", "event_stream": event_stream})
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.fintech.bnpl.lifecycle", "accepted": True}

	def estimate_plan_installment(self, plan_id: str, tenant_id: str) -> dict[str, Any]:
		plan = self._tenant_plan_or_none(plan_id, tenant_id)
		if plan is None:
			raise KeyError(f"BNPL plan not found: {plan_id}")
		return {"plan_id": plan_id, "tenant_id": tenant_id, "plan_type": plan.plan_type, "installment": estimate_installment_amount(plan.principal, plan.down_payment, plan.plan_type)}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {"tenant_id": tenant_id, "program_count": sum(1 for item in self.programs.values() if item.tenant_id == tenant_id), "consumer_count": sum(1 for item in self.consumers.values() if item.tenant_id == tenant_id), "merchant_count": sum(1 for item in self.merchants.values() if item.tenant_id == tenant_id), "checkout_count": sum(1 for item in self.checkouts.values() if item.tenant_id == tenant_id), "affordability_count": sum(1 for item in self.affordability.values() if item.tenant_id == tenant_id), "plan_count": sum(1 for item in self.plans.values() if item.tenant_id == tenant_id), "installment_count": sum(1 for item in self.installments.values() if item.tenant_id == tenant_id), "settlement_count": sum(1 for item in self.settlements.values() if item.tenant_id == tenant_id), "dispute_count": sum(1 for item in self.disputes.values() if item.tenant_id == tenant_id), "audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id), "streaming": get_capability_contract(tenant_id)["streaming"]}

	def list_plans(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = self.plans.values()
		if tenant_id is not None:
			items = [item for item in items if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(items, key=lambda item: item.id)]

	def list_checkouts(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = self.checkouts.values()
		if tenant_id is not None:
			items = [item for item in items if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(items, key=lambda item: item.id)]

	def list_settlements(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = self.settlements.values()
		if tenant_id is not None:
			items = [item for item in items if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(items, key=lambda item: item.id)]

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

	def _record_evidence(self, evidence_id: str, tenant_id: str, kind: str, reference_id: str, status: str, metadata: dict[str, Any]) -> dict[str, Any]:
		evidence = BNPLevidence(evidence_id, tenant_id, kind, reference_id, status, metadata)
		self.evidence[evidence_id] = evidence
		return evidence.to_dict()

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({"tenant_id": tenant_id, "event_type": event_type, "reference_id": reference_id})

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", "bnpl_policy_denied") for action in result["actions"])
		raise PermissionError(reasons or "bnpl_policy_denied")


BuyNowPayLaterService = BNPLService
