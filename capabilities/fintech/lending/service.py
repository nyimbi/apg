"""Executable service layer for APG Digital Lending."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_APPLICATION_PURPOSES, SUPPORTED_COLLECTION_REASONS, SUPPORTED_COUNTRIES, SUPPORTED_CURRENCIES, SUPPORTED_DISBURSEMENT_RAILS, SUPPORTED_OFFER_STATUSES, SUPPORTED_PRODUCT_TYPES, SUPPORTED_REPAYMENT_FREQUENCIES, SUPPORTED_UNDERWRITING_DECISIONS, evaluate_capability_rules, get_capability_contract
	from .lending_runtime import decision_category, estimate_installment, normalize_amount, normalize_code, normalize_country, normalize_currency, normalize_rate, normalize_score
	from .models import BorrowerProfile, CollectionCase, Disbursement, LendingEvidence, LoanApplication, LoanOffer, LoanProduct, RepaymentSchedule, UnderwritingDecision
except ImportError:  # pragma: no cover - supports direct file loading in tests
	from capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_APPLICATION_PURPOSES, SUPPORTED_COLLECTION_REASONS, SUPPORTED_COUNTRIES, SUPPORTED_CURRENCIES, SUPPORTED_DISBURSEMENT_RAILS, SUPPORTED_OFFER_STATUSES, SUPPORTED_PRODUCT_TYPES, SUPPORTED_REPAYMENT_FREQUENCIES, SUPPORTED_UNDERWRITING_DECISIONS, evaluate_capability_rules, get_capability_contract  # type: ignore
	from lending_runtime import decision_category, estimate_installment, normalize_amount, normalize_code, normalize_country, normalize_currency, normalize_rate, normalize_score  # type: ignore
	from models import BorrowerProfile, CollectionCase, Disbursement, LendingEvidence, LoanApplication, LoanOffer, LoanProduct, RepaymentSchedule, UnderwritingDecision  # type: ignore


class LendingService:
	"""Dependency-light lending runtime for generated applications."""

	def __init__(self) -> None:
		self.products: dict[str, LoanProduct] = {}
		self.borrowers: dict[str, BorrowerProfile] = {}
		self.applications: dict[str, LoanApplication] = {}
		self.underwriting: dict[str, UnderwritingDecision] = {}
		self.offers: dict[str, LoanOffer] = {}
		self.disbursements: dict[str, Disbursement] = {}
		self.repayments: dict[str, RepaymentSchedule] = {}
		self.collections: dict[str, CollectionCase] = {}
		self.evidence: dict[str, LendingEvidence] = {}
		self.audit_events: list[dict[str, Any]] = []

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def register_product(self, product_id: str, tenant_id: str, name: str, owner_id: str, product_type: str, currency: str, min_amount: float | int | str, max_amount: float | int | str, min_term_days: int, max_term_days: int, annual_rate: float | int | str, repayment_frequency: str, policy_attached: bool = True) -> dict[str, Any]:
		product_type = normalize_code(product_type)
		currency = normalize_currency(currency)
		minimum = normalize_amount(min_amount)
		maximum = normalize_amount(max_amount)
		rate = normalize_rate(annual_rate)
		frequency = normalize_code(repayment_frequency)
		term_valid = 7 <= int(min_term_days) <= int(max_term_days) <= 3650
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "register_product", "owner_present": bool(owner_id), "currency_supported": currency in SUPPORTED_CURRENCIES, "product_type_supported": product_type in SUPPORTED_PRODUCT_TYPES, "term_valid": term_valid, "rate_valid": 0 <= rate <= 0.75, "amount_limits_valid": 0 < minimum <= maximum <= 1000000, "repayment_frequency_supported": frequency in SUPPORTED_REPAYMENT_FREQUENCIES})
		if product_id in self.products:
			raise ValueError(f"loan product already exists: {product_id}")
		product = LoanProduct(product_id, tenant_id, name, owner_id, product_type, currency, minimum, maximum, int(min_term_days), int(max_term_days), rate, frequency)
		self.products[product_id] = product
		self._audit(tenant_id, "loan_product_registered", product_id)
		return product.to_dict()

	def onboard_borrower(self, borrower_id: str, tenant_id: str, customer_reference: str, kyc_profile_id: str, country: str, income_evidence_id: str, consent_reference: str, policy_attached: bool = True) -> dict[str, Any]:
		country = normalize_country(country)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "onboard_borrower", "customer_present": bool(customer_reference), "kyc_present": bool(kyc_profile_id), "country_supported": country in SUPPORTED_COUNTRIES, "income_evidence_present": bool(income_evidence_id), "consent_present": bool(consent_reference)})
		if borrower_id in self.borrowers:
			raise ValueError(f"borrower already exists: {borrower_id}")
		borrower = BorrowerProfile(borrower_id, tenant_id, customer_reference, kyc_profile_id, country, income_evidence_id, consent_reference)
		self.borrowers[borrower_id] = borrower
		self._audit(tenant_id, "borrower_onboarded", borrower_id)
		return borrower.to_dict()

	def submit_application(self, application_id: str, tenant_id: str, borrower_id: str, product_id: str, requested_amount: float | int | str, purpose: str, affordability_reference: str, bank_statement_reference: str, aml_reference: str, fraud_reference: str, behavior_evidence_reference: str = "", human_review: str = "", policy_attached: bool = True) -> dict[str, Any]:
		borrower = self._tenant_borrower_or_none(borrower_id, tenant_id)
		product = self._tenant_product_or_none(product_id, tenant_id)
		amount = normalize_amount(requested_amount)
		purpose = normalize_code(purpose)
		currency = product.currency if product else ""
		high_amount = amount >= 100000
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "submit_application", "borrower_present": borrower is not None, "product_present": product is not None, "positive_amount": amount > 0, "amount_within_limits": product is not None and product.min_amount <= amount <= product.max_amount, "purpose_supported": purpose in SUPPORTED_APPLICATION_PURPOSES, "affordability_present": bool(affordability_reference), "bank_statement_present": bool(bank_statement_reference), "aml_present": bool(aml_reference), "fraud_present": bool(fraud_reference), "remittance_or_card_evidence_present": bool(behavior_evidence_reference), "high_amount": high_amount, "human_review_recorded": bool(human_review)})
		if application_id in self.applications:
			raise ValueError(f"loan application already exists: {application_id}")
		application = LoanApplication(application_id, tenant_id, borrower_id, product_id, amount, currency, purpose, affordability_reference, bank_statement_reference, aml_reference, fraud_reference, behavior_evidence_reference)
		self.applications[application_id] = application
		self._audit(tenant_id, "loan_application_submitted", application_id)
		return application.to_dict()

	def record_underwriting(self, underwriting_id: str, tenant_id: str, application_id: str, score: float | int | str, decision: str, evidence_references: list[str], human_approval: str, adverse_reason: str = "") -> dict[str, Any]:
		application = self._tenant_application_or_none(application_id, tenant_id)
		score_value = normalize_score(score)
		decision = normalize_code(decision)
		category = decision_category(decision)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_underwriting", "application_present": application is not None, "score_in_range": 0 <= score_value <= 1000, "decision_supported": decision in SUPPORTED_UNDERWRITING_DECISIONS, "decision_evidence_present": bool(evidence_references), "adverse_decision": decision == "decline", "adverse_reason_present": bool(adverse_reason), "final_decision": category == "final", "human_approval_recorded": bool(human_approval)})
		record = UnderwritingDecision(underwriting_id, tenant_id, application_id, score_value, decision, list(evidence_references), human_approval, adverse_reason)
		self.underwriting[underwriting_id] = record
		self._audit(tenant_id, "underwriting_recorded", underwriting_id)
		return record.to_dict()

	def issue_offer(self, offer_id: str, tenant_id: str, application_id: str, underwriting_id: str, amount: float | int | str, apr: float | int | str, term_days: int, expiry_date: str, status: str = "issued", borrower_acceptance_reference: str = "") -> dict[str, Any]:
		application = self._tenant_application_or_none(application_id, tenant_id)
		underwriting = self._tenant_underwriting_or_none(underwriting_id, tenant_id)
		amount_value = normalize_amount(amount)
		apr_value = normalize_rate(apr)
		status = normalize_code(status)
		product = self._tenant_product_or_none(application.product_id, tenant_id) if application else None
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "issue_offer", "application_present": application is not None, "underwriting_present": underwriting is not None, "apr_valid": 0 <= apr_value <= 0.75, "term_valid": product is not None and product.min_term_days <= int(term_days) <= product.max_term_days, "offer_status_supported": status in SUPPORTED_OFFER_STATUSES, "expiry_present": bool(expiry_date), "accepted_offer": status == "accepted", "borrower_acceptance_present": bool(borrower_acceptance_reference)})
		offer = LoanOffer(offer_id, tenant_id, application_id, underwriting_id, amount_value, application.currency if application else "", apr_value, int(term_days), expiry_date, status, borrower_acceptance_reference)
		self.offers[offer_id] = offer
		self._audit(tenant_id, "loan_offer_issued", offer_id)
		return offer.to_dict()

	def record_disbursement(self, disbursement_id: str, tenant_id: str, offer_id: str, amount: float | int | str, rail: str, funding_account: str, destination_reference: str, human_approval: str) -> dict[str, Any]:
		offer = self._tenant_offer_or_none(offer_id, tenant_id)
		amount_value = normalize_amount(amount)
		rail = normalize_code(rail)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_disbursement", "offer_present": offer is not None and offer.status == "accepted", "funding_account_present": bool(funding_account), "rail_supported": rail in SUPPORTED_DISBURSEMENT_RAILS, "destination_present": bool(destination_reference), "human_approval_recorded": bool(human_approval)})
		disbursement = Disbursement(disbursement_id, tenant_id, offer_id, amount_value, offer.currency if offer else "", rail, funding_account, destination_reference, human_approval)
		self.disbursements[disbursement_id] = disbursement
		self._audit(tenant_id, "loan_disbursement_recorded", disbursement_id)
		return disbursement.to_dict()

	def schedule_repayment(self, schedule_id: str, tenant_id: str, offer_id: str, due_amount: float | int | str, due_date: str, frequency: str, installment_count: int) -> dict[str, Any]:
		offer = self._tenant_offer_or_none(offer_id, tenant_id)
		amount_value = normalize_amount(due_amount)
		frequency = normalize_code(frequency)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "schedule_repayment", "offer_present": offer is not None, "positive_due_amount": amount_value > 0, "due_date_present": bool(due_date), "repayment_frequency_supported": frequency in SUPPORTED_REPAYMENT_FREQUENCIES})
		schedule = RepaymentSchedule(schedule_id, tenant_id, offer_id, amount_value, offer.currency if offer else "", due_date, frequency, int(installment_count))
		self.repayments[schedule_id] = schedule
		self._audit(tenant_id, "repayment_schedule_created", schedule_id)
		return schedule.to_dict()

	def open_collection_case(self, case_id: str, tenant_id: str, overdue_account_reference: str, reason: str, reviewer_id: str, contact_policy_reference: str) -> dict[str, Any]:
		reason = normalize_code(reason)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "open_collection_case", "overdue_account_present": bool(overdue_account_reference), "collection_reason_supported": reason in SUPPORTED_COLLECTION_REASONS, "reviewer_present": bool(reviewer_id), "contact_policy_present": bool(contact_policy_reference)})
		case = CollectionCase(case_id, tenant_id, overdue_account_reference, reason, reviewer_id, contact_policy_reference)
		self.collections[case_id] = case
		self._audit(tenant_id, "collection_case_opened", case_id)
		return case.to_dict()

	def register_lending_agent(self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_lending_agent", "agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES, "agent_role_supported": role in SUPPORTED_AGENT_ROLES})
		evidence = self._record_evidence(agent_id, tenant_id, "agent", agent_id, "registered", {"name": name, "runtime": runtime, "role": role, "scope": scope})
		self._audit(tenant_id, "lending_agent_registered", agent_id)
		return evidence

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "lending_batch", "event_stream": event_stream})
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.fintech.lending.lifecycle", "accepted": True}

	def estimate_offer_installment(self, offer_id: str, tenant_id: str) -> dict[str, Any]:
		offer = self._tenant_offer_or_none(offer_id, tenant_id)
		if offer is None:
			raise KeyError(f"loan offer not found: {offer_id}")
		product = self._tenant_product_or_none(self.applications[offer.application_id].product_id, tenant_id)
		frequency = product.repayment_frequency if product else "monthly"
		return {"offer_id": offer_id, "tenant_id": tenant_id, "frequency": frequency, "installment": estimate_installment(offer.amount, offer.apr, offer.term_days, frequency)}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {"tenant_id": tenant_id, "product_count": sum(1 for item in self.products.values() if item.tenant_id == tenant_id), "borrower_count": sum(1 for item in self.borrowers.values() if item.tenant_id == tenant_id), "application_count": sum(1 for item in self.applications.values() if item.tenant_id == tenant_id), "underwriting_count": sum(1 for item in self.underwriting.values() if item.tenant_id == tenant_id), "offer_count": sum(1 for item in self.offers.values() if item.tenant_id == tenant_id), "disbursement_count": sum(1 for item in self.disbursements.values() if item.tenant_id == tenant_id), "repayment_count": sum(1 for item in self.repayments.values() if item.tenant_id == tenant_id), "collection_count": sum(1 for item in self.collections.values() if item.tenant_id == tenant_id), "audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id), "streaming": get_capability_contract(tenant_id)["streaming"]}

	def list_applications(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = self.applications.values()
		if tenant_id is not None:
			items = [item for item in items if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(items, key=lambda item: item.id)]

	def list_offers(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = self.offers.values()
		if tenant_id is not None:
			items = [item for item in items if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(items, key=lambda item: item.id)]

	def list_repayments(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = self.repayments.values()
		if tenant_id is not None:
			items = [item for item in items if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(items, key=lambda item: item.id)]

	def _tenant_product_or_none(self, product_id: str, tenant_id: str) -> LoanProduct | None:
		product = self.products.get(product_id)
		return product if product is not None and product.tenant_id == tenant_id else None

	def _tenant_borrower_or_none(self, borrower_id: str, tenant_id: str) -> BorrowerProfile | None:
		borrower = self.borrowers.get(borrower_id)
		return borrower if borrower is not None and borrower.tenant_id == tenant_id else None

	def _tenant_application_or_none(self, application_id: str, tenant_id: str) -> LoanApplication | None:
		application = self.applications.get(application_id)
		return application if application is not None and application.tenant_id == tenant_id else None

	def _tenant_underwriting_or_none(self, underwriting_id: str, tenant_id: str) -> UnderwritingDecision | None:
		record = self.underwriting.get(underwriting_id)
		return record if record is not None and record.tenant_id == tenant_id else None

	def _tenant_offer_or_none(self, offer_id: str, tenant_id: str) -> LoanOffer | None:
		offer = self.offers.get(offer_id)
		return offer if offer is not None and offer.tenant_id == tenant_id else None

	def _record_evidence(self, evidence_id: str, tenant_id: str, kind: str, reference_id: str, status: str, metadata: dict[str, Any]) -> dict[str, Any]:
		evidence = LendingEvidence(evidence_id, tenant_id, kind, reference_id, status, metadata)
		self.evidence[evidence_id] = evidence
		return evidence.to_dict()

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({"tenant_id": tenant_id, "event_type": event_type, "reference_id": reference_id})

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", "lending_policy_denied") for action in result["actions"])
		raise PermissionError(reasons or "lending_policy_denied")


DigitalLendingService = LendingService
