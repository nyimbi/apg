"""Executable service layer for APG Digital Lending."""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Any

try:
	from .capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_APPLICATION_PURPOSES, SUPPORTED_COLLECTION_REASONS, SUPPORTED_COUNTRIES, SUPPORTED_CURRENCIES, SUPPORTED_DISBURSEMENT_RAILS, SUPPORTED_OFFER_STATUSES, SUPPORTED_PRODUCT_TYPES, SUPPORTED_REPAYMENT_FREQUENCIES, SUPPORTED_UNDERWRITING_DECISIONS, evaluate_capability_rules, get_capability_contract
	from .lending_runtime import decision_category, estimate_installment, normalize_amount, normalize_code, normalize_country, normalize_currency, normalize_rate, normalize_score
	from .models import BorrowerProfile, CollectionCase, Disbursement, LendingEvidence, LoanApplication, LoanOffer, LoanProduct, RepaymentSchedule, UnderwritingDecision
except ImportError:  # pragma: no cover
	from capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_APPLICATION_PURPOSES, SUPPORTED_COLLECTION_REASONS, SUPPORTED_COUNTRIES, SUPPORTED_CURRENCIES, SUPPORTED_DISBURSEMENT_RAILS, SUPPORTED_OFFER_STATUSES, SUPPORTED_PRODUCT_TYPES, SUPPORTED_REPAYMENT_FREQUENCIES, SUPPORTED_UNDERWRITING_DECISIONS, evaluate_capability_rules, get_capability_contract  # type: ignore
	from lending_runtime import decision_category, estimate_installment, normalize_amount, normalize_code, normalize_country, normalize_currency, normalize_rate, normalize_score  # type: ignore
	from _domain_models import BorrowerProfile, CollectionCase, Disbursement, LendingEvidence, LoanApplication, LoanOffer, LoanProduct, RepaymentSchedule, UnderwritingDecision  # type: ignore


# ---------------------------------------------------------------------------
# Internal domain models for expanded service (adapter/store pattern)
# ---------------------------------------------------------------------------

class _Loan:
	"""Active loan record derived from disbursement + repayment schedule."""

	def __init__(
		self,
		loan_id: str,
		tenant_id: str,
		application_id: str,
		offer_id: str,
		disbursement_id: str,
		borrower_id: str,
		product_id: str,
		principal: float,
		currency: str,
		annual_rate: float,
		tenor_months: int,
		disbursement_date: str,
		bank_account: str,
		schedule_type: str = "reducing_balance",
	) -> None:
		self.loan_id = loan_id
		self.tenant_id = tenant_id
		self.application_id = application_id
		self.offer_id = offer_id
		self.disbursement_id = disbursement_id
		self.borrower_id = borrower_id
		self.product_id = product_id
		self.principal = principal
		self.outstanding_principal = principal
		self.currency = currency
		self.annual_rate = annual_rate
		self.tenor_months = tenor_months
		self.disbursement_date = disbursement_date
		self.bank_account = bank_account
		self.schedule_type = schedule_type
		self.status = "active"
		self.repayments: list[dict[str, Any]] = []
		self.fees: list[dict[str, Any]] = []
		self.installments: list[dict[str, Any]] = []
		self.collateral_ids: list[str] = []
		self.write_off_reason: str = ""
		self.written_off_by: str = ""
		self.written_off_date: str = ""
		self.closure_reason: str = ""
		self.restructure_history: list[dict[str, Any]] = []
		self.collection_activities: list[dict[str, Any]] = []
		self.demand_notices: list[dict[str, Any]] = []
		self.legal_actions: list[dict[str, Any]] = []
		self.assigned_collector: str = ""
		self.assigned_lawyer: str = ""

	def to_dict(self) -> dict[str, Any]:
		return {
			"loan_id": self.loan_id,
			"tenant_id": self.tenant_id,
			"application_id": self.application_id,
			"offer_id": self.offer_id,
			"disbursement_id": self.disbursement_id,
			"borrower_id": self.borrower_id,
			"product_id": self.product_id,
			"principal": self.principal,
			"outstanding_principal": self.outstanding_principal,
			"currency": self.currency,
			"annual_rate": self.annual_rate,
			"tenor_months": self.tenor_months,
			"disbursement_date": self.disbursement_date,
			"bank_account": self.bank_account,
			"schedule_type": self.schedule_type,
			"status": self.status,
			"collateral_ids": list(self.collateral_ids),
			"assigned_collector": self.assigned_collector,
			"assigned_lawyer": self.assigned_lawyer,
		}


class _Collateral:
	"""Collateral item record."""

	def __init__(self, collateral_id: str, loan_id: str, collateral_type: str, description: str, market_value: float, currency: str) -> None:
		self.collateral_id = collateral_id
		self.loan_id = loan_id
		self.collateral_type = collateral_type  # property | vehicle | cash | other
		self.description = description
		self.market_value = market_value
		self.currency = currency
		self.status = "held"
		self.released_by: str = ""
		self.release_reason: str = ""
		self.release_date: str = ""

	def forced_sale_value(self) -> float:
		# Property: 60% of market; Vehicle: 70%; Cash/other: 90%
		haircut = {"property": 0.60, "vehicle": 0.70, "cash": 0.90}.get(self.collateral_type, 0.70)
		return round(self.market_value * haircut, 2)

	def to_dict(self) -> dict[str, Any]:
		return {
			"collateral_id": self.collateral_id,
			"loan_id": self.loan_id,
			"collateral_type": self.collateral_type,
			"description": self.description,
			"market_value": self.market_value,
			"forced_sale_value": self.forced_sale_value(),
			"currency": self.currency,
			"status": self.status,
			"released_by": self.released_by,
			"release_reason": self.release_reason,
			"release_date": self.release_date,
		}


class _CreditScore:
	"""Credit score record."""

	def __init__(self, customer_id: str, score: int, behavioural: int, demographic: int, bureau: int, risk_grade: str, probability_of_default: float, components: dict[str, Any]) -> None:
		self.customer_id = customer_id
		self.score = score  # 300–850
		self.behavioural = behavioural
		self.demographic = demographic
		self.bureau = bureau
		self.risk_grade = risk_grade  # A–F
		self.probability_of_default = probability_of_default
		self.components = components
		self.computed_at = date.today().isoformat()

	def to_dict(self) -> dict[str, Any]:
		return {
			"customer_id": self.customer_id,
			"score": self.score,
			"behavioural_score": self.behavioural,
			"demographic_score": self.demographic,
			"bureau_score": self.bureau,
			"risk_grade": self.risk_grade,
			"probability_of_default": self.probability_of_default,
			"components": self.components,
			"computed_at": self.computed_at,
		}


class _BureauReport:
	"""CRB bureau report record."""

	def __init__(self, customer_id: str, id_number: str, country: str, score: int, accounts: list[dict], payment_history: list[dict], defaults: list[dict], enquiries: list[dict], fraud_flags: list[str]) -> None:
		self.customer_id = customer_id
		self.id_number = id_number
		self.country = country
		self.score = score
		self.accounts = accounts
		self.payment_history = payment_history
		self.defaults = defaults
		self.enquiries = enquiries
		self.fraud_flags = fraud_flags
		self.bureau_name = self._bureau_name(country)
		self.fetched_at = date.today().isoformat()

	@staticmethod
	def _bureau_name(country: str) -> str:
		return {"KE": "CRB Africa / TransUnion Kenya", "NG": "Creditinfo West Africa", "ZA": "TransUnion South Africa", "GH": "Creditinfo Ghana", "UG": "TransUnion Uganda", "TZ": "TransUnion Tanzania"}.get(country, "TransUnion Africa")

	def to_dict(self) -> dict[str, Any]:
		return {
			"customer_id": self.customer_id,
			"id_number": self.id_number,
			"country": self.country,
			"bureau_name": self.bureau_name,
			"score": self.score,
			"accounts": self.accounts,
			"payment_history": self.payment_history,
			"defaults": self.defaults,
			"enquiries": self.enquiries,
			"fraud_flags": self.fraud_flags,
			"fetched_at": self.fetched_at,
		}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _risk_grade(score: int) -> str:
	if score >= 750:
		return "A"
	if score >= 680:
		return "B"
	if score >= 620:
		return "C"
	if score >= 560:
		return "D"
	if score >= 480:
		return "E"
	return "F"


def _pd_from_score(score: int) -> float:
	"""Logistic-curve PD mapping from 300–850 range. Grade A ~0.5%, Grade F ~25%."""
	normalised = (score - 300) / 550  # 0→1
	return round(0.30 * math.exp(-3.5 * normalised), 4)


def _emi(principal: float, monthly_rate: float, n_months: int) -> float:
	if monthly_rate == 0:
		return round(principal / n_months, 2)
	return round((principal * monthly_rate * (1 + monthly_rate) ** n_months) / ((1 + monthly_rate) ** n_months - 1), 2)


def _add_months(d: date, months: int) -> date:
	month = d.month - 1 + months
	year = d.year + month // 12
	month = month % 12 + 1
	import calendar
	day = min(d.day, calendar.monthrange(year, month)[1])
	return date(year, month, day)


def _parse_date(s: str) -> date:
	return date.fromisoformat(s)


def _today() -> date:
	return date.today()


# ---------------------------------------------------------------------------
# Main service
# ---------------------------------------------------------------------------

class LendingService:
	"""Dependency-light lending runtime for generated applications."""

	def __init__(self) -> None:
		# Core domain stores (original)
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

		# Expanded stores
		self.loans: dict[str, _Loan] = {}
		self.collateral: dict[str, _Collateral] = {}
		self.credit_scores: dict[str, _CreditScore] = {}
		self.bureau_reports: dict[str, _BureauReport] = {}

		# In-memory stores for new extended entities
		self._underwriters: dict[str, str] = {}      # application_id -> underwriter_id
		self._doc_requests: dict[str, list[str]] = {}  # application_id -> [doc_type]
		self._site_visits: dict[str, list[dict]] = {}  # application_id -> [visit]
		self._income_verifications: dict[str, dict] = {}  # customer_id -> result
		self._demand_notices: dict[str, list[dict]] = {}   # loan_id -> [notice]
		self._collectors: dict[str, str] = {}          # loan_id -> collector_id
		self._writeoffs: dict[str, dict] = {}          # loan_id -> write-off record
		self._product_rates_history: dict[str, list[dict]] = {}  # product_code -> [rate_change]

	# ------------------------------------------------------------------
	# Capability contract
	# ------------------------------------------------------------------

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------
	# Internal guards
	# ------------------------------------------------------------------

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", "lending_policy_denied") for action in result["actions"])
		raise PermissionError(reasons or "lending_policy_denied")

	def _audit(self, tenant_id: str, event_type: str, reference_id: str, meta: dict[str, Any] | None = None) -> None:
		entry: dict[str, Any] = {"tenant_id": tenant_id, "event_type": event_type, "reference_id": reference_id, "ts": _today().isoformat()}
		if meta:
			entry["meta"] = meta
		self.audit_events.append(entry)

	def _require_loan(self, loan_id: str) -> _Loan:
		loan = self.loans.get(loan_id)
		if loan is None:
			raise KeyError(f"loan not found: {loan_id}")
		return loan

	def _require_application(self, application_id: str) -> LoanApplication:
		app = self.applications.get(application_id)
		if app is None:
			raise KeyError(f"application not found: {application_id}")
		return app

	def _require_product(self, product_id: str) -> LoanProduct:
		p = self.products.get(product_id)
		if p is None:
			raise KeyError(f"product not found: {product_id}")
		return p

	# ------------------------------------------------------------------
	# Original core methods (preserved exactly)
	# ------------------------------------------------------------------

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

	# ------------------------------------------------------------------
	# APPLICATION PROCESSING (8 methods)
	# ------------------------------------------------------------------

	def retrieve_application(self, application_id: str) -> dict[str, Any]:
		"""Return full application record including latest status and linked entities."""
		app = self._require_application(application_id)
		result = app.to_dict()
		result["underwriter"] = self._underwriters.get(application_id, "")
		result["required_documents"] = self._doc_requests.get(application_id, [])
		result["site_visits"] = self._site_visits.get(application_id, [])
		# Attach linked underwriting decisions
		linked_uw = [uw.to_dict() for uw in self.underwriting.values() if uw.application_id == application_id]
		result["underwriting_decisions"] = linked_uw
		return result

	def list_applications(self, filters: dict[str, Any] | None = None, tenant_id: str | None = None) -> list[dict[str, Any]]:
		"""List applications with optional filter dict (status, borrower_id, product_id, purpose)."""
		filters = filters or {}
		items = list(self.applications.values())
		if tenant_id is not None:
			items = [a for a in items if a.tenant_id == tenant_id]
		for key in ("status", "borrower_id", "product_id", "purpose"):
			val = filters.get(key)
			if val is not None:
				items = [a for a in items if getattr(a, key, None) == val]
		return [a.to_dict() for a in sorted(items, key=lambda a: a.id)]

	def withdraw_application(self, application_id: str, reason: str) -> dict[str, Any]:
		"""Withdraw a pending application. Allowed while status == submitted."""
		app = self._require_application(application_id)
		if app.status not in ("submitted", "under_review"):
			raise ValueError(f"cannot withdraw application in status '{app.status}'")
		assert reason, "withdrawal reason required"
		app.status = "withdrawn"
		self._audit(app.tenant_id, "application_withdrawn", application_id, {"reason": reason})
		result = app.to_dict()
		result["withdrawal_reason"] = reason
		return result

	def assign_underwriter(self, application_id: str, underwriter_id: str) -> dict[str, Any]:
		"""Assign an underwriter to an application and move it to under_review."""
		app = self._require_application(application_id)
		assert underwriter_id, "underwriter_id required"
		assert app.status not in ("approved", "declined", "withdrawn", "disbursed"), \
			f"cannot assign underwriter to application in status '{app.status}'"
		self._underwriters[application_id] = underwriter_id
		app.status = "under_review"
		self._audit(app.tenant_id, "underwriter_assigned", application_id, {"underwriter_id": underwriter_id})
		return {"application_id": application_id, "underwriter_id": underwriter_id, "status": app.status}

	def request_documents(self, application_id: str, required_docs: list[str]) -> dict[str, Any]:
		"""Record a request for additional documents against an application."""
		app = self._require_application(application_id)
		assert required_docs, "required_docs must be a non-empty list"
		existing = self._doc_requests.get(application_id, [])
		new_docs = [d for d in required_docs if d not in existing]
		self._doc_requests[application_id] = existing + new_docs
		self._audit(app.tenant_id, "documents_requested", application_id, {"docs": new_docs})
		return {
			"application_id": application_id,
			"requested_documents": self._doc_requests[application_id],
			"newly_requested": new_docs,
		}

	def record_site_visit(self, application_id: str, visit_notes: str, inspector_id: str, visit_date: str) -> dict[str, Any]:
		"""Record an on-site inspection visit for an application."""
		app = self._require_application(application_id)
		assert inspector_id, "inspector_id required"
		assert visit_date, "visit_date required (ISO format)"
		assert visit_notes, "visit_notes required"
		visit = {
			"inspector_id": inspector_id,
			"visit_date": visit_date,
			"notes": visit_notes,
			"recorded_at": _today().isoformat(),
		}
		self._site_visits.setdefault(application_id, []).append(visit)
		self._audit(app.tenant_id, "site_visit_recorded", application_id, {"inspector_id": inspector_id, "visit_date": visit_date})
		return {"application_id": application_id, "visit": visit, "total_visits": len(self._site_visits[application_id])}

	def application_analytics(self, period: str) -> dict[str, Any]:
		"""Aggregate application analytics for a given period label (e.g. '2026-Q1', '2026-05')."""
		assert period, "period required"
		apps = list(self.applications.values())
		total = len(apps)
		by_status: dict[str, int] = {}
		by_purpose: dict[str, int] = {}
		by_product: dict[str, int] = {}
		total_amount = 0.0
		for app in apps:
			by_status[app.status] = by_status.get(app.status, 0) + 1
			by_purpose[app.purpose] = by_purpose.get(app.purpose, 0) + 1
			by_product[app.product_id] = by_product.get(app.product_id, 0) + 1
			total_amount += app.requested_amount
		approved = by_status.get("approved", 0)
		declined = by_status.get("declined", 0)
		approval_rate = round(approved / total, 4) if total else 0.0
		decline_rate = round(declined / total, 4) if total else 0.0
		avg_amount = round(total_amount / total, 2) if total else 0.0
		return {
			"period": period,
			"total_applications": total,
			"total_requested_amount": round(total_amount, 2),
			"average_requested_amount": avg_amount,
			"approval_rate": approval_rate,
			"decline_rate": decline_rate,
			"by_status": by_status,
			"by_purpose": by_purpose,
			"by_product": by_product,
		}

	# ------------------------------------------------------------------
	# CREDIT ASSESSMENT (8 methods)
	# ------------------------------------------------------------------

	def credit_score_calculate(self, customer_id: str) -> dict[str, Any]:
		"""
		Compute a composite credit score 300–850 from three weighted pillars:
		  Behavioural 45%: payment history, utilisation, delinquency history
		  Demographic 20%: employment, income stability, tenure
		  Bureau 35%:      CRB data, existing loan count, defaults, enquiries

		Returns score, risk_grade (A–F), probability_of_default.
		Uses existing bureau report if present, otherwise synthesises from
		borrower profile and repayment history.
		"""
		assert customer_id, "customer_id required"

		# Collect all loans for this customer via borrower profiles
		borrower_ids = [b.id for b in self.borrowers.values() if b.customer_reference == customer_id or b.id == customer_id]

		# --- Behavioural pillar (45%) ---
		total_repayments = 0
		on_time = 0
		late = 0
		delinquent_loans = 0
		utilisation_total = 0.0
		utilisation_count = 0

		for loan in self.loans.values():
			if loan.borrower_id not in borrower_ids and loan.borrower_id != customer_id:
				continue
			for pmt in loan.repayments:
				total_repayments += 1
				dpd = pmt.get("dpd", 0)
				if dpd == 0:
					on_time += 1
				elif dpd <= 30:
					late += 1
				else:
					delinquent_loans += 1
			if loan.principal > 0:
				utilisation_total += loan.outstanding_principal / loan.principal
				utilisation_count += 1

		payment_ratio = (on_time / total_repayments) if total_repayments else 0.80
		delinquency_penalty = min(delinquent_loans * 0.05, 0.40)
		utilisation = (utilisation_total / utilisation_count) if utilisation_count else 0.50
		utilisation_score = max(0, 1 - utilisation) * 0.8 + 0.2  # favour low utilisation

		behavioural_raw = (payment_ratio * 0.55 + utilisation_score * 0.25 + max(0, 1 - delinquency_penalty) * 0.20)

		# --- Demographic pillar (20%) ---
		# Income evidence presence is a proxy for stable employment
		has_income_evidence = any(b.income_evidence_id for b in self.borrowers.values() if b.id in borrower_ids or b.customer_reference == customer_id)
		demographic_raw = 0.75 if has_income_evidence else 0.50

		# --- Bureau pillar (35%) ---
		bureau = self.bureau_reports.get(customer_id)
		if bureau:
			# Normalise bureau score: assume bureau range 300-900
			bureau_normalised = (bureau.score - 300) / 600
			default_count = len(bureau.defaults)
			fraud_flag_count = len(bureau.fraud_flags)
			bureau_raw = max(0, bureau_normalised - default_count * 0.08 - fraud_flag_count * 0.15)
		else:
			# No bureau data: conservative estimate
			bureau_raw = 0.55

		# Composite: map [0,1] -> [300, 850]
		composite = behavioural_raw * 0.45 + demographic_raw * 0.20 + bureau_raw * 0.35
		score = int(300 + composite * 550)
		score = max(300, min(850, score))

		grade = _risk_grade(score)
		pd = _pd_from_score(score)

		components = {
			"behavioural": {
				"weight": 0.45,
				"raw": round(behavioural_raw, 4),
				"payment_ratio": round(payment_ratio, 4),
				"delinquency_penalty": round(delinquency_penalty, 4),
				"utilisation": round(utilisation, 4),
				"total_repayments": total_repayments,
			},
			"demographic": {
				"weight": 0.20,
				"raw": round(demographic_raw, 4),
				"income_evidence_present": has_income_evidence,
			},
			"bureau": {
				"weight": 0.35,
				"raw": round(bureau_raw, 4),
				"bureau_score": bureau.score if bureau else None,
				"defaults": len(bureau.defaults) if bureau else 0,
				"fraud_flags": len(bureau.fraud_flags) if bureau else 0,
			},
		}

		cs = _CreditScore(customer_id, score, int(behavioural_raw * 550 + 300), int(demographic_raw * 550 + 300), int(bureau_raw * 550 + 300), grade, pd, components)
		self.credit_scores[customer_id] = cs
		return cs.to_dict()

	def credit_bureau_check(self, customer_id: str, id_number: str, country: str) -> dict[str, Any]:
		"""
		Simulate a credit bureau query (TransUnion Africa / Creditinfo Africa / CRB Africa).
		In production, replace with adapter.call('crb', ...).
		Returns: score, accounts, payment_history, defaults, enquiries, fraud_flags.
		"""
		assert customer_id, "customer_id required"
		assert id_number, "id_number required"
		country = normalize_country(country)
		assert country in SUPPORTED_COUNTRIES, f"country not supported: {country}"

		# Synthesise from internal data where possible
		borrower_ids = [b.id for b in self.borrowers.values() if b.customer_reference == customer_id or b.id == customer_id]

		active_loans = [ln for ln in self.loans.values() if ln.borrower_id in borrower_ids or ln.borrower_id == customer_id]
		accounts = [{"loan_id": ln.loan_id, "principal": ln.principal, "outstanding": ln.outstanding_principal, "status": ln.status, "currency": ln.currency} for ln in active_loans]

		defaults = [{"loan_id": ln.loan_id, "reason": ln.write_off_reason, "date": ln.written_off_date} for ln in active_loans if ln.status == "written_off"]

		payment_history: list[dict] = []
		for ln in active_loans:
			for pmt in ln.repayments:
				payment_history.append({"loan_id": ln.loan_id, "amount": pmt.get("amount", 0), "date": pmt.get("payment_date", ""), "dpd": pmt.get("dpd", 0)})

		fraud_flags: list[str] = []

		# Score synthesis: 600 base, penalise defaults and delinquent payments
		base = 640
		base -= len(defaults) * 60
		for ph in payment_history:
			if ph["dpd"] > 90:
				base -= 15
			elif ph["dpd"] > 30:
				base -= 5
		bureau_score = max(300, min(900, base))

		report = _BureauReport(customer_id, id_number, country, bureau_score, accounts, payment_history, defaults, [], fraud_flags)
		self.bureau_reports[customer_id] = report
		return report.to_dict()

	def income_verification(self, customer_id: str, income_source: str, stated_amount: float | int | str, docs: list[str]) -> dict[str, Any]:
		"""
		Verify income:
		  employed     -> payslip cross-check: accept if docs present, flag if stated > 3x industry median
		  self_employed -> bank statement cashflow analysis: 3-month average
		  mobile_money  -> M-Pesa statement analysis: net inflow average

		Returns: verified, verified_amount, confidence, method, flags.
		"""
		assert customer_id, "customer_id required"
		income_source = normalize_code(income_source)
		amount = normalize_amount(stated_amount)
		assert amount > 0, "stated_amount must be positive"
		assert docs, "docs list required"

		flags: list[str] = []
		verified_amount = amount
		confidence = 0.0

		if income_source == "employed":
			has_payslip = any("payslip" in d.lower() or "slip" in d.lower() for d in docs)
			has_bank_stmt = any("bank" in d.lower() or "statement" in d.lower() for d in docs)
			if has_payslip:
				confidence = 0.90
				# Flag if amount seems unreasonably high (heuristic: > 500k local currency)
				if amount > 500000:
					flags.append("high_income_outlier_review")
			elif has_bank_stmt:
				confidence = 0.70
				flags.append("payslip_not_provided")
			else:
				confidence = 0.40
				flags.append("income_docs_insufficient")
				verified_amount = amount * 0.60  # haircut

		elif income_source in ("self_employed", "business"):
			has_bank_stmt = any("bank" in d.lower() or "statement" in d.lower() for d in docs)
			has_tax = any("tax" in d.lower() or "kra" in d.lower() or "revenue" in d.lower() for d in docs)
			if has_bank_stmt and has_tax:
				confidence = 0.85
			elif has_bank_stmt:
				confidence = 0.70
				flags.append("tax_return_not_provided")
			else:
				confidence = 0.35
				flags.append("bank_statement_required")
				verified_amount = amount * 0.50

		elif income_source in ("mobile_money", "mpesa"):
			has_mpesa_stmt = any("mpesa" in d.lower() or "mobile" in d.lower() or "mtn" in d.lower() or "airtel" in d.lower() for d in docs)
			if has_mpesa_stmt:
				confidence = 0.80
				# Mobile money income gets a conservative 85% recognition rate
				verified_amount = round(amount * 0.85, 2)
				flags.append("mobile_money_cashflow_recognised_at_85pct")
			else:
				confidence = 0.30
				flags.append("mobile_money_statement_required")
				verified_amount = amount * 0.40

		else:
			confidence = 0.50
			flags.append(f"unknown_income_source_{income_source}")

		result = {
			"customer_id": customer_id,
			"income_source": income_source,
			"stated_amount": amount,
			"verified_amount": round(verified_amount, 2),
			"confidence": round(confidence, 4),
			"verified": confidence >= 0.65,
			"method": income_source,
			"docs_provided": docs,
			"flags": flags,
			"verified_at": _today().isoformat(),
		}
		self._income_verifications[customer_id] = result
		return result

	def debt_service_ratio(self, customer_id: str, new_loan_amount: float | int | str, new_loan_rate: float | int | str, tenor_months: int) -> dict[str, Any]:
		"""
		DSR = (sum of existing EMIs + new EMI) / net_monthly_income.
		Threshold: 40%. Returns DSR, pass/fail, existing_emi_total, new_emi.
		"""
		assert customer_id, "customer_id required"
		new_amount = normalize_amount(new_loan_amount)
		new_rate = normalize_rate(new_loan_rate)
		assert tenor_months > 0, "tenor_months must be positive"

		# Existing EMI from active loans
		borrower_ids = {b.id for b in self.borrowers.values() if b.customer_reference == customer_id or b.id == customer_id}
		existing_emi = 0.0
		existing_loans: list[dict] = []
		for loan in self.loans.values():
			if loan.borrower_id not in borrower_ids and loan.borrower_id != customer_id:
				continue
			if loan.status != "active":
				continue
			months_remaining = max(1, loan.tenor_months)
			monthly_rate = loan.annual_rate / 12
			emi = _emi(loan.outstanding_principal, monthly_rate, months_remaining)
			existing_emi += emi
			existing_loans.append({"loan_id": loan.loan_id, "emi": emi, "outstanding": loan.outstanding_principal})

		new_emi = _emi(new_amount, new_rate / 12, tenor_months)

		# Net income from verification
		income_rec = self._income_verifications.get(customer_id)
		net_income = income_rec["verified_amount"] if income_rec else 0.0

		total_emi = existing_emi + new_emi
		dsr = round(total_emi / net_income, 4) if net_income > 0 else float("inf")
		threshold = 0.40
		passes = dsr <= threshold and net_income > 0

		return {
			"customer_id": customer_id,
			"new_loan_amount": new_amount,
			"new_loan_rate": new_rate,
			"tenor_months": tenor_months,
			"new_emi": round(new_emi, 2),
			"existing_emi_total": round(existing_emi, 2),
			"total_emi": round(total_emi, 2),
			"net_monthly_income": round(net_income, 2),
			"dsr": dsr,
			"dsr_threshold": threshold,
			"passes": passes,
			"existing_loans": existing_loans,
		}

	def calculate_loan_eligibility(self, customer_id: str, product_code: str) -> dict[str, Any]:
		"""
		Returns: max_amount, max_tenor, indicative_rate, required_docs, eligibility_reasons.
		Uses credit score, income verification, and DSR to size the offer.
		"""
		assert customer_id, "customer_id required"
		assert product_code, "product_code required"

		product = self._require_product(product_code)
		score_rec = self.credit_scores.get(customer_id)
		income_rec = self._income_verifications.get(customer_id)

		reasons: list[str] = []
		eligible = True

		if score_rec is None:
			reasons.append("credit_score_not_computed")
			eligible = False
			score = 500
			grade = "D"
		else:
			score = score_rec.score
			grade = score_rec.risk_grade

		if income_rec is None:
			reasons.append("income_not_verified")
			eligible = False
			verified_income = 0.0
		else:
			if not income_rec["verified"]:
				reasons.append("income_verification_failed")
				eligible = False
			verified_income = income_rec["verified_amount"]

		# Rate pricing: base rate + credit spread based on grade
		spread_map = {"A": 0.00, "B": 0.02, "C": 0.04, "D": 0.07, "E": 0.12, "F": 0.20}
		indicative_rate = round(product.annual_rate + spread_map.get(grade, 0.10), 4)

		# Max amount: 40% DSR constraint — solve for P given income and tenor
		max_tenor = product.max_term_days // 30
		monthly_rate = indicative_rate / 12
		dsr_threshold = 0.40

		existing_emi = 0.0
		borrower_ids = {b.id for b in self.borrowers.values() if b.customer_reference == customer_id or b.id == customer_id}
		for loan in self.loans.values():
			if loan.borrower_id not in borrower_ids and loan.borrower_id != customer_id:
				continue
			if loan.status != "active":
				continue
			existing_emi += _emi(loan.outstanding_principal, loan.annual_rate / 12, max(1, loan.tenor_months))

		affordable_emi = max(0, verified_income * dsr_threshold - existing_emi)
		if monthly_rate > 0 and affordable_emi > 0:
			max_amount_dsr = round(affordable_emi * (1 - (1 + monthly_rate) ** -max_tenor) / monthly_rate, 2)
		else:
			max_amount_dsr = 0.0

		# Grade-based cap: Grade A gets 100% of product max, F gets 20%
		grade_cap = {"A": 1.0, "B": 0.85, "C": 0.70, "D": 0.50, "E": 0.30, "F": 0.20}.get(grade, 0.50)
		max_amount = min(product.max_amount, max_amount_dsr, product.max_amount * grade_cap)
		max_amount = max(product.min_amount, max_amount) if eligible else 0.0

		required_docs = ["national_id", "bank_statement_3months"]
		if grade in ("D", "E", "F"):
			required_docs.append("income_payslip")
		if max_amount >= 500000:
			required_docs.append("collateral_valuation")

		return {
			"customer_id": customer_id,
			"product_code": product_code,
			"eligible": eligible,
			"credit_grade": grade,
			"credit_score": score,
			"max_amount": round(max_amount, 2),
			"max_tenor_months": max_tenor,
			"indicative_annual_rate": indicative_rate,
			"indicative_monthly_rate": round(indicative_rate / 12, 6),
			"required_docs": required_docs,
			"eligibility_reasons": reasons,
			"computed_at": _today().isoformat(),
		}

	def assess_collateral(self, collateral_items: list[dict[str, Any]]) -> dict[str, Any]:
		"""
		Assess a list of collateral items:
		  property -> FSV = 60% of market value
		  vehicle  -> FSV = 70% of book value
		  cash / other -> FSV = 90%

		Returns: total_market_value, total_collateral_value (FSV), coverage_ratio vs requested amount.
		"""
		assert collateral_items, "collateral_items required"

		total_market = 0.0
		total_fsv = 0.0
		assessed: list[dict] = []

		for item in collateral_items:
			ctype = normalize_code(item.get("type", item.get("collateral_type", "other")))
			market_val = normalize_amount(item.get("market_value", item.get("value", 0)))
			haircut = {"property": 0.60, "vehicle": 0.70, "cash": 0.90, "shares": 0.50}.get(ctype, 0.70)
			fsv = round(market_val * haircut, 2)
			total_market += market_val
			total_fsv += fsv
			assessed.append({
				"collateral_type": ctype,
				"market_value": market_val,
				"haircut": haircut,
				"forced_sale_value": fsv,
				"description": item.get("description", ""),
				"currency": item.get("currency", "KES"),
			})

		requested = normalize_amount(collateral_items[0].get("requested_amount", 0)) if collateral_items else 0.0
		coverage_ratio = round(total_fsv / requested, 4) if requested > 0 else None

		return {
			"items": assessed,
			"total_market_value": round(total_market, 2),
			"total_collateral_value": round(total_fsv, 2),
			"coverage_ratio": coverage_ratio,
			"sufficient": coverage_ratio is not None and coverage_ratio >= 1.0,
			"assessed_at": _today().isoformat(),
		}

	def generate_loan_offers(self, application_id: str) -> list[dict[str, Any]]:
		"""
		Generate three tiered offers for an application:
		  conservative: 60% of max eligible amount, +200bp rate premium
		  standard:     80% of max eligible, base rate
		  aggressive:   100% of max eligible, -100bp (only for grade A/B)

		Returns list of offer dicts with indicative terms.
		"""
		app = self._require_application(application_id)
		eligibility = self.calculate_loan_eligibility(app.borrower_id, app.product_id)
		max_amount = eligibility["max_amount"]
		base_rate = eligibility["indicative_annual_rate"]
		max_tenor = eligibility["max_tenor_months"]
		grade = eligibility["credit_grade"]

		product = self._require_product(app.product_id)

		offers = []
		tiers = [
			("conservative", 0.60, 0.02, max_tenor // 2 or 6),
			("standard",     0.80, 0.00, max_tenor),
		]
		if grade in ("A", "B"):
			tiers.append(("aggressive", 1.00, -0.01, max_tenor))

		for tier_name, amount_factor, rate_delta, tenor in tiers:
			amount = round(max(product.min_amount, min(max_amount * amount_factor, product.max_amount)), 2)
			rate = round(max(0.01, base_rate + rate_delta), 6)
			monthly_rate = rate / 12
			emi = _emi(amount, monthly_rate, tenor)
			total_interest = round(emi * tenor - amount, 2)
			offers.append({
				"tier": tier_name,
				"application_id": application_id,
				"amount": amount,
				"annual_rate": rate,
				"tenor_months": tenor,
				"monthly_emi": emi,
				"total_interest": total_interest,
				"total_cost": round(amount + total_interest, 2),
				"currency": app.currency,
				"generated_at": _today().isoformat(),
			})

		return offers

	def underwriting_decision(self, application_id: str, decision: str, conditions: list[str], underwriter_id: str) -> dict[str, Any]:
		"""
		Record final underwriting outcome: approve / decline / refer / conditional_approve.
		Updates application status accordingly.
		"""
		app = self._require_application(application_id)
		decision = normalize_code(decision)
		valid_decisions = ("approve", "decline", "refer", "conditional_approve")
		assert decision in valid_decisions, f"decision must be one of {valid_decisions}"
		assert underwriter_id, "underwriter_id required"

		status_map = {
			"approve": "approved",
			"decline": "declined",
			"refer": "referred",
			"conditional_approve": "conditionally_approved",
		}
		app.status = status_map[decision]
		self._underwriters[application_id] = underwriter_id

		self._audit(app.tenant_id, f"underwriting_{decision}", application_id, {
			"underwriter_id": underwriter_id,
			"conditions": conditions,
		})

		return {
			"application_id": application_id,
			"decision": decision,
			"status": app.status,
			"underwriter_id": underwriter_id,
			"conditions": conditions,
			"decided_at": _today().isoformat(),
		}

	# ------------------------------------------------------------------
	# LOAN MANAGEMENT (10 methods)
	# ------------------------------------------------------------------

	def disburse_loan(self, loan_id: str, application_id: str, bank_account: str, disbursement_date: str) -> dict[str, Any]:
		"""
		Create an active loan record from an approved application.
		Generates the full reducing-balance repayment schedule.
		Enforces: application must be approved or conditionally_approved.
		"""
		app = self._require_application(application_id)
		assert app.status in ("approved", "conditionally_approved"), \
			f"cannot disburse application in status '{app.status}'"
		assert bank_account, "bank_account required"
		assert disbursement_date, "disbursement_date required"
		assert loan_id not in self.loans, f"loan already exists: {loan_id}"

		product = self._require_product(app.product_id)
		# Tenor: use product max as default, or find from linked offer
		linked_offer = next((o for o in self.offers.values() if o.application_id == application_id), None)
		if linked_offer:
			tenor_months = max(1, linked_offer.term_days // 30)
			annual_rate = linked_offer.apr
			principal = linked_offer.amount
		else:
			tenor_months = product.max_term_days // 30
			annual_rate = product.annual_rate
			principal = app.requested_amount

		loan = _Loan(
			loan_id=loan_id,
			tenant_id=app.tenant_id,
			application_id=application_id,
			offer_id=linked_offer.id if linked_offer else "",
			disbursement_id="",
			borrower_id=app.borrower_id,
			product_id=app.product_id,
			principal=principal,
			currency=app.currency,
			annual_rate=annual_rate,
			tenor_months=tenor_months,
			disbursement_date=disbursement_date,
			bank_account=bank_account,
		)

		# Build schedule now
		sched = self._build_schedule(principal, annual_rate, tenor_months, disbursement_date, "reducing_balance")
		loan.installments = sched["installments"]

		app.status = "disbursed"
		self.loans[loan_id] = loan
		self._audit(app.tenant_id, "loan_disbursed", loan_id, {"application_id": application_id, "principal": principal})

		result = loan.to_dict()
		result["schedule_summary"] = {"installment_count": len(loan.installments), "first_due": loan.installments[0]["due_date"] if loan.installments else None, "last_due": loan.installments[-1]["due_date"] if loan.installments else None, "total_repayable": round(sum(i["emi"] for i in loan.installments), 2)}
		return result

	def _build_schedule(self, principal: float, annual_rate: float, tenor_months: int, start_date: str, schedule_type: str = "reducing_balance") -> dict[str, Any]:
		"""Internal: builds an installment list for a given loan."""
		monthly_rate = annual_rate / 12
		emi = _emi(principal, monthly_rate, tenor_months)
		balance = principal
		installments = []
		d = _parse_date(start_date)
		cumulative_interest = 0.0

		for n in range(1, tenor_months + 1):
			due_date = _add_months(d, n)
			if schedule_type == "flat_rate":
				interest = round(principal * monthly_rate, 2)
				principal_portion = round(principal / tenor_months, 2)
			else:  # reducing balance
				interest = round(balance * monthly_rate, 2)
				principal_portion = round(emi - interest, 2)

			balance = round(max(0.0, balance - principal_portion), 2)
			cumulative_interest = round(cumulative_interest + interest, 2)

			# Last installment: clear any floating-point residual
			if n == tenor_months and balance > 0:
				principal_portion = round(principal_portion + balance, 2)
				balance = 0.0

			installments.append({
				"installment_no": n,
				"due_date": due_date.isoformat(),
				"emi": round(emi, 2),
				"principal": principal_portion,
				"interest": interest,
				"balance": balance,
				"cumulative_interest": cumulative_interest,
				"status": "pending",
				"paid_amount": 0.0,
				"paid_date": None,
			})

		return {
			"schedule_type": schedule_type,
			"principal": principal,
			"annual_rate": annual_rate,
			"tenor_months": tenor_months,
			"monthly_emi": round(emi, 2),
			"total_repayable": round(emi * tenor_months, 2),
			"total_interest": round(emi * tenor_months - principal, 2),
			"installments": installments,
		}

	def generate_repayment_schedule(self, loan_id: str, schedule_type: str = "reducing_balance") -> dict[str, Any]:
		"""
		Return the full repayment schedule for a loan.
		Supports reducing_balance (default) and flat_rate.
		"""
		loan = self._require_loan(loan_id)
		sched = self._build_schedule(loan.principal, loan.annual_rate, loan.tenor_months, loan.disbursement_date, schedule_type)
		sched["loan_id"] = loan_id
		sched["currency"] = loan.currency
		return sched

	def process_repayment(self, loan_id: str, amount: float | int | str, payment_date: str, payment_method: str, reference: str) -> dict[str, Any]:
		"""
		Apply a repayment to a loan.
		Allocation order: outstanding fees first, then interest, then principal.
		Updates installment statuses. Returns allocation breakdown.
		"""
		loan = self._require_loan(loan_id)
		assert loan.status == "active", f"cannot process repayment on loan with status '{loan.status}'"
		pmt_amount = normalize_amount(amount)
		assert pmt_amount > 0, "payment amount must be positive"
		assert payment_date, "payment_date required"
		assert reference, "reference required"

		remaining = pmt_amount
		fees_cleared = 0.0
		interest_cleared = 0.0
		principal_cleared = 0.0
		allocations: list[dict] = []

		# Step 1: clear outstanding fees
		for fee in loan.fees:
			if fee.get("status") == "outstanding" and remaining > 0:
				fee_outstanding = fee.get("amount", 0.0) - fee.get("paid", 0.0)
				applied = min(remaining, fee_outstanding)
				fee["paid"] = fee.get("paid", 0.0) + applied
				if fee["paid"] >= fee["amount"]:
					fee["status"] = "cleared"
				remaining -= applied
				fees_cleared += applied
				allocations.append({"type": "fee", "fee_type": fee.get("fee_type", "fee"), "applied": round(applied, 2)})

		# Step 2: apply to installments (oldest first)
		pending = sorted([i for i in loan.installments if i["status"] in ("pending", "partial")], key=lambda x: x["installment_no"])
		for inst in pending:
			if remaining <= 0:
				break
			inst_outstanding = inst["emi"] - inst.get("paid_amount", 0.0)
			if inst_outstanding <= 0:
				inst["status"] = "paid"
				continue

			# Interest portion of this installment
			inst_interest = inst["interest"]
			inst_principal = inst["principal"]
			inst_interest_paid = inst.get("interest_paid", 0.0)
			inst_principal_paid = inst.get("principal_paid", 0.0)
			interest_due = inst_interest - inst_interest_paid
			principal_due = inst_principal - inst_principal_paid

			# Pay interest first
			interest_applied = min(remaining, interest_due)
			remaining -= interest_applied
			inst["interest_paid"] = inst_interest_paid + interest_applied
			interest_cleared += interest_applied

			# Then principal
			principal_applied = min(remaining, principal_due)
			remaining -= principal_applied
			inst["principal_paid"] = inst_principal_paid + principal_applied
			principal_cleared += principal_applied
			inst["paid_amount"] = inst.get("paid_amount", 0.0) + interest_applied + principal_applied

			if inst["paid_amount"] >= inst["emi"] - 0.01:
				inst["status"] = "paid"
				inst["paid_date"] = payment_date
			else:
				inst["status"] = "partial"

			allocations.append({"installment_no": inst["installment_no"], "interest_applied": round(interest_applied, 2), "principal_applied": round(principal_applied, 2)})

		# Reduce outstanding principal
		loan.outstanding_principal = round(max(0.0, loan.outstanding_principal - principal_cleared), 2)

		# Compute DPD at time of payment
		today = _today()
		pmt_d = _parse_date(payment_date)
		dpd = (pmt_d - today).days  # negative means late; treat as 0 for on-time

		pmt_record = {
			"reference": reference,
			"amount": pmt_amount,
			"payment_date": payment_date,
			"payment_method": payment_method,
			"fees_cleared": round(fees_cleared, 2),
			"interest_cleared": round(interest_cleared, 2),
			"principal_cleared": round(principal_cleared, 2),
			"overpayment": round(remaining, 2),
			"dpd": 0,
		}
		loan.repayments.append(pmt_record)

		# Check if fully repaid
		if loan.outstanding_principal <= 0.01:
			loan.status = "settled"
			self._audit(loan.tenant_id, "loan_settled", loan_id)

		self._audit(loan.tenant_id, "repayment_processed", loan_id, {"amount": pmt_amount, "reference": reference})

		return {
			"loan_id": loan_id,
			"payment_amount": pmt_amount,
			"fees_cleared": round(fees_cleared, 2),
			"interest_cleared": round(interest_cleared, 2),
			"principal_cleared": round(principal_cleared, 2),
			"overpayment": round(remaining, 2),
			"outstanding_principal": loan.outstanding_principal,
			"loan_status": loan.status,
			"allocations": allocations,
		}

	def early_settlement(self, loan_id: str, settlement_date: str) -> dict[str, Any]:
		"""
		Calculate the total amount required to settle a loan early.
		= outstanding principal + accrued interest to settlement date + early settlement fee (1% of principal).
		"""
		loan = self._require_loan(loan_id)
		assert loan.status == "active", f"loan not active: status '{loan.status}'"
		assert settlement_date, "settlement_date required"

		disburse_d = _parse_date(loan.disbursement_date)
		settle_d = _parse_date(settlement_date)
		days_elapsed = (settle_d - disburse_d).days
		accrued_interest = round(loan.outstanding_principal * loan.annual_rate * days_elapsed / 365, 2)

		# Early settlement fee: 1% of outstanding principal (product-configurable, hardcoded here)
		early_settlement_fee = round(loan.outstanding_principal * 0.01, 2)

		total_settlement = round(loan.outstanding_principal + accrued_interest + early_settlement_fee, 2)

		self._audit(loan.tenant_id, "early_settlement_calculated", loan_id, {"settlement_date": settlement_date, "total": total_settlement})

		return {
			"loan_id": loan_id,
			"settlement_date": settlement_date,
			"outstanding_principal": loan.outstanding_principal,
			"accrued_interest": accrued_interest,
			"early_settlement_fee": early_settlement_fee,
			"total_settlement_amount": total_settlement,
			"currency": loan.currency,
		}

	def restructure_loan(self, loan_id: str, new_terms: dict[str, Any], reason: str, approved_by: str) -> dict[str, Any]:
		"""
		Restructure: extend tenor, reduce rate, or capitalise arrears.
		Requires approval. Rebuilds repayment schedule from today.
		"""
		loan = self._require_loan(loan_id)
		assert loan.status == "active", f"can only restructure active loan, current: '{loan.status}'"
		assert reason, "reason required"
		assert approved_by, "approved_by required"

		old_terms = {"annual_rate": loan.annual_rate, "tenor_months": loan.tenor_months, "outstanding_principal": loan.outstanding_principal}

		# Apply new terms
		new_rate = normalize_rate(new_terms.get("annual_rate", loan.annual_rate))
		new_tenor = int(new_terms.get("tenor_months", loan.tenor_months))
		capitalise_arrears = bool(new_terms.get("capitalise_arrears", False))
		assert new_tenor > 0, "tenor_months must be positive"

		if capitalise_arrears:
			# Find total arrears (unpaid interest and fees)
			arrears = sum(fee.get("amount", 0) - fee.get("paid", 0) for fee in loan.fees if fee.get("status") == "outstanding")
			loan.outstanding_principal = round(loan.outstanding_principal + arrears, 2)

		loan.annual_rate = new_rate
		loan.tenor_months = new_tenor

		# Rebuild schedule from today
		new_sched = self._build_schedule(loan.outstanding_principal, new_rate, new_tenor, _today().isoformat(), loan.schedule_type)
		loan.installments = new_sched["installments"]

		restructure_record = {
			"restructure_date": _today().isoformat(),
			"old_terms": old_terms,
			"new_terms": {"annual_rate": new_rate, "tenor_months": new_tenor, "capitalise_arrears": capitalise_arrears},
			"reason": reason,
			"approved_by": approved_by,
		}
		loan.restructure_history.append(restructure_record)
		self._audit(loan.tenant_id, "loan_restructured", loan_id, {"approved_by": approved_by, "reason": reason})

		return {
			"loan_id": loan_id,
			"restructure_record": restructure_record,
			"new_outstanding_principal": loan.outstanding_principal,
			"new_monthly_emi": new_sched["monthly_emi"],
			"installment_count": len(new_sched["installments"]),
			"restructure_history_count": len(loan.restructure_history),
		}

	def add_loan_fee(self, loan_id: str, fee_type: str, amount: float | int | str, reason: str) -> dict[str, Any]:
		"""
		Charge a fee to a loan: late_payment_penalty, restructuring_fee, legal_fee, insurance_premium, etc.
		"""
		loan = self._require_loan(loan_id)
		valid_fee_types = ("late_payment_penalty", "restructuring_fee", "legal_fee", "insurance_premium", "processing_fee", "other")
		fee_type = normalize_code(fee_type)
		assert fee_type in valid_fee_types, f"fee_type must be one of {valid_fee_types}"
		fee_amount = normalize_amount(amount)
		assert fee_amount > 0, "fee amount must be positive"
		assert reason, "reason required"

		fee_id = f"fee_{loan_id}_{len(loan.fees) + 1}"
		fee = {
			"fee_id": fee_id,
			"fee_type": fee_type,
			"amount": fee_amount,
			"reason": reason,
			"status": "outstanding",
			"paid": 0.0,
			"charged_at": _today().isoformat(),
		}
		loan.fees.append(fee)
		self._audit(loan.tenant_id, "loan_fee_added", loan_id, {"fee_id": fee_id, "fee_type": fee_type, "amount": fee_amount})

		return {"loan_id": loan_id, "fee": fee, "total_outstanding_fees": round(sum(f["amount"] - f.get("paid", 0) for f in loan.fees if f["status"] == "outstanding"), 2)}

	def waive_fee_or_penalty(self, loan_id: str, fee_id: str, waiver_reason: str, approved_by: str) -> dict[str, Any]:
		"""Waive a specific fee or penalty on a loan. Requires approval."""
		loan = self._require_loan(loan_id)
		assert waiver_reason, "waiver_reason required"
		assert approved_by, "approved_by required"

		fee = next((f for f in loan.fees if f.get("fee_id") == fee_id), None)
		if fee is None:
			raise KeyError(f"fee not found: {fee_id} on loan {loan_id}")

		waived_amount = fee["amount"] - fee.get("paid", 0.0)
		fee["status"] = "waived"
		fee["waiver_reason"] = waiver_reason
		fee["waived_by"] = approved_by
		fee["waived_at"] = _today().isoformat()

		self._audit(loan.tenant_id, "fee_waived", loan_id, {"fee_id": fee_id, "waived_amount": waived_amount, "approved_by": approved_by})

		return {"loan_id": loan_id, "fee_id": fee_id, "waived_amount": round(waived_amount, 2), "waiver_reason": waiver_reason, "approved_by": approved_by}

	def record_collateral_release(self, loan_id: str, collateral_id: str, reason: str, released_by: str) -> dict[str, Any]:
		"""Release a collateral item when loan is settled or partially repaid."""
		loan = self._require_loan(loan_id)
		coll = self.collateral.get(collateral_id)
		if coll is None:
			raise KeyError(f"collateral not found: {collateral_id}")
		assert reason, "reason required"
		assert released_by, "released_by required"
		assert coll.status == "held", f"collateral not held: status '{coll.status}'"

		coll.status = "released"
		coll.release_reason = reason
		coll.released_by = released_by
		coll.release_date = _today().isoformat()

		if collateral_id in loan.collateral_ids:
			loan.collateral_ids.remove(collateral_id)

		self._audit(loan.tenant_id, "collateral_released", loan_id, {"collateral_id": collateral_id, "released_by": released_by})

		return {"loan_id": loan_id, "collateral": coll.to_dict()}

	def get_loan_statement(self, loan_id: str) -> dict[str, Any]:
		"""
		Return full transaction history for a loan:
		disbursement, all repayments, all fees, restructures, collateral.
		"""
		loan = self._require_loan(loan_id)
		collateral_items = [c.to_dict() for c in self.collateral.values() if c.loan_id == loan_id]

		return {
			"loan_id": loan_id,
			"borrower_id": loan.borrower_id,
			"product_id": loan.product_id,
			"currency": loan.currency,
			"principal": loan.principal,
			"outstanding_principal": loan.outstanding_principal,
			"annual_rate": loan.annual_rate,
			"tenor_months": loan.tenor_months,
			"disbursement_date": loan.disbursement_date,
			"status": loan.status,
			"repayments": list(loan.repayments),
			"fees": list(loan.fees),
			"installments": [
				{k: v for k, v in inst.items()} for inst in loan.installments
			],
			"restructure_history": list(loan.restructure_history),
			"collateral": collateral_items,
			"collection_activities": list(loan.collection_activities),
			"demand_notices": list(loan.demand_notices),
			"legal_actions": list(loan.legal_actions),
			"total_repaid": round(sum(r["principal_cleared"] for r in loan.repayments), 2),
			"total_interest_paid": round(sum(r["interest_cleared"] for r in loan.repayments), 2),
			"total_fees_paid": round(sum(r.get("fees_cleared", 0) for r in loan.repayments), 2),
			"statement_date": _today().isoformat(),
		}

	def close_loan(self, loan_id: str, closing_reason: str) -> dict[str, Any]:
		"""
		Close a loan. closing_reason: 'settled' | 'written_off' | 'restructured_out' | 'cancelled'.
		Validates that outstanding principal is zero for settled closure.
		"""
		loan = self._require_loan(loan_id)
		valid_reasons = ("settled", "written_off", "restructured_out", "cancelled", "early_settlement")
		assert closing_reason in valid_reasons, f"closing_reason must be one of {valid_reasons}"

		if closing_reason == "settled":
			if loan.outstanding_principal > 1.0:
				raise ValueError(f"cannot close as settled: outstanding principal {loan.outstanding_principal} remains")

		loan.status = "closed"
		loan.closure_reason = closing_reason
		self._audit(loan.tenant_id, "loan_closed", loan_id, {"reason": closing_reason})

		return {"loan_id": loan_id, "status": loan.status, "closing_reason": closing_reason, "closed_at": _today().isoformat()}

	# ------------------------------------------------------------------
	# DELINQUENCY & COLLECTIONS (8 methods)
	# ------------------------------------------------------------------

	def calculate_dpd(self, loan_id: str) -> dict[str, Any]:
		"""
		Compute Days Past Due for each installment as of today.
		Returns per-installment DPD and the maximum DPD for the loan.
		"""
		loan = self._require_loan(loan_id)
		today = _today()
		dpd_list = []
		max_dpd = 0

		for inst in loan.installments:
			if inst["status"] == "paid":
				dpd_list.append({"installment_no": inst["installment_no"], "due_date": inst["due_date"], "dpd": 0, "status": inst["status"]})
				continue
			due = _parse_date(inst["due_date"])
			dpd = max(0, (today - due).days)
			max_dpd = max(max_dpd, dpd)
			dpd_list.append({"installment_no": inst["installment_no"], "due_date": inst["due_date"], "dpd": dpd, "status": inst["status"], "outstanding": round(inst["emi"] - inst.get("paid_amount", 0), 2)})

		# Update repayment records with dpd
		for i, pmt in enumerate(loan.repayments):
			inst_no = None
			for alloc in (dpd_list or []):
				pass  # simplified: dpd is per installment not per payment
			loan.repayments[i]["dpd"] = 0

		return {
			"loan_id": loan_id,
			"as_of_date": today.isoformat(),
			"max_dpd": max_dpd,
			"installments": dpd_list,
			"delinquency_bucket": self._dpd_bucket(max_dpd),
		}

	@staticmethod
	def _dpd_bucket(dpd: int) -> str:
		if dpd == 0:
			return "current"
		if dpd <= 30:
			return "1-30"
		if dpd <= 60:
			return "31-60"
		if dpd <= 90:
			return "61-90"
		if dpd <= 120:
			return "91-120"
		return "120+"

	def delinquency_report(self, as_of_date: str | None = None) -> dict[str, Any]:
		"""
		Portfolio delinquency report with DPD buckets and PAR ratios.
		Buckets: current, 1-30, 31-60, 61-90, 91-120, 120+.
		PAR = outstanding balance in bucket / total portfolio outstanding.
		"""
		as_of = _parse_date(as_of_date) if as_of_date else _today()
		buckets: dict[str, dict] = {
			"current": {"count": 0, "outstanding": 0.0},
			"1-30":   {"count": 0, "outstanding": 0.0},
			"31-60":  {"count": 0, "outstanding": 0.0},
			"61-90":  {"count": 0, "outstanding": 0.0},
			"91-120": {"count": 0, "outstanding": 0.0},
			"120+":   {"count": 0, "outstanding": 0.0},
		}
		total_outstanding = 0.0

		for loan in self.loans.values():
			if loan.status not in ("active",):
				continue
			max_dpd = 0
			for inst in loan.installments:
				if inst["status"] == "paid":
					continue
				due = _parse_date(inst["due_date"])
				dpd = max(0, (as_of - due).days)
				max_dpd = max(max_dpd, dpd)
			bucket = self._dpd_bucket(max_dpd)
			buckets[bucket]["count"] += 1
			buckets[bucket]["outstanding"] += loan.outstanding_principal
			total_outstanding += loan.outstanding_principal

		par_ratios = {}
		npl_outstanding = 0.0
		for bucket, data in buckets.items():
			data["outstanding"] = round(data["outstanding"], 2)
			par = round(data["outstanding"] / total_outstanding, 4) if total_outstanding else 0.0
			par_ratios[f"par_{bucket.replace('-', '_').replace('+', 'plus')}"] = par
			if bucket != "current":
				npl_outstanding += data["outstanding"]

		npl_ratio = round(npl_outstanding / total_outstanding, 4) if total_outstanding else 0.0
		par30 = round((buckets["31-60"]["outstanding"] + buckets["61-90"]["outstanding"] + buckets["91-120"]["outstanding"] + buckets["120+"]["outstanding"]) / total_outstanding, 4) if total_outstanding else 0.0

		return {
			"as_of_date": as_of.isoformat(),
			"total_active_loans": sum(b["count"] for b in buckets.values()),
			"total_outstanding": round(total_outstanding, 2),
			"npl_outstanding": round(npl_outstanding, 2),
			"npl_ratio": npl_ratio,
			"par_30": par30,
			"buckets": buckets,
			"par_ratios": par_ratios,
		}

	def generate_demand_notice(self, loan_id: str, level: int) -> dict[str, Any]:
		"""
		Generate a demand notice at escalating severity level 1–4.
		  1: Friendly reminder
		  2: Formal notice to pay
		  3: Final demand before legal action
		  4: Notice of intent to sue / list on CRB
		"""
		loan = self._require_loan(loan_id)
		assert 1 <= level <= 4, "demand notice level must be 1–4"

		dpd_info = self.calculate_dpd(loan_id)
		max_dpd = dpd_info["max_dpd"]

		level_text = {
			1: "Friendly reminder: your loan instalment is overdue. Please pay immediately to avoid penalties.",
			2: "Formal notice: this is a formal demand for immediate payment of your outstanding loan balance. Failure to pay within 7 days will result in additional penalties.",
			3: "Final demand: this is your final notice before legal action is initiated and your account is listed with the Credit Reference Bureau.",
			4: "NOTICE OF INTENT TO SUE: Legal proceedings will commence in 48 hours. Your account will be listed with the CRB. Settle immediately to avoid further consequences.",
		}[level]

		notice = {
			"loan_id": loan_id,
			"level": level,
			"max_dpd": max_dpd,
			"outstanding_principal": loan.outstanding_principal,
			"currency": loan.currency,
			"notice_text": level_text,
			"issued_at": _today().isoformat(),
			"response_deadline": (_today() + timedelta(days=7 if level <= 2 else 2)).isoformat(),
		}

		loan.demand_notices.append(notice)
		self._demand_notices.setdefault(loan_id, []).append(notice)
		self._audit(loan.tenant_id, f"demand_notice_level_{level}_issued", loan_id, {"max_dpd": max_dpd})

		return notice

	def assign_to_collector(self, loan_id: str, collector_id: str) -> dict[str, Any]:
		"""Assign a delinquent loan to a collections officer."""
		loan = self._require_loan(loan_id)
		assert collector_id, "collector_id required"
		loan.assigned_collector = collector_id
		self._collectors[loan_id] = collector_id
		self._audit(loan.tenant_id, "loan_assigned_to_collector", loan_id, {"collector_id": collector_id})
		return {"loan_id": loan_id, "collector_id": collector_id, "assigned_at": _today().isoformat()}

	def record_collection_activity(self, loan_id: str, activity_type: str, outcome: str, notes: str, next_action: str) -> dict[str, Any]:
		"""
		Record a collections activity (call, visit, letter, legal).
		activity_type: call | field_visit | sms | email | legal | promise_to_pay | payment_received
		outcome: contacted | no_answer | promise_to_pay | paid | refused | escalate
		"""
		loan = self._require_loan(loan_id)
		assert activity_type, "activity_type required"
		assert outcome, "outcome required"

		activity = {
			"activity_type": normalize_code(activity_type),
			"outcome": normalize_code(outcome),
			"notes": notes,
			"next_action": next_action,
			"collector_id": loan.assigned_collector,
			"recorded_at": _today().isoformat(),
		}
		loan.collection_activities.append(activity)
		self._audit(loan.tenant_id, "collection_activity_recorded", loan_id, {"type": activity_type, "outcome": outcome})
		return {"loan_id": loan_id, "activity": activity, "total_activities": len(loan.collection_activities)}

	def legal_action(self, loan_id: str, action_type: str, lawyer_id: str, court_date: str | None = None) -> dict[str, Any]:
		"""
		Record legal action: file_suit | serve_summons | obtain_judgment | garnish_wages | attach_property | crb_listing.
		"""
		loan = self._require_loan(loan_id)
		valid_actions = ("file_suit", "serve_summons", "obtain_judgment", "garnish_wages", "attach_property", "crb_listing", "out_of_court_settlement")
		action_type = normalize_code(action_type)
		assert action_type in valid_actions, f"action_type must be one of {valid_actions}"
		assert lawyer_id, "lawyer_id required"

		loan.assigned_lawyer = lawyer_id
		legal_record = {
			"action_type": action_type,
			"lawyer_id": lawyer_id,
			"court_date": court_date,
			"initiated_at": _today().isoformat(),
			"status": "initiated",
		}
		loan.legal_actions.append(legal_record)
		self._audit(loan.tenant_id, f"legal_action_{action_type}", loan_id, {"lawyer_id": lawyer_id, "court_date": court_date})
		return {"loan_id": loan_id, "legal_action": legal_record, "total_legal_actions": len(loan.legal_actions)}

	def write_off_loan(self, loan_id: str, reason: str, write_off_date: str, approved_by: str) -> dict[str, Any]:
		"""
		Write off a non-performing loan. Moves to 'written_off' status.
		Requires approval. Records write-off amount for provision reversal.
		"""
		loan = self._require_loan(loan_id)
		assert loan.status == "active", f"can only write off active loan, current: '{loan.status}'"
		assert reason, "reason required"
		assert write_off_date, "write_off_date required"
		assert approved_by, "approved_by required"

		write_off_amount = loan.outstanding_principal
		total_fees_outstanding = sum(f["amount"] - f.get("paid", 0) for f in loan.fees if f.get("status") == "outstanding")

		loan.status = "written_off"
		loan.write_off_reason = reason
		loan.written_off_by = approved_by
		loan.written_off_date = write_off_date

		write_off_record = {
			"loan_id": loan_id,
			"write_off_amount": round(write_off_amount, 2),
			"fees_written_off": round(total_fees_outstanding, 2),
			"total_written_off": round(write_off_amount + total_fees_outstanding, 2),
			"reason": reason,
			"write_off_date": write_off_date,
			"approved_by": approved_by,
			"currency": loan.currency,
		}
		self._writeoffs[loan_id] = write_off_record
		self._audit(loan.tenant_id, "loan_written_off", loan_id, {"approved_by": approved_by, "amount": write_off_amount})
		return write_off_record

	def collection_performance_report(self, period: str, collector_id: str | None = None) -> dict[str, Any]:
		"""
		Summarise collections performance: loans assigned, activities, outcomes, amount recovered.
		Optionally filtered to a specific collector.
		"""
		assert period, "period required"

		loans_in_scope = [ln for ln in self.loans.values() if collector_id is None or ln.assigned_collector == collector_id]

		total_assigned = len(loans_in_scope)
		total_outstanding = round(sum(ln.outstanding_principal for ln in loans_in_scope if ln.status == "active"), 2)
		total_recovered = round(sum(sum(r["principal_cleared"] for r in ln.repayments) for ln in loans_in_scope), 2)
		total_activities = sum(len(ln.collection_activities) for ln in loans_in_scope)

		outcomes: dict[str, int] = {}
		for ln in loans_in_scope:
			for act in ln.collection_activities:
				o = act.get("outcome", "unknown")
				outcomes[o] = outcomes.get(o, 0) + 1

		total_demand_notices = sum(len(ln.demand_notices) for ln in loans_in_scope)
		total_legal = sum(len(ln.legal_actions) for ln in loans_in_scope)
		total_written_off = sum(1 for ln in loans_in_scope if ln.status == "written_off")
		total_settled = sum(1 for ln in loans_in_scope if ln.status in ("settled", "closed"))

		recovery_rate = round(total_recovered / (total_recovered + total_outstanding), 4) if (total_recovered + total_outstanding) > 0 else 0.0

		return {
			"period": period,
			"collector_id": collector_id,
			"total_loans_assigned": total_assigned,
			"total_outstanding": total_outstanding,
			"total_recovered": total_recovered,
			"recovery_rate": recovery_rate,
			"total_activities": total_activities,
			"outcome_breakdown": outcomes,
			"demand_notices_issued": total_demand_notices,
			"legal_actions_initiated": total_legal,
			"loans_written_off": total_written_off,
			"loans_settled": total_settled,
		}

	# ------------------------------------------------------------------
	# PORTFOLIO ANALYTICS (5 methods)
	# ------------------------------------------------------------------

	def portfolio_summary(self, as_of_date: str | None = None) -> dict[str, Any]:
		"""
		Total book, PAR 30/60/90, NPL ratio, average ticket, yield.
		"""
		as_of = _parse_date(as_of_date) if as_of_date else _today()

		active_loans = [ln for ln in self.loans.values() if ln.status == "active"]
		total_book = round(sum(ln.outstanding_principal for ln in active_loans), 2)
		total_disbursed = round(sum(ln.principal for ln in self.loans.values()), 2)
		average_ticket = round(total_book / len(active_loans), 2) if active_loans else 0.0

		# PAR buckets
		par_30_balance = 0.0
		par_60_balance = 0.0
		par_90_balance = 0.0
		npl_balance = 0.0

		for loan in active_loans:
			max_dpd = 0
			for inst in loan.installments:
				if inst["status"] == "paid":
					continue
				due = _parse_date(inst["due_date"])
				dpd = max(0, (as_of - due).days)
				max_dpd = max(max_dpd, dpd)
			if max_dpd > 30:
				par_30_balance += loan.outstanding_principal
			if max_dpd > 60:
				par_60_balance += loan.outstanding_principal
			if max_dpd > 90:
				par_90_balance += loan.outstanding_principal
				npl_balance += loan.outstanding_principal

		par_30 = round(par_30_balance / total_book, 4) if total_book else 0.0
		par_60 = round(par_60_balance / total_book, 4) if total_book else 0.0
		par_90 = round(par_90_balance / total_book, 4) if total_book else 0.0
		npl_ratio = round(npl_balance / total_book, 4) if total_book else 0.0

		# Portfolio yield: weighted average APR
		if active_loans and total_book:
			weighted_rate = sum(ln.outstanding_principal * ln.annual_rate for ln in active_loans)
			portfolio_yield = round(weighted_rate / total_book, 4)
		else:
			portfolio_yield = 0.0

		written_off = round(sum(self._writeoffs.get(ln.loan_id, {}).get("write_off_amount", 0) for ln in self.loans.values()), 2)

		return {
			"as_of_date": as_of.isoformat(),
			"total_active_loans": len(active_loans),
			"total_book": total_book,
			"total_disbursed": total_disbursed,
			"average_ticket": average_ticket,
			"portfolio_yield": portfolio_yield,
			"par_30": par_30,
			"par_60": par_60,
			"par_90": par_90,
			"npl_ratio": npl_ratio,
			"npl_balance": round(npl_balance, 2),
			"written_off_total": written_off,
		}

	def provision_calculation(self, method: str = "ifrs9") -> dict[str, Any]:
		"""
		IFRS 9 Expected Credit Loss (ECL) calculation.
		Stage 1: 12-month ECL for performing loans (DPD 0)
		Stage 2: Lifetime ECL for significant credit deterioration (DPD 1–90)
		Stage 3: Lifetime ECL for credit-impaired loans (DPD > 90)
		LGD assumption: 40% for unsecured, 25% for collateralised.
		EAD: outstanding principal.
		"""
		assert method in ("ifrs9", "incurred_loss"), f"method must be 'ifrs9' or 'incurred_loss'"

		stage1_balance = 0.0; stage1_ecl = 0.0; stage1_count = 0
		stage2_balance = 0.0; stage2_ecl = 0.0; stage2_count = 0
		stage3_balance = 0.0; stage3_ecl = 0.0; stage3_count = 0

		today = _today()
		for loan in self.loans.values():
			if loan.status not in ("active",):
				continue

			max_dpd = 0
			for inst in loan.installments:
				if inst["status"] == "paid":
					continue
				due = _parse_date(inst["due_date"])
				dpd = max(0, (today - due).days)
				max_dpd = max(max_dpd, dpd)

			ead = loan.outstanding_principal
			cs = self.credit_scores.get(loan.borrower_id)
			pd_12m = cs.probability_of_default if cs else 0.05
			pd_lifetime = min(1.0, pd_12m * loan.tenor_months / 12)
			has_collateral = bool(loan.collateral_ids)
			lgd = 0.25 if has_collateral else 0.40

			if max_dpd == 0:
				# Stage 1: 12-month ECL
				ecl = round(pd_12m * lgd * ead, 2)
				stage1_balance += ead; stage1_ecl += ecl; stage1_count += 1
			elif max_dpd <= 90:
				# Stage 2: lifetime ECL
				ecl = round(pd_lifetime * lgd * ead, 2)
				stage2_balance += ead; stage2_ecl += ecl; stage2_count += 1
			else:
				# Stage 3: lifetime ECL (credit-impaired, PD ~= 1)
				ecl = round(lgd * ead, 2)
				stage3_balance += ead; stage3_ecl += ecl; stage3_count += 1

		total_ecl = round(stage1_ecl + stage2_ecl + stage3_ecl, 2)
		total_balance = round(stage1_balance + stage2_balance + stage3_balance, 2)
		provision_coverage = round(total_ecl / total_balance, 4) if total_balance else 0.0

		return {
			"method": method,
			"as_of_date": today.isoformat(),
			"stage1": {"loan_count": stage1_count, "outstanding_balance": round(stage1_balance, 2), "ecl": round(stage1_ecl, 2), "basis": "12_month_pd"},
			"stage2": {"loan_count": stage2_count, "outstanding_balance": round(stage2_balance, 2), "ecl": round(stage2_ecl, 2), "basis": "lifetime_pd"},
			"stage3": {"loan_count": stage3_count, "outstanding_balance": round(stage3_balance, 2), "ecl": round(stage3_ecl, 2), "basis": "credit_impaired_lgd"},
			"total_ecl": total_ecl,
			"total_outstanding": total_balance,
			"provision_coverage_ratio": provision_coverage,
		}

	def vintage_analysis(self, cohort_months: int = 12) -> dict[str, Any]:
		"""
		Analyse default rates by origination cohort (disbursement month).
		cohort_months: number of recent months to include.
		Returns per-cohort: disbursed count, written-off count, default rate, total principal.
		"""
		assert cohort_months > 0, "cohort_months must be positive"

		today = _today()
		cutoff = _add_months(today, -cohort_months)

		cohorts: dict[str, dict] = {}
		for loan in self.loans.values():
			try:
				d = _parse_date(loan.disbursement_date)
			except ValueError:
				continue
			if d < cutoff:
				continue
			cohort_key = f"{d.year}-{d.month:02d}"
			if cohort_key not in cohorts:
				cohorts[cohort_key] = {"cohort": cohort_key, "disbursed_count": 0, "written_off_count": 0, "total_principal": 0.0, "outstanding": 0.0}
			cohorts[cohort_key]["disbursed_count"] += 1
			cohorts[cohort_key]["total_principal"] += loan.principal
			cohorts[cohort_key]["outstanding"] += loan.outstanding_principal
			if loan.status == "written_off":
				cohorts[cohort_key]["written_off_count"] += 1

		results = []
		for c in sorted(cohorts.values(), key=lambda x: x["cohort"]):
			n = c["disbursed_count"]
			c["default_rate"] = round(c["written_off_count"] / n, 4) if n else 0.0
			c["total_principal"] = round(c["total_principal"], 2)
			c["outstanding"] = round(c["outstanding"], 2)
			results.append(c)

		return {
			"cohort_months": cohort_months,
			"as_of_date": today.isoformat(),
			"cohorts": results,
			"total_cohort_count": len(results),
		}

	def concentration_risk_report(self) -> dict[str, Any]:
		"""
		Concentration by sector (product_type), geography (country via borrower), and ticket size band.
		Returns balance-weighted percentages.
		"""
		by_product: dict[str, float] = {}
		by_country: dict[str, float] = {}
		ticket_bands: dict[str, dict] = {
			"micro_0_50k":        {"min": 0,       "max": 50000,   "count": 0, "balance": 0.0},
			"small_50k_200k":     {"min": 50000,    "max": 200000,  "count": 0, "balance": 0.0},
			"mid_200k_1m":        {"min": 200000,   "max": 1000000, "count": 0, "balance": 0.0},
			"large_1m_plus":      {"min": 1000000,  "max": float("inf"), "count": 0, "balance": 0.0},
		}
		total_balance = 0.0

		for loan in self.loans.values():
			if loan.status not in ("active",):
				continue
			bal = loan.outstanding_principal
			total_balance += bal

			# Sector via product type
			product = self.products.get(loan.product_id)
			sector = product.product_type if product else "unknown"
			by_product[sector] = by_product.get(sector, 0.0) + bal

			# Geography via borrower
			borrower = self.borrowers.get(loan.borrower_id)
			country = borrower.country if borrower else "UNKNOWN"
			by_country[country] = by_country.get(country, 0.0) + bal

			# Ticket size
			for band_data in ticket_bands.values():
				if band_data["min"] <= loan.principal < band_data["max"]:
					band_data["count"] += 1
					band_data["balance"] += bal
					break

		def pct(v: float) -> float:
			return round(v / total_balance, 4) if total_balance else 0.0

		return {
			"as_of_date": _today().isoformat(),
			"total_portfolio_balance": round(total_balance, 2),
			"by_sector": {k: {"balance": round(v, 2), "pct": pct(v)} for k, v in sorted(by_product.items(), key=lambda x: -x[1])},
			"by_geography": {k: {"balance": round(v, 2), "pct": pct(v)} for k, v in sorted(by_country.items(), key=lambda x: -x[1])},
			"by_ticket_size": {
				k: {
					"count": v["count"],
					"balance": round(v["balance"], 2),
					"pct": pct(v["balance"]),
				}
				for k, v in ticket_bands.items()
			},
		}

	def stress_test(self, scenarios: list[dict[str, Any]]) -> dict[str, Any]:
		"""
		Run default rate sensitivity scenarios against the portfolio.
		Each scenario: {"name": str, "additional_default_rate": float, "lgd": float}
		Returns per-scenario: incremental losses, stressed NLP ratio, capital impact.
		"""
		assert scenarios, "scenarios list required"

		base_summary = self.portfolio_summary()
		total_book = base_summary["total_book"]
		base_npl = base_summary["npl_ratio"]

		results = []
		for scenario in scenarios:
			name = scenario.get("name", "unnamed")
			add_default = float(scenario.get("additional_default_rate", 0.05))
			lgd = float(scenario.get("lgd", 0.40))
			assert 0 <= add_default <= 1, "additional_default_rate must be in [0,1]"
			assert 0 <= lgd <= 1, "lgd must be in [0,1]"

			incremental_loss = round(total_book * add_default * lgd, 2)
			stressed_npl = round(min(1.0, base_npl + add_default), 4)
			stressed_npl_balance = round(total_book * stressed_npl, 2)

			# Capital impact assuming 100% provisioning on incremental defaults
			capital_charge = incremental_loss

			results.append({
				"scenario": name,
				"additional_default_rate": add_default,
				"lgd": lgd,
				"incremental_loss": incremental_loss,
				"stressed_npl_ratio": stressed_npl,
				"stressed_npl_balance": stressed_npl_balance,
				"capital_charge": capital_charge,
				"total_portfolio_balance": total_book,
			})

		return {
			"as_of_date": _today().isoformat(),
			"base_npl_ratio": base_npl,
			"base_portfolio_balance": total_book,
			"scenarios": results,
		}

	# ------------------------------------------------------------------
	# PRODUCT CONFIG (4 methods)
	# ------------------------------------------------------------------

	def create_loan_product(
		self,
		product_code: str,
		name: str,
		product_type: str,
		rate_type: str,
		min_amount: float | int | str,
		max_amount: float | int | str,
		min_tenor: int,
		max_tenor: int,
		fees: list[dict[str, Any]],
		tenant_id: str = "default",
		owner_id: str = "system",
		currency: str = "KES",
		repayment_frequency: str = "monthly",
	) -> dict[str, Any]:
		"""
		Create a new loan product with full fee schedule.
		rate_type: fixed | variable | reducing_balance
		fees: list of {"fee_type": str, "rate": float, "amount": float, "basis": "flat|pct_principal"}
		"""
		assert product_code, "product_code required"
		assert name, "name required"
		rate_type = normalize_code(rate_type)
		assert rate_type in ("fixed", "variable", "reducing_balance"), f"rate_type invalid: {rate_type}"

		# Use existing register_product for core validation
		result = self.register_product(
			product_id=product_code,
			tenant_id=tenant_id,
			name=name,
			owner_id=owner_id,
			product_type=product_type,
			currency=currency,
			min_amount=min_amount,
			max_amount=max_amount,
			min_term_days=min_tenor * 30,
			max_term_days=max_tenor * 30,
			annual_rate=0.18,  # default base rate; caller updates via update_product_rates
			repayment_frequency=repayment_frequency,
		)
		result["rate_type"] = rate_type
		result["fees"] = fees
		result["min_tenor_months"] = min_tenor
		result["max_tenor_months"] = max_tenor
		return result

	def update_product_rates(self, product_code: str, new_rates: dict[str, float], effective_date: str) -> dict[str, Any]:
		"""
		Update pricing on a product. new_rates: {"annual_rate": float, ...}.
		Records rate history. Effective_date is informational (rates applied immediately in this runtime).
		"""
		assert product_code, "product_code required"
		assert new_rates, "new_rates required"
		assert effective_date, "effective_date required"

		product = self._require_product(product_code)
		old_rate = product.annual_rate

		if "annual_rate" in new_rates:
			new_rate = normalize_rate(new_rates["annual_rate"])
			assert 0 < new_rate <= 0.75, f"annual_rate must be in (0, 0.75], got {new_rate}"
			product.annual_rate = new_rate

		change_record = {
			"product_code": product_code,
			"old_rates": {"annual_rate": old_rate},
			"new_rates": new_rates,
			"effective_date": effective_date,
			"recorded_at": _today().isoformat(),
		}
		self._product_rates_history.setdefault(product_code, []).append(change_record)
		self._audit(product.tenant_id, "product_rates_updated", product_code, change_record)

		return {
			"product_code": product_code,
			"new_annual_rate": product.annual_rate,
			"effective_date": effective_date,
			"rate_history_count": len(self._product_rates_history.get(product_code, [])),
		}

	def product_performance_report(self, product_code: str, period: str) -> dict[str, Any]:
		"""
		Performance report for a specific product: applications, disbursal rate, default rate,
		average ticket, yield, PAR 30/60/90.
		"""
		assert product_code, "product_code required"
		assert period, "period required"

		product = self._require_product(product_code)
		apps = [a for a in self.applications.values() if a.product_id == product_code]
		active_loans = [ln for ln in self.loans.values() if ln.product_id == product_code and ln.status == "active"]
		all_loans = [ln for ln in self.loans.values() if ln.product_id == product_code]

		total_apps = len(apps)
		disbursed = len([a for a in apps if a.status == "disbursed"])
		disbursal_rate = round(disbursed / total_apps, 4) if total_apps else 0.0

		total_book = round(sum(ln.outstanding_principal for ln in active_loans), 2)
		avg_ticket = round(sum(ln.principal for ln in all_loans) / len(all_loans), 2) if all_loans else 0.0
		written_off = sum(1 for ln in all_loans if ln.status == "written_off")
		default_rate = round(written_off / len(all_loans), 4) if all_loans else 0.0

		weighted_rate = sum(ln.outstanding_principal * ln.annual_rate for ln in active_loans)
		product_yield = round(weighted_rate / total_book, 4) if total_book else 0.0

		today = _today()
		par30 = par60 = par90 = 0.0
		for loan in active_loans:
			max_dpd = 0
			for inst in loan.installments:
				if inst["status"] == "paid":
					continue
				due = _parse_date(inst["due_date"])
				dpd = max(0, (today - due).days)
				max_dpd = max(max_dpd, dpd)
			if max_dpd > 30:
				par30 += loan.outstanding_principal
			if max_dpd > 60:
				par60 += loan.outstanding_principal
			if max_dpd > 90:
				par90 += loan.outstanding_principal

		return {
			"product_code": product_code,
			"product_name": product.name,
			"period": period,
			"total_applications": total_apps,
			"disbursed_count": disbursed,
			"disbursal_rate": disbursal_rate,
			"active_loan_count": len(active_loans),
			"total_book": total_book,
			"average_ticket": avg_ticket,
			"default_rate": default_rate,
			"written_off_count": written_off,
			"product_yield": product_yield,
			"par_30": round(par30 / total_book, 4) if total_book else 0.0,
			"par_60": round(par60 / total_book, 4) if total_book else 0.0,
			"par_90": round(par90 / total_book, 4) if total_book else 0.0,
		}

	def list_products(self, active_only: bool = True) -> list[dict[str, Any]]:
		"""List all loan products, optionally filtering to active only."""
		items = list(self.products.values())
		if active_only:
			items = [p for p in items if p.status == "active"]
		return [p.to_dict() for p in sorted(items, key=lambda p: p.id)]

	# ------------------------------------------------------------------
	# Listing helpers (original + extended)
	# ------------------------------------------------------------------

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

	def list_loans(self, status: str | None = None, borrower_id: str | None = None) -> list[dict[str, Any]]:
		loans = list(self.loans.values())
		if status is not None:
			loans = [ln for ln in loans if ln.status == status]
		if borrower_id is not None:
			loans = [ln for ln in loans if ln.borrower_id == borrower_id]
		return [ln.to_dict() for ln in sorted(loans, key=lambda ln: ln.loan_id)]

	# ------------------------------------------------------------------
	# Private helpers (original)
	# ------------------------------------------------------------------

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


DigitalLendingService = LendingService
