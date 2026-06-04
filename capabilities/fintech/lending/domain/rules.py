"""
Business rules for APG Digital Lending.

Every rule is a pure callable. RuleViolation for violations.
assert_* for validation guards. calculate_* for rule-gated derivations.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from __future__ import annotations

from datetime import date
from typing import Any


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class RuleViolation(Exception):
	"""Raised when a lending business rule is violated."""

	def __init__(self, rule_name: str, reason: str, required_action: str = "") -> None:
		self.rule_name = rule_name
		self.reason = reason
		self.required_action = required_action
		super().__init__(f"[{rule_name}] {reason}")


# ---------------------------------------------------------------------------
# Tenant / access rules
# ---------------------------------------------------------------------------

def assert_tenant_context(tenant_id: str | None) -> None:
	"""All operations require a non-empty tenant_id."""
	if not tenant_id:
		raise RuleViolation("tenant_context_required", "tenant_id is required", "attach_tenant_context")


def assert_no_cross_tenant(actor_tenant: str, resource_tenant: str) -> None:
	"""Actor must belong to the same tenant as the resource."""
	if actor_tenant != resource_tenant:
		raise RuleViolation(
			"cross_tenant_access_denied",
			f"actor tenant '{actor_tenant}' cannot access resource of tenant '{resource_tenant}'",
			"use_own_tenant_resources",
		)


def assert_actor_present(actor_id: str | None) -> None:
	"""All write operations require an actor_id for audit."""
	if not actor_id:
		raise RuleViolation("actor_required", "actor_id is required for write operations", "provide_actor_id")


# ---------------------------------------------------------------------------
# Loan product rules
# ---------------------------------------------------------------------------

SUPPORTED_CURRENCIES = {"KES", "USD", "EUR", "GBP", "NGN", "GHS", "UGX", "TZS", "ZAR", "XOF", "ETB", "RWF", "MWK", "ZMW"}
SUPPORTED_PRODUCT_TYPES = {
	"term_loan", "revolving", "overdraft", "microfinance", "mortgage",
	"asset_finance", "invoice_discounting", "bnpl", "salary_advance",
	"emergency", "agri", "sme", "group",
}
SUPPORTED_REPAYMENT_FREQUENCIES = {"daily", "weekly", "biweekly", "monthly", "quarterly", "bullet"}
MAX_ANNUAL_RATE = 0.72  # 72% — most African regulatory caps
MIN_ANNUAL_RATE = 0.005  # 0.5% floor


def assert_valid_currency(currency: str) -> None:
	if currency not in SUPPORTED_CURRENCIES:
		raise RuleViolation("unsupported_currency", f"currency '{currency}' not supported", "use_supported_currency")


def assert_valid_product_type(product_type: str) -> None:
	if product_type not in SUPPORTED_PRODUCT_TYPES:
		raise RuleViolation("unsupported_product_type", f"product_type '{product_type}' not supported")


def assert_valid_rate(annual_rate: float) -> None:
	if not (MIN_ANNUAL_RATE <= annual_rate <= MAX_ANNUAL_RATE):
		raise RuleViolation(
			"rate_out_of_bounds",
			f"annual_rate {annual_rate:.4f} must be between {MIN_ANNUAL_RATE} and {MAX_ANNUAL_RATE}",
			"adjust_rate",
		)


def assert_valid_amount_limits(min_amount: float, max_amount: float) -> None:
	if min_amount <= 0:
		raise RuleViolation("invalid_min_amount", "min_amount must be positive")
	if max_amount <= 0:
		raise RuleViolation("invalid_max_amount", "max_amount must be positive")
	if max_amount < min_amount:
		raise RuleViolation("invalid_amount_limits", "max_amount must be >= min_amount")
	if max_amount > 100_000_000:
		raise RuleViolation("amount_exceeds_single_obligor_limit", "max_amount exceeds 100M single-obligor limit")


def assert_valid_tenor(min_tenor_months: int, max_tenor_months: int) -> None:
	if min_tenor_months < 1:
		raise RuleViolation("invalid_min_tenor", "min_tenor_months must be >= 1")
	if max_tenor_months > 360:
		raise RuleViolation("tenor_exceeds_maximum", "max_tenor_months must not exceed 360 (30 years)")
	if max_tenor_months < min_tenor_months:
		raise RuleViolation("invalid_tenor_limits", "max_tenor_months must be >= min_tenor_months")


def assert_product_active(is_active: bool, product_id: str) -> None:
	if not is_active:
		raise RuleViolation("product_inactive", f"loan product '{product_id}' is not active", "activate_product")


# ---------------------------------------------------------------------------
# Application rules
# ---------------------------------------------------------------------------

SUPPORTED_PURPOSES = {
	"business", "education", "medical", "home_improvement", "vehicle",
	"personal", "agriculture", "debt_consolidation", "emergency",
	"salary_advance", "working_capital", "asset_purchase", "other",
}


def assert_kyc_present(kyc_ref: str) -> None:
	if not kyc_ref:
		raise RuleViolation("kyc_required", "KYC reference is required for loan applications", "complete_kyc")


def assert_valid_purpose(purpose: str) -> None:
	if purpose not in SUPPORTED_PURPOSES:
		raise RuleViolation("unsupported_purpose", f"purpose '{purpose}' not supported")


def assert_amount_within_product_limits(amount: float, min_amount: float, max_amount: float) -> None:
	if amount < min_amount:
		raise RuleViolation(
			"amount_below_minimum",
			f"requested amount {amount:.2f} is below product minimum {min_amount:.2f}",
			"increase_amount_or_change_product",
		)
	if amount > max_amount:
		raise RuleViolation(
			"amount_above_maximum",
			f"requested amount {amount:.2f} exceeds product maximum {max_amount:.2f}",
			"reduce_amount_or_change_product",
		)


def assert_tenor_within_product_limits(tenor_months: int, min_tenor: int, max_tenor: int) -> None:
	if tenor_months < min_tenor:
		raise RuleViolation("tenor_below_minimum", f"tenor {tenor_months}m < product minimum {min_tenor}m")
	if tenor_months > max_tenor:
		raise RuleViolation("tenor_above_maximum", f"tenor {tenor_months}m > product maximum {max_tenor}m")


def assert_no_duplicate_application(existing_statuses: list[str]) -> None:
	"""Borrower must not have an open application for the same product."""
	open_statuses = {"submitted", "under_review", "referred", "conditionally_approved", "approved"}
	if any(s in open_statuses for s in existing_statuses):
		raise RuleViolation(
			"duplicate_application",
			"borrower already has an open application for this product",
			"withdraw_existing_application_first",
		)


def assert_borrower_not_blacklisted(is_blacklisted: bool, borrower_id: str) -> None:
	if is_blacklisted:
		raise RuleViolation("borrower_blacklisted", f"borrower '{borrower_id}' is blacklisted", "review_blacklist_status")


# ---------------------------------------------------------------------------
# Credit assessment rules
# ---------------------------------------------------------------------------

MIN_CREDIT_SCORE_FOR_APPROVAL = 480  # Grade E lower bound


def assert_minimum_credit_score(score: int, product_min_score: int = MIN_CREDIT_SCORE_FOR_APPROVAL) -> None:
	if score < product_min_score:
		raise RuleViolation(
			"credit_score_below_minimum",
			f"composite score {score} is below minimum {product_min_score}",
			"decline_application",
		)


def assert_income_verified(verified: bool) -> None:
	if not verified:
		raise RuleViolation("income_not_verified", "income must be verified before underwriting", "complete_income_verification")


def assert_dsr_passes(dsr: float, threshold: float = 0.40) -> None:
	if dsr > threshold:
		raise RuleViolation(
			"dsr_exceeds_threshold",
			f"DSR {dsr:.2%} exceeds threshold {threshold:.2%}",
			"reduce_loan_amount_or_increase_income_evidence",
		)


def assert_no_active_defaults(defaults_count: int) -> None:
	if defaults_count > 0:
		raise RuleViolation(
			"active_defaults_present",
			f"borrower has {defaults_count} active default(s) on bureau",
			"clear_defaults_first",
		)


def assert_no_fraud_flags(fraud_flags: list[str]) -> None:
	if fraud_flags:
		raise RuleViolation(
			"fraud_flags_present",
			f"borrower has fraud flags: {', '.join(fraud_flags)}",
			"refer_to_fraud_team",
		)


# ---------------------------------------------------------------------------
# Underwriting rules
# ---------------------------------------------------------------------------

VALID_UNDERWRITING_DECISIONS = {"approve", "decline", "refer", "conditional_approve"}


def assert_valid_underwriting_decision(decision: str) -> None:
	if decision not in VALID_UNDERWRITING_DECISIONS:
		raise RuleViolation(
			"invalid_underwriting_decision",
			f"decision '{decision}' not valid; must be one of {VALID_UNDERWRITING_DECISIONS}",
		)


def assert_decline_has_reason(decision: str, adverse_reason: str) -> None:
	if decision == "decline" and not adverse_reason:
		raise RuleViolation(
			"decline_requires_adverse_reason",
			"decline decisions must include an adverse action reason",
			"provide_adverse_reason",
		)


def assert_human_approval_for_high_value(amount: float, human_approval: str, threshold: float = 500_000) -> None:
	if amount >= threshold and not human_approval:
		raise RuleViolation(
			"high_value_requires_human_approval",
			f"loans >= {threshold:,.0f} require human approval",
			"obtain_human_approval",
		)


def assert_application_status_for_underwriting(status: str) -> None:
	valid = {"submitted", "under_review", "referred"}
	if status not in valid:
		raise RuleViolation(
			"invalid_application_status_for_underwriting",
			f"application status '{status}' is not valid for underwriting; must be in {valid}",
		)


# ---------------------------------------------------------------------------
# Offer rules
# ---------------------------------------------------------------------------

def assert_application_approved_for_offer(status: str) -> None:
	valid = {"approved", "conditionally_approved"}
	if status not in valid:
		raise RuleViolation(
			"application_not_approved",
			f"application status '{status}' not eligible for offer issuance",
			"approve_application_first",
		)


def assert_offer_not_expired(expiry_date: date, as_of: date | None = None) -> None:
	check = as_of or date.today()
	if expiry_date < check:
		raise RuleViolation("offer_expired", f"offer expired on {expiry_date.isoformat()}", "issue_new_offer")


def assert_offer_accepted_for_disbursement(offer_status: str) -> None:
	if offer_status != "accepted":
		raise RuleViolation(
			"offer_not_accepted",
			f"offer status '{offer_status}' cannot be disbursed; must be 'accepted'",
			"borrower_must_accept_offer",
		)


# ---------------------------------------------------------------------------
# Disbursement rules
# ---------------------------------------------------------------------------

SUPPORTED_DISBURSEMENT_RAILS = {"bank_transfer", "mobile_money", "cash", "cheque", "internal"}


def assert_valid_disbursement_rail(rail: str) -> None:
	if rail not in SUPPORTED_DISBURSEMENT_RAILS:
		raise RuleViolation("unsupported_disbursement_rail", f"rail '{rail}' not supported")


def assert_disbursement_account_present(bank_account: str) -> None:
	if not bank_account:
		raise RuleViolation("disbursement_account_missing", "bank_account is required for disbursement")


def assert_no_duplicate_loan_for_application(existing_loan_ids: list[str], application_id: str) -> None:
	if existing_loan_ids:
		raise RuleViolation(
			"duplicate_disbursement",
			f"application '{application_id}' already has loan(s): {existing_loan_ids}",
			"cannot_disburse_twice",
		)


# ---------------------------------------------------------------------------
# Repayment rules
# ---------------------------------------------------------------------------

def assert_loan_active_for_repayment(status: str) -> None:
	if status != "active":
		raise RuleViolation(
			"loan_not_active",
			f"repayment cannot be applied to loan with status '{status}'",
		)


def assert_positive_payment_amount(amount: float) -> None:
	if amount <= 0:
		raise RuleViolation("non_positive_payment", "payment amount must be positive")


def assert_payment_reference_present(reference: str) -> None:
	if not reference:
		raise RuleViolation("payment_reference_missing", "payment reference is required for audit trail")


# ---------------------------------------------------------------------------
# Delinquency & collection rules
# ---------------------------------------------------------------------------

COLLECTION_ESCALATION_THRESHOLDS = {
	"stage1_monitoring": 1,
	"stage2_collections": 31,
	"stage3_legal": 91,
	"stage4_writeoff_eligible": 181,
}


def assert_dpd_for_demand_notice(dpd: int, level: int) -> None:
	required_dpd = {1: 1, 2: 15, 3: 30, 4: 60}
	min_dpd = required_dpd.get(level, 0)
	if dpd < min_dpd:
		raise RuleViolation(
			"dpd_insufficient_for_notice_level",
			f"DPD {dpd} is insufficient for demand notice level {level} (requires >={min_dpd} DPD)",
		)


def calculate_required_provision_rate(dpd: int, has_collateral: bool) -> float:
	"""
	Determine minimum provision rate for a loan based on DPD and collateral.
	Based on Central Bank of Kenya prudential guidelines (generalised).
	"""
	if dpd == 0:
		return 0.01  # 1% general provision
	if dpd <= 30:
		return 0.03
	if dpd <= 60:
		return 0.20 if not has_collateral else 0.10
	if dpd <= 90:
		return 0.50 if not has_collateral else 0.25
	if dpd <= 180:
		return 0.75 if not has_collateral else 0.50
	return 1.00  # 100% fully provided


# ---------------------------------------------------------------------------
# Restructure rules
# ---------------------------------------------------------------------------

MAX_RESTRUCTURES = 3


def assert_loan_eligible_for_restructure(status: str, restructure_count: int) -> None:
	if status != "active":
		raise RuleViolation("loan_not_active_for_restructure", f"can only restructure active loans, status: '{status}'")
	if restructure_count >= MAX_RESTRUCTURES:
		raise RuleViolation(
			"max_restructures_reached",
			f"loan has reached maximum {MAX_RESTRUCTURES} restructures",
			"escalate_to_credit_committee",
		)


def assert_restructure_approved(approved_by: str) -> None:
	if not approved_by:
		raise RuleViolation("restructure_requires_approval", "restructure must be approved by authorised officer")


def assert_new_tenor_not_shorter(old_tenor: int, new_tenor: int) -> None:
	if new_tenor < old_tenor:
		raise RuleViolation(
			"tenor_cannot_be_shortened_in_restructure",
			f"new tenor {new_tenor}m must be >= existing tenor {old_tenor}m",
		)


# ---------------------------------------------------------------------------
# Write-off rules
# ---------------------------------------------------------------------------

MIN_DPD_FOR_WRITEOFF = 90


def assert_eligible_for_writeoff(dpd: int, approved_by: str) -> None:
	if dpd < MIN_DPD_FOR_WRITEOFF:
		raise RuleViolation(
			"insufficient_dpd_for_writeoff",
			f"loan must be at least {MIN_DPD_FOR_WRITEOFF} DPD to write off; current DPD: {dpd}",
			"escalate_collections_first",
		)
	if not approved_by:
		raise RuleViolation("writeoff_requires_approval", "write-off must be authorised by a credit officer")


# ---------------------------------------------------------------------------
# Collateral rules
# ---------------------------------------------------------------------------

def assert_collateral_coverage(coverage_ratio: float, min_coverage: float = 1.0) -> None:
	if coverage_ratio < min_coverage:
		raise RuleViolation(
			"insufficient_collateral_coverage",
			f"collateral coverage {coverage_ratio:.2f}x is below required {min_coverage:.2f}x",
			"provide_additional_collateral",
		)


def assert_collateral_held_for_release(status: str) -> None:
	if status != "held":
		raise RuleViolation(
			"collateral_not_held",
			f"collateral cannot be released from status '{status}'",
		)


# ---------------------------------------------------------------------------
# Concentraction / single obligor limits
# ---------------------------------------------------------------------------

SINGLE_OBLIGOR_LIMIT_PCT = 0.25  # 25% of total portfolio


def assert_single_obligor_limit(
	borrower_total_exposure: float,
	total_portfolio: float,
	new_loan_amount: float,
) -> None:
	if total_portfolio <= 0:
		return
	new_total = borrower_total_exposure + new_loan_amount
	if new_total / total_portfolio > SINGLE_OBLIGOR_LIMIT_PCT:
		raise RuleViolation(
			"single_obligor_limit_exceeded",
			f"new total exposure {new_total:,.2f} would exceed {SINGLE_OBLIGOR_LIMIT_PCT:.0%} single-obligor limit",
			"reduce_loan_amount",
		)


# ---------------------------------------------------------------------------
# Offer tier calculations (rule-governed)
# ---------------------------------------------------------------------------

def calculate_offer_tiers(
	max_eligible_amount: float,
	base_annual_rate: float,
	max_tenor_months: int,
	risk_grade: str,
) -> list[dict[str, Any]]:
	"""
	Generate conservative / standard / aggressive offer tiers.
	Aggressive only for grade A/B.
	"""
	tiers = [
		("conservative", 0.60, 0.02, max(1, max_tenor_months // 2)),
		("standard",     0.80, 0.00, max_tenor_months),
	]
	if risk_grade.upper() in ("A", "B"):
		tiers.append(("aggressive", 1.00, -0.01, max_tenor_months))

	offers = []
	for tier_name, amount_factor, rate_delta, tenor in tiers:
		amount = round(max_eligible_amount * amount_factor, 2)
		rate = round(max(0.01, base_annual_rate + rate_delta), 6)
		offers.append({
			"tier": tier_name,
			"amount_factor": amount_factor,
			"offered_amount": amount,
			"annual_rate": rate,
			"tenor_months": tenor,
			"rate_delta": rate_delta,
		})
	return offers
