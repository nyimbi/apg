"""Deterministic domain rules for Lease Management (IFRS 16 / ASC 842).

All rules are pure callables — no I/O, no side-effects.
Each assert_* function raises RuleViolation on breach.
Each calculate_* function returns a typed result.

Rule coverage
─────────────
• Tenant isolation
• Lease lifecycle gates (create / activate / execute / terminate)
• IFRS 16 accounting gates (classification / remeasurement triggers)
• Rent management (escalation, review, arrears)
• Options (exercise window, notice, reasonably-certain assessment)
• Modifications (scope change, partial surrender, rate change)
• Subleases (head-lease must exist, term cannot exceed head)
• Exemptions (short-term < 12 months, low-value < USD 5 000)
• Approval thresholds (property manager / asset manager / IC / board)
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any


# ── Exception ─────────────────────────────────────────────────────────────────

class RuleViolation(Exception):
	"""Raised when a business rule is violated."""

	def __init__(self, rule_name: str, reason: str, required_action: str = "") -> None:
		self.rule_name = rule_name
		self.reason = reason
		self.required_action = required_action
		super().__init__(f"[{rule_name}] {reason}")


# ── Tenant / Auth ─────────────────────────────────────────────────────────────

def assert_tenant_context(tenant_id: str | None) -> None:
	"""All operations require a non-empty tenant_id."""
	if not tenant_id or not str(tenant_id).strip():
		raise RuleViolation(
			"tenant_context_required",
			"tenant_id is required for all operations",
			"attach_tenant_context",
		)


def assert_no_cross_tenant(actor_tenant: str, resource_tenant: str) -> None:
	"""Cross-tenant access is always denied."""
	if actor_tenant != resource_tenant:
		raise RuleViolation(
			"cross_tenant_access_denied",
			f"actor tenant '{actor_tenant}' cannot access resource owned by '{resource_tenant}'",
			"use_own_tenant_resources",
		)


def assert_write_policy(operation_type: str, policy_attached: bool) -> None:
	"""Write operations require an attached lease policy."""
	if operation_type == "write" and not policy_attached:
		raise RuleViolation(
			"write_requires_policy",
			"write operations require an attached lease policy",
			"attach_lease_policy",
		)


# ── Lease Lifecycle ───────────────────────────────────────────────────────────

def assert_dates_valid(commencement: date, expiry: date) -> None:
	"""Expiry must be strictly after commencement."""
	if expiry <= commencement:
		raise RuleViolation(
			"expiry_must_follow_commencement",
			f"expiry_date {expiry} must be after commencement_date {commencement}",
			"correct_lease_dates",
		)


def assert_lease_term_positive(term_months: int) -> None:
	"""Lease term must be at least 1 month."""
	if term_months < 1:
		raise RuleViolation(
			"lease_term_too_short",
			f"lease term {term_months} months is less than 1 month",
			"extend_lease_term",
		)


def assert_rent_non_negative(rent: Decimal) -> None:
	"""Rent must be non-negative (zero rent allowed for peppercorn leases)."""
	if rent < 0:
		raise RuleViolation(
			"rent_negative",
			f"rent {rent} must be non-negative",
			"set_valid_rent",
		)


def assert_lease_activatable(
	status: str,
	abstraction_verified: bool,
	commencement_present: bool,
	expiry_present: bool,
) -> None:
	"""A lease may only be activated if all pre-conditions are met."""
	if status not in ("heads_of_terms", "negotiating", "signed", "draft"):
		raise RuleViolation(
			"invalid_status_for_activation",
			f"lease in status '{status}' cannot be activated",
			"check_lease_status",
		)
	if not commencement_present:
		raise RuleViolation(
			"commencement_date_required",
			"commencement_date must be set before activation",
			"set_commencement_date",
		)
	if not expiry_present:
		raise RuleViolation(
			"expiry_date_required",
			"expiry_date must be set before activation",
			"set_expiry_date",
		)
	if not abstraction_verified:
		raise RuleViolation(
			"abstraction_not_verified",
			"lease abstraction must be verified before activation",
			"verify_lease_abstraction",
		)


def assert_lease_surrenderable(status: str) -> None:
	"""Only active or holding-over leases may be surrendered."""
	if status not in ("active", "holding_over"):
		raise RuleViolation(
			"lease_not_surrenderable",
			f"lease in status '{status}' cannot be surrendered",
			"check_lease_status",
		)


def assert_lease_terminatable(status: str) -> None:
	"""Only active, holding-over, or notice-served leases may be terminated."""
	if status not in ("active", "holding_over", "notice_served"):
		raise RuleViolation(
			"lease_not_terminatable",
			f"lease in status '{status}' cannot be terminated",
			"check_lease_status",
		)


def assert_forfeiture_legal_process(legal_process_complete: bool) -> None:
	"""Forfeiture requires completion of legal process."""
	if not legal_process_complete:
		raise RuleViolation(
			"forfeiture_requires_legal_process",
			"legal forfeiture process must be completed before recording forfeiture",
			"complete_legal_forfeiture_process",
		)


# ── IFRS 16 Accounting ────────────────────────────────────────────────────────

def assert_discount_rate_present(rate: Decimal | None) -> None:
	"""IFRS 16 calculations require a discount rate."""
	if rate is None or rate <= 0:
		raise RuleViolation(
			"discount_rate_required",
			"a positive discount rate (IBR or implicit rate) is required for IFRS 16 calculations",
			"set_discount_rate",
		)


def assert_ifrs16_reclassification_auditor_approved(auditor_approved: bool) -> None:
	"""IFRS 16 category reclassifications require auditor sign-off."""
	if not auditor_approved:
		raise RuleViolation(
			"ifrs16_reclassification_requires_auditor",
			"IFRS 16 category reclassification requires auditor approval",
			"obtain_auditor_approval",
		)


def assert_rou_asset_calculated(rou_asset: Decimal | None) -> None:
	"""ROU asset must be calculated before amortisation can proceed."""
	if not rou_asset or rou_asset <= 0:
		raise RuleViolation(
			"rou_asset_not_calculated",
			"ROU asset has not been calculated; run calculate_rou_asset first",
			"calculate_rou_asset",
		)


def assert_lease_liability_calculated(liability: Decimal | None) -> None:
	"""Lease liability must be calculated before payment processing."""
	if not liability or liability <= 0:
		raise RuleViolation(
			"lease_liability_not_calculated",
			"lease liability has not been calculated; run calculate_lease_liability first",
			"calculate_lease_liability",
		)


def assert_not_short_term_exempt(lease_term_months: int) -> None:
	"""Reject IFRS 16 recognition if the short-term exemption applies."""
	if lease_term_months <= 12:
		raise RuleViolation(
			"short_term_exemption_applies",
			f"lease term {lease_term_months} months qualifies for short-term exemption (≤12 months); "
			"IFRS 16 recognition is not required",
			"apply_short_term_exemption_or_opt_out",
		)


def assert_not_low_value_exempt(fair_value_new_usd: Decimal, threshold_usd: Decimal = Decimal("5000")) -> None:
	"""Reject IFRS 16 recognition if the low-value exemption applies."""
	if fair_value_new_usd > 0 and fair_value_new_usd <= threshold_usd:
		raise RuleViolation(
			"low_value_exemption_applies",
			f"underlying asset fair value USD {fair_value_new_usd} is at or below USD {threshold_usd}; "
			"low-value exemption applies",
			"apply_low_value_exemption_or_opt_out",
		)


# ── Rent Management ───────────────────────────────────────────────────────────

def assert_escalation_type_supported(escalation_type: str, supported: list[str]) -> None:
	if escalation_type not in supported:
		raise RuleViolation(
			"escalation_type_not_supported",
			f"escalation_type '{escalation_type}' is not supported; choose from {supported}",
			"select_supported_escalation_type",
		)


def assert_cpi_escalation_has_base_index(escalation_type: str, cpi_base_index: Decimal | None) -> None:
	if escalation_type == "cpi_linked" and (cpi_base_index is None or cpi_base_index <= 0):
		raise RuleViolation(
			"cpi_base_index_required",
			"cpi_base_index must be provided and positive for CPI-linked escalation",
			"provide_cpi_base_index",
		)


def assert_rent_review_not_backdated_without_authorisation(
	review_date: date,
	as_of: date,
	backdating_authorised_by: str | None,
) -> None:
	"""Backdated rent reviews require explicit authorisation."""
	if review_date < as_of and not backdating_authorised_by:
		raise RuleViolation(
			"rent_review_backdating_requires_authorisation",
			f"rent review date {review_date} is in the past; backdating authorisation required",
			"obtain_backdating_authorisation",
		)


def assert_no_arrears_before_renewal(total_arrears: Decimal) -> None:
	"""Leases with outstanding arrears cannot be renewed without clearing arrears."""
	if total_arrears > 0:
		raise RuleViolation(
			"arrears_must_be_cleared_before_renewal",
			f"lease has {total_arrears} in outstanding arrears; clear arrears before renewal",
			"clear_rent_arrears",
		)


# ── Options ───────────────────────────────────────────────────────────────────

def assert_option_type_supported(option_type: str, supported: list[str]) -> None:
	if option_type not in supported:
		raise RuleViolation(
			"option_type_not_supported",
			f"option_type '{option_type}' is not in supported list {supported}",
			"select_supported_option_type",
		)


def assert_option_notice_served(notice_served: bool) -> None:
	"""Option exercise requires notice to be served first."""
	if not notice_served:
		raise RuleViolation(
			"option_notice_required",
			"notice must be served before exercising an option",
			"serve_option_notice",
		)


def assert_within_option_exercise_window(
	exercise_from: date,
	exercise_to: date,
	today: date,
) -> None:
	if not (exercise_from <= today <= exercise_to):
		raise RuleViolation(
			"outside_option_exercise_window",
			f"today {today} is outside the option exercise window {exercise_from} – {exercise_to}",
			"check_option_exercise_dates",
		)


def assert_option_not_lapsed(option_status: str) -> None:
	if option_status in ("lapsed", "exercised", "waived"):
		raise RuleViolation(
			"option_no_longer_open",
			f"option has status '{option_status}' and cannot be exercised",
			"check_option_status",
		)


# ── Modifications ─────────────────────────────────────────────────────────────

def assert_partial_surrender_proportion(proportion: Decimal) -> None:
	"""Surrendered proportion must be strictly between 0 and 1."""
	if not (Decimal("0") < proportion < Decimal("1")):
		raise RuleViolation(
			"invalid_surrender_proportion",
			f"surrendered_proportion {proportion} must be strictly between 0 and 1",
			"set_valid_surrender_proportion",
		)


def assert_modification_approved(status: str) -> None:
	"""A modification must be approved before it can be applied."""
	if status != "approved":
		raise RuleViolation(
			"modification_not_approved",
			f"modification has status '{status}'; must be 'approved' before applying",
			"approve_modification",
		)


def assert_modification_not_already_applied(applied: bool) -> None:
	if applied:
		raise RuleViolation(
			"modification_already_applied",
			"this modification has already been applied and cannot be re-applied",
			"check_modification_status",
		)


# ── Sublease ──────────────────────────────────────────────────────────────────

def assert_sublease_within_head_lease(
	sublease_end: date,
	head_lease_end: date,
) -> None:
	"""Sublease term cannot extend beyond the head lease expiry."""
	if sublease_end > head_lease_end:
		raise RuleViolation(
			"sublease_exceeds_head_lease",
			f"sublease end {sublease_end} is after head lease expiry {head_lease_end}",
			"shorten_sublease_term",
		)


def assert_sublease_rent_not_exceeds_head(
	sublease_rent: Decimal,
	head_lease_rent: Decimal,
) -> None:
	"""Sublease rent may not exceed head lease rent (profit-making subleases may require landlord consent)."""
	if sublease_rent > head_lease_rent:
		raise RuleViolation(
			"sublease_rent_exceeds_head_rent",
			f"sublease rent {sublease_rent} exceeds head lease rent {head_lease_rent}; "
			"landlord consent is required for profit subleasing",
			"obtain_landlord_consent_for_profit_sublease",
		)


# ── Assignments ───────────────────────────────────────────────────────────────

def assert_landlord_consent_obtained(consent_ref: str | None) -> None:
	if not consent_ref:
		raise RuleViolation(
			"landlord_consent_required",
			"landlord consent reference must be provided for lease assignments",
			"obtain_landlord_consent",
		)


def assert_assignment_type_supported(assignment_type: str, supported: list[str]) -> None:
	if assignment_type not in supported:
		raise RuleViolation(
			"assignment_type_not_supported",
			f"assignment_type '{assignment_type}' not in {supported}",
			"select_supported_assignment_type",
		)


# ── Approval Thresholds ───────────────────────────────────────────────────────

# Annual rent thresholds (KES) for escalating approval requirements
_APPROVAL_THRESHOLDS: dict[str, Decimal] = {
	"property_manager":      Decimal("5_000_000"),   # up to KES 5M annual rent
	"asset_manager":         Decimal("20_000_000"),  # up to KES 20M
	"investment_committee":  Decimal("100_000_000"), # up to KES 100M
	"board":                 Decimal("9_999_999_999"),  # above KES 100M
}

_APPROVAL_HIERARCHY = ["property_manager", "asset_manager", "investment_committee", "board"]


def required_approval_level(annual_rent: Decimal) -> str:
	"""Return the minimum approval level required for a lease with the given annual rent."""
	for level in _APPROVAL_HIERARCHY:
		if annual_rent <= _APPROVAL_THRESHOLDS[level]:
			return level
	return "board"


def assert_approval_sufficient(
	annual_rent: Decimal,
	approver_level: str,
) -> None:
	"""Assert that the approver's level is sufficient for the lease value."""
	required = required_approval_level(annual_rent)
	required_idx = _APPROVAL_HIERARCHY.index(required)
	approver_idx = _APPROVAL_HIERARCHY.index(approver_level) if approver_level in _APPROVAL_HIERARCHY else -1
	if approver_idx < required_idx:
		raise RuleViolation(
			"approval_level_insufficient",
			f"annual rent {annual_rent} requires at least '{required}' approval; "
			f"'{approver_level}' is insufficient",
			f"escalate_to_{required}",
		)


# ── Payment / Finance ─────────────────────────────────────────────────────────

def assert_payment_positive(amount: Decimal) -> None:
	if amount <= 0:
		raise RuleViolation(
			"payment_must_be_positive",
			f"payment amount {amount} must be positive",
			"set_valid_payment_amount",
		)


def assert_security_deposit_non_negative(deposit: Decimal) -> None:
	if deposit < 0:
		raise RuleViolation(
			"security_deposit_negative",
			f"security deposit {deposit} must be non-negative",
			"set_valid_security_deposit",
		)


# ── Convenience composite assertions ─────────────────────────────────────────

def assert_lease_create_valid(
	tenant_id: str,
	commencement: date,
	expiry: date,
	rent: Decimal,
	security_deposit: Decimal,
	lease_term_months: int,
) -> None:
	"""Run all create-time validations in one call."""
	assert_tenant_context(tenant_id)
	assert_dates_valid(commencement, expiry)
	assert_lease_term_positive(lease_term_months)
	assert_rent_non_negative(rent)
	assert_security_deposit_non_negative(security_deposit)


def assert_ifrs16_recognition_valid(
	lease_term_months: int,
	fair_value_new_usd: Decimal | None,
	discount_rate: Decimal | None,
	short_term_opt_out: bool = False,
	low_value_opt_out: bool = False,
) -> None:
	"""Run all IFRS 16 recognition gate-checks."""
	if not short_term_opt_out:
		assert_not_short_term_exempt(lease_term_months)
	if fair_value_new_usd is not None and not low_value_opt_out:
		assert_not_low_value_exempt(fair_value_new_usd)
	assert_discount_rate_present(discount_rate)
