"""Deterministic domain rules for Tax Administration.

Every business rule is a callable function. Violations raise RuleViolation.
assert_* functions are entry-point guards; calculate_* helpers live in calculations.py.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class RuleViolation(Exception):
	"""Raised when a business rule is violated."""

	def __init__(self, rule_name: str, reason: str, required_action: str = "") -> None:
		self.rule_name = rule_name
		self.reason = reason
		self.required_action = required_action
		super().__init__(f"Rule '{rule_name}' violated: {reason}")


# ---------------------------------------------------------------------------
# Tenant isolation
# ---------------------------------------------------------------------------

def assert_tenant_context(context: dict[str, Any]) -> None:
	"""All operations require a tenant context."""
	if not context.get("tenant_id"):
		raise RuleViolation(
			"tenant_context_required",
			"tenant_id is required",
			"attach_tenant_context",
		)


def assert_no_cross_tenant_access(actor_tenant: str, resource_tenant: str) -> None:
	"""Cross-tenant access is always denied."""
	if actor_tenant != resource_tenant:
		raise RuleViolation(
			"cross_tenant_access_denied",
			"cross-tenant access is not permitted",
			"use_own_tenant_resources",
		)


def assert_write_policy(context: dict[str, Any]) -> None:
	"""Write operations require an attached policy."""
	if context.get("operation_type") == "write" and not context.get("policy_attached"):
		raise RuleViolation(
			"write_requires_policy",
			"write operations require an attached policy",
			"attach_policy",
		)


# ---------------------------------------------------------------------------
# Taxpayer registration
# ---------------------------------------------------------------------------

_SUPPORTED_TAX_TYPES = {
	"income_tax", "vat", "corporate_tax", "withholding_tax",
	"capital_gains_tax", "excise_duty", "customs_duty", "stamp_duty",
	"rental_income_tax", "turnover_tax", "digital_services_tax", "presumptive_tax",
}

_SUPPORTED_TAXPAYER_TYPES = {
	"individual", "company", "partnership", "trust",
	"government_entity", "ngo", "foreign_entity",
}


def assert_tax_type_supported(tax_type: str) -> None:
	"""Tax type must be in the supported list."""
	if tax_type.lower() not in _SUPPORTED_TAX_TYPES:
		raise RuleViolation(
			"tax_type_not_supported",
			f"tax_type '{tax_type}' is not supported",
			f"use one of: {sorted(_SUPPORTED_TAX_TYPES)}",
		)


def assert_taxpayer_type_supported(taxpayer_type: str) -> None:
	"""Taxpayer type must be valid."""
	if taxpayer_type.lower() not in _SUPPORTED_TAXPAYER_TYPES:
		raise RuleViolation(
			"taxpayer_type_not_supported",
			f"taxpayer_type '{taxpayer_type}' is not supported",
			f"use one of: {sorted(_SUPPORTED_TAXPAYER_TYPES)}",
		)


def assert_legal_name_present(legal_name: str | None) -> None:
	if not legal_name or not legal_name.strip():
		raise RuleViolation(
			"legal_name_required",
			"taxpayer legal_name is required",
			"provide_legal_name",
		)


def assert_id_number_present(id_number: str | None) -> None:
	if not id_number or not id_number.strip():
		raise RuleViolation(
			"id_number_required",
			"national_id or business_registration_number is required",
			"provide_id_number",
		)


def assert_pin_unique(existing_pins: set[str], new_pin: str) -> None:
	"""TIN must be globally unique within the tenant."""
	if new_pin.upper() in {p.upper() for p in existing_pins}:
		raise RuleViolation(
			"duplicate_pin_denied",
			f"PIN '{new_pin}' is already registered",
			"use_unique_pin",
		)


def assert_taxpayer_active(status: str) -> None:
	"""Operations on inactive taxpayers must be blocked."""
	if status in ("deregistered", "blocked"):
		raise RuleViolation(
			"taxpayer_inactive",
			f"taxpayer status '{status}' does not permit this operation",
			"reactivate_taxpayer",
		)


# ---------------------------------------------------------------------------
# Return filing
# ---------------------------------------------------------------------------

def assert_taxpayer_pin_present(pin: str | None) -> None:
	if not pin or not pin.strip():
		raise RuleViolation(
			"taxpayer_pin_required",
			"taxpayer PIN is required to file a return",
			"provide_taxpayer_pin",
		)


def assert_period_present(period: str | None) -> None:
	if not period or not period.strip():
		raise RuleViolation(
			"period_required",
			"tax period is required",
			"provide_period",
		)


def assert_return_type_supported(return_type: str) -> None:
	_supported = {
		"monthly_vat", "annual_income", "quarterly_advance",
		"withholding_tax_return", "corporate_annual", "customs_entry",
		"turnover_tax_monthly", "capital_gains",
	}
	if return_type.lower() not in _supported:
		raise RuleViolation(
			"return_type_not_supported",
			f"return_type '{return_type}' is not supported",
			f"use one of: {sorted(_supported)}",
		)


def assert_no_duplicate_return(
	existing_returns: list[Any],
	tax_pin: str,
	return_type: str,
	period_start: date,
	period_end: date,
) -> None:
	"""Prevent duplicate returns for same PIN/type/period (not amendments)."""
	for r in existing_returns:
		if (
			r.tax_pin.upper() == tax_pin.upper()
			and r.return_type.value == return_type
			and r.tax_period_start == period_start
			and r.tax_period_end == period_end
			and not r.is_amended
			and r.status.value not in ("rejected",)
		):
			raise RuleViolation(
				"duplicate_return_denied",
				f"a return for PIN={tax_pin} type={return_type} period={period_start}/{period_end} already exists",
				"amend_existing_return",
			)


def assert_return_amounts_consistent(
	gross_income: Decimal,
	allowable_deductions: Decimal,
	taxable_income: Decimal,
	tolerance: Decimal = Decimal("1.00"),
) -> None:
	"""taxable_income must equal gross_income - allowable_deductions (within tolerance)."""
	computed = gross_income - allowable_deductions
	if abs(computed - taxable_income) > tolerance:
		raise RuleViolation(
			"return_amounts_inconsistent",
			f"taxable_income {taxable_income} != gross {gross_income} - deductions {allowable_deductions} = {computed}",
			"correct_return_amounts",
		)


def assert_non_negative_amounts(*amounts: Decimal, field: str = "amount") -> None:
	for a in amounts:
		if a < Decimal("0"):
			raise RuleViolation(
				"negative_amount_denied",
				f"{field} cannot be negative: {a}",
				"provide_non_negative_amount",
			)


# ---------------------------------------------------------------------------
# Assessment
# ---------------------------------------------------------------------------

_SUPPORTED_ASSESSMENT_TYPES = {
	"self_assessment", "amended_assessment", "best_judgement",
	"audit_assessment", "estimated_assessment", "agency_assessment",
}


def assert_assessment_type_supported(assessment_type: str) -> None:
	if assessment_type.lower() not in _SUPPORTED_ASSESSMENT_TYPES:
		raise RuleViolation(
			"assessment_type_not_supported",
			f"assessment_type '{assessment_type}' is not supported",
			f"use one of: {sorted(_SUPPORTED_ASSESSMENT_TYPES)}",
		)


def assert_return_exists(return_record: Any | None, return_id: str) -> None:
	if return_record is None:
		raise RuleViolation(
			"return_not_found",
			f"return '{return_id}' does not exist",
			"provide_valid_return_id",
		)


def assert_assessed_amount_positive(amount: Decimal) -> None:
	if amount <= Decimal("0"):
		raise RuleViolation(
			"assessed_amount_must_be_positive",
			f"assessed_amount must be > 0, got {amount}",
			"provide_positive_assessed_amount",
		)


def assert_assessor_present(assessor_id: str | None) -> None:
	if not assessor_id or not assessor_id.strip():
		raise RuleViolation(
			"assessor_required",
			"assessor_id is required for an assessment",
			"provide_assessor_id",
		)


# ---------------------------------------------------------------------------
# Objection
# ---------------------------------------------------------------------------

_OBJECTION_DEADLINE_DAYS = 30


def assert_objection_within_deadline(
	assessment_date: date,
	objection_date: date,
	deadline_days: int = _OBJECTION_DEADLINE_DAYS,
) -> None:
	"""Objection must be filed within 30 days of assessment."""
	days_elapsed = (objection_date - assessment_date).days
	if days_elapsed > deadline_days:
		raise RuleViolation(
			"objection_deadline_passed",
			f"objection filed {days_elapsed} days after assessment (limit {deadline_days})",
			"file_objection_within_deadline",
		)


def assert_objection_within_deadline_flag(within_deadline: bool) -> None:
	"""Legacy flag-based deadline check."""
	if not within_deadline:
		raise RuleViolation(
			"objection_deadline_passed",
			"objection deadline has passed",
			"file_objection_within_deadline",
		)


def assert_grounds_present(grounds: str | None) -> None:
	if not grounds or not grounds.strip():
		raise RuleViolation(
			"grounds_required",
			"grounds for objection are required",
			"provide_grounds",
		)


def assert_amount_disputed_positive(amount: Decimal) -> None:
	if amount <= Decimal("0"):
		raise RuleViolation(
			"amount_disputed_must_be_positive",
			f"amount_disputed must be > 0, got {amount}",
			"provide_positive_amount_disputed",
		)


def assert_objection_appealable(objection_status: str) -> None:
	"""Appeals only valid after dismissed or partially_upheld objections."""
	if objection_status not in ("dismissed", "partially_upheld"):
		raise RuleViolation(
			"objection_not_appealable",
			f"objection status '{objection_status}' cannot be appealed (must be dismissed or partially_upheld)",
			"obtain_dismissal_or_partial_uphold_first",
		)


# ---------------------------------------------------------------------------
# Debt collection
# ---------------------------------------------------------------------------

_SUPPORTED_COLLECTION_METHODS = {
	"payment_plan", "garnishment", "asset_seizure", "third_party_demand",
	"legal_proceedings", "write_off", "salary_attachment", "bank_levy",
	"distress", "court_order", "employer_attachment", "bank_attachment",
}


def assert_demand_notice_issued(demand_notice_reference: str | None) -> None:
	"""A demand notice must be issued before collection action."""
	if not demand_notice_reference or not demand_notice_reference.strip():
		raise RuleViolation(
			"demand_notice_required",
			"a demand notice must be issued before initiating collection",
			"issue_demand_notice_first",
		)


def assert_collection_method_supported(collection_method: str) -> None:
	if collection_method.lower() not in _SUPPORTED_COLLECTION_METHODS:
		raise RuleViolation(
			"collection_method_not_supported",
			f"collection_method '{collection_method}' is not supported",
			f"use one of: {sorted(_SUPPORTED_COLLECTION_METHODS)}",
		)


def assert_debt_outstanding(debt_status: str) -> None:
	"""Collection actions only valid on outstanding/partially_paid debts."""
	if debt_status not in ("outstanding", "partially_paid", "under_arrangement"):
		raise RuleViolation(
			"debt_not_actionable",
			f"debt with status '{debt_status}' cannot be collected",
			"check_debt_status",
		)


def assert_payment_amount_positive(amount: Decimal) -> None:
	if amount <= Decimal("0"):
		raise RuleViolation(
			"payment_amount_must_be_positive",
			f"payment amount must be > 0, got {amount}",
			"provide_positive_payment_amount",
		)


def assert_payment_reference_present(reference: str | None) -> None:
	if not reference or not reference.strip():
		raise RuleViolation(
			"payment_reference_required",
			"payment reference is required",
			"provide_payment_reference",
		)


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

_SUPPORTED_AUDIT_TYPES = {
	"desk_audit", "field_audit", "it_audit", "transfer_pricing",
	"vat_refund_audit", "forensic_audit", "compliance_audit", "sector_audit",
}


def assert_audit_type_supported(audit_type: str) -> None:
	if audit_type.lower() not in _SUPPORTED_AUDIT_TYPES:
		raise RuleViolation(
			"audit_type_not_supported",
			f"audit_type '{audit_type}' is not supported",
			f"use one of: {sorted(_SUPPORTED_AUDIT_TYPES)}",
		)


def assert_auditor_present(auditor_id: str | None) -> None:
	if not auditor_id or not auditor_id.strip():
		raise RuleViolation(
			"auditor_required",
			"auditor_id is required to open an audit case",
			"assign_auditor",
		)


def assert_audit_period_valid(period_start: date, period_end: date) -> None:
	if period_end < period_start:
		raise RuleViolation(
			"audit_period_invalid",
			f"period_end {period_end} must be >= period_start {period_start}",
			"correct_audit_period",
		)


def assert_audit_open(audit_status: str) -> None:
	"""Findings can only be recorded on open audits."""
	if audit_status not in ("planned", "in_progress"):
		raise RuleViolation(
			"audit_not_open",
			f"audit with status '{audit_status}' cannot accept new findings",
			"reopen_or_create_new_audit",
		)


# ---------------------------------------------------------------------------
# Refund
# ---------------------------------------------------------------------------

def assert_refund_amount_positive(amount: Decimal) -> None:
	if amount <= Decimal("0"):
		raise RuleViolation(
			"refund_amount_must_be_positive",
			f"refund amount must be > 0, got {amount}",
			"provide_positive_refund_amount",
		)


def assert_no_outstanding_debt_for_clearance(outstanding_debts: list[Any]) -> None:
	"""Tax Clearance Certificate requires zero outstanding debt."""
	if outstanding_debts:
		total = sum(getattr(d, "balance", Decimal("0")) for d in outstanding_debts)
		raise RuleViolation(
			"outstanding_debt_blocks_clearance",
			f"{len(outstanding_debts)} outstanding debt(s) totalling {total} prevent clearance",
			"settle_outstanding_debts_first",
		)


# ---------------------------------------------------------------------------
# Exchange of Information
# ---------------------------------------------------------------------------

_SUPPORTED_EOI_URGENCY = {"routine", "urgent", "spontaneous"}


def assert_eoi_urgency_valid(urgency: str) -> None:
	if urgency.lower() not in _SUPPORTED_EOI_URGENCY:
		raise RuleViolation(
			"eoi_urgency_invalid",
			f"urgency '{urgency}' is not valid",
			f"use one of: {sorted(_SUPPORTED_EOI_URGENCY)}",
		)


def assert_treaty_partner_present(treaty_partner: str | None) -> None:
	if not treaty_partner or not treaty_partner.strip():
		raise RuleViolation(
			"treaty_partner_required",
			"treaty_partner (ISO country code) is required",
			"provide_treaty_partner",
		)


# ---------------------------------------------------------------------------
# Penalty & interest
# ---------------------------------------------------------------------------

def assert_penalty_rate_valid(rate: Decimal) -> None:
	if rate < Decimal("0") or rate > Decimal("1"):
		raise RuleViolation(
			"penalty_rate_invalid",
			f"penalty rate {rate} must be in [0, 1]",
			"provide_valid_penalty_rate",
		)


def assert_interest_rate_valid(annual_rate: Decimal) -> None:
	if annual_rate < Decimal("0") or annual_rate > Decimal("1"):
		raise RuleViolation(
			"interest_rate_invalid",
			f"annual interest rate {annual_rate} must be in [0, 1]",
			"provide_valid_interest_rate",
		)


# ---------------------------------------------------------------------------
# Compliance helpers
# ---------------------------------------------------------------------------

def assert_evidence_present(evidence_reference: str | None) -> None:
	if not evidence_reference or not evidence_reference.strip():
		raise RuleViolation(
			"evidence_required",
			"evidence_reference is required for audit trail",
			"provide_evidence_reference",
		)


def assert_officer_present(officer_id: str | None) -> None:
	if not officer_id or not officer_id.strip():
		raise RuleViolation(
			"officer_required",
			"officer_id is required",
			"provide_officer_id",
		)


def assert_reviewer_present(reviewer_id: str | None) -> None:
	if not reviewer_id or not reviewer_id.strip():
		raise RuleViolation(
			"reviewer_required",
			"reviewer_id is required",
			"provide_reviewer_id",
		)


# ---------------------------------------------------------------------------
# Agent runtime
# ---------------------------------------------------------------------------

_SUPPORTED_AGENT_RUNTIMES = {"codex", "langgraph", "crewai", "autogen", "bytewax", "custom"}
_SUPPORTED_AGENT_ROLES = {
	"return_processor", "audit_analyst", "debt_collector",
	"refund_reviewer", "compliance_officer", "risk_analyst",
}


def assert_agent_runtime_supported(runtime: str) -> None:
	if runtime.lower() not in _SUPPORTED_AGENT_RUNTIMES:
		raise RuleViolation(
			"agent_runtime_not_supported",
			f"agent runtime '{runtime}' is not supported",
			f"use one of: {sorted(_SUPPORTED_AGENT_RUNTIMES)}",
		)


def assert_agent_role_supported(role: str) -> None:
	if role.lower() not in _SUPPORTED_AGENT_ROLES:
		raise RuleViolation(
			"agent_role_not_supported",
			f"agent role '{role}' is not supported",
			f"use one of: {sorted(_SUPPORTED_AGENT_ROLES)}",
		)


def assert_event_stream_supported(stream: str) -> None:
	"""Only bytewax is the approved APG event stream."""
	if stream.lower() != "bytewax":
		raise RuleViolation(
			"event_stream_not_supported",
			f"event stream '{stream}' is not supported (only bytewax)",
			"use_bytewax_event_stream",
		)
