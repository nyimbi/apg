"""
Employee Data Management — domain business rules.

Every rule is a callable function.  Rules raise RuleViolation on violation.
assert_* functions are guards; calculate_* functions are pure computations.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero
"""
from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
from typing import Any


# ---------------------------------------------------------------------------
# Violation
# ---------------------------------------------------------------------------

class RuleViolation(Exception):
	"""Raised when a business rule is violated."""

	def __init__(self, rule_name: str, reason: str, required_action: str = "") -> None:
		self.rule_name = rule_name
		self.reason = reason
		self.required_action = required_action
		super().__init__(f"Rule '{rule_name}' violated: {reason}")


# ---------------------------------------------------------------------------
# Tenant & auth
# ---------------------------------------------------------------------------

def assert_tenant_context(tenant_id: str | None) -> None:
	"""All operations require a tenant context."""
	if not tenant_id:
		raise RuleViolation(
			"tenant_context_required",
			"tenant_id is required for all operations",
			"attach_tenant_context",
		)


def assert_no_cross_tenant(actor_tenant: str, resource_tenant: str) -> None:
	"""Cross-tenant data access is always denied."""
	if actor_tenant != resource_tenant:
		raise RuleViolation(
			"cross_tenant_access_denied",
			f"actor tenant '{actor_tenant}' cannot access resources of '{resource_tenant}'",
			"use_own_tenant_resources",
		)


def assert_actor_present(actor_id: str | None) -> None:
	"""Mutating operations require an identified actor."""
	if not actor_id:
		raise RuleViolation(
			"actor_required",
			"an actor_id is required for write operations",
			"provide_actor_id",
		)


# ---------------------------------------------------------------------------
# Employee number uniqueness
# ---------------------------------------------------------------------------

def assert_employee_number_unique(
	employee_number: str,
	existing_numbers: set[str],
) -> None:
	"""Employee numbers must be unique within a tenant."""
	if employee_number in existing_numbers:
		raise RuleViolation(
			"employee_number_not_unique",
			f"employee_number '{employee_number}' is already in use",
			"assign_unique_employee_number",
		)


# ---------------------------------------------------------------------------
# Hire / pre-employment
# ---------------------------------------------------------------------------

def assert_hire_date_not_future_limit(hire_date: date, max_future_days: int = 365) -> None:
	"""Hire date cannot be more than 1 year in the future."""
	cutoff = date.today() + timedelta(days=max_future_days)
	if hire_date > cutoff:
		raise RuleViolation(
			"hire_date_too_far_future",
			f"hire_date {hire_date} exceeds the {max_future_days}-day forward booking limit",
			"use_realistic_hire_date",
		)


def assert_start_date_not_before_hire(hire_date: date, start_date: date) -> None:
	"""Start date cannot precede hire date."""
	if start_date < hire_date:
		raise RuleViolation(
			"start_date_before_hire_date",
			f"start_date {start_date} cannot be before hire_date {hire_date}",
			"align_start_date_with_hire_date",
		)


def assert_work_permit_for_foreign_national(
	nationality: str | None,
	country_of_work: str,
	has_active_permit: bool,
) -> None:
	"""Foreign nationals working in a country other than their nationality must have a work permit."""
	if nationality and nationality.upper() != country_of_work.upper() and not has_active_permit:
		raise RuleViolation(
			"work_permit_required",
			f"employee with nationality '{nationality}' requires a work permit to work in '{country_of_work}'",
			"obtain_work_permit_before_hire",
		)


def assert_background_check_consent(consent_given: bool) -> None:
	"""Background checks require explicit employee consent."""
	if not consent_given:
		raise RuleViolation(
			"background_check_consent_required",
			"employee consent must be recorded before initiating a background check",
			"obtain_written_consent",
		)


# ---------------------------------------------------------------------------
# Position / headcount
# ---------------------------------------------------------------------------

def assert_position_not_overfilled(
	authorized_headcount: int,
	current_headcount: int,
) -> None:
	"""Hiring into a position cannot exceed its authorized headcount."""
	if current_headcount >= authorized_headcount:
		raise RuleViolation(
			"position_headcount_exceeded",
			f"position is authorized for {authorized_headcount} but already has {current_headcount}",
			"raise_headcount_authorization_or_choose_another_position",
		)


def assert_position_active(is_active: bool, position_id: str) -> None:
	"""Employees can only be hired into active positions."""
	if not is_active:
		raise RuleViolation(
			"position_inactive",
			f"position '{position_id}' is not active",
			"activate_position_before_hire",
		)


# ---------------------------------------------------------------------------
# Salary / grade
# ---------------------------------------------------------------------------

def assert_salary_within_grade(
	salary: Decimal,
	grade_min: Decimal,
	grade_max: Decimal,
	allow_exception: bool = False,
) -> None:
	"""Salary must fall within the job grade band unless an exception is granted."""
	if salary < grade_min or salary > grade_max:
		if not allow_exception:
			raise RuleViolation(
				"salary_outside_grade_band",
				f"salary {salary} is outside grade band [{grade_min}, {grade_max}]",
				"adjust_salary_or_request_off_band_exception",
			)


def assert_promotion_salary_increase(
	current_salary: Decimal,
	new_salary: Decimal,
	min_increase_pct: float = 0.0,
) -> None:
	"""Promotion new salary must be >= current salary (optionally by a minimum pct)."""
	if new_salary < current_salary:
		raise RuleViolation(
			"promotion_salary_decrease",
			f"new salary {new_salary} is less than current {current_salary} — promotions cannot reduce pay",
			"set_new_salary_above_current",
		)
	if min_increase_pct > 0:
		required = current_salary * Decimal(str(1 + min_increase_pct / 100))
		if new_salary < required:
			raise RuleViolation(
				"promotion_salary_below_minimum_increase",
				f"new salary {new_salary} does not meet the minimum {min_increase_pct}% increase to {required}",
				"increase_salary_by_minimum_required_percentage",
			)


# ---------------------------------------------------------------------------
# Employment status transitions
# ---------------------------------------------------------------------------

_VALID_STATUS_TRANSITIONS: dict[str, set[str]] = {
	"probation":   {"active", "terminated", "probation"},
	"active":      {"notice", "on_leave", "suspended", "terminated", "retired"},
	"on_leave":    {"active", "terminated"},
	"notice":      {"terminated", "active"},
	"suspended":   {"active", "terminated"},
	"terminated":  set(),   # terminal — can only rehire as a new record
	"retired":     set(),
	"deceased":    set(),
}


def assert_valid_status_transition(
	current_status: str,
	new_status: str,
) -> None:
	"""Validate that a status change follows the allowed lifecycle FSM."""
	allowed = _VALID_STATUS_TRANSITIONS.get(current_status, set())
	if new_status not in allowed:
		raise RuleViolation(
			"invalid_status_transition",
			f"cannot transition employment status from '{current_status}' to '{new_status}'",
			f"use_one_of: {sorted(allowed)}",
		)


# ---------------------------------------------------------------------------
# Probation
# ---------------------------------------------------------------------------

_MAX_PROBATION_MONTHS = 6


def assert_probation_period_valid(
	hire_date: date,
	probation_end_date: date,
) -> None:
	"""Kenya Employment Act: probation cannot exceed 6 months."""
	months = (
		(probation_end_date.year - hire_date.year) * 12
		+ probation_end_date.month - hire_date.month
	)
	if months > _MAX_PROBATION_MONTHS:
		raise RuleViolation(
			"probation_exceeds_statutory_maximum",
			f"probation of {months} months exceeds the statutory maximum of {_MAX_PROBATION_MONTHS} months",
			"shorten_probation_period",
		)


def assert_probation_outcome_provided(outcome: str) -> None:
	"""Probation review must have a clear outcome."""
	valid = {"pass", "fail", "extend"}
	if outcome not in valid:
		raise RuleViolation(
			"invalid_probation_outcome",
			f"outcome '{outcome}' is not valid; must be one of {valid}",
			"provide_valid_probation_outcome",
		)


# ---------------------------------------------------------------------------
# Termination
# ---------------------------------------------------------------------------

def assert_termination_notice_period(
	notice_date: date | None,
	effective_date: date,
	notice_period_days: int,
) -> None:
	"""For resignation/redundancy, notice must equal the contractual period."""
	if notice_date is None:
		return
	days_notice = (effective_date - notice_date).days
	if days_notice < notice_period_days:
		raise RuleViolation(
			"insufficient_notice_period",
			f"only {days_notice} days notice given; contract requires {notice_period_days}",
			"extend_notice_period_or_pay_in_lieu",
		)


def assert_termination_approved_for_dismissal(
	termination_type: str,
	approved_by: str | None,
) -> None:
	"""Dismissals must be explicitly approved."""
	if termination_type == "dismissal" and not approved_by:
		raise RuleViolation(
			"dismissal_requires_approval",
			"dismissal terminations must be approved by an authorized officer",
			"obtain_dismissal_approval",
		)


def assert_disciplinary_hearing_before_dismissal(
	termination_type: str,
	hearing_completed: bool,
) -> None:
	"""Dismissal requires a completed disciplinary hearing (fair procedure)."""
	if termination_type == "dismissal" and not hearing_completed:
		raise RuleViolation(
			"dismissal_without_hearing",
			"a dismissal termination requires a completed disciplinary hearing",
			"complete_disciplinary_hearing_first",
		)


# ---------------------------------------------------------------------------
# Transfer
# ---------------------------------------------------------------------------

def assert_transfer_not_to_same_position(
	current_position_id: str,
	new_position_id: str,
) -> None:
	"""A transfer must change the position."""
	if current_position_id == new_position_id:
		raise RuleViolation(
			"transfer_no_change",
			"transfer new_position_id is the same as current — no change would occur",
			"choose_different_position_for_transfer",
		)


# ---------------------------------------------------------------------------
# Performance review
# ---------------------------------------------------------------------------

def assert_review_period_valid(
	period_start: date,
	period_end: date,
) -> None:
	"""Review period end must be after start."""
	if period_end <= period_start:
		raise RuleViolation(
			"review_period_invalid",
			f"review_period_end {period_end} must be after review_period_start {period_start}",
			"set_valid_review_period",
		)


def assert_self_assessment_before_manager_review(
	self_rating_present: bool,
	attempting_manager_review: bool,
) -> None:
	"""Manager rating cannot be submitted before the employee self-assessment."""
	if attempting_manager_review and not self_rating_present:
		raise RuleViolation(
			"self_assessment_required_before_manager_review",
			"employee self-assessment must be completed before the manager submits a rating",
			"complete_self_assessment_first",
		)


# ---------------------------------------------------------------------------
# Disciplinary
# ---------------------------------------------------------------------------

_DISCIPLINARY_ESCALATION: list[str] = [
	"verbal_warning",
	"written_warning",
	"final_warning",
	"suspension",
	"demotion",
	"dismissal",
]


def assert_disciplinary_escalation(
	existing_type: str | None,
	new_type: str,
) -> None:
	"""Disciplinary actions should escalate — skip-escalations require explicit override."""
	if existing_type is None:
		return
	try:
		existing_idx = _DISCIPLINARY_ESCALATION.index(existing_type)
		new_idx = _DISCIPLINARY_ESCALATION.index(new_type)
	except ValueError:
		return
	if new_idx < existing_idx:
		raise RuleViolation(
			"disciplinary_regression",
			f"cannot issue a '{new_type}' after a '{existing_type}' without override",
			"use_override_flag_for_non_escalating_action",
		)


def assert_disciplinary_incident_description(description: str) -> None:
	"""Incident description must be substantive."""
	if len(description.strip()) < 20:
		raise RuleViolation(
			"disciplinary_description_too_short",
			"incident_description must be at least 20 characters",
			"provide_detailed_incident_description",
		)


# ---------------------------------------------------------------------------
# Work permit
# ---------------------------------------------------------------------------

def assert_work_permit_not_expired(expiry_date: date | None) -> None:
	"""An expired work permit cannot be used for employment."""
	if expiry_date and expiry_date < date.today():
		raise RuleViolation(
			"work_permit_expired",
			f"work permit expired on {expiry_date}",
			"renew_or_replace_work_permit",
		)


def assert_work_permit_renewal_submitted_on_time(
	expiry_date: date | None,
	renewal_submitted_at: date | None,
	lead_days: int = 60,
) -> None:
	"""Permit renewal should be submitted at least `lead_days` before expiry."""
	if expiry_date and renewal_submitted_at:
		if (expiry_date - renewal_submitted_at).days < lead_days:
			raise RuleViolation(
				"work_permit_renewal_submitted_late",
				f"renewal was submitted only {(expiry_date - renewal_submitted_at).days} days before expiry; "
				f"recommend submitting {lead_days} days in advance",
				"submit_renewal_earlier",
			)


# ---------------------------------------------------------------------------
# Onboarding
# ---------------------------------------------------------------------------

def assert_onboarding_checklist_not_empty(items: list[Any]) -> None:
	"""Onboarding checklists must have at least one item."""
	if not items:
		raise RuleViolation(
			"onboarding_checklist_empty",
			"onboarding checklist must contain at least one item",
			"add_onboarding_tasks",
		)


# ---------------------------------------------------------------------------
# Succession planning
# ---------------------------------------------------------------------------

def assert_succession_candidate_active(employment_status: str) -> None:
	"""Only active employees can be succession candidates."""
	if employment_status not in ("active", "probation"):
		raise RuleViolation(
			"succession_candidate_not_active",
			f"employee with status '{employment_status}' cannot be a succession candidate",
			"only_nominate_active_employees",
		)


# ---------------------------------------------------------------------------
# Benefit enrollment
# ---------------------------------------------------------------------------

def assert_benefit_enrollment_dates(
	coverage_start: date,
	coverage_end: date | None,
) -> None:
	"""Benefit coverage end must be after start if provided."""
	if coverage_end and coverage_end <= coverage_start:
		raise RuleViolation(
			"benefit_coverage_end_before_start",
			f"coverage_end {coverage_end} must be after coverage_start {coverage_start}",
			"correct_benefit_coverage_dates",
		)


# ---------------------------------------------------------------------------
# Emergency contact
# ---------------------------------------------------------------------------

def assert_primary_contact_unique(
	employee_id: str,
	is_primary: bool,
	existing_primary_count: int,
) -> None:
	"""Each employee can only have one primary emergency contact."""
	if is_primary and existing_primary_count > 0:
		raise RuleViolation(
			"duplicate_primary_emergency_contact",
			f"employee '{employee_id}' already has a primary emergency contact",
			"demote_existing_primary_contact_first",
		)


# ---------------------------------------------------------------------------
# Qualification
# ---------------------------------------------------------------------------

def assert_qualification_years_valid(
	start_year: int,
	end_year: int | None,
) -> None:
	"""Qualification end year must be >= start year."""
	if end_year and end_year < start_year:
		raise RuleViolation(
			"qualification_end_year_before_start",
			f"end_year {end_year} is before start_year {start_year}",
			"correct_qualification_years",
		)


# ---------------------------------------------------------------------------
# Compensation change
# ---------------------------------------------------------------------------

def assert_compensation_change_has_reason(reason: str) -> None:
	"""Compensation changes must be justified."""
	if not reason or len(reason.strip()) < 5:
		raise RuleViolation(
			"compensation_change_reason_required",
			"a meaningful reason is required for any compensation change",
			"provide_compensation_change_justification",
		)


# ---------------------------------------------------------------------------
# Attrition prediction helpers
# ---------------------------------------------------------------------------

def calculate_flight_risk_score(
	months_since_last_review: int,
	months_since_last_promotion: int,
	disciplinary_count: int,
	grievance_count: int,
	salary_range_penetration: float,  # 0.0-1.0
) -> float:
	"""
	Heuristic flight-risk score in [0, 100].

	Higher = more likely to leave.  Used for attrition_prediction().
	"""
	score = 0.0

	# Recency of review: >12 months adds up to 20 pts
	score += min(months_since_last_review / 12 * 20, 20)

	# Recency of promotion: >24 months adds up to 20 pts
	score += min(months_since_last_promotion / 24 * 20, 20)

	# Disciplinary events (may indicate disengagement, max 10 pts)
	score += min(disciplinary_count * 5, 10)

	# Grievances (may indicate unhappiness, max 10 pts)
	score += min(grievance_count * 5, 10)

	# Low salary range penetration indicates underpayment, max 40 pts
	score += (1.0 - max(min(salary_range_penetration, 1.0), 0.0)) * 40

	return round(min(score, 100.0), 1)
