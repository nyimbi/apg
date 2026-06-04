"""
APG Budgeting & Forecasting — Domain Rules

All business rules as callable assert_* functions.
RuleViolation is the single exception type for violations.
Calculations are pure functions with typed I/O.

© 2025 Datacraft. Author: Nyimbi Odero <nyimbi@gmail.com>
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
# Context / tenancy rules
# ---------------------------------------------------------------------------

def assert_tenant_context(tenant_id: str) -> None:
	"""All operations require a non-empty tenant_id string."""
	if not tenant_id or not tenant_id.strip():
		raise RuleViolation(
			"tenant_context_required",
			"tenant_id is required for all operations",
			"attach_tenant_context",
		)


def assert_actor_present(actor_id: str) -> None:
	"""All write operations require an identified actor."""
	if not actor_id or not actor_id.strip():
		raise RuleViolation(
			"actor_required",
			"actor_id is required for all write operations",
			"authenticate_user",
		)


def assert_no_cross_tenant(actor_tenant: str, resource_tenant: str) -> None:
	"""Cross-tenant access is always denied."""
	if actor_tenant != resource_tenant:
		raise RuleViolation(
			"cross_tenant_access_denied",
			f"Actor tenant '{actor_tenant}' cannot access resource of tenant '{resource_tenant}'",
			"use_own_tenant_resources",
		)


# ---------------------------------------------------------------------------
# Budget lifecycle rules
# ---------------------------------------------------------------------------

def assert_budget_in_draft(status: str) -> None:
	"""Mutation of lines/fields is only permitted on DRAFT budgets."""
	if status != "draft":
		raise RuleViolation(
			"budget_must_be_draft",
			f"Budget is in status '{status}'; only DRAFT budgets can be edited",
			"reset_to_draft_or_create_new_version",
		)


def assert_budget_submittable(status: str) -> None:
	"""Budget must be DRAFT to be submitted."""
	if status not in ("draft",):
		raise RuleViolation(
			"budget_not_submittable",
			f"Budget in status '{status}' cannot be submitted; must be DRAFT",
			"complete_draft_first",
		)


def assert_budget_approvable(status: str) -> None:
	"""Budget must be SUBMITTED or UNDER_REVIEW to receive an approval decision."""
	if status not in ("submitted", "under_review"):
		raise RuleViolation(
			"budget_not_approvable",
			f"Budget in status '{status}' cannot be approved/rejected",
			"submit_budget_first",
		)


def assert_budget_not_locked(status: str) -> None:
	"""LOCKED budgets cannot be closed or cancelled directly."""
	if status == "locked":
		raise RuleViolation(
			"budget_is_locked",
			"Locked budgets cannot be directly closed or cancelled; unlock first",
			"unlock_budget_first",
		)


def assert_budget_has_lines(line_count: int) -> None:
	"""A budget must have at least one line before submission."""
	if line_count == 0:
		raise RuleViolation(
			"budget_has_no_lines",
			"Budget must have at least one budget line before submission",
			"add_budget_lines",
		)


def assert_budget_period_valid(period_start: date, period_end: date) -> None:
	"""period_end must be strictly after period_start."""
	if period_end <= period_start:
		raise RuleViolation(
			"invalid_budget_period",
			f"period_end ({period_end}) must be after period_start ({period_start})",
			"correct_budget_dates",
		)


def assert_fiscal_year_reasonable(fiscal_year: int) -> None:
	"""Fiscal year must be within ±10 years of current."""
	from datetime import datetime
	current = datetime.now().year
	if not (current - 10 <= fiscal_year <= current + 10):
		raise RuleViolation(
			"fiscal_year_unreasonable",
			f"Fiscal year {fiscal_year} is implausible (expected {current-10}–{current+10})",
			"correct_fiscal_year",
		)


# ---------------------------------------------------------------------------
# Approval workflow rules
# ---------------------------------------------------------------------------

def assert_approval_pending(status: str) -> None:
	"""An approval action can only be taken on PENDING approvals."""
	if status != "pending":
		raise RuleViolation(
			"approval_not_pending",
			f"Approval is in status '{status}'; only PENDING approvals can be acted on",
			"check_approval_status",
		)


def assert_approver_not_self(budget_created_by: str, approver_id: str) -> None:
	"""Four-eyes principle: the budget creator cannot approve their own budget."""
	if budget_created_by == approver_id:
		raise RuleViolation(
			"self_approval_not_permitted",
			"Budget creator cannot approve their own budget (four-eyes principle)",
			"assign_independent_approver",
		)


# ---------------------------------------------------------------------------
# Amounts / balance rules
# ---------------------------------------------------------------------------

def assert_amounts_balanced(total_debits: Decimal, total_credits: Decimal) -> None:
	"""For double-entry: debits must equal credits."""
	if total_debits != total_credits:
		raise RuleViolation(
			"amounts_not_balanced",
			f"Debits ({total_debits}) do not equal credits ({total_credits})",
			"correct_line_amounts",
		)


def assert_zero_based_balanced(
	justified_total: Decimal,
	budget_total: Decimal,
	tolerance: Decimal = Decimal("0.01"),
) -> None:
	"""Zero-based budget: justified amounts must reconcile with budget total."""
	diff = abs(justified_total - budget_total)
	if diff > tolerance:
		raise RuleViolation(
			"zero_based_unbalanced",
			f"Justified total ({justified_total}) differs from budget total ({budget_total}) by {diff}",
			"reconcile_line_justifications",
		)


def assert_driver_value_positive(value: Decimal) -> None:
	"""Driver assumption values must be positive (non-zero, non-negative)."""
	if value <= Decimal("0"):
		raise RuleViolation(
			"driver_value_must_be_positive",
			f"Driver value {value} must be > 0",
			"correct_driver_value",
		)


# ---------------------------------------------------------------------------
# Forecast rules
# ---------------------------------------------------------------------------

def assert_forecast_horizon_valid(horizon: int) -> None:
	"""Forecast horizon must be between 1 and 120 periods."""
	if not (1 <= horizon <= 120):
		raise RuleViolation(
			"invalid_forecast_horizon",
			f"Forecast horizon {horizon} must be between 1 and 120 periods",
			"correct_horizon",
		)


def assert_sufficient_history(history_count: int, min_required: int = 3) -> None:
	"""Statistical forecasting requires a minimum history length."""
	if history_count < min_required:
		raise RuleViolation(
			"insufficient_history",
			f"Only {history_count} data points available; minimum {min_required} required for forecasting",
			"add_more_historical_data",
		)


# ---------------------------------------------------------------------------
# Scenario rules
# ---------------------------------------------------------------------------

def assert_scenarios_non_empty(scenario_ids: list[str]) -> None:
	"""Scenario analysis requires at least one scenario."""
	if not scenario_ids:
		raise RuleViolation(
			"no_scenarios_provided",
			"Scenario analysis requires at least one scenario_id",
			"create_scenarios_first",
		)


def assert_probability_sum_valid(
	probabilities: list[float],
	tolerance: float = 0.05,
) -> None:
	"""
	Scenario probabilities should sum to approximately 1.0.

	A tolerance of ±5% is allowed to account for floating-point rounding.
	"""
	total = sum(probabilities)
	if not (1.0 - tolerance <= total <= 1.0 + tolerance):
		raise RuleViolation(
			"probabilities_do_not_sum_to_one",
			f"Scenario probabilities sum to {total:.4f}; expected ~1.0 (±{tolerance})",
			"adjust_scenario_probabilities",
		)
