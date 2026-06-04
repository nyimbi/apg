"""Deterministic domain rules for Financial Reporting.

Single source of truth for all governance decisions. Every rule is a pure
function — callable from service layer, rule engine, and tests.
"""
from __future__ import annotations

from datetime import date, datetime
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


# ── Tenant & Access ───────────────────────────────────────────────────────────

def assert_tenant_context(context: dict[str, Any]) -> None:
	"""All operations require a tenant context."""
	if not context.get("tenant_id"):
		raise RuleViolation(
			"tenant_context_required",
			"tenant_id is required for all operations",
			"attach_tenant_context",
		)


def assert_write_policy(context: dict[str, Any]) -> None:
	"""Write operations require an attached policy."""
	if context.get("operation_type") == "write" and not context.get("policy_attached"):
		raise RuleViolation(
			"write_requires_policy",
			"write operations require an attached policy",
			"attach_policy",
		)


def assert_no_cross_tenant_access(actor_tenant: str, resource_tenant: str) -> None:
	"""Cross-tenant access is always denied."""
	if actor_tenant != resource_tenant:
		raise RuleViolation(
			"cross_tenant_access_denied",
			f"actor tenant '{actor_tenant}' cannot access resource of tenant '{resource_tenant}'",
			"use_own_tenant_resources",
		)


# ── Report Definition Rules ───────────────────────────────────────────────────

def assert_definition_name_present(name: str | None) -> None:
	if not name or not name.strip():
		raise RuleViolation(
			"definition_name_required",
			"report definition name cannot be blank",
			"provide_definition_name",
		)


def assert_definition_name_length(name: str) -> None:
	if len(name.strip()) > 200:
		raise RuleViolation(
			"definition_name_too_long",
			"report definition name must be ≤ 200 characters",
			"shorten_name",
		)


def assert_statement_type_supported(statement_type: str, supported: set[str]) -> None:
	if statement_type not in supported:
		raise RuleViolation(
			"statement_type_not_supported",
			f"statement_type '{statement_type}' is not supported; valid: {sorted(supported)}",
			"use_supported_statement_type",
		)


def assert_accounting_standard_supported(standard: str, supported: set[str]) -> None:
	if standard not in supported:
		raise RuleViolation(
			"accounting_standard_not_supported",
			f"accounting standard '{standard}' is not supported; valid: {sorted(supported)}",
			"use_supported_standard",
		)


def assert_comparative_periods_in_range(periods: int) -> None:
	if not 0 <= periods <= 5:
		raise RuleViolation(
			"comparative_periods_out_of_range",
			f"comparative_periods must be 0–5, got {periods}",
			"correct_period_count",
		)


# ── Report Line Rules ─────────────────────────────────────────────────────────

def assert_line_account_mapping_present(mapping: str | None) -> None:
	if not mapping or not mapping.strip():
		raise RuleViolation(
			"line_account_mapping_required",
			"account_mapping cannot be blank",
			"provide_account_mapping",
		)


def assert_line_sort_order_present(sort_order: int | None) -> None:
	if sort_order is None:
		raise RuleViolation(
			"line_sort_order_required",
			"sort_order is required for report lines",
			"provide_sort_order",
		)


def assert_line_type_valid(line_type: str) -> None:
	valid = {"detail", "header", "subtotal", "total", "spacer", "formula"}
	if line_type not in valid:
		raise RuleViolation(
			"line_type_invalid",
			f"line_type '{line_type}' is invalid; valid: {sorted(valid)}",
			"use_valid_line_type",
		)


# ── Reporting Period Rules ────────────────────────────────────────────────────

def assert_period_dates_present(start: str | None, end: str | None) -> None:
	if not start or not end:
		raise RuleViolation(
			"period_dates_required",
			"period start_date and end_date are both required",
			"provide_period_dates",
		)


def assert_period_date_range_valid(start: str, end: str) -> None:
	"""end must be strictly after start."""
	try:
		s = date.fromisoformat(start[:10])
		e = date.fromisoformat(end[:10])
	except ValueError as exc:
		raise RuleViolation(
			"period_date_format_invalid",
			f"period dates must be ISO-8601 (YYYY-MM-DD): {exc}",
			"fix_date_format",
		) from exc
	if e <= s:
		raise RuleViolation(
			"period_end_before_start",
			f"period end_date ({end}) must be after start_date ({start})",
			"correct_period_dates",
		)


def assert_period_not_closed(period_status: str) -> None:
	if period_status == "closed":
		raise RuleViolation(
			"period_already_closed",
			"cannot modify a closed reporting period",
			"reopen_period_first",
		)


# ── Report Generation Rules ───────────────────────────────────────────────────

def assert_template_has_lines(line_count: int) -> None:
	if line_count < 1:
		raise RuleViolation(
			"generation_requires_template_lines",
			"report template must have at least one line before generating",
			"add_report_lines",
		)


def assert_output_format_supported(fmt: str, supported: set[str]) -> None:
	if fmt not in supported:
		raise RuleViolation(
			"generation_output_format_supported",
			f"output format '{fmt}' is not supported; valid: {sorted(supported)}",
			"use_supported_output_format",
		)


def assert_data_quality_score_valid(score: float) -> None:
	if not 0.0 <= score <= 1.0:
		raise RuleViolation(
			"data_quality_score_invalid",
			f"data_quality_score must be in [0.0, 1.0], got {score}",
			"correct_quality_score",
		)


def assert_quality_review_for_low_score(score: float, reviewed_by: str | None) -> None:
	"""Scores below 0.95 require explicit quality review."""
	if score < 0.95 and not reviewed_by:
		raise RuleViolation(
			"generation_quality_requires_review",
			f"data_quality_score {score} < 0.95 requires quality_reviewed_by to be set",
			"assign_quality_reviewer",
		)


# ── Statement Publication Rules ───────────────────────────────────────────────

def assert_balance_check_passed(passed: bool) -> None:
	if not passed:
		raise RuleViolation(
			"statement_requires_balance_check",
			"financial statement must pass balance check before publication",
			"resolve_balance_discrepancy",
		)


def assert_approval_recorded(approved_by: str | None) -> None:
	if not approved_by or not approved_by.strip():
		raise RuleViolation(
			"statement_requires_approval",
			"statement must be approved before publication",
			"obtain_approval",
		)


def assert_narrative_review_recorded(reviewed_by: str | None) -> None:
	if not reviewed_by or not reviewed_by.strip():
		raise RuleViolation(
			"statement_requires_narrative_review",
			"narrative must be reviewed before publication",
			"obtain_narrative_review",
		)


def assert_statement_not_already_published(status: str) -> None:
	if status == "published":
		raise RuleViolation(
			"statement_already_published",
			"statement has already been published and cannot be re-published",
			"create_amended_statement",
		)


# ── Consolidation Rules ───────────────────────────────────────────────────────

def assert_no_self_consolidation(parent: str, subsidiary: str) -> None:
	if parent.strip().lower() == subsidiary.strip().lower():
		raise RuleViolation(
			"consolidation_no_self_consolidation",
			f"entity '{parent}' cannot consolidate itself",
			"choose_different_subsidiary",
		)


def assert_ownership_percent_in_range(pct: float) -> None:
	if not 0.0 <= pct <= 100.0:
		raise RuleViolation(
			"consolidation_ownership_within_bounds",
			f"ownership_percent must be 0–100, got {pct}",
			"correct_ownership_percent",
		)


def assert_consolidation_method_valid(method: str) -> None:
	valid = {"full", "proportional", "equity", "none"}
	if method not in valid:
		raise RuleViolation(
			"consolidation_method_invalid",
			f"consolidation method '{method}' is invalid; valid: {sorted(valid)}",
			"use_valid_consolidation_method",
		)


def assert_elimination_review_recorded(reviewed_by: str | None) -> None:
	if not reviewed_by:
		raise RuleViolation(
			"consolidation_elimination_review_required",
			"elimination entries must be reviewed before consolidation is finalised",
			"assign_elimination_reviewer",
		)


# ── Disclosure Rules ──────────────────────────────────────────────────────────

def assert_disclosure_owner_present(owner: str | None) -> None:
	if not owner or not owner.strip():
		raise RuleViolation(
			"disclosure_owner_required",
			"disclosure must have an assigned owner",
			"assign_disclosure_owner",
		)


def assert_disclosure_review_recorded(reviewed_by: str | None) -> None:
	if not reviewed_by or not reviewed_by.strip():
		raise RuleViolation(
			"disclosure_review_required",
			"disclosure must be reviewed before being recorded on a statement",
			"obtain_disclosure_review",
		)


# ── Distribution Rules ────────────────────────────────────────────────────────

def assert_recipients_present(recipients: list[str] | None) -> None:
	if not recipients:
		raise RuleViolation(
			"distribution_recipients_required",
			"at least one recipient is required for statement distribution",
			"add_recipients",
		)


def assert_statement_is_approved(approved_by: str | None) -> None:
	if not approved_by:
		raise RuleViolation(
			"distribution_requires_approved_statement",
			"only approved statements may be distributed",
			"approve_statement_first",
		)


# ── XBRL Rules ────────────────────────────────────────────────────────────────

def assert_xbrl_taxonomy_supported(taxonomy: str, supported: set[str]) -> None:
	if taxonomy not in supported:
		raise RuleViolation(
			"xbrl_taxonomy_not_supported",
			f"XBRL taxonomy '{taxonomy}' is not supported; valid: {sorted(supported)}",
			"use_supported_taxonomy",
		)


def assert_xbrl_element_name_present(element_name: str | None) -> None:
	if not element_name or not element_name.strip():
		raise RuleViolation(
			"xbrl_element_name_required",
			"element_name is required for XBRL tag",
			"provide_element_name",
		)


def assert_xbrl_context_ref_present(context_ref: str | None) -> None:
	if not context_ref or not context_ref.strip():
		raise RuleViolation(
			"xbrl_context_ref_required",
			"context_ref is required for XBRL tag",
			"provide_context_ref",
		)


# ── Regulatory Filing Rules ───────────────────────────────────────────────────

def assert_jurisdiction_supported(jurisdiction: str, supported: set[str]) -> None:
	if jurisdiction not in supported:
		raise RuleViolation(
			"regulatory_jurisdiction_not_supported",
			f"jurisdiction '{jurisdiction}' is not supported; valid: {sorted(supported)}",
			"use_supported_jurisdiction",
		)


def assert_filing_deadline_not_past(deadline: str) -> None:
	try:
		d = date.fromisoformat(deadline[:10])
	except ValueError as exc:
		raise RuleViolation(
			"regulatory_deadline_format_invalid",
			f"filing_deadline must be ISO-8601 (YYYY-MM-DD): {exc}",
			"fix_deadline_format",
		) from exc
	if d < date.today():
		raise RuleViolation(
			"regulatory_deadline_in_past",
			f"filing deadline {deadline} is in the past; obtain extension or waiver",
			"request_deadline_extension",
		)


def assert_regulatory_prepared_by_present(prepared_by: str | None) -> None:
	if not prepared_by or not prepared_by.strip():
		raise RuleViolation(
			"regulatory_prepared_by_required",
			"regulatory submission must identify who prepared it",
			"provide_preparer",
		)


# ── Segment Reporting Rules ───────────────────────────────────────────────────

def assert_segment_name_present(name: str | None) -> None:
	if not name or not name.strip():
		raise RuleViolation(
			"segment_name_required",
			"segment_name cannot be blank",
			"provide_segment_name",
		)


def assert_segment_amounts_non_negative(
	revenue: float, operating_profit: float | None = None
) -> None:
	if revenue < 0:
		raise RuleViolation(
			"segment_revenue_negative",
			f"segment revenue cannot be negative, got {revenue}",
			"correct_segment_revenue",
		)


# ── Schedule Rules ────────────────────────────────────────────────────────────

def assert_schedule_recipients_present(recipients: list[str] | None) -> None:
	if not recipients:
		raise RuleViolation(
			"schedule_recipients_required",
			"at least one recipient is required for a report schedule",
			"add_schedule_recipients",
		)


# ── Calculation helpers (calculate_* convention) ─────────────────────────────

def calculate_balance_equation(
	total_assets: float,
	total_liabilities: float,
	total_equity: float,
	tolerance: float = 0.01,
) -> tuple[bool, float]:
	"""Returns (balanced, discrepancy)."""
	lhs = round(total_assets, 4)
	rhs = round(total_liabilities + total_equity, 4)
	discrepancy = abs(lhs - rhs)
	return discrepancy <= tolerance, round(discrepancy, 4)


def calculate_ownership_adjusted_amount(amount: float, ownership_pct: float) -> float:
	"""Proportional consolidation helper."""
	return round(amount * ownership_pct / 100.0, 4)


def calculate_minority_interest(subsidiary_equity: float, parent_ownership_pct: float) -> float:
	minority_pct = 100.0 - parent_ownership_pct
	return round(subsidiary_equity * minority_pct / 100.0, 4)


def calculate_variance_pct(actual: float, budget: float) -> float | None:
	if budget == 0:
		return None
	return round((actual - budget) / abs(budget) * 100, 4)


def calculate_period_change(current: float, prior: float) -> tuple[float, float | None]:
	abs_change = round(current - prior, 4)
	pct = None if prior == 0 else round((current - prior) / abs(prior) * 100, 4)
	return abs_change, pct
