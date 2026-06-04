"""Payroll domain rules — all business invariants as callable Python.

Every rule is a pure function. Violations raise RuleViolation.
assert_* functions guard pre/post conditions.
calculate_* functions derive values from domain data.
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

	def __init__(self, rule_id: str, reason: str, required_action: str = "") -> None:
		self.rule_id = rule_id
		self.reason = reason
		self.required_action = required_action
		super().__init__(f"[{rule_id}] {reason}")


# ---------------------------------------------------------------------------
# Tenant isolation
# ---------------------------------------------------------------------------

def assert_tenant_match(actor_tenant: str, resource_tenant: str) -> None:
	"""Cross-tenant access is always denied."""
	if actor_tenant != resource_tenant:
		raise RuleViolation(
			"PR-001",
			f"Cross-tenant access denied: actor={actor_tenant} resource={resource_tenant}",
			"use_own_tenant_resources",
		)


def assert_tenant_id_present(tenant_id: str | None) -> None:
	if not tenant_id:
		raise RuleViolation("PR-002", "tenant_id is required for all payroll operations", "attach_tenant_context")


# ---------------------------------------------------------------------------
# Employee rules
# ---------------------------------------------------------------------------

def assert_employee_active(employee: dict[str, Any]) -> None:
	if employee.get("is_deleted"):
		raise RuleViolation("PR-010", f"Employee {employee['id']} is deleted", "restore_employee")
	if not employee.get("is_active", True):
		raise RuleViolation("PR-011", f"Employee {employee['id']} is inactive", "reactivate_employee")


def assert_employee_not_terminated(employee: dict[str, Any], period_end: date) -> None:
	"""Employee must not have been terminated before the period ends."""
	term = employee.get("termination_date")
	if term is None:
		return
	if isinstance(term, str):
		term = date.fromisoformat(term)
	if isinstance(period_end, str):
		period_end = date.fromisoformat(period_end)
	if term < period_end:
		raise RuleViolation(
			"PR-012",
			f"Employee {employee['id']} was terminated {term} before period end {period_end}",
			"process_final_settlement_instead",
		)


def assert_hire_date_before_period(employee: dict[str, Any], period_start: date) -> None:
	"""Employee must be hired on or before period start (or handle pro-ration)."""
	hire = employee.get("hire_date")
	if hire is None:
		raise RuleViolation("PR-013", "Employee has no hire_date", "set_hire_date")
	if isinstance(hire, str):
		hire = date.fromisoformat(hire)
	if isinstance(period_start, str):
		period_start = date.fromisoformat(period_start)
	# Hire after period end means no pay in this period
	if hire > period_start:
		raise RuleViolation(
			"PR-014",
			f"Employee hire date {hire} is after period start {period_start}. Pro-rate required.",
			"calculate_prorated_salary",
		)


def assert_basic_salary_positive(basic_salary: Decimal) -> None:
	if basic_salary <= 0:
		raise RuleViolation("PR-015", "Basic salary must be positive", "set_valid_basic_salary")


# ---------------------------------------------------------------------------
# Pay period rules
# ---------------------------------------------------------------------------

def assert_period_dates_valid(start_date: date, end_date: date, pay_date: date) -> None:
	if end_date <= start_date:
		raise RuleViolation("PR-020", f"Period end {end_date} must be after start {start_date}", "fix_period_dates")
	if pay_date < end_date:
		raise RuleViolation("PR-021", f"Pay date {pay_date} cannot be before period end {end_date}", "fix_pay_date")


def assert_period_open(period: dict[str, Any]) -> None:
	status = period.get("status", "open")
	if status not in ("open",):
		raise RuleViolation(
			"PR-022",
			f"Period {period['id']} is {status}; only open periods can be processed",
			"reopen_period",
		)


def assert_period_not_closed(period: dict[str, Any]) -> None:
	if period.get("status") == "closed":
		raise RuleViolation("PR-023", f"Period {period['id']} is closed; no changes allowed", "create_new_period")


def assert_no_overlapping_period(
	existing_periods: list[dict[str, Any]],
	start_date: date,
	end_date: date,
	tenant_id: str,
) -> None:
	for p in existing_periods:
		if p.get("tenant_id") != tenant_id or p.get("is_deleted"):
			continue
		ps = date.fromisoformat(p["start_date"]) if isinstance(p["start_date"], str) else p["start_date"]
		pe = date.fromisoformat(p["end_date"]) if isinstance(p["end_date"], str) else p["end_date"]
		if start_date <= pe and end_date >= ps:
			raise RuleViolation(
				"PR-024",
				f"New period [{start_date}, {end_date}] overlaps with existing period {p['id']} [{ps}, {pe}]",
				"adjust_period_dates",
			)


# ---------------------------------------------------------------------------
# Payroll run rules
# ---------------------------------------------------------------------------

def assert_run_in_status(run: dict[str, Any], *allowed: str) -> None:
	current = run.get("status", "draft")
	if current not in allowed:
		raise RuleViolation(
			"PR-030",
			f"Run {run['id']} is '{current}'; operation requires one of {allowed}",
			f"move_run_to_{allowed[0]}",
		)


def assert_run_approved_before_post(run: dict[str, Any]) -> None:
	if not run.get("approved_by"):
		raise RuleViolation("PR-031", "Run must be approved before posting", "approve_run_first")


def assert_run_posted_before_payment(run: dict[str, Any]) -> None:
	if run.get("status") not in ("posted", "paid"):
		raise RuleViolation("PR-032", "Run must be posted before generating bank file", "post_run_first")


def assert_run_not_reversed(run: dict[str, Any]) -> None:
	if run.get("status") == "reversed":
		raise RuleViolation("PR-033", f"Run {run['id']} has already been reversed", "create_new_run")


def assert_no_duplicate_run_for_period(
	existing_runs: list[dict[str, Any]],
	period_id: str,
	tenant_id: str,
	is_bonus: bool = False,
) -> None:
	"""One regular run per period per tenant (bonus runs exempt)."""
	if is_bonus:
		return
	for r in existing_runs:
		if (r.get("tenant_id") == tenant_id
				and r.get("period_id") == period_id
				and not r.get("is_bonus_run")
				and not r.get("is_deleted")
				and r.get("status") not in ("cancelled", "reversed")):
			raise RuleViolation(
				"PR-034",
				f"A payroll run already exists for period {period_id}",
				"use_existing_run_or_cancel_it",
			)


# ---------------------------------------------------------------------------
# Payslip / line item rules
# ---------------------------------------------------------------------------

def assert_amount_non_negative_for_earning(element_type: str, amount: Decimal) -> None:
	earning_types = {"basic", "allowance", "overtime", "bonus", "commission", "back_pay"}
	if element_type in earning_types and amount < 0:
		raise RuleViolation(
			"PR-040",
			f"Earning element '{element_type}' cannot have a negative amount ({amount})",
			"use_deduction_element_type_for_negative_amounts",
		)


def assert_deduction_does_not_exceed_net(
	total_deductions: Decimal,
	gross_earnings: Decimal,
	min_net_balance: Decimal = Decimal("0"),
) -> None:
	"""Net pay cannot go negative from deductions."""
	net = gross_earnings - total_deductions
	if net < min_net_balance:
		raise RuleViolation(
			"PR-041",
			f"Total deductions {total_deductions} would reduce net below {min_net_balance} (gross={gross_earnings})",
			"reduce_deduction_amount",
		)


# ---------------------------------------------------------------------------
# Tax rules
# ---------------------------------------------------------------------------

def assert_paye_non_negative(paye: Decimal) -> None:
	if paye < 0:
		raise RuleViolation("PR-050", f"PAYE cannot be negative ({paye})", "recheck_tax_calculation")


def assert_country_has_paye_table(country: str, supported: set[str]) -> None:
	if country.upper() not in supported:
		raise RuleViolation(
			"PR-051",
			f"No PAYE table configured for country '{country}'",
			f"add_paye_table_for_{country}",
		)


def assert_tax_code_valid(tax_code: str | None) -> None:
	"""Tax codes must follow expected formats (lenient — just non-empty)."""
	if tax_code is not None and not tax_code.strip():
		raise RuleViolation("PR-052", "tax_code cannot be an empty string", "provide_valid_tax_code")


# ---------------------------------------------------------------------------
# Statutory deduction rules
# ---------------------------------------------------------------------------

def assert_nssf_number_present_ke(employee: dict[str, Any]) -> None:
	"""Kenya NSSF requires a member number for contributions."""
	if not employee.get("nssf_number"):
		raise RuleViolation("PR-060", "NSSF number required for Kenya payroll", "register_employee_with_nssf")


def assert_nhif_number_present_ke(employee: dict[str, Any]) -> None:
	if not employee.get("nhif_number"):
		raise RuleViolation("PR-061", "NHIF/SHIF number required for Kenya payroll", "register_employee_with_nhif")


def assert_pension_rate_reasonable(rate: Decimal) -> None:
	if rate < 0 or rate > Decimal("0.30"):
		raise RuleViolation(
			"PR-062",
			f"Pension contribution rate {rate} is outside acceptable range [0, 0.30]",
			"correct_pension_rate",
		)


# ---------------------------------------------------------------------------
# Leave rules
# ---------------------------------------------------------------------------

def assert_leave_days_available(entitled: Decimal, taken: Decimal, carried: Decimal, requested: Decimal) -> None:
	available = entitled + carried - taken
	if requested > available:
		raise RuleViolation(
			"PR-070",
			f"Requested {requested} days exceeds available balance {available}",
			"reduce_leave_request",
		)


def assert_leave_type_encashable(leave_type: str) -> None:
	non_encashable = {"sick", "maternity", "paternity", "study", "compassionate", "unpaid"}
	if leave_type.lower() in non_encashable:
		raise RuleViolation(
			"PR-071",
			f"Leave type '{leave_type}' cannot be encashed under standard African employment law",
			"only_annual_leave_is_encashable",
		)


def assert_carry_forward_within_limit(
	carry_days: Decimal,
	max_carry: Decimal = Decimal("30"),
) -> None:
	if carry_days > max_carry:
		raise RuleViolation(
			"PR-072",
			f"Carry-forward of {carry_days} days exceeds maximum of {max_carry}",
			"encash_or_forfeit_excess_days",
		)


# ---------------------------------------------------------------------------
# Salary advance rules
# ---------------------------------------------------------------------------

def assert_advance_within_limit(
	advance_amount: Decimal,
	monthly_salary: Decimal,
	max_months: int = 3,
) -> None:
	"""Advance capped at max_months × monthly salary."""
	limit = monthly_salary * max_months
	if advance_amount > limit:
		raise RuleViolation(
			"PR-080",
			f"Advance {advance_amount} exceeds limit of {max_months} months salary ({limit})",
			"reduce_advance_amount",
		)


def assert_advance_instalment_feasible(
	monthly_instalment: Decimal,
	net_pay_estimate: Decimal,
	max_deduction_pct: Decimal = Decimal("0.50"),
) -> None:
	"""Instalment must not exceed max_deduction_pct of net pay."""
	max_instalment = net_pay_estimate * max_deduction_pct
	if monthly_instalment > max_instalment:
		raise RuleViolation(
			"PR-081",
			f"Instalment {monthly_instalment} exceeds {max_deduction_pct*100}% of net pay {net_pay_estimate}",
			"reduce_instalment_amount",
		)


def assert_no_active_advance(existing_advances: list[dict[str, Any]], employee_id: str, tenant_id: str) -> None:
	for adv in existing_advances:
		if (adv.get("employee_id") == employee_id
				and adv.get("tenant_id") == tenant_id
				and adv.get("status") == "active"
				and not adv.get("is_deleted")):
			raise RuleViolation(
				"PR-082",
				f"Employee {employee_id} already has an active salary advance ({adv['id']})",
				"clear_existing_advance_first",
			)


# ---------------------------------------------------------------------------
# Garnishment rules
# ---------------------------------------------------------------------------

def assert_garnishment_within_legal_limit(
	disposable_earnings: Decimal,
	requested_amount: Decimal,
	max_pct: Decimal = Decimal("0.3333"),
) -> Decimal:
	"""Return capped garnishment amount per Kenya Employment Act s.19."""
	cap = (disposable_earnings * max_pct).quantize(Decimal("0.01"))
	if requested_amount > cap:
		return cap
	return requested_amount


# ---------------------------------------------------------------------------
# GL / accounting rules
# ---------------------------------------------------------------------------

def assert_journal_balanced(total_debits: Decimal, total_credits: Decimal, tolerance: Decimal = Decimal("0.01")) -> None:
	if abs(total_debits - total_credits) > tolerance:
		raise RuleViolation(
			"PR-090",
			f"Journal is not balanced: DR={total_debits} CR={total_credits} diff={abs(total_debits-total_credits)}",
			"recheck_gl_accounts",
		)


def assert_run_not_already_posted_to_gl(run: dict[str, Any]) -> None:
	if run.get("gl_posted"):
		raise RuleViolation(
			"PR-091",
			f"Run {run['id']} has already been posted to GL",
			"reverse_then_repost",
		)


# ---------------------------------------------------------------------------
# Prorated salary calculations
# ---------------------------------------------------------------------------

def calculate_prorated_salary(
	monthly_salary: Decimal,
	days_in_month: int,
	days_worked: int,
) -> Decimal:
	"""Days-based proration. Uses calendar days."""
	if days_in_month <= 0:
		return Decimal("0")
	days_worked = max(0, min(days_worked, days_in_month))
	return (monthly_salary * Decimal(days_worked) / Decimal(days_in_month)).quantize(Decimal("0.01"))


def calculate_days_worked(
	hire_date: date,
	period_start: date,
	period_end: date,
) -> tuple[int, int]:
	"""Return (days_worked, days_in_period) for a mid-month starter."""
	days_in_period = (period_end - period_start).days + 1
	if hire_date <= period_start:
		return days_in_period, days_in_period
	if hire_date > period_end:
		return 0, days_in_period
	return (period_end - hire_date).days + 1, days_in_period


def calculate_termination_proration(
	monthly_salary: Decimal,
	termination_date: date,
	period_start: date,
	period_end: date,
) -> Decimal:
	"""Pro-rate for employee terminated mid-period."""
	days_in_period = (period_end - period_start).days + 1
	if termination_date >= period_end:
		return monthly_salary
	if termination_date < period_start:
		return Decimal("0")
	days_worked = (termination_date - period_start).days + 1
	return calculate_prorated_salary(monthly_salary, days_in_period, days_worked)


# ---------------------------------------------------------------------------
# Terminal benefit calculations
# ---------------------------------------------------------------------------

def calculate_notice_pay(
	monthly_salary: Decimal,
	notice_days_owed: int,
	working_days_per_month: int = 22,
) -> Decimal:
	"""Salary for unserved notice period."""
	if notice_days_owed <= 0:
		return Decimal("0")
	return (monthly_salary / Decimal(working_days_per_month) * Decimal(notice_days_owed)).quantize(Decimal("0.01"))


def calculate_severance_ke(monthly_salary: Decimal, completed_years: int) -> Decimal:
	"""Kenya Employment Act 2007 s.35: 15 days gross per year of service."""
	if completed_years < 1:
		return Decimal("0")
	# 15 working days in a 26-day working month
	return (monthly_salary / Decimal("26") * Decimal("15") * Decimal(completed_years)).quantize(Decimal("0.01"))


def calculate_gratuity(
	monthly_salary: Decimal,
	years_of_service: Decimal,
	rate: Decimal = Decimal("0.25"),
) -> Decimal:
	"""Contractual gratuity: rate × annual_salary × years."""
	return (monthly_salary * Decimal("12") * years_of_service * rate).quantize(Decimal("0.01"))


def calculate_leave_encashment(
	monthly_salary: Decimal,
	leave_days: Decimal,
	working_days_per_month: int = 22,
) -> Decimal:
	daily = monthly_salary / Decimal(working_days_per_month)
	return (daily * leave_days).quantize(Decimal("0.01"))


# ---------------------------------------------------------------------------
# Retroactive salary adjustment
# ---------------------------------------------------------------------------

def calculate_retro_adjustment(
	old_monthly: Decimal,
	new_monthly: Decimal,
	months_affected: int,
) -> Decimal:
	"""Back-pay owed for a salary increase applied retroactively."""
	assert months_affected >= 0, "months_affected must be non-negative"
	return ((new_monthly - old_monthly) * Decimal(months_affected)).quantize(Decimal("0.01"))


# ---------------------------------------------------------------------------
# Overtime
# ---------------------------------------------------------------------------

def assert_overtime_hours_reasonable(hours: Decimal, max_hours: Decimal = Decimal("80")) -> None:
	if hours < 0:
		raise RuleViolation("PR-100", "Overtime hours cannot be negative", "correct_hours")
	if hours > max_hours:
		raise RuleViolation(
			"PR-101",
			f"Overtime of {hours} hours in a period exceeds {max_hours}-hour sanity limit",
			"verify_timesheet_data",
		)


def assert_overtime_approved(overtime: dict[str, Any]) -> None:
	if not overtime.get("approved_by"):
		raise RuleViolation("PR-102", "Overtime must be approved before payroll processing", "approve_overtime")


# ---------------------------------------------------------------------------
# Final settlement completeness
# ---------------------------------------------------------------------------

def assert_final_settlement_complete(settlement: dict[str, Any]) -> None:
	required = ["employee_id", "termination_date", "last_day_worked", "reason_for_leaving"]
	missing = [f for f in required if not settlement.get(f)]
	if missing:
		raise RuleViolation(
			"PR-110",
			f"Final settlement is missing required fields: {missing}",
			"complete_settlement_details",
		)


# ---------------------------------------------------------------------------
# Cross-cutting
# ---------------------------------------------------------------------------

def assert_actor_id_present(actor_id: str | None) -> None:
	if not actor_id:
		raise RuleViolation("PR-120", "actor_id is required for all write operations", "provide_actor_id")


def assert_currency_match(expected: str, actual: str) -> None:
	if expected.upper() != actual.upper():
		raise RuleViolation(
			"PR-121",
			f"Currency mismatch: expected {expected} got {actual}",
			"convert_or_align_currency",
		)
