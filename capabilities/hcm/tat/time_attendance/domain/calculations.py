"""
Time & Attendance — Financial & Domain Calculations

Pure functions, type-safe, comprehensive edge-case handling.
All monetary/hour values use Decimal for precision.
Tabs, not spaces. Python 3.12+.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class HoursBreakdown(NamedTuple):
	regular: Decimal
	overtime: Decimal
	double_time: Decimal
	holiday: Decimal
	total: Decimal


class PayBreakdown(NamedTuple):
	regular_pay: Decimal
	overtime_pay: Decimal
	double_time_pay: Decimal
	holiday_pay: Decimal
	gross_pay: Decimal


class LeaveEntitlement(NamedTuple):
	annual_days: Decimal
	accrued_to_date: Decimal
	used_to_date: Decimal
	balance: Decimal
	pending: Decimal
	available: Decimal


class FlexiBalance(NamedTuple):
	credit_hours: Decimal      # positive = ahead
	debit_hours: Decimal       # positive = behind (absolute value)
	net_hours: Decimal         # credit - debit (signed)


ZERO = Decimal("0")
TWO_PLACES = Decimal("0.01")
HALF_DAY = Decimal("0.5")


def _q(v: Decimal) -> Decimal:
	"""Quantize to 2 decimal places, ROUND_HALF_UP."""
	return v.quantize(TWO_PLACES, rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Core hour calculations
# ---------------------------------------------------------------------------

def calculate_worked_hours(clock_in: datetime, clock_out: datetime, break_minutes: int = 0) -> Decimal:
	"""
	Net hours worked after deducting unpaid break time.

	Args:
		clock_in: Inclusive start of attendance.
		clock_out: Inclusive end of attendance.
		break_minutes: Unpaid break minutes to deduct.

	Returns:
		Decimal hours, quantized to 2 dp. Minimum 0.
	"""
	gross_seconds = (clock_out - clock_in).total_seconds()
	net_seconds = max(gross_seconds - break_minutes * 60, 0)
	return _q(Decimal(str(net_seconds)) / Decimal("3600"))


def calculate_hours_breakdown(
	worked_hours: Decimal,
	daily_ot_threshold: Decimal = Decimal("8"),
	daily_double_threshold: Decimal = Decimal("12"),
	is_holiday: bool = False,
) -> HoursBreakdown:
	"""
	Split worked hours into regular / overtime / double-time / holiday buckets.

	Double-time kicks in above daily_double_threshold (e.g., >12h/day in California).
	Holiday hours are flagged separately; they still use the threshold logic for OT/DT
	but the holiday multiplier is applied at pay calculation time.
	"""
	if worked_hours <= ZERO:
		return HoursBreakdown(ZERO, ZERO, ZERO, ZERO, ZERO)

	if is_holiday:
		return HoursBreakdown(
			regular=ZERO,
			overtime=ZERO,
			double_time=ZERO,
			holiday=worked_hours,
			total=worked_hours,
		)

	regular = min(worked_hours, daily_ot_threshold)
	overtime_raw = max(worked_hours - daily_ot_threshold, ZERO)
	overtime = min(overtime_raw, daily_double_threshold - daily_ot_threshold)
	double_time = max(worked_hours - daily_double_threshold, ZERO)

	return HoursBreakdown(
		regular=_q(regular),
		overtime=_q(overtime),
		double_time=_q(double_time),
		holiday=ZERO,
		total=_q(worked_hours),
	)


def calculate_weekly_hours(daily_hours: list[Decimal]) -> Decimal:
	"""Sum daily hours for a 7-day week."""
	return _q(sum(daily_hours, ZERO))


def calculate_weekly_hours_breakdown(
	daily_breakdowns: list[HoursBreakdown],
	weekly_ot_threshold: Decimal = Decimal("40"),
) -> HoursBreakdown:
	"""
	Re-bucket hours once weekly totals are known.
	California uses daily OT; federal (FLSA) uses weekly OT.
	This implements the FLSA weekly model: first daily totals, then weekly excess.
	"""
	total_worked = sum((bd.total for bd in daily_breakdowns), ZERO)
	total_holiday = sum((bd.holiday for bd in daily_breakdowns), ZERO)
	non_holiday = total_worked - total_holiday

	regular = min(non_holiday, weekly_ot_threshold)
	overtime = max(non_holiday - weekly_ot_threshold, ZERO)

	return HoursBreakdown(
		regular=_q(regular),
		overtime=_q(overtime),
		double_time=ZERO,
		holiday=_q(total_holiday),
		total=_q(total_worked),
	)


# ---------------------------------------------------------------------------
# Pay calculations
# ---------------------------------------------------------------------------

def calculate_pay(
	breakdown: HoursBreakdown,
	hourly_rate: Decimal,
	overtime_multiplier: Decimal = Decimal("1.5"),
	double_time_multiplier: Decimal = Decimal("2.0"),
	holiday_multiplier: Decimal = Decimal("2.0"),
) -> PayBreakdown:
	"""
	Compute gross pay from hours breakdown and rate.

	Args:
		breakdown: Output of calculate_hours_breakdown or calculate_weekly_hours_breakdown.
		hourly_rate: Base hourly rate.
		overtime_multiplier: Multiplier for overtime hours (default 1.5×).
		double_time_multiplier: Multiplier for double-time hours (default 2.0×).
		holiday_multiplier: Multiplier for holiday hours (default 2.0×).
	"""
	regular_pay = _q(breakdown.regular * hourly_rate)
	overtime_pay = _q(breakdown.overtime * hourly_rate * overtime_multiplier)
	double_time_pay = _q(breakdown.double_time * hourly_rate * double_time_multiplier)
	holiday_pay = _q(breakdown.holiday * hourly_rate * holiday_multiplier)
	gross_pay = _q(regular_pay + overtime_pay + double_time_pay + holiday_pay)

	return PayBreakdown(
		regular_pay=regular_pay,
		overtime_pay=overtime_pay,
		double_time_pay=double_time_pay,
		holiday_pay=holiday_pay,
		gross_pay=gross_pay,
	)


def calculate_on_call_pay(
	on_call_hours: Decimal,
	hourly_rate: Decimal,
	on_call_multiplier: Decimal = Decimal("0.5"),
) -> Decimal:
	"""On-call premium: fraction of hourly rate for standby hours."""
	return _q(on_call_hours * hourly_rate * on_call_multiplier)


# ---------------------------------------------------------------------------
# Leave entitlement calculations
# ---------------------------------------------------------------------------

def calculate_leave_entitlement(
	start_date: date,
	reference_date: date,
	annual_days: Decimal,
	used_days: Decimal,
	pending_days: Decimal,
) -> LeaveEntitlement:
	"""
	Compute leave entitlement accrual and balance.

	Accrual is pro-rated by the fraction of the year elapsed.
	Partial years accrue from first day of employment (inclusive).
	"""
	year_start = date(reference_date.year, 1, 1)
	year_end = date(reference_date.year, 12, 31)
	days_in_year = Decimal(str((year_end - year_start).days + 1))

	# Earliest of reference_date and year_end
	effective_ref = min(reference_date, year_end)
	effective_start = max(start_date, year_start)

	if effective_ref < effective_start:
		accrued = ZERO
	else:
		days_worked_in_year = Decimal(str((effective_ref - effective_start).days + 1))
		accrued = _q(annual_days * days_worked_in_year / days_in_year)

	balance = max(_q(accrued - used_days), ZERO)
	available = max(_q(balance - pending_days), ZERO)

	return LeaveEntitlement(
		annual_days=annual_days,
		accrued_to_date=accrued,
		used_to_date=used_days,
		balance=balance,
		pending=pending_days,
		available=available,
	)


def calculate_prorata_leave(
	full_time_annual_days: Decimal,
	fte: Decimal,
) -> Decimal:
	"""
	Part-time pro-rata leave entitlement, rounded to nearest half-day.
	"""
	raw = full_time_annual_days * fte
	# Round to nearest 0.5
	return (raw * 2).to_integral_value(rounding=ROUND_HALF_UP) / 2


def working_days_between(start: date, end: date, public_holidays: list[date] | None = None) -> int:
	"""
	Count Mon–Fri working days between start (inclusive) and end (inclusive),
	excluding any dates in public_holidays.
	"""
	holidays = set(public_holidays or [])
	days = 0
	current = start
	while current <= end:
		if current.weekday() < 5 and current not in holidays:  # Mon=0 … Fri=4
			days += 1
		current += timedelta(days=1)
	return days


# ---------------------------------------------------------------------------
# Flexitime calculations
# ---------------------------------------------------------------------------

def calculate_flexi_balance(
	worked_hours_log: list[tuple[date, Decimal]],
	standard_daily_hours: Decimal,
	carry_forward_hours: Decimal = ZERO,
) -> FlexiBalance:
	"""
	Compute cumulative flexitime balance from a log of (date, hours_worked) tuples.

	Args:
		worked_hours_log: List of (date, hours_worked) for each day in the period.
		standard_daily_hours: Contracted standard hours per day.
		carry_forward_hours: Opening balance carried from previous period (signed).
	"""
	net = carry_forward_hours
	for _, hours in worked_hours_log:
		net += hours - standard_daily_hours

	net = _q(net)
	credit = max(net, ZERO)
	debit = max(-net, ZERO)
	return FlexiBalance(credit_hours=credit, debit_hours=debit, net_hours=net)


def flexi_hours_to_take(
	flexi_balance: Decimal,
	policy_max_carry: Decimal = Decimal("16"),
) -> Decimal:
	"""
	Hours that must be taken before year-end if balance exceeds policy_max_carry.
	Returns 0 if within limits.
	"""
	return max(_q(flexi_balance - policy_max_carry), ZERO)


# ---------------------------------------------------------------------------
# Annualised hours reconciliation
# ---------------------------------------------------------------------------

def annualised_hours_remaining(
	contracted_annual_hours: Decimal,
	hours_worked: Decimal,
) -> Decimal:
	"""Hours still to be worked to fulfil annual contract."""
	return max(_q(contracted_annual_hours - hours_worked), ZERO)


def annualised_hours_owed(
	contracted_annual_hours: Decimal,
	hours_worked: Decimal,
) -> Decimal:
	"""Negative value means employee has already exceeded contract (banked time)."""
	return _q(contracted_annual_hours - hours_worked)


def calculate_annualised_expected_hours(
	contracted_annual_hours: Decimal,
	weeks_elapsed: Decimal,
	total_contract_weeks: Decimal = Decimal("52"),
) -> Decimal:
	"""Expected hours worked at the current point in the contract year."""
	return _q(contracted_annual_hours * weeks_elapsed / total_contract_weeks)


# ---------------------------------------------------------------------------
# Shift differential calculations
# ---------------------------------------------------------------------------

def calculate_shift_differential(
	hours_worked: Decimal,
	hourly_rate: Decimal,
	shift_type: str,
	night_shift_premium: Decimal = Decimal("0.15"),
	weekend_premium: Decimal = Decimal("0.10"),
) -> Decimal:
	"""
	Additional pay for non-standard shift types.

	Args:
		shift_type: One of 'day', 'evening', 'night', 'weekend', 'overnight'.
		night_shift_premium: Fraction of hourly_rate added per night-shift hour.
		weekend_premium: Fraction of hourly_rate added per weekend hour.
	"""
	premiums: dict[str, Decimal] = {
		"day": ZERO,
		"evening": Decimal("0.05"),
		"night": night_shift_premium,
		"weekend": weekend_premium,
		"overnight": night_shift_premium + Decimal("0.05"),
	}
	premium_rate = premiums.get(shift_type, ZERO)
	return _q(hours_worked * hourly_rate * premium_rate)


# ---------------------------------------------------------------------------
# Absence / sickness calculations
# ---------------------------------------------------------------------------

def calculate_bradford_factor(
	absence_instances: int,
	total_days_absent: int,
) -> int:
	"""
	Bradford Factor = S² × D
	S = number of separate absence instances.
	D = total days of absence.

	Higher scores indicate problematic short-burst absenteeism.
	"""
	return (absence_instances ** 2) * total_days_absent


BRADFORD_THRESHOLDS: dict[str, int] = {
	"green": 0,
	"amber": 51,
	"red": 201,
	"critical": 451,
}


def bradford_rating(score: int) -> str:
	"""Map a Bradford Factor score to a risk category."""
	if score >= BRADFORD_THRESHOLDS["critical"]:
		return "critical"
	if score >= BRADFORD_THRESHOLDS["red"]:
		return "red"
	if score >= BRADFORD_THRESHOLDS["amber"]:
		return "amber"
	return "green"


# ---------------------------------------------------------------------------
# TOIL (Time Off In Lieu) calculations
# ---------------------------------------------------------------------------

def calculate_toil_balance(
	toil_accrued: Decimal,
	toil_used: Decimal,
	carry_forward: Decimal = ZERO,
	expiry_multiplier: Decimal = ZERO,
) -> Decimal:
	"""
	Net TOIL balance.

	expiry_multiplier: If TOIL expires after a period, subtract accrued × multiplier.
	"""
	return _q(carry_forward + toil_accrued - toil_used - (toil_accrued * expiry_multiplier))


def calculate_toil_from_overtime(
	overtime_hours: Decimal,
	overtime_multiplier: Decimal = Decimal("1.0"),
) -> Decimal:
	"""Convert approved overtime hours into TOIL credits."""
	return _q(overtime_hours * overtime_multiplier)


# ---------------------------------------------------------------------------
# Comp-time calculations
# ---------------------------------------------------------------------------

def calculate_comp_time_balance(
	comp_time_earned: Decimal,
	comp_time_used: Decimal,
	max_accrual: Decimal = Decimal("240"),
) -> Decimal:
	"""
	Net comp-time balance, capped at max_accrual.
	Federal employees: max 240h (FLSA section 7(o)).
	"""
	balance = _q(comp_time_earned - comp_time_used)
	return min(balance, max_accrual)


# ---------------------------------------------------------------------------
# Roster / schedule utilities
# ---------------------------------------------------------------------------

def hours_per_week_from_shifts(
	shifts: list[tuple[datetime, datetime]],
) -> Decimal:
	"""
	Sum shift durations for a list of (start, end) tuples within a week.
	Night shifts spanning midnight are handled correctly.
	"""
	total = sum(
		(_q(Decimal(str((end - start).total_seconds())) / Decimal("3600")) for start, end in shifts),
		ZERO,
	)
	return _q(total)


def roster_coverage_gap(
	required_headcount: dict[str, int],
	scheduled_headcount: dict[str, int],
) -> dict[str, int]:
	"""
	Return the headcount shortfall for each slot key.
	Positive value = understaffed; 0 or negative = covered.
	"""
	return {
		slot: max(required_headcount.get(slot, 0) - scheduled_headcount.get(slot, 0), 0)
		for slot in required_headcount
	}
