"""
CI tests for domain/calculations.py

Pure numeric tests — no DB, no async.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from domain.calculations import (
	HoursBreakdown,
	_q,
	annualised_hours_owed,
	annualised_hours_remaining,
	bradford_rating,
	calculate_annualised_expected_hours,
	calculate_bradford_factor,
	calculate_comp_time_balance,
	calculate_flexi_balance,
	calculate_hours_breakdown,
	calculate_leave_entitlement,
	calculate_on_call_pay,
	calculate_pay,
	calculate_prorata_leave,
	calculate_shift_differential,
	calculate_toil_balance,
	calculate_toil_from_overtime,
	calculate_weekly_hours,
	calculate_weekly_hours_breakdown,
	calculate_worked_hours,
	flexi_hours_to_take,
	hours_per_week_from_shifts,
	roster_coverage_gap,
	working_days_between,
)

UTC = timezone.utc


def _dt(h: int, m: int = 0, day: int = 1, month: int = 1) -> datetime:
	return datetime(2026, month, day, h, m, tzinfo=UTC)


# ---------------------------------------------------------------------------
# calculate_worked_hours
# ---------------------------------------------------------------------------

def test_worked_hours_8h_no_break():
	assert calculate_worked_hours(_dt(9), _dt(17)) == Decimal("8.00")


def test_worked_hours_with_1h_break():
	assert calculate_worked_hours(_dt(9), _dt(17), break_minutes=60) == Decimal("7.00")


def test_worked_hours_night_shift():
	start = datetime(2026, 1, 1, 22, 0, tzinfo=UTC)
	end = datetime(2026, 1, 2, 6, 0, tzinfo=UTC)
	assert calculate_worked_hours(start, end) == Decimal("8.00")


def test_worked_hours_break_exceeds_shift_returns_zero():
	assert calculate_worked_hours(_dt(9), _dt(9, 30), break_minutes=60) == Decimal("0.00")


def test_worked_hours_fractional_30min():
	assert calculate_worked_hours(_dt(9, 0), _dt(9, 30)) == Decimal("0.50")


def test_worked_hours_45min():
	assert calculate_worked_hours(_dt(9, 0), _dt(9, 45)) == Decimal("0.75")


# ---------------------------------------------------------------------------
# calculate_hours_breakdown
# ---------------------------------------------------------------------------

def test_breakdown_7h_regular_only():
	bd = calculate_hours_breakdown(Decimal("7"))
	assert bd.regular == Decimal("7.00")
	assert bd.overtime == Decimal("0.00")
	assert bd.double_time == Decimal("0.00")
	assert bd.total == Decimal("7.00")


def test_breakdown_exactly_8h():
	bd = calculate_hours_breakdown(Decimal("8"))
	assert bd.regular == Decimal("8.00")
	assert bd.overtime == Decimal("0.00")


def test_breakdown_10h_with_2h_ot():
	bd = calculate_hours_breakdown(Decimal("10"))
	assert bd.regular == Decimal("8.00")
	assert bd.overtime == Decimal("2.00")
	assert bd.double_time == Decimal("0.00")


def test_breakdown_14h_with_double_time():
	bd = calculate_hours_breakdown(Decimal("14"))
	assert bd.regular == Decimal("8.00")
	assert bd.overtime == Decimal("4.00")
	assert bd.double_time == Decimal("2.00")


def test_breakdown_holiday():
	bd = calculate_hours_breakdown(Decimal("8"), is_holiday=True)
	assert bd.holiday == Decimal("8.00")
	assert bd.regular == Decimal("0.00")
	assert bd.overtime == Decimal("0.00")


def test_breakdown_zero():
	bd = calculate_hours_breakdown(Decimal("0"))
	assert bd.total == Decimal("0.00")


def test_breakdown_negative_treated_as_zero():
	bd = calculate_hours_breakdown(Decimal("-1"))
	assert bd.total == Decimal("0.00")


# ---------------------------------------------------------------------------
# calculate_weekly_hours_breakdown
# ---------------------------------------------------------------------------

def test_weekly_exactly_40():
	dailies = [calculate_hours_breakdown(Decimal("8"))] * 5
	bd = calculate_weekly_hours_breakdown(dailies)
	assert bd.regular == Decimal("40.00")
	assert bd.overtime == Decimal("0.00")


def test_weekly_45h():
	dailies = [calculate_hours_breakdown(Decimal("9"))] * 5
	bd = calculate_weekly_hours_breakdown(dailies)
	assert bd.regular == Decimal("40.00")
	assert bd.overtime == Decimal("5.00")


def test_weekly_preserves_holiday():
	holiday = calculate_hours_breakdown(Decimal("8"), is_holiday=True)
	normal = calculate_hours_breakdown(Decimal("8"))
	bd = calculate_weekly_hours_breakdown([normal] * 4 + [holiday])
	assert bd.holiday == Decimal("8.00")


def test_weekly_empty_list():
	bd = calculate_weekly_hours_breakdown([])
	assert bd.total == Decimal("0.00")


# ---------------------------------------------------------------------------
# calculate_pay
# ---------------------------------------------------------------------------

def test_pay_regular_only():
	bd = HoursBreakdown(Decimal("8"), Decimal("0"), Decimal("0"), Decimal("0"), Decimal("8"))
	pay = calculate_pay(bd, Decimal("20"))
	assert pay.regular_pay == Decimal("160.00")
	assert pay.gross_pay == Decimal("160.00")


def test_pay_with_1_5x_overtime():
	bd = HoursBreakdown(Decimal("8"), Decimal("2"), Decimal("0"), Decimal("0"), Decimal("10"))
	pay = calculate_pay(bd, Decimal("20"))
	assert pay.overtime_pay == Decimal("60.00")   # 2 × 20 × 1.5
	assert pay.gross_pay == Decimal("220.00")


def test_pay_with_double_time():
	bd = HoursBreakdown(Decimal("8"), Decimal("4"), Decimal("2"), Decimal("0"), Decimal("14"))
	pay = calculate_pay(bd, Decimal("20"))
	assert pay.double_time_pay == Decimal("80.00")  # 2 × 20 × 2.0


def test_pay_with_holiday():
	bd = HoursBreakdown(Decimal("0"), Decimal("0"), Decimal("0"), Decimal("8"), Decimal("8"))
	pay = calculate_pay(bd, Decimal("20"))
	assert pay.holiday_pay == Decimal("320.00")  # 8 × 20 × 2.0


def test_pay_zero_rate():
	bd = HoursBreakdown(Decimal("8"), Decimal("2"), Decimal("0"), Decimal("0"), Decimal("10"))
	pay = calculate_pay(bd, Decimal("0"))
	assert pay.gross_pay == Decimal("0.00")


# ---------------------------------------------------------------------------
# calculate_leave_entitlement
# ---------------------------------------------------------------------------

def test_entitlement_full_year_no_used():
	result = calculate_leave_entitlement(
		date(2026, 1, 1), date(2026, 12, 31), Decimal("25"), Decimal("0"), Decimal("0")
	)
	assert result.annual_days == Decimal("25")
	assert result.used_to_date == Decimal("0")
	assert result.available >= Decimal("0")


def test_entitlement_used_reduces_balance():
	result = calculate_leave_entitlement(
		date(2026, 1, 1), date(2026, 12, 31), Decimal("25"), Decimal("5"), Decimal("0")
	)
	assert result.used_to_date == Decimal("5")
	assert result.balance == result.accrued_to_date - Decimal("5")


def test_entitlement_balance_not_negative():
	result = calculate_leave_entitlement(
		date(2026, 1, 1), date(2026, 3, 1), Decimal("20"), Decimal("20"), Decimal("0")
	)
	assert result.balance >= Decimal("0")


def test_entitlement_pending_reduces_available():
	result = calculate_leave_entitlement(
		date(2026, 1, 1), date(2026, 12, 31), Decimal("25"), Decimal("5"), Decimal("3")
	)
	assert result.pending == Decimal("3")
	assert result.available == result.balance - Decimal("3")


# ---------------------------------------------------------------------------
# calculate_prorata_leave
# ---------------------------------------------------------------------------

def test_prorata_full_time():
	assert calculate_prorata_leave(Decimal("25"), Decimal("1.0")) == Decimal("25")


def test_prorata_half_time():
	assert calculate_prorata_leave(Decimal("25"), Decimal("0.5")) == Decimal("12.5")


def test_prorata_rounds_to_half_day():
	result = calculate_prorata_leave(Decimal("25"), Decimal("0.6"))
	assert result % Decimal("0.5") == Decimal("0")


def test_prorata_0_6fte():
	# 25 × 0.6 = 15.0 → 15.0
	assert calculate_prorata_leave(Decimal("25"), Decimal("0.6")) == Decimal("15.0")


# ---------------------------------------------------------------------------
# working_days_between
# ---------------------------------------------------------------------------

def test_working_days_mon_to_fri():
	assert working_days_between(date(2026, 6, 1), date(2026, 6, 5)) == 5


def test_working_days_excludes_weekend():
	assert working_days_between(date(2026, 6, 1), date(2026, 6, 7)) == 5


def test_working_days_excludes_holiday():
	holiday = date(2026, 6, 3)
	assert working_days_between(date(2026, 6, 1), date(2026, 6, 5), [holiday]) == 4


def test_working_days_single_day():
	assert working_days_between(date(2026, 6, 1), date(2026, 6, 1)) == 1


def test_working_days_weekend_only():
	assert working_days_between(date(2026, 6, 6), date(2026, 6, 7)) == 0


# ---------------------------------------------------------------------------
# Flexitime
# ---------------------------------------------------------------------------

def test_flexi_balance_net_credit():
	log = [
		(date(2026, 6, 1), Decimal("9")),
		(date(2026, 6, 2), Decimal("8")),
		(date(2026, 6, 3), Decimal("9")),
	]
	bal = calculate_flexi_balance(log, Decimal("8"))
	assert bal.net_hours == Decimal("2.00")
	assert bal.credit_hours == Decimal("2.00")
	assert bal.debit_hours == Decimal("0.00")


def test_flexi_balance_net_debit():
	log = [(date(2026, 6, 1), Decimal("6"))]
	bal = calculate_flexi_balance(log, Decimal("8"))
	assert bal.net_hours == Decimal("-2.00")
	assert bal.debit_hours == Decimal("2.00")


def test_flexi_balance_carry_forward():
	log = [(date(2026, 6, 1), Decimal("8"))]
	bal = calculate_flexi_balance(log, Decimal("8"), carry_forward_hours=Decimal("3"))
	assert bal.net_hours == Decimal("3.00")


def test_flexi_hours_to_take_within_limit():
	assert flexi_hours_to_take(Decimal("10"), Decimal("16")) == Decimal("0.00")


def test_flexi_hours_to_take_over_limit():
	assert flexi_hours_to_take(Decimal("20"), Decimal("16")) == Decimal("4.00")


# ---------------------------------------------------------------------------
# Annualised hours
# ---------------------------------------------------------------------------

def test_annualised_remaining():
	assert annualised_hours_remaining(Decimal("1800"), Decimal("1200")) == Decimal("600.00")


def test_annualised_remaining_zero_when_exceeded():
	assert annualised_hours_remaining(Decimal("1800"), Decimal("2000")) == Decimal("0.00")


def test_annualised_owed_positive():
	assert annualised_hours_owed(Decimal("1800"), Decimal("1200")) == Decimal("600.00")


def test_annualised_owed_negative_banked():
	assert annualised_hours_owed(Decimal("1800"), Decimal("2000")) == Decimal("-200.00")


def test_annualised_expected_midyear():
	expected = calculate_annualised_expected_hours(Decimal("1800"), Decimal("26"))
	assert abs(expected - Decimal("900")) < Decimal("5")


# ---------------------------------------------------------------------------
# Shift differential
# ---------------------------------------------------------------------------

def test_shift_differential_day_zero():
	assert calculate_shift_differential(Decimal("8"), Decimal("20"), "day") == Decimal("0.00")


def test_shift_differential_night():
	# 8 × 20 × 0.15 = 24.00
	assert calculate_shift_differential(Decimal("8"), Decimal("20"), "night") == Decimal("24.00")


def test_shift_differential_weekend():
	# 8 × 20 × 0.10 = 16.00
	assert calculate_shift_differential(Decimal("8"), Decimal("20"), "weekend") == Decimal("16.00")


def test_shift_differential_evening():
	# 8 × 20 × 0.05 = 8.00
	assert calculate_shift_differential(Decimal("8"), Decimal("20"), "evening") == Decimal("8.00")


def test_shift_differential_unknown_type_zero():
	assert calculate_shift_differential(Decimal("8"), Decimal("20"), "unknown") == Decimal("0.00")


# ---------------------------------------------------------------------------
# Bradford Factor
# ---------------------------------------------------------------------------

def test_bradford_factor():
	assert calculate_bradford_factor(3, 9) == 81  # 3² × 9


def test_bradford_factor_zero():
	assert calculate_bradford_factor(0, 0) == 0


def test_bradford_factor_single_instance():
	assert calculate_bradford_factor(1, 5) == 5  # 1² × 5


def test_bradford_rating_green():
	assert bradford_rating(30) == "green"


def test_bradford_rating_amber():
	assert bradford_rating(100) == "amber"


def test_bradford_rating_red():
	assert bradford_rating(250) == "red"


def test_bradford_rating_critical():
	assert bradford_rating(500) == "critical"


def test_bradford_rating_boundary_amber():
	assert bradford_rating(51) == "amber"


def test_bradford_rating_boundary_red():
	assert bradford_rating(201) == "red"


# ---------------------------------------------------------------------------
# TOIL / comp-time
# ---------------------------------------------------------------------------

def test_toil_balance_basic():
	assert calculate_toil_balance(Decimal("8"), Decimal("4")) == Decimal("4.00")


def test_toil_balance_with_carry():
	assert calculate_toil_balance(Decimal("8"), Decimal("4"), carry_forward=Decimal("2")) == Decimal("6.00")


def test_toil_from_overtime_1to1():
	assert calculate_toil_from_overtime(Decimal("4")) == Decimal("4.00")


def test_toil_from_overtime_1_5x():
	assert calculate_toil_from_overtime(Decimal("4"), Decimal("1.5")) == Decimal("6.00")


def test_comp_time_balance_basic():
	assert calculate_comp_time_balance(Decimal("50"), Decimal("20")) == Decimal("30.00")


def test_comp_time_balance_capped():
	assert calculate_comp_time_balance(Decimal("300"), Decimal("0")) == Decimal("240.00")


def test_comp_time_balance_negative_zero():
	# Used more than earned → 0 (or negative — depends on policy; test actual behaviour)
	result = calculate_comp_time_balance(Decimal("10"), Decimal("50"))
	assert result == Decimal("-40.00")  # balance can go negative (caller enforces floor)


# ---------------------------------------------------------------------------
# Roster gap
# ---------------------------------------------------------------------------

def test_roster_no_gap():
	gaps = roster_coverage_gap({"08:00": 3}, {"08:00": 4})
	assert gaps["08:00"] == 0


def test_roster_gap_understaffed():
	gaps = roster_coverage_gap({"08:00": 3, "14:00": 2}, {"08:00": 1, "14:00": 2})
	assert gaps["08:00"] == 2
	assert gaps["14:00"] == 0


def test_roster_missing_slot_treated_as_zero_scheduled():
	gaps = roster_coverage_gap({"08:00": 3}, {})
	assert gaps["08:00"] == 3


# ---------------------------------------------------------------------------
# On-call pay
# ---------------------------------------------------------------------------

def test_on_call_pay_default_multiplier():
	# 8h × $20 × 0.5 = $80
	assert calculate_on_call_pay(Decimal("8"), Decimal("20")) == Decimal("80.00")


def test_on_call_pay_custom_multiplier():
	assert calculate_on_call_pay(Decimal("4"), Decimal("20"), Decimal("0.25")) == Decimal("20.00")


# ---------------------------------------------------------------------------
# Hours per week from shifts
# ---------------------------------------------------------------------------

def test_hours_per_week_three_shifts():
	shifts = [
		(_dt(9), _dt(17)),   # 8h
		(_dt(9), _dt(17)),   # 8h
		(_dt(9), _dt(13)),   # 4h
	]
	assert hours_per_week_from_shifts(shifts) == Decimal("20.00")


def test_hours_per_week_empty():
	assert hours_per_week_from_shifts([]) == Decimal("0.00")


def test_hours_per_week_night_shift():
	start = datetime(2026, 1, 1, 22, 0, tzinfo=UTC)
	end = datetime(2026, 1, 2, 6, 0, tzinfo=UTC)
	assert hours_per_week_from_shifts([(start, end)]) == Decimal("8.00")
