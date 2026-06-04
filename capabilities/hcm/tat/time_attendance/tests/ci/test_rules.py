"""
CI tests for domain/rules.py

Pure unit tests — no DB, no network, no async.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal

import pytest

from domain.rules import (
	RuleViolation,
	assert_annualised_hours_deficit_manageable,
	assert_biometric_confidence,
	assert_clock_in_before_clock_out,
	assert_core_hours_covered,
	assert_device_registered,
	assert_fmla_eligibility,
	assert_import_row_valid,
	assert_leave_balance_sufficient,
	assert_leave_dates_valid,
	assert_leave_not_overlapping,
	assert_maximum_consecutive_days,
	assert_maximum_weekly_hours,
	assert_medical_certificate_for_extended_sick,
	assert_minimum_rest_between_shifts,
	assert_night_shift_midnight_span,
	assert_no_cross_tenant_access,
	assert_not_already_clocked_in,
	assert_not_already_clocked_out,
	assert_overtime_threshold_positive,
	assert_shift_duration_reasonable,
	assert_timesheet_approved_before_export,
	assert_within_geofence,
	assert_zero_hours_contract_minimum,
	calculate_daily_overtime,
	calculate_flexitime_credit,
	calculate_prorata_entitlement,
	calculate_toil_accrual,
	calculate_weekly_overtime,
	get_holiday_date_in_tz,
	is_night_shift,
)

UTC = timezone.utc


def _dt(h: int, m: int = 0, d: int = 1) -> datetime:
	return datetime(2026, 1, d, h, m, tzinfo=UTC)


# ---------------------------------------------------------------------------
# RuleViolation
# ---------------------------------------------------------------------------

def test_rule_violation_attrs():
	exc = RuleViolation("test_rule", "bad thing", "fix it")
	assert exc.rule_name == "test_rule"
	assert exc.reason == "bad thing"
	assert exc.required_action == "fix it"
	assert "test_rule" in str(exc)


def test_rule_violation_default_action():
	exc = RuleViolation("r", "reason")
	assert exc.required_action == ""


# ---------------------------------------------------------------------------
# Cross-tenant
# ---------------------------------------------------------------------------

def test_cross_tenant_raises():
	with pytest.raises(RuleViolation) as exc_info:
		assert_no_cross_tenant_access("tenant-A", "tenant-B")
	assert exc_info.value.rule_name == "cross_tenant_access_denied"


def test_same_tenant_ok():
	assert_no_cross_tenant_access("tenant-A", "tenant-A")


# ---------------------------------------------------------------------------
# Clock-in / clock-out
# ---------------------------------------------------------------------------

def test_clock_in_before_out_ok():
	assert_clock_in_before_clock_out(_dt(9), _dt(17))


def test_clock_out_before_in_raises():
	with pytest.raises(RuleViolation):
		assert_clock_in_before_clock_out(_dt(17), _dt(9))


def test_same_time_raises():
	ts = _dt(9)
	with pytest.raises(RuleViolation):
		assert_clock_in_before_clock_out(ts, ts)


def test_already_clocked_in_raises():
	with pytest.raises(RuleViolation) as ei:
		assert_not_already_clocked_in(_dt(9))
	assert ei.value.rule_name == "already_clocked_in"


def test_not_clocked_in_ok():
	assert_not_already_clocked_in(None)


def test_already_clocked_out_raises():
	with pytest.raises(RuleViolation):
		assert_not_already_clocked_out(_dt(17))


def test_not_clocked_out_ok():
	assert_not_already_clocked_out(None)


def test_shift_duration_ok():
	assert_shift_duration_reasonable(_dt(8), _dt(16))


def test_shift_duration_excessive_raises():
	with pytest.raises(RuleViolation):
		# 24-hour shift
		assert_shift_duration_reasonable(_dt(8, d=1), _dt(8, d=2))


# ---------------------------------------------------------------------------
# Night shift
# ---------------------------------------------------------------------------

def test_is_night_shift_false():
	assert not is_night_shift(_dt(9), _dt(17))


def test_is_night_shift_true():
	assert is_night_shift(_dt(22), _dt(6, d=2))


def test_night_shift_midnight_span_ok():
	assert_night_shift_midnight_span(_dt(22), _dt(6, d=2))


def test_night_shift_multi_day_raises():
	with pytest.raises(RuleViolation):
		assert_night_shift_midnight_span(
			datetime(2026, 1, 1, 22, tzinfo=UTC),
			datetime(2026, 1, 3, 6, tzinfo=UTC),
		)


# ---------------------------------------------------------------------------
# Overtime
# ---------------------------------------------------------------------------

def test_overtime_threshold_positive_ok():
	assert_overtime_threshold_positive(8.0)


def test_overtime_threshold_zero_raises():
	with pytest.raises(RuleViolation):
		assert_overtime_threshold_positive(0)


def test_overtime_threshold_negative_raises():
	with pytest.raises(RuleViolation):
		assert_overtime_threshold_positive(-1)


def test_calculate_daily_overtime_none():
	assert calculate_daily_overtime(Decimal("7")) == Decimal("0")


def test_calculate_daily_overtime_some():
	assert calculate_daily_overtime(Decimal("10")) == Decimal("2")


def test_calculate_weekly_overtime_none():
	assert calculate_weekly_overtime(Decimal("38")) == Decimal("0")


def test_calculate_weekly_overtime_some():
	assert calculate_weekly_overtime(Decimal("45")) == Decimal("5")


# ---------------------------------------------------------------------------
# Leave rules
# ---------------------------------------------------------------------------

def test_leave_dates_valid_ok():
	assert_leave_dates_valid(date(2026, 6, 1), date(2026, 6, 5))


def test_leave_end_before_start_raises():
	with pytest.raises(RuleViolation):
		assert_leave_dates_valid(date(2026, 6, 5), date(2026, 6, 1))


def test_leave_same_day_ok():
	assert_leave_dates_valid(date(2026, 6, 1), date(2026, 6, 1))


def test_leave_balance_sufficient():
	assert_leave_balance_sufficient(Decimal("10"), Decimal("5"))


def test_leave_balance_exact_ok():
	assert_leave_balance_sufficient(Decimal("5"), Decimal("5"))


def test_leave_balance_insufficient_raises():
	with pytest.raises(RuleViolation):
		assert_leave_balance_sufficient(Decimal("3"), Decimal("5"))


def test_leave_no_overlap_ok():
	existing = [{"start_date": date(2026, 7, 1), "end_date": date(2026, 7, 5)}]
	assert_leave_not_overlapping(existing, date(2026, 7, 10), date(2026, 7, 12))


def test_leave_adjacent_no_overlap():
	existing = [{"start_date": date(2026, 7, 1), "end_date": date(2026, 7, 5)}]
	assert_leave_not_overlapping(existing, date(2026, 7, 6), date(2026, 7, 8))


def test_leave_overlap_raises():
	existing = [{"start_date": date(2026, 7, 1), "end_date": date(2026, 7, 10)}]
	with pytest.raises(RuleViolation):
		assert_leave_not_overlapping(existing, date(2026, 7, 8), date(2026, 7, 12))


def test_medical_cert_not_needed_short_sick():
	assert_medical_certificate_for_extended_sick("sick", 2, False)


def test_medical_cert_needed_raises():
	with pytest.raises(RuleViolation):
		assert_medical_certificate_for_extended_sick("sick", 5, False)


def test_medical_cert_attached_ok():
	assert_medical_certificate_for_extended_sick("sick", 5, True)


def test_medical_cert_vacation_exempt():
	assert_medical_certificate_for_extended_sick("vacation", 20, False)


# ---------------------------------------------------------------------------
# TOIL / pro-rata
# ---------------------------------------------------------------------------

def test_prorata_full_time():
	result = calculate_prorata_entitlement(Decimal("25"), Decimal("1"))
	assert result == Decimal("25")


def test_prorata_half_time():
	result = calculate_prorata_entitlement(Decimal("25"), Decimal("0.5"))
	assert result == Decimal("12.5")


def test_prorata_invalid_fte_zero_raises():
	with pytest.raises(RuleViolation):
		calculate_prorata_entitlement(Decimal("25"), Decimal("0"))


def test_prorata_invalid_fte_over_one_raises():
	with pytest.raises(RuleViolation):
		calculate_prorata_entitlement(Decimal("25"), Decimal("1.5"))


def test_toil_accrual_1to1():
	assert calculate_toil_accrual(Decimal("4")) == Decimal("4.00")


def test_toil_accrual_with_multiplier():
	assert calculate_toil_accrual(Decimal("4"), Decimal("1.5")) == Decimal("6.00")


# ---------------------------------------------------------------------------
# Flexitime credit
# ---------------------------------------------------------------------------

def test_flexitime_credit_over():
	result = calculate_flexitime_credit(Decimal("9"), Decimal("8"))
	assert result == Decimal("1.00")


def test_flexitime_credit_under():
	result = calculate_flexitime_credit(Decimal("7"), Decimal("8"))
	assert result == Decimal("-1.00")


def test_flexitime_credit_exact():
	result = calculate_flexitime_credit(Decimal("8"), Decimal("8"))
	assert result == Decimal("0.00")


# ---------------------------------------------------------------------------
# Zero-hours contract
# ---------------------------------------------------------------------------

def test_zero_hours_no_minimum_ok():
	assert_zero_hours_contract_minimum(Decimal("0"), Decimal("0"))


def test_zero_hours_minimum_met_ok():
	assert_zero_hours_contract_minimum(Decimal("10"), Decimal("8"))


def test_zero_hours_minimum_not_met_raises():
	with pytest.raises(RuleViolation):
		assert_zero_hours_contract_minimum(Decimal("6"), Decimal("8"))


# ---------------------------------------------------------------------------
# Geofence
# ---------------------------------------------------------------------------

def test_within_geofence_same_point():
	assert_within_geofence(0.0, 0.0, 0.0, 0.0, 200.0)


def test_outside_geofence_raises():
	# Nairobi vs Mombasa ~480km
	with pytest.raises(RuleViolation):
		assert_within_geofence(-1.286, 36.820, -4.043, 39.668, 200.0)


def test_geofence_within_100m():
	# ~100m north
	lat_offset = 100 / 111_320
	assert_within_geofence(lat_offset, 0.0, 0.0, 0.0, 200.0)


# ---------------------------------------------------------------------------
# Biometric confidence
# ---------------------------------------------------------------------------

def test_biometric_confidence_high_ok():
	assert_biometric_confidence(0.95)


def test_biometric_confidence_at_threshold_ok():
	assert_biometric_confidence(0.85, threshold=0.85)


def test_biometric_confidence_below_raises():
	with pytest.raises(RuleViolation):
		assert_biometric_confidence(0.70)


def test_biometric_confidence_custom_threshold():
	with pytest.raises(RuleViolation):
		assert_biometric_confidence(0.89, threshold=0.90)


# ---------------------------------------------------------------------------
# Shift scheduling
# ---------------------------------------------------------------------------

def test_min_rest_ok():
	assert_minimum_rest_between_shifts(_dt(18), _dt(8, d=2))  # 14h rest


def test_min_rest_too_short_raises():
	with pytest.raises(RuleViolation):
		assert_minimum_rest_between_shifts(_dt(18), _dt(22))  # 4h rest


def test_max_consecutive_days_ok():
	assert_maximum_consecutive_days(5)


def test_max_consecutive_6_ok():
	assert_maximum_consecutive_days(6)


def test_max_consecutive_7_raises():
	with pytest.raises(RuleViolation):
		assert_maximum_consecutive_days(7)


def test_max_weekly_hours_ok():
	assert_maximum_weekly_hours(Decimal("45"))


def test_max_weekly_hours_exactly_48_ok():
	assert_maximum_weekly_hours(Decimal("48"))


def test_max_weekly_hours_over_raises():
	with pytest.raises(RuleViolation):
		assert_maximum_weekly_hours(Decimal("55"))


# ---------------------------------------------------------------------------
# Core hours (flexitime)
# ---------------------------------------------------------------------------

def test_core_hours_covered_ok():
	assert_core_hours_covered(_dt(8), _dt(17), time(10, 0), time(15, 0))


def test_core_hours_late_arrival_raises():
	with pytest.raises(RuleViolation):
		assert_core_hours_covered(_dt(11), _dt(17), time(10, 0), time(15, 0))


def test_core_hours_early_departure_raises():
	with pytest.raises(RuleViolation):
		assert_core_hours_covered(_dt(8), _dt(14), time(10, 0), time(15, 0))


# ---------------------------------------------------------------------------
# Annualised hours
# ---------------------------------------------------------------------------

def test_annualised_deficit_positive_ok():
	# Positive = ahead of schedule, no problem
	assert_annualised_hours_deficit_manageable(Decimal("20"))


def test_annualised_deficit_zero_ok():
	assert_annualised_hours_deficit_manageable(Decimal("0"))


def test_annualised_deficit_small_negative_ok():
	# -20h behind — within 40h tolerance
	assert_annualised_hours_deficit_manageable(Decimal("-20"))


def test_annualised_deficit_exactly_at_limit_ok():
	# -40h exactly at limit — should NOT raise
	assert_annualised_hours_deficit_manageable(Decimal("-40"))


def test_annualised_deficit_too_large_raises():
	# -41h exceeds -40h max_deficit
	with pytest.raises(RuleViolation):
		assert_annualised_hours_deficit_manageable(Decimal("-41"))


# ---------------------------------------------------------------------------
# Timesheet export guard
# ---------------------------------------------------------------------------

def test_timesheet_approved_ok():
	assert_timesheet_approved_before_export("approved")


def test_timesheet_submitted_raises():
	with pytest.raises(RuleViolation):
		assert_timesheet_approved_before_export("submitted")


def test_timesheet_pending_raises():
	with pytest.raises(RuleViolation):
		assert_timesheet_approved_before_export("pending")


# ---------------------------------------------------------------------------
# Device required
# ---------------------------------------------------------------------------

def test_device_required_mobile_no_device_raises():
	with pytest.raises(RuleViolation):
		assert_device_registered(None, "mobile")


def test_device_required_kiosk_no_device_raises():
	with pytest.raises(RuleViolation):
		assert_device_registered(None, "kiosk")


def test_device_required_biometric_no_device_raises():
	with pytest.raises(RuleViolation):
		assert_device_registered(None, "biometric")


def test_device_web_no_device_ok():
	assert_device_registered(None, "web")


def test_device_registered_mobile_ok():
	assert_device_registered("device-123", "mobile")


# ---------------------------------------------------------------------------
# Import row validation
# ---------------------------------------------------------------------------

def test_import_row_valid_ok():
	assert_import_row_valid(
		{"employee_id": "emp1", "clock_in": "2026-01-01T09:00:00"},
		["employee_id", "clock_in"],
	)


def test_import_row_missing_column_raises():
	with pytest.raises(RuleViolation):
		assert_import_row_valid({"employee_id": "emp1"}, ["employee_id", "clock_in"])


def test_import_row_empty_value_raises():
	with pytest.raises(RuleViolation):
		assert_import_row_valid(
			{"employee_id": "emp1", "clock_in": ""},
			["employee_id", "clock_in"],
		)


# ---------------------------------------------------------------------------
# Holiday timezone
# ---------------------------------------------------------------------------

def test_holiday_date_zero_offset():
	assert get_holiday_date_in_tz(date(2026, 1, 1), 0) == date(2026, 1, 1)


def test_holiday_date_positive_offset_same_day():
	assert get_holiday_date_in_tz(date(2026, 1, 1), 3) == date(2026, 1, 1)


def test_holiday_date_negative_offset_prev_day():
	d = get_holiday_date_in_tz(date(2026, 1, 1), -5)
	assert d == date(2025, 12, 31)
