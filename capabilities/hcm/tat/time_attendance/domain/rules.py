"""Deterministic domain rules for Time and Attendance Tracking.

These rules are the single source of truth for all governance decisions within
this capability.  Every function is pure (no I/O, no side effects) and raises
RuleViolation on constraint failure.

Tabs, not spaces.  Python 3.12+.
Copyright © 2025 Datacraft.  Author: Nyimbi Odero
"""
from __future__ import annotations

import math
from datetime import date, datetime, time, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
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
# Tenancy
# ---------------------------------------------------------------------------

def assert_tenant_context(context: dict[str, Any]) -> None:
	"""All operations require a tenant context."""
	if not context.get("tenant_id"):
		raise RuleViolation(
			"tenant_context_required",
			"tenant_id is required",
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
			"cross-tenant access is not permitted",
			"use_own_tenant_resources",
		)


# ---------------------------------------------------------------------------
# Clock-in / clock-out
# ---------------------------------------------------------------------------

def assert_clock_in_before_clock_out(clock_in: datetime, clock_out: datetime) -> None:
	"""Clock-out must be strictly after clock-in."""
	if clock_out <= clock_in:
		raise RuleViolation(
			"clock_out_before_clock_in",
			f"clock_out ({clock_out.isoformat()}) must be after clock_in ({clock_in.isoformat()})",
			"correct_clock_times",
		)


def assert_not_already_clocked_in(existing_clock_in: datetime | None) -> None:
	"""Employee must not already have an open clock-in today."""
	if existing_clock_in is not None:
		raise RuleViolation(
			"already_clocked_in",
			f"employee already clocked in at {existing_clock_in.isoformat()}",
			"clock_out_first",
		)


def assert_not_already_clocked_out(existing_clock_out: datetime | None) -> None:
	"""Time entry must not already have a clock-out recorded."""
	if existing_clock_out is not None:
		raise RuleViolation(
			"already_clocked_out",
			f"entry already has clock_out at {existing_clock_out.isoformat()}",
			"create_new_entry",
		)


def assert_shift_duration_reasonable(
	clock_in: datetime,
	clock_out: datetime,
	max_hours: float = 20.0,
) -> None:
	"""A single shift must not exceed max_hours (default 20)."""
	duration_h = (clock_out - clock_in).total_seconds() / 3600
	if duration_h > max_hours:
		raise RuleViolation(
			"shift_duration_unreasonable",
			f"shift duration {duration_h:.1f}h exceeds maximum {max_hours}h",
			"verify_clock_times_or_split_shift",
		)


# ---------------------------------------------------------------------------
# Night shift
# ---------------------------------------------------------------------------

def is_night_shift(
	clock_in: datetime,
	clock_out: datetime,
	night_start_hour: int = 22,
	night_end_hour: int = 6,
) -> bool:
	"""Return True if the shift falls (at least partially) in night hours."""
	in_h = clock_in.hour
	out_h = clock_out.hour
	# Crosses midnight
	if clock_out.date() > clock_in.date():
		return True
	# Starts in or reaches night window on same day
	return in_h >= night_start_hour or out_h <= night_end_hour


def assert_night_shift_midnight_span(clock_in: datetime, clock_out: datetime) -> None:
	"""Night shifts may span midnight but must not span more than one calendar day boundary."""
	day_delta = (clock_out.date() - clock_in.date()).days
	if day_delta > 1:
		raise RuleViolation(
			"night_shift_spans_multiple_days",
			f"shift crosses {day_delta} calendar day boundaries; maximum is 1",
			"split_shift_into_separate_entries",
		)


# ---------------------------------------------------------------------------
# Overtime thresholds
# ---------------------------------------------------------------------------

def assert_overtime_threshold_positive(threshold: float) -> None:
	"""Overtime threshold must be a positive number."""
	if threshold <= 0:
		raise RuleViolation(
			"overtime_threshold_not_positive",
			f"overtime threshold must be > 0, got {threshold}",
			"set_positive_threshold",
		)


def calculate_daily_overtime(
	worked_hours: Decimal,
	daily_threshold: Decimal = Decimal("8"),
) -> Decimal:
	"""Hours worked above the daily threshold (FLSA/California daily model)."""
	return max(worked_hours - daily_threshold, Decimal("0"))


def calculate_weekly_overtime(
	weekly_hours: Decimal,
	weekly_threshold: Decimal = Decimal("40"),
) -> Decimal:
	"""Hours worked above the weekly threshold."""
	return max(weekly_hours - weekly_threshold, Decimal("0"))


# ---------------------------------------------------------------------------
# Rest between shifts
# ---------------------------------------------------------------------------

def assert_minimum_rest_between_shifts(
	previous_clock_out: datetime,
	next_clock_in: datetime,
	min_rest_hours: float = 11.0,
) -> None:
	"""Minimum rest period between consecutive shifts (EU Working Time Directive default: 11h)."""
	rest_h = (next_clock_in - previous_clock_out).total_seconds() / 3600
	if rest_h < min_rest_hours:
		raise RuleViolation(
			"insufficient_rest_between_shifts",
			f"only {rest_h:.1f}h rest between shifts; minimum is {min_rest_hours}h",
			"reschedule_shift_or_get_manager_override",
		)


def assert_maximum_consecutive_days(consecutive_days: int, max_days: int = 6) -> None:
	"""Employee must have at least one rest day in any seven-day period."""
	if consecutive_days > max_days:
		raise RuleViolation(
			"maximum_consecutive_days_exceeded",
			f"{consecutive_days} consecutive working days exceeds maximum {max_days}",
			"schedule_a_rest_day",
		)


def assert_maximum_weekly_hours(
	weekly_hours: Decimal,
	max_hours: Decimal = Decimal("48"),
) -> None:
	"""Weekly hours must not exceed the statutory maximum (default 48h, EU WTD)."""
	if weekly_hours > max_hours:
		raise RuleViolation(
			"maximum_weekly_hours_exceeded",
			f"{weekly_hours}h weekly hours exceeds maximum {max_hours}h",
			"reduce_hours_or_obtain_opt_out",
		)


# ---------------------------------------------------------------------------
# Leave
# ---------------------------------------------------------------------------

def assert_leave_dates_valid(start_date: date, end_date: date) -> None:
	"""Leave end date must be on or after start date."""
	if end_date < start_date:
		raise RuleViolation(
			"leave_end_before_start",
			f"leave end_date ({end_date}) is before start_date ({start_date})",
			"correct_leave_dates",
		)


def assert_leave_balance_sufficient(
	available_balance: Decimal,
	requested_days: Decimal,
) -> None:
	"""Employee must have enough leave balance to cover the request."""
	if available_balance < requested_days:
		raise RuleViolation(
			"insufficient_leave_balance",
			f"available balance {available_balance} days < requested {requested_days} days",
			"reduce_leave_request_or_request_unpaid_leave",
		)


def assert_leave_not_overlapping(
	existing_leaves: list[dict[str, Any]],
	new_start: date,
	new_end: date,
) -> None:
	"""New leave must not overlap with any already-approved/pending leave."""
	for leave in existing_leaves:
		ex_start = leave["start_date"]
		ex_end = leave["end_date"]
		# Overlap when: new_start <= ex_end AND new_end >= ex_start
		if new_start <= ex_end and new_end >= ex_start:
			raise RuleViolation(
				"leave_dates_overlap",
				f"requested leave ({new_start}–{new_end}) overlaps with existing leave ({ex_start}–{ex_end})",
				"choose_non_overlapping_dates",
			)


def assert_medical_certificate_for_extended_sick(
	leave_type: str,
	duration_days: int,
	certificate_attached: bool,
	threshold_days: int = 3,
) -> None:
	"""Sick leave exceeding threshold_days requires a medical certificate."""
	if leave_type != "sick":
		return
	if duration_days > threshold_days and not certificate_attached:
		raise RuleViolation(
			"medical_certificate_required",
			f"sick leave of {duration_days} days exceeds {threshold_days}-day threshold; "
			"medical certificate is required",
			"attach_medical_certificate",
		)


# ---------------------------------------------------------------------------
# Flexitime / core hours
# ---------------------------------------------------------------------------

def assert_core_hours_covered(
	clock_in: datetime,
	clock_out: datetime,
	core_start: time,
	core_end: time,
) -> None:
	"""Under flexitime, employee must be present during the mandatory core hours."""
	in_time = clock_in.time().replace(tzinfo=None)
	out_time = clock_out.time().replace(tzinfo=None)

	# Strip tzinfo from core hours for comparison
	cs = core_start.replace(tzinfo=None)
	ce = core_end.replace(tzinfo=None)

	if in_time > cs:
		raise RuleViolation(
			"core_hours_not_covered",
			f"clock-in {in_time} is after core start {cs}",
			"arrive_by_core_hours_start",
		)
	if out_time < ce:
		raise RuleViolation(
			"core_hours_not_covered",
			f"clock-out {out_time} is before core end {ce}",
			"stay_until_core_hours_end",
		)


def calculate_flexitime_credit(
	worked_hours: Decimal,
	standard_hours: Decimal,
) -> Decimal:
	"""
	Signed flexitime credit for a single day.

	Positive = banked credit; negative = deficit.
	"""
	_q = lambda v: v.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
	return _q(worked_hours - standard_hours)


# ---------------------------------------------------------------------------
# Annualised hours
# ---------------------------------------------------------------------------

def assert_annualised_hours_deficit_manageable(
	hours_delta: Decimal,
	max_deficit: Decimal = Decimal("40"),
) -> None:
	"""
	Assert that the running annualised hours deficit is within acceptable limits.

	hours_delta: contracted_hours_to_date - worked_hours_to_date.
	  Positive → ahead of schedule (no problem).
	  Negative → behind schedule (deficit).
	max_deficit: maximum allowable deficit in absolute hours.
	"""
	if hours_delta < -max_deficit:
		raise RuleViolation(
			"annualised_hours_deficit_too_large",
			f"hours deficit {abs(hours_delta):.1f}h exceeds maximum {max_deficit}h",
			"increase_hours_worked_or_adjust_contract",
		)


# ---------------------------------------------------------------------------
# Timesheet export guard
# ---------------------------------------------------------------------------

def assert_timesheet_approved_before_export(status: str) -> None:
	"""Timesheet must be in 'approved' status before payroll export."""
	if status != "approved":
		raise RuleViolation(
			"timesheet_not_approved",
			f"timesheet status is '{status}'; must be 'approved' before export",
			"get_manager_approval_first",
		)


# ---------------------------------------------------------------------------
# Device / biometric
# ---------------------------------------------------------------------------

_DEVICE_REQUIRED_METHODS = {"mobile", "kiosk", "biometric"}


def assert_device_registered(
	device_id: str | None,
	method: str,
) -> None:
	"""Device-based clock methods require a registered device_id."""
	if method in _DEVICE_REQUIRED_METHODS and not device_id:
		raise RuleViolation(
			"device_not_registered",
			f"clock method '{method}' requires a registered device_id",
			"register_device_first",
		)


def assert_biometric_confidence(
	confidence_score: float,
	threshold: float = 0.80,
	# Legacy kwarg alias used in service.py
	min_confidence: float | None = None,
	context: str = "clock_in",
) -> None:
	"""Assert biometric verification confidence meets the minimum threshold."""
	effective_threshold = min_confidence if min_confidence is not None else threshold
	if confidence_score < effective_threshold:
		raise RuleViolation(
			"biometric_confidence_too_low",
			f"biometric confidence {confidence_score:.0%} below minimum "
			f"{effective_threshold:.0%} for {context}",
			"retry_biometric_verification_or_use_pin",
		)


# ---------------------------------------------------------------------------
# Geofencing
# ---------------------------------------------------------------------------

_EARTH_RADIUS_M = 6_371_000.0


def _haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
	"""Great-circle distance in metres between two WGS-84 coordinates."""
	r = _EARTH_RADIUS_M
	phi1, phi2 = math.radians(lat1), math.radians(lat2)
	dphi = math.radians(lat2 - lat1)
	dlam = math.radians(lng2 - lng1)
	a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
	return 2 * r * math.asin(math.sqrt(a))


def assert_within_geofence(
	employee_lat: float,
	employee_lng: float,
	fence_lat: float,
	fence_lng: float,
	radius_metres: float,
) -> None:
	"""Employee's GPS position must be within the geofence radius."""
	distance = _haversine_m(employee_lat, employee_lng, fence_lat, fence_lng)
	if distance > radius_metres:
		raise RuleViolation(
			"outside_geofence",
			f"employee is {distance:.0f}m from fence centre; "
			f"maximum allowed radius is {radius_metres:.0f}m",
			"move_within_geofence_before_clocking",
		)


# ---------------------------------------------------------------------------
# Import validation
# ---------------------------------------------------------------------------

def assert_import_row_valid(
	row: dict[str, Any],
	required_columns: list[str],
) -> None:
	"""CSV import row must contain all required columns with non-empty values."""
	for col in required_columns:
		if col not in row:
			raise RuleViolation(
				"import_row_missing_column",
				f"required column '{col}' is absent in import row",
				"fix_csv_headers",
			)
		if not str(row[col]).strip():
			raise RuleViolation(
				"import_row_empty_value",
				f"column '{col}' has an empty value in import row",
				"populate_missing_values",
			)


# ---------------------------------------------------------------------------
# Zero-hours contracts
# ---------------------------------------------------------------------------

def assert_zero_hours_contract_minimum(
	hours_offered: Decimal,
	minimum_hours: Decimal,
) -> None:
	"""
	On zero-hours contracts with a stipulated minimum (e.g. 8h/week),
	offered hours must meet or exceed that minimum.

	If minimum_hours is 0 (true zero-hours, no floor), always passes.
	"""
	if minimum_hours > Decimal("0") and hours_offered < minimum_hours:
		raise RuleViolation(
			"zero_hours_minimum_not_met",
			f"offered hours {hours_offered}h below contractual minimum {minimum_hours}h",
			"offer_at_least_minimum_contracted_hours",
		)


# ---------------------------------------------------------------------------
# FMLA / statutory leave eligibility
# ---------------------------------------------------------------------------

def assert_fmla_eligibility(
	months_employed: int,
	hours_worked_last_12_months: float,
	min_months: int = 12,
	min_hours: float = 1250.0,
) -> None:
	"""
	US FMLA eligibility: ≥12 months employed and ≥1 250 hours in last 12 months.
	Jurisdiction-specific — callers should gate on jurisdiction before invoking.
	"""
	if months_employed < min_months:
		raise RuleViolation(
			"fmla_ineligible_months",
			f"employee has {months_employed} months service; FMLA requires {min_months}",
			"wait_until_12_months_service",
		)
	if hours_worked_last_12_months < min_hours:
		raise RuleViolation(
			"fmla_ineligible_hours",
			f"employee worked {hours_worked_last_12_months:.0f}h in last 12 months; "
			f"FMLA requires {min_hours:.0f}h",
			"accumulate_required_hours",
		)


# ---------------------------------------------------------------------------
# Pro-rata / TOIL helpers
# ---------------------------------------------------------------------------

def calculate_prorata_entitlement(
	full_time_days: Decimal,
	fte: Decimal,
) -> Decimal:
	"""
	Pro-rata leave entitlement for part-time employees.

	fte must be in (0, 1].
	"""
	if fte <= Decimal("0") or fte > Decimal("1"):
		raise RuleViolation(
			"invalid_fte",
			f"FTE must be in (0, 1], got {fte}",
			"set_valid_fte",
		)
	return full_time_days * fte


def calculate_toil_accrual(
	overtime_hours: Decimal,
	multiplier: Decimal = Decimal("1.0"),
) -> Decimal:
	"""Convert approved overtime hours into TOIL credits at the given multiplier."""
	_q = lambda v: v.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
	return _q(overtime_hours * multiplier)


# ---------------------------------------------------------------------------
# Holiday timezone helper
# ---------------------------------------------------------------------------

def get_holiday_date_in_tz(
	holiday_utc: date,
	utc_offset_hours: int,
) -> date:
	"""
	Return the local calendar date of a public holiday given a UTC offset.

	A holiday stored as 2026-01-01 UTC in timezone UTC-5 falls on 2025-12-31 locally
	if the standard time is 00:00 UTC (i.e. the previous day in UTC-5).

	This implementation uses a 00:00 UTC reference point, which is the conventional
	storage format for public holidays (date-only, not datetime).  Negative offsets
	can shift the holiday to the previous calendar day.
	"""
	from datetime import timedelta
	if utc_offset_hours >= 0:
		return holiday_utc
	# For negative offsets: if the local time would be before midnight, it's previous day
	# Using 00:00 UTC as the canonical holiday time
	local_dt = datetime(holiday_utc.year, holiday_utc.month, holiday_utc.day, 0, 0) + timedelta(hours=utc_offset_hours)
	return local_dt.date()
