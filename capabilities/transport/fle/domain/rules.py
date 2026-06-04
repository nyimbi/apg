"""Deterministic domain rules for Fleet Management.

Every governance decision in the fleet domain is encoded here as a callable
function.  assert_* functions raise RuleViolation on failure.  calculate_*
functions return computed values.

Rules are the single source of truth for:
- Driver compliance (licence, CPC, medical, HOS tachograph)
- Vehicle compliance (registration, COF, insurance, roadworthiness)
- Load/axle regulations
- Cross-border customs documentation
- Incident reporting windows
- Dispatch pre-flight checks
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any


# ──────────────────────────────────────────────────────────────────
# Exceptions
# ──────────────────────────────────────────────────────────────────

class RuleViolation(Exception):
	"""Raised when a fleet business rule is violated."""

	def __init__(self, rule_name: str, reason: str, required_action: str = "") -> None:
		self.rule_name = rule_name
		self.reason = reason
		self.required_action = required_action
		super().__init__(f"[{rule_name}] {reason}")


# ──────────────────────────────────────────────────────────────────
# Tenant / security
# ──────────────────────────────────────────────────────────────────

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
			f"actor tenant {actor_tenant!r} cannot access resource of tenant {resource_tenant!r}",
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


# ──────────────────────────────────────────────────────────────────
# Vehicle identity & registration
# ──────────────────────────────────────────────────────────────────

def assert_vehicle_registration_present(registration: str) -> None:
	"""A vehicle must have a non-empty registration plate."""
	if not registration or not registration.strip():
		raise RuleViolation(
			"vehicle_registration_required",
			"vehicle registration plate is required",
			"provide_registration_plate",
		)


def assert_vin_present(vin: str) -> None:
	"""A vehicle must have a non-empty VIN."""
	if not vin or not vin.strip():
		raise RuleViolation(
			"vin_required",
			"Vehicle Identification Number (VIN) is required",
			"provide_vin",
		)


def assert_no_duplicate_vin(vin: str, existing_vins: list[str]) -> None:
	"""VIN must be unique within the tenant fleet."""
	if vin in existing_vins:
		raise RuleViolation(
			"duplicate_vin",
			f"VIN {vin!r} is already registered in this fleet",
			"use_unique_vin",
		)


def assert_road_worthiness_valid(roadworthiness_ref: str, expires_at: datetime | None) -> None:
	"""Road worthiness certificate must be present and not expired."""
	if not roadworthiness_ref:
		raise RuleViolation(
			"roadworthiness_required",
			"road worthiness certificate reference is required",
			"obtain_roadworthiness_certificate",
		)
	if expires_at and expires_at < datetime.utcnow():
		raise RuleViolation(
			"roadworthiness_expired",
			f"road worthiness certificate expired {expires_at.date()}",
			"renew_roadworthiness_certificate",
		)


# ──────────────────────────────────────────────────────────────────
# Vehicle dispatch pre-flight
# ──────────────────────────────────────────────────────────────────

def assert_vehicle_active_for_dispatch(status: str) -> None:
	"""Vehicle must be ACTIVE before it can be dispatched."""
	if status != "active":
		raise RuleViolation(
			"vehicle_not_active",
			f"vehicle status is {status!r} — only 'active' vehicles may be dispatched",
			"return_vehicle_to_active_status",
		)


def assert_maintenance_not_overdue_for_dispatch(overdue_maint_ids: list[str]) -> None:
	"""Vehicles with overdue mandatory maintenance must not be dispatched."""
	if overdue_maint_ids:
		raise RuleViolation(
			"maintenance_overdue",
			f"{len(overdue_maint_ids)} overdue maintenance item(s) block dispatch",
			"complete_overdue_maintenance_before_dispatch",
		)


def assert_no_concurrent_trip(vehicle_id: str, active_trip_ids: list[str]) -> None:
	"""A vehicle may not be on two trips at the same time."""
	if active_trip_ids:
		raise RuleViolation(
			"vehicle_already_on_trip",
			f"vehicle {vehicle_id!r} is already on trip(s): {active_trip_ids}",
			"complete_or_cancel_current_trip_first",
		)


# ──────────────────────────────────────────────────────────────────
# Driver compliance
# ──────────────────────────────────────────────────────────────────

def assert_driver_active(status: str) -> None:
	"""Driver must be ACTIVE to be assigned or dispatched."""
	if status != "active":
		raise RuleViolation(
			"driver_not_active",
			f"driver status is {status!r} — only 'active' drivers may be assigned",
			"reinstate_or_assign_different_driver",
		)


def assert_driver_licence_valid(licence_expiry: datetime) -> None:
	"""Driver's licence must not be expired."""
	if licence_expiry < datetime.utcnow():
		raise RuleViolation(
			"driver_licence_expired",
			f"driver licence expired on {licence_expiry.date()}",
			"renew_driver_licence_before_dispatch",
		)


def assert_driver_cpc_valid(cpc_expiry: datetime) -> None:
	"""Driver CPC qualification must not be expired (EU/EAC requirement)."""
	if cpc_expiry < datetime.utcnow():
		raise RuleViolation(
			"driver_cpc_expired",
			f"Driver CPC expired on {cpc_expiry.date()}",
			"complete_cpc_training_before_dispatch",
		)


def assert_driver_medical_valid(medical_expiry: datetime) -> None:
	"""Driver medical certificate must not be expired."""
	if medical_expiry < datetime.utcnow():
		raise RuleViolation(
			"driver_medical_expired",
			f"driver medical certificate expired on {medical_expiry.date()}",
			"renew_medical_certificate_before_dispatch",
		)


def assert_driver_not_already_on_trip(driver_id: str, active_trip_ids: list[str]) -> None:
	"""A driver may not be concurrently assigned to two active trips."""
	if active_trip_ids:
		raise RuleViolation(
			"driver_already_on_trip",
			f"driver {driver_id!r} is already on trip(s): {active_trip_ids}",
			"complete_current_trip_or_change_driver",
		)


# ──────────────────────────────────────────────────────────────────
# EU Tachograph / Hours of Service (Regulation EC 561/2006)
# ──────────────────────────────────────────────────────────────────

# EU limits (minutes)
_EU_MAX_CONTINUOUS_DRIVING_MIN = 270       # 4.5 hours continuous
_EU_MAX_DAILY_DRIVING_MIN = 540            # 9 hours/day (extendable to 10 twice/week)
_EU_MAX_DAILY_DRIVING_EXTENDED_MIN = 600   # 10 hours extended
_EU_MAX_WEEKLY_DRIVING_MIN = 3360          # 56 hours/week
_EU_MAX_FORTNIGHTLY_DRIVING_MIN = 5400     # 90 hours per 2 weeks
_EU_MIN_DAILY_REST_MIN = 660               # 11 hours (reducible to 9 three times/week)
_EU_MIN_BREAK_AFTER_CONTINUOUS_MIN = 45    # 45 min break after 4.5 h driving


def assert_eu_continuous_driving(driving_minutes: int) -> None:
	"""EU: No more than 4.5 h (270 min) continuous driving without a 45 min break."""
	if driving_minutes > _EU_MAX_CONTINUOUS_DRIVING_MIN:
		raise RuleViolation(
			"eu_continuous_driving_exceeded",
			f"continuous driving {driving_minutes} min exceeds EU limit of {_EU_MAX_CONTINUOUS_DRIVING_MIN} min",
			"record_mandatory_45_min_break_then_resume",
		)


def assert_eu_daily_driving(daily_driving_minutes: int, extended: bool = False) -> None:
	"""EU: Daily driving limit (9 h standard, 10 h if extended day permitted)."""
	limit = _EU_MAX_DAILY_DRIVING_EXTENDED_MIN if extended else _EU_MAX_DAILY_DRIVING_MIN
	if daily_driving_minutes > limit:
		raise RuleViolation(
			"eu_daily_driving_exceeded",
			f"daily driving {daily_driving_minutes} min exceeds EU limit of {limit} min",
			"end_duty_period_and_take_daily_rest",
		)


def assert_eu_weekly_driving(weekly_driving_minutes: int) -> None:
	"""EU: Weekly driving must not exceed 56 hours (3360 min)."""
	if weekly_driving_minutes > _EU_MAX_WEEKLY_DRIVING_MIN:
		raise RuleViolation(
			"eu_weekly_driving_exceeded",
			f"weekly driving {weekly_driving_minutes} min exceeds EU limit of {_EU_MAX_WEEKLY_DRIVING_MIN} min",
			"take_weekly_rest_before_driving_again",
		)


def assert_eu_daily_rest(rest_minutes: int, reduced: bool = False) -> None:
	"""EU: Daily rest period — 11 h standard, 9 h reduced (max 3x per week)."""
	minimum = 540 if reduced else _EU_MIN_DAILY_REST_MIN
	if rest_minutes < minimum:
		raise RuleViolation(
			"eu_daily_rest_insufficient",
			f"daily rest {rest_minutes} min is below EU minimum of {minimum} min",
			"extend_rest_period_to_comply",
		)


# US Hours of Service (49 CFR Part 395) — property-carrying drivers
_US_HOS_MAX_DRIVING_MIN = 660    # 11 hours driving
_US_HOS_MAX_ON_DUTY_MIN = 840    # 14-hour window
_US_HOS_MIN_OFF_DUTY_MIN = 600   # 10 hours off duty


def assert_us_hos_driving(driving_minutes: int) -> None:
	"""US HOS: Property-carrying driver may not drive after 11 hours on duty."""
	if driving_minutes > _US_HOS_MAX_DRIVING_MIN:
		raise RuleViolation(
			"us_hos_driving_exceeded",
			f"US HOS driving {driving_minutes} min exceeds limit of {_US_HOS_MAX_DRIVING_MIN} min",
			"take_10_hour_off_duty_period",
		)


def assert_us_hos_on_duty_window(on_duty_minutes: int) -> None:
	"""US HOS: Driver may not drive beyond 14 hours after coming on duty."""
	if on_duty_minutes > _US_HOS_MAX_ON_DUTY_MIN:
		raise RuleViolation(
			"us_hos_on_duty_window_exceeded",
			f"US HOS on-duty window {on_duty_minutes} min exceeds 14-hour limit",
			"take_10_hour_off_duty_period",
		)


# ──────────────────────────────────────────────────────────────────
# Insurance
# ──────────────────────────────────────────────────────────────────

def assert_insurance_valid(cover_end: datetime) -> None:
	"""Vehicle must have active insurance cover for dispatch."""
	if cover_end < datetime.utcnow():
		raise RuleViolation(
			"insurance_expired",
			f"vehicle insurance expired on {cover_end.date()}",
			"renew_insurance_before_dispatch",
		)


# ──────────────────────────────────────────────────────────────────
# Certificate of Fitness (COF)
# ──────────────────────────────────────────────────────────────────

def assert_cof_valid(expires_at: datetime | None) -> None:
	"""COF must be present and not expired (East Africa / NTSA requirement)."""
	if expires_at is None:
		raise RuleViolation(
			"cof_missing",
			"Certificate of Fitness (COF) is missing",
			"obtain_cof_from_ntsa",
		)
	if expires_at < datetime.utcnow():
		raise RuleViolation(
			"cof_expired",
			f"COF expired on {expires_at.date()}",
			"renew_cof_before_dispatch",
		)


# ──────────────────────────────────────────────────────────────────
# Load / overloading
# ──────────────────────────────────────────────────────────────────

def assert_vehicle_not_overloaded(
	load_kg: Decimal,
	payload_capacity_kg: Decimal,
) -> None:
	"""Load must not exceed vehicle's rated payload capacity."""
	if payload_capacity_kg > 0 and load_kg > payload_capacity_kg:
		excess = load_kg - payload_capacity_kg
		raise RuleViolation(
			"vehicle_overloaded",
			f"load {load_kg} kg exceeds capacity {payload_capacity_kg} kg (excess: {excess} kg)",
			"reduce_load_or_use_larger_vehicle",
		)


def assert_axle_load_within_limits(
	axle_load_kg: Decimal,
	legal_axle_limit_kg: Decimal = Decimal("10000"),
) -> None:
	"""Each axle load must not exceed legal limit (default 10 t per axle)."""
	if axle_load_kg > legal_axle_limit_kg:
		raise RuleViolation(
			"axle_load_exceeded",
			f"axle load {axle_load_kg} kg exceeds legal limit of {legal_axle_limit_kg} kg",
			"redistribute_load_or_obtain_overload_permit",
		)


def calculate_overloading_fine(
	excess_kg: Decimal,
	fine_per_kg: Decimal = Decimal("10"),
) -> Decimal:
	"""Calculate overloading fine — Kenya/EAC: KES 10 per excess kg (configurable)."""
	if excess_kg <= 0:
		return Decimal("0")
	return (excess_kg * fine_per_kg).quantize(Decimal("0.01"))


# ──────────────────────────────────────────────────────────────────
# Trip rules
# ──────────────────────────────────────────────────────────────────

def assert_trip_arrival_after_departure(
	departure: datetime,
	arrival: datetime,
) -> None:
	"""Planned arrival must be strictly after departure."""
	if arrival <= departure:
		raise RuleViolation(
			"arrival_not_after_departure",
			f"planned arrival {arrival} must be after departure {departure}",
			"correct_trip_schedule",
		)


def assert_odometer_not_regressing(
	new_odometer: Decimal,
	last_odometer: Decimal,
) -> None:
	"""Odometer readings must never decrease."""
	if new_odometer < last_odometer:
		raise RuleViolation(
			"odometer_regression",
			f"new odometer {new_odometer} km is less than previous {last_odometer} km",
			"correct_odometer_reading_or_report_tamper",
		)


# ──────────────────────────────────────────────────────────────────
# Cross-border / customs
# ──────────────────────────────────────────────────────────────────

def assert_customs_docs_present_for_cross_border(
	customs_required: bool,
	cross_border_countries: list[str],
	customs_docs_present: bool,
) -> None:
	"""Cross-border trips require customs documentation to be marked present."""
	if cross_border_countries and (customs_required or not customs_docs_present):
		if not customs_docs_present:
			raise RuleViolation(
				"customs_docs_missing",
				f"cross-border trip to {cross_border_countries} requires customs documentation",
				"attach_customs_clearance_docs_before_dispatch",
			)


def assert_hired_vehicle_within_hire_period(
	hire_start: datetime,
	hire_end: datetime,
	planned_departure: datetime,
) -> None:
	"""Hired vehicle trips must fall within the contracted hire period."""
	if planned_departure < hire_start or planned_departure > hire_end:
		raise RuleViolation(
			"outside_hire_period",
			f"trip departure {planned_departure.date()} is outside hire period "
			f"{hire_start.date()}–{hire_end.date()}",
			"adjust_trip_date_or_extend_hire_contract",
		)


# ──────────────────────────────────────────────────────────────────
# Incidents
# ──────────────────────────────────────────────────────────────────

_INCIDENT_REPORTING_WINDOW_HOURS = 72


def assert_incident_reported_within_window(occurred_at: datetime) -> None:
	"""Incidents must be reported within 72 hours of occurrence."""
	hours_elapsed = (datetime.utcnow() - occurred_at).total_seconds() / 3600
	if hours_elapsed > _INCIDENT_REPORTING_WINDOW_HOURS:
		raise RuleViolation(
			"incident_reporting_window_exceeded",
			f"incident occurred {hours_elapsed:.1f} h ago — reporting window is "
			f"{_INCIDENT_REPORTING_WINDOW_HOURS} h",
			"report_incident_with_late_justification",
		)


def assert_fatal_incident_requires_police_ref(
	severity: str,
	police_ref: str,
) -> None:
	"""Fatal incidents require a police reference number."""
	if severity == "fatal" and not (police_ref and police_ref.strip()):
		raise RuleViolation(
			"fatal_incident_police_ref_required",
			"fatal incidents must include a police occurrence/reference number",
			"obtain_police_reference_number",
		)


# ──────────────────────────────────────────────────────────────────
# Fuel
# ──────────────────────────────────────────────────────────────────

def assert_fuel_litres_positive(litres: Decimal) -> None:
	"""Fuel records must record a positive quantity."""
	if litres <= 0:
		raise RuleViolation(
			"fuel_litres_not_positive",
			f"fuel litres {litres} must be > 0",
			"enter_correct_fuel_quantity",
		)


def assert_fuel_cost_non_negative(cost: Decimal) -> None:
	"""Fuel cost must be non-negative."""
	if cost < 0:
		raise RuleViolation(
			"fuel_cost_negative",
			f"fuel cost {cost} must be >= 0",
			"correct_fuel_cost_entry",
		)


# ──────────────────────────────────────────────────────────────────
# Tachograph infringement coding
# ──────────────────────────────────────────────────────────────────

EU_INFRINGEMENT_CODES: dict[str, str] = {
	"C1": "Continuous driving time exceeded (minor — < 3 h over limit)",
	"C2": "Continuous driving time exceeded (serious — > 3 h over limit)",
	"D1": "Daily driving time exceeded (minor — < 1 h over limit)",
	"D2": "Daily driving time exceeded (serious — > 1 h over limit)",
	"D3": "Daily driving time exceeded (most serious — > 2 h over limit)",
	"WK1": "Weekly driving time exceeded",
	"WK2": "Fortnightly driving time exceeded",
	"R1": "Daily rest period reduced without entitlement",
	"R2": "Daily rest period insufficient (most serious)",
	"B1": "45-minute break not taken after 4.5 h driving",
}


def classify_eu_infringement(driving_minutes: int, limit_minutes: int) -> str | None:
	"""Return the appropriate EU infringement code for a driving time breach."""
	excess_minutes = driving_minutes - limit_minutes
	if excess_minutes <= 0:
		return None
	if limit_minutes == _EU_MAX_CONTINUOUS_DRIVING_MIN:
		return "C2" if excess_minutes > 60 else "C1"
	if limit_minutes == _EU_MAX_DAILY_DRIVING_MIN:
		if excess_minutes > 120:
			return "D3"
		if excess_minutes > 60:
			return "D2"
		return "D1"
	if limit_minutes == _EU_MAX_WEEKLY_DRIVING_MIN:
		return "WK1"
	return None


# ──────────────────────────────────────────────────────────────────
# Maintenance
# ──────────────────────────────────────────────────────────────────

def assert_maintenance_schedule_valid(
	scheduled_date: datetime,
	current_date: datetime | None = None,
) -> None:
	"""Maintenance must not be scheduled in the past (more than 1 day ago)."""
	ref = current_date or datetime.utcnow()
	if (ref - scheduled_date).days > 1:
		raise RuleViolation(
			"maintenance_scheduled_in_past",
			f"maintenance scheduled_date {scheduled_date.date()} is in the past",
			"schedule_maintenance_for_future_date_or_record_as_completed",
		)


# ──────────────────────────────────────────────────────────────────
# Compound pre-dispatch check
# ──────────────────────────────────────────────────────────────────

def run_dispatch_pre_flight(
	vehicle_status: str,
	driver_status: str,
	driver_licence_expiry: datetime,
	load_kg: Decimal,
	payload_capacity_kg: Decimal,
	active_vehicle_trips: list[str],
	active_driver_trips: list[str],
	insurance_cover_end: datetime | None = None,
	cof_expires_at: datetime | None = None,
	driver_cpc_expiry: datetime | None = None,
	driver_medical_expiry: datetime | None = None,
) -> list[str]:
	"""
	Run all pre-dispatch checks in sequence.

	Returns a list of violation messages (empty = cleared for dispatch).
	Does NOT raise — callers decide whether to hard-block or warn.
	"""
	violations: list[str] = []

	checks = [
		lambda: assert_vehicle_active_for_dispatch(vehicle_status),
		lambda: assert_driver_active(driver_status),
		lambda: assert_driver_licence_valid(driver_licence_expiry),
		lambda: assert_no_concurrent_trip("vehicle", active_vehicle_trips),
		lambda: assert_driver_not_already_on_trip("driver", active_driver_trips),
		lambda: assert_vehicle_not_overloaded(load_kg, payload_capacity_kg),
	]

	if insurance_cover_end:
		checks.append(lambda: assert_insurance_valid(insurance_cover_end))
	if cof_expires_at is not None:
		checks.append(lambda: assert_cof_valid(cof_expires_at))
	if driver_cpc_expiry:
		checks.append(lambda: assert_driver_cpc_valid(driver_cpc_expiry))
	if driver_medical_expiry:
		checks.append(lambda: assert_driver_medical_valid(driver_medical_expiry))

	for check in checks:
		try:
			check()
		except RuleViolation as e:
			violations.append(str(e))

	return violations
