"""
Fleet Management business rules.

Every rule is a callable function.  Violations raise RuleViolation.
assert_* functions enforce; calculate_* functions compute values.

EU Tachograph: Regulation (EC) 561/2006 + (EU) 165/2014
US HOS: 49 CFR Part 395 (property-carrying, 11/14-hr rule)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any


# ──────────────────────────────────────────────────────────────────
# Exception
# ──────────────────────────────────────────────────────────────────

@dataclass
class RuleViolation(Exception):
	"""Raised when a business rule is violated."""
	rule: str
	message: str
	context: dict[str, Any]

	def __str__(self) -> str:
		return f"[{self.rule}] {self.message}"


# ──────────────────────────────────────────────────────────────────
# Vehicle rules
# ──────────────────────────────────────────────────────────────────

def assert_vehicle_registration_present(registration: str) -> None:
	"""A vehicle must have a non-empty registration plate."""
	if not registration or not registration.strip():
		raise RuleViolation(
			"VEH-001",
			"Vehicle registration plate is required.",
			{"registration": registration},
		)


def assert_vin_present(vin: str) -> None:
	"""A vehicle must have a VIN of at least 11 characters."""
	if not vin or len(vin.strip()) < 11:
		raise RuleViolation(
			"VEH-002",
			f"VIN must be at least 11 characters, got: '{vin}'.",
			{"vin": vin},
		)


def assert_no_duplicate_vin(vin: str, existing_vins: list[str], exclude_id: str | None = None) -> None:
	"""VIN must be unique within tenant."""
	# Caller passes list of (vin, vehicle_id) tuples or just vins
	if vin.strip().upper() in [v.upper() for v in existing_vins]:
		raise RuleViolation(
			"VEH-003",
			f"VIN '{vin}' is already registered.",
			{"vin": vin},
		)


def assert_vehicle_not_overloaded(load_kg: Decimal, payload_capacity_kg: Decimal) -> None:
	"""Load must not exceed declared payload capacity."""
	if payload_capacity_kg > 0 and load_kg > payload_capacity_kg:
		raise RuleViolation(
			"VEH-004",
			f"Load {load_kg} kg exceeds payload capacity {payload_capacity_kg} kg.",
			{"load_kg": str(load_kg), "payload_capacity_kg": str(payload_capacity_kg)},
		)


def assert_vehicle_active_for_dispatch(status: str) -> None:
	"""Only active vehicles may be dispatched."""
	allowed = {"active"}
	if status not in allowed:
		raise RuleViolation(
			"VEH-005",
			f"Vehicle status '{status}' does not permit dispatch; must be 'active'.",
			{"status": status},
		)


def assert_vehicle_not_in_maintenance_for_trip(status: str) -> None:
	"""A vehicle in maintenance cannot start a trip."""
	if status == "in_maintenance":
		raise RuleViolation(
			"VEH-006",
			"Vehicle is currently in maintenance and cannot be assigned to a trip.",
			{"status": status},
		)


def assert_cof_valid(cof_expires_at: datetime | None, check_date: datetime | None = None) -> None:
	"""Certificate of Fitness must be current."""
	now = check_date or datetime.utcnow()
	if cof_expires_at is None or cof_expires_at < now:
		raise RuleViolation(
			"VEH-007",
			f"Vehicle COF has expired or is missing (expiry: {cof_expires_at}).",
			{"cof_expires_at": str(cof_expires_at)},
		)


def assert_insurance_valid(cover_end: datetime | None, check_date: datetime | None = None) -> None:
	"""Insurance cover must be current on dispatch date."""
	now = check_date or datetime.utcnow()
	if cover_end is None or cover_end < now:
		raise RuleViolation(
			"VEH-008",
			f"Vehicle insurance cover has lapsed (ends: {cover_end}).",
			{"cover_end": str(cover_end)},
		)


def assert_road_worthiness_valid(expires_at: datetime | None, check_date: datetime | None = None) -> None:
	"""Road worthiness certificate must be current."""
	now = check_date or datetime.utcnow()
	if expires_at is None or expires_at < now:
		raise RuleViolation(
			"VEH-009",
			f"Road worthiness certificate has expired (expiry: {expires_at}).",
			{"expires_at": str(expires_at)},
		)


# ──────────────────────────────────────────────────────────────────
# Driver rules
# ──────────────────────────────────────────────────────────────────

def assert_driver_licence_valid(licence_expiry: datetime, check_date: datetime | None = None) -> None:
	"""Driver licence must not be expired."""
	now = check_date or datetime.utcnow()
	if licence_expiry < now:
		raise RuleViolation(
			"DRV-001",
			f"Driver licence expired on {licence_expiry.date()}.",
			{"licence_expiry": str(licence_expiry)},
		)


def assert_driver_cpc_valid(cpc_expiry: datetime | None, check_date: datetime | None = None) -> None:
	"""Driver CPC (Certificate of Professional Competence) must be current."""
	now = check_date or datetime.utcnow()
	if cpc_expiry is None or cpc_expiry < now:
		raise RuleViolation(
			"DRV-002",
			f"Driver CPC has expired or is missing (expiry: {cpc_expiry}).",
			{"cpc_expiry": str(cpc_expiry)},
		)


def assert_driver_medical_valid(medical_expiry: datetime | None, check_date: datetime | None = None) -> None:
	"""Driver medical certificate must be current (if tracked)."""
	if medical_expiry is None:
		return  # not all jurisdictions require this
	now = check_date or datetime.utcnow()
	if medical_expiry < now:
		raise RuleViolation(
			"DRV-003",
			f"Driver medical certificate expired on {medical_expiry.date()}.",
			{"medical_expiry": str(medical_expiry)},
		)


def assert_driver_active(status: str) -> None:
	"""Only active drivers may be assigned to trips."""
	if status != "active":
		raise RuleViolation(
			"DRV-004",
			f"Driver status '{status}' does not permit trip assignment.",
			{"status": status},
		)


def assert_driver_licence_class_valid(licence_class: str, required_class: str) -> None:
	"""Driver must hold licence class required for the vehicle type."""
	# Simplified hierarchy: CE > C > BE > B etc.
	hierarchy = ["b", "be", "c1", "c1e", "c", "ce", "d1", "d1e", "d", "de"]
	driver_idx = hierarchy.index(licence_class.lower()) if licence_class.lower() in hierarchy else -1
	required_idx = hierarchy.index(required_class.lower()) if required_class.lower() in hierarchy else -1
	if driver_idx < required_idx:
		raise RuleViolation(
			"DRV-005",
			f"Driver holds '{licence_class}' but vehicle requires at least '{required_class}'.",
			{"driver_licence": licence_class, "required": required_class},
		)


# ──────────────────────────────────────────────────────────────────
# EU Tachograph / HOS rules (EC 561/2006)
# ──────────────────────────────────────────────────────────────────

# EU limits (minutes)
EU_MAX_CONTINUOUS_DRIVING_MIN = 270  # 4.5 hours
EU_BREAK_REQUIRED_MIN = 45           # 45-minute break (or 15+30)
EU_MAX_DAILY_DRIVING_MIN = 540       # 9 hours (extendable to 10h twice/week)
EU_MAX_DAILY_DRIVING_EXTENDED_MIN = 600  # 10 hours
EU_MAX_WEEKLY_DRIVING_MIN = 3360     # 56 hours
EU_MAX_FORTNIGHTLY_DRIVING_MIN = 5400  # 90 hours
EU_MIN_DAILY_REST_MIN = 660          # 11 hours (reducible to 9h three times/week)
EU_MIN_WEEKLY_REST_MIN = 2700        # 45 hours (reducible to 24h)

# US HOS (property-carrying — 11/14 hr rule)
US_MAX_DRIVING_HRS = 11
US_ON_DUTY_WINDOW_HRS = 14
US_MANDATORY_OFF_DUTY_HRS = 10
US_60HRS_7DAY = 60
US_70HRS_8DAY = 70


def assert_eu_continuous_driving(continuous_driving_min: int) -> None:
	"""EU 561/2006: max 4h30m continuous driving before mandatory break."""
	if continuous_driving_min > EU_MAX_CONTINUOUS_DRIVING_MIN:
		raise RuleViolation(
			"TACHO-001",
			f"Continuous driving {continuous_driving_min}min exceeds EU limit of {EU_MAX_CONTINUOUS_DRIVING_MIN}min.",
			{"continuous_driving_min": continuous_driving_min, "limit_min": EU_MAX_CONTINUOUS_DRIVING_MIN},
		)


def assert_eu_daily_driving(daily_driving_min: int, extended: bool = False) -> None:
	"""EU 561/2006: daily driving limit 9h (10h extended, max twice/week)."""
	limit = EU_MAX_DAILY_DRIVING_EXTENDED_MIN if extended else EU_MAX_DAILY_DRIVING_MIN
	if daily_driving_min > limit:
		raise RuleViolation(
			"TACHO-002",
			f"Daily driving {daily_driving_min}min exceeds {'extended' if extended else 'standard'} EU limit of {limit}min.",
			{"daily_driving_min": daily_driving_min, "limit_min": limit},
		)


def assert_eu_weekly_driving(weekly_driving_min: int) -> None:
	"""EU 561/2006: weekly driving must not exceed 56 hours."""
	if weekly_driving_min > EU_MAX_WEEKLY_DRIVING_MIN:
		raise RuleViolation(
			"TACHO-003",
			f"Weekly driving {weekly_driving_min}min exceeds EU limit of {EU_MAX_WEEKLY_DRIVING_MIN}min.",
			{"weekly_driving_min": weekly_driving_min},
		)


def assert_eu_fortnightly_driving(fortnightly_driving_min: int) -> None:
	"""EU 561/2006: fortnightly driving must not exceed 90 hours."""
	if fortnightly_driving_min > EU_MAX_FORTNIGHTLY_DRIVING_MIN:
		raise RuleViolation(
			"TACHO-004",
			f"Fortnightly driving {fortnightly_driving_min}min exceeds EU limit of {EU_MAX_FORTNIGHTLY_DRIVING_MIN}min.",
			{"fortnightly_driving_min": fortnightly_driving_min},
		)


def assert_eu_daily_rest(rest_min: int, reduced: bool = False) -> None:
	"""EU 561/2006: minimum daily rest 11h (reducible to 9h max 3×/week)."""
	min_rest = 540 if reduced else EU_MIN_DAILY_REST_MIN
	if rest_min < min_rest:
		raise RuleViolation(
			"TACHO-005",
			f"Daily rest {rest_min}min is below EU minimum of {min_rest}min.",
			{"rest_min": rest_min, "min_rest": min_rest},
		)


def assert_us_hos_driving(driving_hrs: float) -> None:
	"""US HOS 49 CFR 395: max 11 hours driving after 10 consecutive off-duty."""
	if driving_hrs > US_MAX_DRIVING_HRS:
		raise RuleViolation(
			"HOS-001",
			f"US HOS: driving {driving_hrs:.1f}h exceeds 11-hour limit.",
			{"driving_hrs": driving_hrs},
		)


def assert_us_hos_on_duty_window(on_duty_hrs: float) -> None:
	"""US HOS: must not drive after 14 consecutive hours on-duty."""
	if on_duty_hrs > US_ON_DUTY_WINDOW_HRS:
		raise RuleViolation(
			"HOS-002",
			f"US HOS: on-duty window {on_duty_hrs:.1f}h exceeds 14-hour limit.",
			{"on_duty_hrs": on_duty_hrs},
		)


def assert_us_hos_cumulative(cumulative_hrs: float, cycle_days: int = 7) -> None:
	"""US HOS: 60/7 or 70/8 day cycle."""
	limit = US_60HRS_7DAY if cycle_days == 7 else US_70HRS_8DAY
	if cumulative_hrs > limit:
		raise RuleViolation(
			"HOS-003",
			f"US HOS: cumulative {cumulative_hrs:.1f}h exceeds {limit}h/{cycle_days}-day limit.",
			{"cumulative_hrs": cumulative_hrs, "limit": limit, "cycle_days": cycle_days},
		)


# ──────────────────────────────────────────────────────────────────
# Trip rules
# ──────────────────────────────────────────────────────────────────

def assert_trip_departure_in_future(planned_departure: datetime, buffer_minutes: int = 0) -> None:
	"""Trip departure must be in the future (with optional buffer)."""
	threshold = datetime.utcnow() - timedelta(minutes=buffer_minutes)
	if planned_departure < threshold:
		raise RuleViolation(
			"TRIP-001",
			f"Planned departure {planned_departure} is in the past.",
			{"planned_departure": str(planned_departure)},
		)


def assert_trip_arrival_after_departure(planned_departure: datetime, planned_arrival: datetime) -> None:
	"""Arrival must be after departure."""
	if planned_arrival <= planned_departure:
		raise RuleViolation(
			"TRIP-002",
			"Planned arrival must be after planned departure.",
			{"departure": str(planned_departure), "arrival": str(planned_arrival)},
		)


def assert_customs_docs_present_for_cross_border(
	customs_required: bool, cross_border_countries: list[str], documents_attached: bool
) -> None:
	"""Cross-border trips must have customs documentation."""
	if customs_required and cross_border_countries and not documents_attached:
		raise RuleViolation(
			"TRIP-003",
			f"Cross-border trip to {cross_border_countries} requires customs documentation.",
			{"countries": cross_border_countries, "documents_attached": documents_attached},
		)


def assert_no_concurrent_trip(vehicle_id: str, active_trip_ids: list[str]) -> None:
	"""A vehicle cannot have two concurrent active trips."""
	if active_trip_ids:
		raise RuleViolation(
			"TRIP-004",
			f"Vehicle '{vehicle_id}' already has active trip(s): {active_trip_ids}.",
			{"vehicle_id": vehicle_id, "active_trips": active_trip_ids},
		)


def assert_driver_not_already_on_trip(driver_id: str, active_trip_ids: list[str]) -> None:
	"""A driver cannot be on two trips simultaneously."""
	if active_trip_ids:
		raise RuleViolation(
			"TRIP-005",
			f"Driver '{driver_id}' is already assigned to active trip(s): {active_trip_ids}.",
			{"driver_id": driver_id, "active_trips": active_trip_ids},
		)


# ──────────────────────────────────────────────────────────────────
# Overloading / fines
# ──────────────────────────────────────────────────────────────────

def assert_axle_load_within_limits(axle_load_kg: Decimal, axle_limit_kg: Decimal) -> None:
	"""Individual axle must not exceed statutory limit."""
	if axle_load_kg > axle_limit_kg:
		raise RuleViolation(
			"OVL-001",
			f"Axle load {axle_load_kg} kg exceeds statutory limit {axle_limit_kg} kg.",
			{"axle_load_kg": str(axle_load_kg), "limit_kg": str(axle_limit_kg)},
		)


def calculate_overloading_fine(excess_kg: Decimal, rate_per_kg: Decimal) -> Decimal:
	"""Calculate overloading fine: excess_kg × rate_per_kg."""
	if excess_kg <= 0:
		return Decimal("0")
	return (excess_kg * rate_per_kg).quantize(Decimal("0.01"))


def allocate_overloading_fine(
	fine_total: Decimal,
	driver_share_pct: float,
	owner_share_pct: float,
) -> dict[str, Decimal]:
	"""Allocate overloading fine between driver and vehicle owner."""
	assert abs(driver_share_pct + owner_share_pct - 100) < 0.01, "Shares must sum to 100%"
	driver_share = (fine_total * Decimal(str(driver_share_pct / 100))).quantize(Decimal("0.01"))
	owner_share = fine_total - driver_share
	return {"driver": driver_share, "owner": owner_share}


# ──────────────────────────────────────────────────────────────────
# Maintenance rules
# ──────────────────────────────────────────────────────────────────

def assert_maintenance_not_overdue_for_dispatch(overdue: bool) -> None:
	"""Vehicle with overdue critical maintenance must not be dispatched."""
	if overdue:
		raise RuleViolation(
			"MNT-001",
			"Vehicle has overdue critical maintenance items and cannot be dispatched.",
			{"overdue": True},
		)


def calculate_next_service_date(
	last_service_date: datetime,
	interval_days: int,
) -> datetime:
	"""Return next service date based on interval."""
	return last_service_date + timedelta(days=interval_days)


def calculate_next_service_odometer(
	last_odometer_km: Decimal,
	interval_km: Decimal,
) -> Decimal:
	"""Return odometer reading at next service."""
	return last_odometer_km + interval_km


# ──────────────────────────────────────────────────────────────────
# Fuel rules
# ──────────────────────────────────────────────────────────────────

def assert_fuel_volume_reasonable(litres: Decimal, tank_capacity_l: Decimal | None) -> None:
	"""Fuel fill must not exceed tank capacity (with 5% tolerance)."""
	if tank_capacity_l and litres > tank_capacity_l * Decimal("1.05"):
		raise RuleViolation(
			"FUEL-001",
			f"Fuel fill {litres}L exceeds tank capacity {tank_capacity_l}L.",
			{"litres": str(litres), "capacity": str(tank_capacity_l)},
		)


def assert_odometer_not_regressing(new_odometer: Decimal, last_odometer: Decimal) -> None:
	"""Odometer reading must not decrease."""
	if new_odometer < last_odometer:
		raise RuleViolation(
			"ODO-001",
			f"Odometer reading {new_odometer} km is less than last reading {last_odometer} km.",
			{"new_km": str(new_odometer), "last_km": str(last_odometer)},
		)


# ──────────────────────────────────────────────────────────────────
# Incident / breakdown rules
# ──────────────────────────────────────────────────────────────────

def assert_incident_reported_within_window(occurred_at: datetime, hours_limit: int = 24) -> None:
	"""Incidents must be reported within the statutory window."""
	age = datetime.utcnow() - occurred_at
	if age > timedelta(hours=hours_limit):
		raise RuleViolation(
			"INC-001",
			f"Incident occurred {age.total_seconds()/3600:.1f}h ago; must be reported within {hours_limit}h.",
			{"occurred_at": str(occurred_at), "hours_limit": hours_limit},
		)


def assert_fatal_incident_requires_police_ref(severity: str, police_ref: str) -> None:
	"""Fatal/critical incidents must have a police reference number."""
	if severity in ("fatal", "critical") and not (police_ref and police_ref.strip()):
		raise RuleViolation(
			"INC-002",
			f"Severity '{severity}' incidents require a police reference number.",
			{"severity": severity},
		)


# ──────────────────────────────────────────────────────────────────
# Hired/rental vehicle rules
# ──────────────────────────────────────────────────────────────────

def assert_hired_vehicle_within_hire_period(
	ownership_type: str,
	hire_start: datetime | None,
	hire_end: datetime | None,
	check_date: datetime | None = None,
) -> None:
	"""Hired/rental vehicles must be within the hire contract window."""
	if ownership_type not in ("hired", "contract_hire"):
		return
	now = check_date or datetime.utcnow()
	if hire_end and now > hire_end:
		raise RuleViolation(
			"HIRE-001",
			f"Hired vehicle hire period ended {hire_end.date()}.",
			{"hire_end": str(hire_end)},
		)
	if hire_start and now < hire_start:
		raise RuleViolation(
			"HIRE-002",
			f"Hired vehicle hire period starts {hire_start.date()}.",
			{"hire_start": str(hire_start)},
		)
