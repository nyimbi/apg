"""
Fleet Management financial and operational calculations.

All functions are pure, type-safe, and handle edge cases explicitly.
Currency amounts use Decimal throughout — never float for money.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Any


TWO_DP = Decimal("0.01")
FOUR_DP = Decimal("0.0001")


# ──────────────────────────────────────────────────────────────────
# Fuel calculations
# ──────────────────────────────────────────────────────────────────

def calculate_fuel_cost(litres: Decimal, cost_per_litre: Decimal) -> Decimal:
	"""Total fuel cost for a fill-up."""
	if litres <= 0 or cost_per_litre < 0:
		return Decimal("0")
	return (litres * cost_per_litre).quantize(TWO_DP, rounding=ROUND_HALF_UP)


def calculate_fuel_efficiency_l100km(
	litres_consumed: Decimal,
	distance_km: Decimal,
) -> Decimal:
	"""Litres per 100 km.  Returns 0 if distance is zero."""
	if distance_km <= 0:
		return Decimal("0")
	return ((litres_consumed / distance_km) * 100).quantize(TWO_DP, rounding=ROUND_HALF_UP)


def calculate_fuel_efficiency_mpg(
	litres_consumed: Decimal,
	distance_km: Decimal,
	imperial: bool = False,
) -> Decimal:
	"""Miles per gallon (UK imperial or US customary)."""
	if distance_km <= 0 or litres_consumed <= 0:
		return Decimal("0")
	miles = distance_km * Decimal("0.621371")
	gallons = litres_consumed / (Decimal("4.54609") if imperial else Decimal("3.78541"))
	return (miles / gallons).quantize(TWO_DP, rounding=ROUND_HALF_UP)


def calculate_co2_emissions_kg(litres_diesel: Decimal) -> Decimal:
	"""CO2 equivalent from diesel combustion (DEFRA factor: 2.68 kg CO2/litre)."""
	return (litres_diesel * Decimal("2.68")).quantize(TWO_DP, rounding=ROUND_HALF_UP)


# ──────────────────────────────────────────────────────────────────
# Trip / distance calculations
# ──────────────────────────────────────────────────────────────────

def calculate_trip_distance_km(
	odometer_start: Decimal | None,
	odometer_end: Decimal | None,
) -> Decimal | None:
	"""Distance from odometer delta.  Returns None if either reading is absent."""
	if odometer_start is None or odometer_end is None:
		return None
	delta = odometer_end - odometer_start
	return max(Decimal("0"), delta).quantize(TWO_DP, rounding=ROUND_HALF_UP)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> Decimal:
	"""Great-circle distance between two coordinates (km)."""
	import math
	R = 6371.0
	phi1, phi2 = math.radians(lat1), math.radians(lat2)
	dphi = math.radians(lat2 - lat1)
	dlam = math.radians(lon2 - lon1)
	a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
	c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
	return Decimal(str(round(R * c, 2)))


def calculate_trip_duration_hours(
	departure: datetime,
	arrival: datetime,
) -> Decimal:
	"""Trip duration in hours (including fractions)."""
	delta = arrival - departure
	return Decimal(str(round(delta.total_seconds() / 3600, 4)))


def calculate_avg_speed_kmh(distance_km: Decimal, duration_hours: Decimal) -> Decimal:
	"""Average speed from distance and duration."""
	if duration_hours <= 0:
		return Decimal("0")
	return (distance_km / duration_hours).quantize(TWO_DP, rounding=ROUND_HALF_UP)


# ──────────────────────────────────────────────────────────────────
# Total Cost of Ownership (TCO)
# ──────────────────────────────────────────────────────────────────

def calculate_depreciation_straight_line(
	purchase_price: Decimal,
	residual_value: Decimal,
	useful_life_years: int,
	years_held: Decimal,
) -> Decimal:
	"""Straight-line depreciation for years_held."""
	if useful_life_years <= 0 or purchase_price <= residual_value:
		return Decimal("0")
	annual = (purchase_price - residual_value) / Decimal(str(useful_life_years))
	return (annual * years_held).quantize(TWO_DP, rounding=ROUND_HALF_UP)


def calculate_depreciation_reducing_balance(
	purchase_price: Decimal,
	annual_rate_pct: Decimal,
	years_held: Decimal,
) -> Decimal:
	"""Reducing-balance depreciation.  Returns accumulated depreciation."""
	rate = annual_rate_pct / 100
	current_value = purchase_price * ((1 - rate) ** float(years_held))
	return (purchase_price - Decimal(str(current_value))).quantize(TWO_DP, rounding=ROUND_HALF_UP)


def calculate_tco(
	fuel_cost: Decimal,
	maintenance_cost: Decimal,
	insurance_cost: Decimal,
	registration_cost: Decimal,
	depreciation: Decimal,
	driver_cost: Decimal,
	toll_cost: Decimal = Decimal("0"),
	fine_cost: Decimal = Decimal("0"),
	other_cost: Decimal = Decimal("0"),
) -> Decimal:
	"""Total Cost of Ownership sum."""
	return (
		fuel_cost + maintenance_cost + insurance_cost + registration_cost
		+ depreciation + driver_cost + toll_cost + fine_cost + other_cost
	).quantize(TWO_DP, rounding=ROUND_HALF_UP)


def calculate_cost_per_km(total_cost: Decimal, distance_km: Decimal) -> Decimal:
	"""Cost per km.  Returns 0 if no distance."""
	if distance_km <= 0:
		return Decimal("0")
	return (total_cost / distance_km).quantize(FOUR_DP, rounding=ROUND_HALF_UP)


def calculate_cost_per_trip(total_cost: Decimal, trip_count: int) -> Decimal:
	if trip_count <= 0:
		return Decimal("0")
	return (total_cost / Decimal(str(trip_count))).quantize(TWO_DP, rounding=ROUND_HALF_UP)


# ──────────────────────────────────────────────────────────────────
# Utilisation
# ──────────────────────────────────────────────────────────────────

def calculate_utilisation_pct(
	active_hours: Decimal,
	available_hours: Decimal,
) -> float:
	"""Fleet utilisation percentage."""
	if available_hours <= 0:
		return 0.0
	return round(float(active_hours / available_hours) * 100, 2)


def calculate_fleet_utilisation(
	vehicle_active_hours: list[Decimal],
	total_available_hours: Decimal,
) -> float:
	"""Aggregate fleet utilisation across vehicles."""
	if not vehicle_active_hours or total_available_hours <= 0:
		return 0.0
	total_active = sum(vehicle_active_hours, Decimal("0"))
	return round(float(total_active / total_available_hours) * 100, 2)


def calculate_idle_time_pct(idle_minutes: int, total_engine_on_minutes: int) -> float:
	"""Idle time as percentage of engine-on time."""
	if total_engine_on_minutes <= 0:
		return 0.0
	return round((idle_minutes / total_engine_on_minutes) * 100, 2)


# ──────────────────────────────────────────────────────────────────
# Driver behaviour scoring
# ──────────────────────────────────────────────────────────────────

def calculate_driver_score(
	speeding_events: int,
	harsh_braking_events: int,
	harsh_acceleration_events: int,
	cornering_events: int,
	idle_events: int,
	seatbelt_events: int,
	distraction_events: int,
	distance_km: Decimal,
) -> dict[str, float]:
	"""
	Score each dimension per 100 km of driving.
	Perfect = 100.  Each event deducts points per 100 km.
	Weights tuned to industry benchmarks (Samsara / Geotab / Lytx).
	"""
	if distance_km <= 0:
		return {
			"overall": 0.0, "speeding": 0.0, "harsh_braking": 0.0,
			"harsh_acceleration": 0.0, "cornering": 0.0, "idle": 0.0,
			"seatbelt": 0.0, "distraction": 0.0, "grade": "F",
		}

	km = float(distance_km)
	per_100 = 100 / km

	def _score(events: int, deduction_per_event: float) -> float:
		return max(0.0, round(100 - (events * per_100 * deduction_per_event), 2))

	speeding_s = _score(speeding_events, 8.0)
	harsh_brake_s = _score(harsh_braking_events, 6.0)
	harsh_acc_s = _score(harsh_acceleration_events, 5.0)
	cornering_s = _score(cornering_events, 4.0)
	idle_s = _score(idle_events, 2.0)
	seatbelt_s = _score(seatbelt_events, 15.0)
	distraction_s = _score(distraction_events, 12.0)

	weights = {
		"speeding": 0.25, "harsh_braking": 0.20, "harsh_acceleration": 0.15,
		"cornering": 0.10, "idle": 0.05, "seatbelt": 0.15, "distraction": 0.10,
	}
	scores = {
		"speeding": speeding_s, "harsh_braking": harsh_brake_s,
		"harsh_acceleration": harsh_acc_s, "cornering": cornering_s,
		"idle": idle_s, "seatbelt": seatbelt_s, "distraction": distraction_s,
	}
	overall = round(sum(scores[k] * weights[k] for k in weights), 2)

	if overall >= 90:
		grade = "A"
	elif overall >= 80:
		grade = "B"
	elif overall >= 70:
		grade = "C"
	elif overall >= 60:
		grade = "D"
	else:
		grade = "F"

	return {"overall": overall, **scores, "grade": grade}


# ──────────────────────────────────────────────────────────────────
# Compliance calendar helpers
# ──────────────────────────────────────────────────────────────────

def days_until(due_date: datetime, reference: datetime | None = None) -> int:
	"""Days from reference (default: now) until due_date.  Negative = overdue."""
	now = reference or datetime.utcnow()
	return (due_date.replace(tzinfo=None) - now.replace(tzinfo=None)).days


def compliance_severity(days: int) -> str:
	"""Map days-until-due to severity level."""
	if days < 0:
		return "critical"
	if days <= 7:
		return "critical"
	if days <= 30:
		return "warning"
	return "info"


# ──────────────────────────────────────────────────────────────────
# Predictive maintenance signals
# ──────────────────────────────────────────────────────────────────

def predict_brake_wear_failure(
	current_thickness_mm: float,
	wear_rate_mm_per_1000km: float,
	minimum_thickness_mm: float,
	current_odometer_km: Decimal,
) -> dict[str, Any]:
	"""
	Linear wear-to-failure prediction for brake pads.
	Returns distance_to_failure_km and confidence score.
	"""
	remaining_mm = current_thickness_mm - minimum_thickness_mm
	if wear_rate_mm_per_1000km <= 0 or remaining_mm <= 0:
		return {"distance_to_failure_km": 0, "confidence_pct": 95.0, "urgency": "critical"}

	km_to_failure = (remaining_mm / wear_rate_mm_per_1000km) * 1000
	confidence = min(95.0, 60.0 + (remaining_mm / current_thickness_mm) * 35.0)

	if km_to_failure < 2000:
		urgency = "critical"
	elif km_to_failure < 5000:
		urgency = "high"
	elif km_to_failure < 10000:
		urgency = "medium"
	else:
		urgency = "low"

	return {
		"distance_to_failure_km": round(km_to_failure, 0),
		"failure_odometer_km": float(current_odometer_km) + km_to_failure,
		"confidence_pct": round(confidence, 1),
		"urgency": urgency,
	}


def predict_oil_change_due(
	last_oil_change_km: Decimal,
	current_odometer_km: Decimal,
	oil_change_interval_km: Decimal = Decimal("10000"),
	last_oil_change_date: datetime | None = None,
	oil_change_interval_days: int = 180,
) -> dict[str, Any]:
	"""
	Determine if oil change is due based on odometer or calendar, whichever comes first.
	"""
	km_since = current_odometer_km - last_oil_change_km
	km_remaining = oil_change_interval_km - km_since

	calendar_overdue = False
	days_remaining_calendar: int | None = None
	if last_oil_change_date:
		next_by_date = last_oil_change_date + timedelta(days=oil_change_interval_days)
		days_remaining_calendar = days_until(next_by_date)
		calendar_overdue = days_remaining_calendar < 0

	odometer_overdue = km_remaining <= 0
	is_due = odometer_overdue or calendar_overdue

	urgency = "low"
	if is_due:
		urgency = "critical"
	elif km_remaining < 1000 or (days_remaining_calendar is not None and days_remaining_calendar < 14):
		urgency = "high"
	elif km_remaining < 2500 or (days_remaining_calendar is not None and days_remaining_calendar < 30):
		urgency = "medium"

	return {
		"is_due": is_due,
		"km_remaining": float(km_remaining),
		"days_remaining_calendar": days_remaining_calendar,
		"urgency": urgency,
		"odometer_overdue": odometer_overdue,
		"calendar_overdue": calendar_overdue,
	}
