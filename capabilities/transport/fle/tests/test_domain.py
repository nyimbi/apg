"""
Tests for domain rules and calculations.
Pure functions — no async required.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from capabilities.transport.fle.domain.calculations import (
	calculate_avg_speed_kmh,
	calculate_co2_emissions_kg,
	calculate_cost_per_km,
	calculate_depreciation_straight_line,
	calculate_driver_score,
	calculate_fleet_utilisation,
	calculate_fuel_cost,
	calculate_fuel_efficiency_l100km,
	calculate_fuel_efficiency_mpg,
	calculate_tco,
	calculate_trip_distance_km,
	calculate_trip_duration_hours,
	calculate_utilisation_pct,
	compliance_severity,
	days_until,
	haversine_km,
	predict_brake_wear_failure,
	predict_oil_change_due,
)
from capabilities.transport.fle.domain.rules import (
	RuleViolation,
	allocate_overloading_fine,
	assert_axle_load_within_limits,
	assert_cof_valid,
	assert_customs_docs_present_for_cross_border,
	assert_driver_active,
	assert_driver_cpc_valid,
	assert_driver_licence_valid,
	assert_eu_continuous_driving,
	assert_eu_daily_driving,
	assert_eu_daily_rest,
	assert_eu_weekly_driving,
	assert_fatal_incident_requires_police_ref,
	assert_fuel_volume_reasonable,
	assert_hired_vehicle_within_hire_period,
	assert_incident_reported_within_window,
	assert_insurance_valid,
	assert_maintenance_not_overdue_for_dispatch,
	assert_no_concurrent_trip,
	assert_no_duplicate_vin,
	assert_odometer_not_regressing,
	assert_trip_arrival_after_departure,
	assert_us_hos_driving,
	assert_us_hos_on_duty_window,
	assert_vehicle_active_for_dispatch,
	assert_vehicle_not_overloaded,
	assert_vehicle_registration_present,
	assert_vin_present,
	calculate_next_service_date,
	calculate_overloading_fine,
)


FUTURE = datetime.utcnow() + timedelta(days=365 * 3)
PAST = datetime.utcnow() - timedelta(days=1)


# ──────────────────────────────────────────────────────────────────
# Vehicle rules
# ──────────────────────────────────────────────────────────────────

def test_registration_required():
	with pytest.raises(RuleViolation, match="VEH-001"):
		assert_vehicle_registration_present("")


def test_vin_too_short():
	with pytest.raises(RuleViolation, match="VEH-002"):
		assert_vin_present("TOOSHORT")


def test_vin_valid():
	assert_vin_present("WAUZZZ8K9BA123456")  # no raise


def test_duplicate_vin():
	with pytest.raises(RuleViolation, match="VEH-003"):
		assert_no_duplicate_vin("VIN123456789", ["VIN123456789"])


def test_unique_vin():
	assert_no_duplicate_vin("VIN123456789", ["DIFFERENT999"])  # no raise


def test_overload():
	with pytest.raises(RuleViolation, match="VEH-004"):
		assert_vehicle_not_overloaded(Decimal("20000"), Decimal("16000"))


def test_not_overloaded():
	assert_vehicle_not_overloaded(Decimal("15000"), Decimal("16000"))  # no raise


def test_vehicle_inactive_dispatch_rejected():
	with pytest.raises(RuleViolation, match="VEH-005"):
		assert_vehicle_active_for_dispatch("in_maintenance")


def test_vehicle_active_dispatch_ok():
	assert_vehicle_active_for_dispatch("active")  # no raise


def test_cof_expired():
	with pytest.raises(RuleViolation, match="VEH-007"):
		assert_cof_valid(PAST)


def test_cof_valid():
	assert_cof_valid(FUTURE)  # no raise


def test_insurance_expired():
	with pytest.raises(RuleViolation, match="VEH-008"):
		assert_insurance_valid(PAST)


# ──────────────────────────────────────────────────────────────────
# Driver rules
# ──────────────────────────────────────────────────────────────────

def test_driver_expired_licence():
	with pytest.raises(RuleViolation, match="DRV-001"):
		assert_driver_licence_valid(PAST)


def test_driver_valid_licence():
	assert_driver_licence_valid(FUTURE)  # no raise


def test_driver_expired_cpc():
	with pytest.raises(RuleViolation, match="DRV-002"):
		assert_driver_cpc_valid(PAST)


def test_driver_inactive():
	with pytest.raises(RuleViolation, match="DRV-004"):
		assert_driver_active("suspended")


def test_driver_active_ok():
	assert_driver_active("active")  # no raise


# ──────────────────────────────────────────────────────────────────
# EU Tachograph rules
# ──────────────────────────────────────────────────────────────────

def test_eu_continuous_driving_limit():
	with pytest.raises(RuleViolation, match="TACHO-001"):
		assert_eu_continuous_driving(280)  # >270 min


def test_eu_continuous_driving_ok():
	assert_eu_continuous_driving(270)  # no raise


def test_eu_daily_driving_limit():
	with pytest.raises(RuleViolation, match="TACHO-002"):
		assert_eu_daily_driving(545)  # >540 min standard


def test_eu_daily_driving_extended():
	assert_eu_daily_driving(600, extended=True)  # no raise


def test_eu_weekly_driving_limit():
	with pytest.raises(RuleViolation, match="TACHO-003"):
		assert_eu_weekly_driving(3400)  # >3360 min


def test_eu_daily_rest_insufficient():
	with pytest.raises(RuleViolation, match="TACHO-005"):
		assert_eu_daily_rest(500)  # <660 min


# ──────────────────────────────────────────────────────────────────
# US HOS rules
# ──────────────────────────────────────────────────────────────────

def test_us_hos_driving_limit():
	with pytest.raises(RuleViolation, match="HOS-001"):
		assert_us_hos_driving(12.0)  # >11h


def test_us_hos_on_duty_window():
	with pytest.raises(RuleViolation, match="HOS-002"):
		assert_us_hos_on_duty_window(15.0)  # >14h


# ──────────────────────────────────────────────────────────────────
# Trip rules
# ──────────────────────────────────────────────────────────────────

def test_arrival_before_departure():
	dep = datetime.utcnow() + timedelta(hours=2)
	arr = datetime.utcnow() + timedelta(hours=1)
	with pytest.raises(RuleViolation, match="TRIP-002"):
		assert_trip_arrival_after_departure(dep, arr)


def test_customs_docs_missing():
	with pytest.raises(RuleViolation, match="TRIP-003"):
		assert_customs_docs_present_for_cross_border(True, ["TZ", "UG"], False)


def test_customs_docs_not_required():
	assert_customs_docs_present_for_cross_border(False, [], False)  # no raise


def test_concurrent_trip():
	with pytest.raises(RuleViolation, match="TRIP-004"):
		assert_no_concurrent_trip("V1", ["trip-001"])


# ──────────────────────────────────────────────────────────────────
# Overloading / fines
# ──────────────────────────────────────────────────────────────────

def test_overloading_fine_calculation():
	fine = calculate_overloading_fine(Decimal("500"), Decimal("100"))
	assert fine == Decimal("50000.00")


def test_overloading_fine_zero_excess():
	fine = calculate_overloading_fine(Decimal("0"), Decimal("100"))
	assert fine == Decimal("0")


def test_fine_allocation():
	shares = allocate_overloading_fine(Decimal("10000"), 30.0, 70.0)
	assert shares["driver"] == Decimal("3000.00")
	assert shares["owner"] == Decimal("7000.00")


def test_axle_overload():
	with pytest.raises(RuleViolation, match="OVL-001"):
		assert_axle_load_within_limits(Decimal("12000"), Decimal("10000"))


# ──────────────────────────────────────────────────────────────────
# Fuel rules
# ──────────────────────────────────────────────────────────────────

def test_fuel_exceeds_tank():
	with pytest.raises(RuleViolation, match="FUEL-001"):
		assert_fuel_volume_reasonable(Decimal("450"), Decimal("400"))


def test_fuel_within_tank():
	assert_fuel_volume_reasonable(Decimal("390"), Decimal("400"))  # no raise


def test_odometer_regression():
	with pytest.raises(RuleViolation, match="ODO-001"):
		assert_odometer_not_regressing(Decimal("49000"), Decimal("50000"))


# ──────────────────────────────────────────────────────────────────
# Incidents
# ──────────────────────────────────────────────────────────────────

def test_incident_stale_reporting():
	with pytest.raises(RuleViolation, match="INC-001"):
		assert_incident_reported_within_window(datetime.utcnow() - timedelta(hours=30))


def test_fatal_no_police_ref():
	with pytest.raises(RuleViolation, match="INC-002"):
		assert_fatal_incident_requires_police_ref("fatal", "")


def test_minor_no_police_ref_ok():
	assert_fatal_incident_requires_police_ref("minor", "")  # no raise


# ──────────────────────────────────────────────────────────────────
# Hired vehicle
# ──────────────────────────────────────────────────────────────────

def test_hired_vehicle_outside_period():
	with pytest.raises(RuleViolation, match="HIRE-001"):
		assert_hired_vehicle_within_hire_period(
			"hired",
			datetime.utcnow() - timedelta(days=60),
			datetime.utcnow() - timedelta(days=5),  # ended 5 days ago
		)


def test_owned_vehicle_no_hire_check():
	assert_hired_vehicle_within_hire_period("owned", None, None)  # no raise


# ──────────────────────────────────────────────────────────────────
# Maintenance rules
# ──────────────────────────────────────────────────────────────────

def test_overdue_maintenance_blocks_dispatch():
	with pytest.raises(RuleViolation, match="MNT-001"):
		assert_maintenance_not_overdue_for_dispatch(True)


def test_next_service_date():
	last = datetime(2025, 1, 1)
	next_d = calculate_next_service_date(last, 90)
	assert next_d == datetime(2025, 4, 1)


# ──────────────────────────────────────────────────────────────────
# Calculations
# ──────────────────────────────────────────────────────────────────

def test_fuel_cost():
	assert calculate_fuel_cost(Decimal("100"), Decimal("185")) == Decimal("18500.00")


def test_fuel_cost_zero_litres():
	assert calculate_fuel_cost(Decimal("0"), Decimal("185")) == Decimal("0")


def test_fuel_efficiency_l100km():
	eff = calculate_fuel_efficiency_l100km(Decimal("80"), Decimal("400"))
	assert eff == Decimal("20.00")


def test_fuel_efficiency_zero_distance():
	assert calculate_fuel_efficiency_l100km(Decimal("80"), Decimal("0")) == Decimal("0")


def test_fuel_mpg():
	mpg = calculate_fuel_efficiency_mpg(Decimal("80"), Decimal("400"), imperial=True)
	assert mpg > 0


def test_co2_emissions():
	co2 = calculate_co2_emissions_kg(Decimal("100"))
	assert co2 == Decimal("268.00")


def test_trip_distance():
	d = calculate_trip_distance_km(Decimal("50000"), Decimal("50450"))
	assert d == Decimal("450.00")


def test_trip_distance_none_on_missing():
	assert calculate_trip_distance_km(None, Decimal("50000")) is None


def test_trip_duration():
	dep = datetime(2025, 1, 1, 8, 0)
	arr = datetime(2025, 1, 1, 14, 30)
	dur = calculate_trip_duration_hours(dep, arr)
	assert dur == Decimal("6.5")


def test_avg_speed():
	spd = calculate_avg_speed_kmh(Decimal("390"), Decimal("6.5"))
	assert spd == Decimal("60.00")


def test_haversine():
	# Nairobi → Mombasa ≈ 441 km
	dist = haversine_km(-1.2921, 36.8219, -4.0435, 39.6682)
	assert 420 < float(dist) < 460


def test_depreciation_straight_line():
	dep = calculate_depreciation_straight_line(
		Decimal("5000000"), Decimal("500000"), 10, Decimal("3")
	)
	assert dep == Decimal("1350000.00")


def test_tco_sum():
	total = calculate_tco(
		Decimal("100000"), Decimal("50000"), Decimal("45000"),
		Decimal("5000"), Decimal("300000"), Decimal("0"),
	)
	assert total == Decimal("500000.00")


def test_cost_per_km():
	cpk = calculate_cost_per_km(Decimal("500000"), Decimal("100000"))
	assert cpk == Decimal("5.0000")


def test_cost_per_km_zero_distance():
	assert calculate_cost_per_km(Decimal("500000"), Decimal("0")) == Decimal("0")


def test_utilisation_pct():
	assert calculate_utilisation_pct(Decimal("160"), Decimal("200")) == 80.0


def test_utilisation_zero():
	assert calculate_utilisation_pct(Decimal("0"), Decimal("0")) == 0.0


def test_fleet_utilisation():
	assert calculate_fleet_utilisation([Decimal("100"), Decimal("80")], Decimal("200")) == 90.0


def test_driver_score_perfect():
	scores = calculate_driver_score(0, 0, 0, 0, 0, 0, 0, Decimal("500"))
	assert scores["overall"] == 100.0
	assert scores["grade"] == "A"


def test_driver_score_with_events():
	scores = calculate_driver_score(5, 3, 2, 1, 2, 0, 1, Decimal("200"))
	assert 0 <= scores["overall"] <= 100
	assert scores["grade"] in ("A", "B", "C", "D", "F")


def test_driver_score_zero_distance():
	scores = calculate_driver_score(5, 3, 2, 1, 2, 0, 1, Decimal("0"))
	assert scores["overall"] == 0.0
	assert scores["grade"] == "F"


def test_days_until_future():
	future = datetime.utcnow() + timedelta(days=30)
	d = days_until(future)
	assert 28 <= d <= 31


def test_days_until_past():
	past = datetime.utcnow() - timedelta(days=5)
	d = days_until(past)
	assert d < 0


def test_compliance_severity():
	assert compliance_severity(-1) == "critical"
	assert compliance_severity(3) == "critical"
	assert compliance_severity(20) == "warning"
	assert compliance_severity(60) == "info"


def test_predict_brake_wear():
	result = predict_brake_wear_failure(8.0, 1.0, 3.0, Decimal("50000"))
	assert result["urgency"] in ("low", "medium", "high", "critical")
	assert result["distance_to_failure_km"] == pytest.approx(5000.0)


def test_predict_oil_change_due_by_odometer():
	result = predict_oil_change_due(
		last_oil_change_km=Decimal("40000"),
		current_odometer_km=Decimal("49500"),
	)
	assert result["is_due"] is False
	assert result["km_remaining"] == pytest.approx(500.0)
	assert result["urgency"] in ("high", "critical")


def test_predict_oil_change_overdue():
	result = predict_oil_change_due(
		last_oil_change_km=Decimal("30000"),
		current_odometer_km=Decimal("50000"),
	)
	assert result["is_due"] is True
	assert result["urgency"] == "critical"
