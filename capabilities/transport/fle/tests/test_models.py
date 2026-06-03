"""
Unit tests for FLE Pydantic v2 models.
No async required — pure model validation tests.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from capabilities.transport.fle.models import (
	DriverCreate, FuelRecordCreate, IncidentCreate, IncidentSeverity,
	InspectionCreate, InspectionResult, InspectionType,
	InsurancePolicyCreate, MaintenanceCreate, MaintenanceType,
	RegistrationCreate, TachographRecordCreate, TachographMode,
	TelematicsEventCreate, TripCreate, VehicleAssignmentCreate,
	VehicleCreate, VehicleStatus, VehicleType, FuelType, OwnershipType,
	LicenceClass, uuid7str,
)

FUTURE = datetime.utcnow() + timedelta(days=365 * 3)


# ──────────────────────────────────────────────────────────────────
# uuid7str
# ──────────────────────────────────────────────────────────────────

def test_uuid7str_unique():
	ids = {uuid7str() for _ in range(100)}
	assert len(ids) == 100


def test_uuid7str_is_string():
	assert isinstance(uuid7str(), str)
	assert len(uuid7str()) == 36  # xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx


# ──────────────────────────────────────────────────────────────────
# VehicleCreate
# ──────────────────────────────────────────────────────────────────

def test_vehicle_create_valid():
	v = VehicleCreate(
		tenant_id="t1",
		vehicle_type=VehicleType.RIGID_TRUCK,
		registration="KCA 001A",
		vin="WAUZZZ8K9BA123456",
		make="Mercedes",
		model="Actros",
		year=2022,
		fuel_type=FuelType.DIESEL,
		ownership_type=OwnershipType.OWNED,
	)
	assert v.tenant_id == "t1"
	assert v.vehicle_type == VehicleType.RIGID_TRUCK


def test_vehicle_create_empty_tenant_rejected():
	with pytest.raises(ValidationError):
		VehicleCreate(
			tenant_id="  ",
			vehicle_type="rigid_truck",
			registration="KCA 001A",
			vin="WAUZZZ8K9BA123456",
			make="M", model="A", year=2022,
			fuel_type="diesel", ownership_type="owned",
		)


def test_vehicle_create_invalid_year():
	with pytest.raises(ValidationError):
		VehicleCreate(
			tenant_id="t1", vehicle_type="van",
			registration="X", vin="12345678901",
			make="M", model="A", year=1970,
			fuel_type="petrol", ownership_type="owned",
		)


def test_vehicle_create_extra_field_rejected():
	with pytest.raises(ValidationError):
		VehicleCreate(
			tenant_id="t1", vehicle_type="van",
			registration="X", vin="12345678901",
			make="M", model="A", year=2020,
			fuel_type="petrol", ownership_type="owned",
			bogus_field="should_fail",
		)


# ──────────────────────────────────────────────────────────────────
# DriverCreate
# ──────────────────────────────────────────────────────────────────

def test_driver_create_valid():
	d = DriverCreate(
		tenant_id="t1",
		name="Jane Otieno",
		licence_number="DLN-999",
		licence_class=LicenceClass.CE,
		licence_expiry=FUTURE,
	)
	assert d.name == "Jane Otieno"
	assert d.licence_class == LicenceClass.CE


def test_driver_create_empty_name_rejected():
	with pytest.raises(ValidationError):
		DriverCreate(
			tenant_id="t1", name="   ",
			licence_number="DLN-001",
			licence_class="ce",
			licence_expiry=FUTURE,
		)


# ──────────────────────────────────────────────────────────────────
# TripCreate
# ──────────────────────────────────────────────────────────────────

def test_trip_create_valid():
	t = TripCreate(
		tenant_id="t1",
		vehicle_id="v1",
		driver_id="d1",
		origin="Nairobi",
		destination="Mombasa",
		planned_departure=datetime.utcnow() + timedelta(hours=1),
		load_kg=Decimal("5000"),
	)
	assert t.origin == "Nairobi"
	assert t.load_kg == Decimal("5000")


def test_trip_create_empty_origin_rejected():
	with pytest.raises(ValidationError):
		TripCreate(
			tenant_id="t1", vehicle_id="v1", driver_id="d1",
			origin="  ", destination="Mombasa",
			planned_departure=datetime.utcnow() + timedelta(hours=1),
		)


# ──────────────────────────────────────────────────────────────────
# FuelRecordCreate
# ──────────────────────────────────────────────────────────────────

def test_fuel_record_valid():
	f = FuelRecordCreate(
		tenant_id="t1",
		vehicle_id="v1",
		litres=Decimal("120.5"),
		cost_per_litre=Decimal("185.00"),
		odometer_km=Decimal("50000"),
	)
	assert f.litres == Decimal("120.5")


def test_fuel_record_zero_litres_rejected():
	with pytest.raises(ValidationError):
		FuelRecordCreate(
			tenant_id="t1", vehicle_id="v1",
			litres=Decimal("0"), cost_per_litre=Decimal("185"),
			odometer_km=Decimal("50000"),
		)


# ──────────────────────────────────────────────────────────────────
# MaintenanceCreate
# ──────────────────────────────────────────────────────────────────

def test_maintenance_create_valid():
	m = MaintenanceCreate(
		tenant_id="t1",
		vehicle_id="v1",
		maintenance_type=MaintenanceType.SCHEDULED,
		description="Oil change + filter",
		scheduled_date=datetime.utcnow() + timedelta(days=14),
		estimated_cost=Decimal("8500"),
	)
	assert m.maintenance_type == MaintenanceType.SCHEDULED


# ──────────────────────────────────────────────────────────────────
# TachographRecordCreate
# ──────────────────────────────────────────────────────────────────

def test_tachograph_record_valid():
	now = datetime.utcnow()
	r = TachographRecordCreate(
		tenant_id="t1",
		vehicle_id="v1",
		driver_id="d1",
		period_start=now - timedelta(hours=9),
		period_end=now,
		mode=TachographMode.DRIVING,
		driving_minutes=270,
		break_minutes=45,
		distance_km=Decimal("320"),
	)
	assert r.driving_minutes == 270


# ──────────────────────────────────────────────────────────────────
# TelematicsEventCreate
# ──────────────────────────────────────────────────────────────────

def test_telematics_event_valid():
	e = TelematicsEventCreate(
		tenant_id="t1",
		vehicle_id="v1",
		event_type="position",
		lat=-1.2921,
		lon=36.8219,
		speed_kmh=60.0,
	)
	assert e.event_type == "position"
	assert e.lat == pytest.approx(-1.2921)


# ──────────────────────────────────────────────────────────────────
# IncidentCreate
# ──────────────────────────────────────────────────────────────────

def test_incident_create_valid():
	i = IncidentCreate(
		tenant_id="t1",
		vehicle_id="v1",
		occurred_at=datetime.utcnow() - timedelta(hours=1),
		severity=IncidentSeverity.MINOR,
		description="Minor fender bender at depot",
	)
	assert i.severity == IncidentSeverity.MINOR


# ──────────────────────────────────────────────────────────────────
# InsurancePolicyCreate
# ──────────────────────────────────────────────────────────────────

def test_insurance_policy_valid():
	p = InsurancePolicyCreate(
		tenant_id="t1",
		vehicle_id="v1",
		policy_number="POL-2024-001",
		insurer="Jubilee Insurance",
		policy_type="comprehensive",
		cover_start=datetime.utcnow(),
		cover_end=datetime.utcnow() + timedelta(days=365),
		premium=Decimal("45000"),
	)
	assert p.policy_number == "POL-2024-001"
