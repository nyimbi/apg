"""Transport capability integration tests: FleetService (fle).

All tests are sync; async service methods called via asyncio.run().
Uses the in-memory store (plain object as db_session) — zero config.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal


# ── helpers ───────────────────────────────────────────────────────────────────

class _DB:
	"""Minimal stub that satisfies FleetService's _store() duck-type."""
	pass


def _svc():
	from capabilities.transport.fle.service import FleetService
	return FleetService(db_session=_DB(), tenant_id="test-tenant", actor_id="test-actor")


def _vehicle_payload(tenant_id: str = "test-tenant", vin: str = "VIN001TEST0000001"):
	from capabilities.transport.fle.models import (
		VehicleCreate, VehicleType, FuelType, OwnershipType,
	)
	return VehicleCreate(
		tenant_id=tenant_id,
		vehicle_type=VehicleType.VAN,
		registration="KCA 001A",
		vin=vin,
		make="Toyota",
		model="Hiace",
		year=2022,
		fuel_type=FuelType.DIESEL,
		ownership_type=OwnershipType.OWNED,
	)


def _driver_payload(tenant_id: str = "test-tenant"):
	from capabilities.transport.fle.models import DriverCreate, LicenceClass
	return DriverCreate(
		tenant_id=tenant_id,
		name="John Driver",
		licence_number="DL-TEST-001",
		licence_class=LicenceClass.C,
		licence_expiry=datetime.utcnow() + timedelta(days=730),
	)


# ── 1. vehicle registration ───────────────────────────────────────────────────

def test_fleet_vehicle_registration():
	"""FleetService can register a vehicle and returns a VehicleResponse."""
	svc = _svc()
	payload = _vehicle_payload()
	result = asyncio.run(svc.register_vehicle(payload))
	assert result.registration == "KCA 001A"
	assert result.vin == "VIN001TEST0000001"
	assert result.make == "Toyota"
	assert result.tenant_id == "test-tenant"
	assert result.id


# ── 2. driver assignment ──────────────────────────────────────────────────────

def test_fleet_assignment():
	"""Assign a driver to a vehicle; returns VehicleAssignmentResponse."""
	from capabilities.transport.fle.models import VehicleAssignmentCreate
	svc = _svc()

	vehicle = asyncio.run(svc.register_vehicle(_vehicle_payload()))
	driver = asyncio.run(svc.register_driver(_driver_payload()))

	assignment_payload = VehicleAssignmentCreate(
		tenant_id="test-tenant",
		vehicle_id=vehicle.id,
		driver_id=driver.id,
	)
	result = asyncio.run(svc.assign_driver(assignment_payload))
	assert result.vehicle_id == vehicle.id
	assert result.driver_id == driver.id
	assert result.is_active is True


# ── 3. trip planning and tracking ────────────────────────────────────────────

def test_fleet_trip_tracking():
	"""Plan a trip, dispatch it, start it, then complete it."""
	from capabilities.transport.fle.models import TripCreate
	svc = _svc()

	vehicle = asyncio.run(svc.register_vehicle(_vehicle_payload()))
	driver = asyncio.run(svc.register_driver(_driver_payload()))

	trip_payload = TripCreate(
		tenant_id="test-tenant",
		vehicle_id=vehicle.id,
		driver_id=driver.id,
		origin="Nairobi",
		destination="Mombasa",
		planned_departure=datetime.utcnow() + timedelta(hours=1),
	)
	trip = asyncio.run(svc.plan_trip(trip_payload))
	assert trip.status.value == "planned"

	dispatched = asyncio.run(svc.dispatch_trip(trip.id))
	assert dispatched.status.value == "dispatched"

	started = asyncio.run(svc.start_trip(trip.id, Decimal("10000")))
	assert started.status.value == "in_progress"

	completed = asyncio.run(svc.complete_trip(trip.id, Decimal("10480"), Decimal("45")))
	assert completed.status.value == "completed"
	assert completed.distance_km is not None


# ── 4. maintenance scheduling ─────────────────────────────────────────────────

def test_fleet_maintenance_schedule():
	"""Schedule maintenance, start it, complete it; vehicle status transitions correctly."""
	from capabilities.transport.fle.models import MaintenanceCreate, MaintenanceType
	svc = _svc()

	vehicle = asyncio.run(svc.register_vehicle(_vehicle_payload()))

	maint_payload = MaintenanceCreate(
		tenant_id="test-tenant",
		vehicle_id=vehicle.id,
		maintenance_type=MaintenanceType.SCHEDULED,
		description="6-month service",
		scheduled_date=datetime.utcnow() + timedelta(days=1),
		estimated_cost=Decimal("15000"),
	)
	scheduled = asyncio.run(svc.schedule_maintenance(maint_payload))
	assert scheduled.status.value == "scheduled"

	started = asyncio.run(svc.start_maintenance(scheduled.id))
	assert started.status.value == "in_progress"

	completed = asyncio.run(svc.complete_maintenance(scheduled.id, Decimal("14500"), "All done"))
	assert completed.status.value == "completed"
	assert float(completed.actual_cost) == 14500.0


# ── 5. rule evaluation — allow ────────────────────────────────────────────────

def test_fleet_rule_evaluation():
	"""evaluate_rules('transport_fle', context) returns allow for valid read context."""
	from capabilities.transport.fle.capability_contract import evaluate_capability_rules

	ctx = {"tenant_context_present": True, "operation_type": "read"}
	result = evaluate_capability_rules(ctx)
	assert result["decision"] == "allow"


# ── 6. fuel management ────────────────────────────────────────────────────────

def test_fleet_fuel_management():
	"""Record a fuel purchase; total_cost is computed and odometer is updated."""
	from capabilities.transport.fle.models import FuelRecordCreate
	svc = _svc()

	vehicle = asyncio.run(svc.register_vehicle(_vehicle_payload()))

	fuel_payload = FuelRecordCreate(
		tenant_id="test-tenant",
		vehicle_id=vehicle.id,
		litres=Decimal("60"),
		cost_per_litre=Decimal("185"),
		odometer_km=Decimal("5000"),
		station_name="Total Westlands",
	)
	record = asyncio.run(svc.record_fuel_purchase(fuel_payload))
	assert float(record.total_cost) == 60 * 185
	assert float(record.litres) == 60.0
	assert record.id


# ── 7. manifest navigation ────────────────────────────────────────────────────

def test_fleet_manifest_navigation():
	"""Transport domain contains exactly 10 capabilities in the manifest."""
	from capabilities.manifest import get_domain
	caps = get_domain("transport")
	assert len(caps) == 10, f"expected 10 transport capabilities, got {len(caps)}"
	# manifest entries use 'id' key for capability id
	ids = {c.get("capability_id") or c.get("id") or c.get("code") for c in caps}
	assert any("fle" in str(cid) for cid in ids), f"transport_fle not found in: {ids}"


# ── 8. composability — all requires satisfied ─────────────────────────────────

def test_transport_composability():
	"""transport_fle REQUIRES list is non-empty and all entries are known APG codes."""
	from capabilities.transport.fle.capability_contract import REQUIRES
	known_codes = {
		"auth", "audl", "mten", "conf", "ntfy", "wflo", "moni",
		"comp", "mqeb", "schd", "nlpc", "keym", "stor",
	}
	assert len(REQUIRES) > 0, "REQUIRES must not be empty"
	for req in REQUIRES:
		assert req in known_codes, f"unknown requirement: {req}"
