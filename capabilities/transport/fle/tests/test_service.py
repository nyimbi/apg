"""
Service-layer tests for FLE Fleet Management.

Uses real FleetService with in-memory store — no mocks.
Async via plain asyncio.get_event_loop().run_until_complete().
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from capabilities.transport.fle.models import (
	COFInspectionCreate, DriverCreate, FuelRecordCreate,
	IncidentCreate, IncidentSeverity, InspectionCreate, InspectionResult,
	InspectionType, InsurancePolicyCreate, MaintenanceCreate,
	MaintenanceStatus, MaintenanceType, RegistrationCreate,
	TachographRecordCreate, TachographMode, TelematicsEventCreate,
	TripCreate, TripStatus, VehicleAssignmentCreate, VehicleCreate,
	VehicleStatus, VehicleUpdate, DriverUpdate,
)
from capabilities.transport.fle.service import FleetService


class _DB:
	pass


FUTURE = datetime.utcnow() + timedelta(days=365 * 3)
loop = asyncio.get_event_loop()


def run(coro):
	return loop.run_until_complete(coro)


def make_svc(tenant="t1", actor="actor"):
	return FleetService(_DB(), tenant, actor)


def reg_vehicle(svc, registration="KCA 001T", vin="WAUZZZ8K9BA000001"):
	return run(svc.register_vehicle(VehicleCreate(
		tenant_id=svc._tenant_id,
		vehicle_type="rigid_truck",
		registration=registration,
		vin=vin,
		make="Mercedes", model="Actros", year=2022,
		fuel_type="diesel", ownership_type="owned",
		gross_vehicle_weight_kg=Decimal("26000"),
		payload_capacity_kg=Decimal("16000"),
	)))


def reg_driver(svc, name="John Kamau", licence="DLN-001"):
	return run(svc.register_driver(DriverCreate(
		tenant_id=svc._tenant_id,
		name=name,
		licence_number=licence,
		licence_class="ce",
		licence_expiry=FUTURE,
		cpc_expiry=FUTURE,
		tacho_card_number="TC-001",
	)))


def make_trip(svc, v_id, d_id, origin="Nairobi", dest="Mombasa"):
	return run(svc.plan_trip(TripCreate(
		tenant_id=svc._tenant_id,
		vehicle_id=v_id,
		driver_id=d_id,
		origin=origin,
		destination=dest,
		planned_departure=datetime.utcnow() + timedelta(hours=1),
		planned_arrival=datetime.utcnow() + timedelta(hours=9),
		load_kg=Decimal("5000"),
	)))


# ──────────────────────────────────────────────────────────────────
# Vehicle CRUD
# ──────────────────────────────────────────────────────────────────

def test_register_vehicle():
	svc = make_svc()
	v = reg_vehicle(svc)
	assert v.registration == "KCA 001T"
	assert v.status == VehicleStatus.ACTIVE
	assert v.tenant_id == "t1"


def test_get_vehicle():
	svc = make_svc()
	v = reg_vehicle(svc)
	fetched = run(svc.get_vehicle(v.id))
	assert fetched.id == v.id


def test_list_vehicles_empty():
	svc = make_svc()
	assert run(svc.list_vehicles()) == []


def test_list_vehicles_status_filter():
	svc = make_svc()
	v = reg_vehicle(svc)
	run(svc.set_vehicle_status(v.id, VehicleStatus.IN_MAINTENANCE))
	assert len(run(svc.list_vehicles(status=VehicleStatus.ACTIVE))) == 0
	assert len(run(svc.list_vehicles(status=VehicleStatus.IN_MAINTENANCE))) == 1


def test_update_vehicle():
	svc = make_svc()
	v = reg_vehicle(svc)
	updated = run(svc.update_vehicle(v.id, VehicleUpdate(colour="White", notes="Depot A")))
	assert updated.colour == "White"
	assert updated.notes == "Depot A"


def test_delete_vehicle():
	svc = make_svc()
	v = reg_vehicle(svc)
	run(svc.delete_vehicle(v.id))
	assert run(svc.list_vehicles()) == []


def test_duplicate_vin_rejected():
	svc = make_svc()
	reg_vehicle(svc, vin="VIN999999999999")
	with pytest.raises(Exception):
		reg_vehicle(svc, registration="KBC 002T", vin="VIN999999999999")


def test_cross_tenant_isolation():
	svc1 = make_svc("tenant_A")
	svc2 = make_svc("tenant_B")
	reg_vehicle(svc1)
	assert run(svc2.list_vehicles()) == []


# ──────────────────────────────────────────────────────────────────
# Driver CRUD
# ──────────────────────────────────────────────────────────────────

def test_register_driver():
	svc = make_svc()
	d = reg_driver(svc)
	assert d.name == "John Kamau"
	assert d.licence_class.value == "ce"


def test_expired_licence_rejected():
	svc = make_svc()
	with pytest.raises(Exception):
		run(svc.register_driver(DriverCreate(
			tenant_id="t1",
			name="Bad Driver",
			licence_number="DLN-BAD",
			licence_class="b",
			licence_expiry=datetime.utcnow() - timedelta(days=1),
		)))


def test_driver_list():
	svc = make_svc()
	reg_driver(svc, name="A", licence="L1")
	reg_driver(svc, name="B", licence="L2")
	assert len(run(svc.list_drivers())) == 2


def test_update_driver():
	svc = make_svc()
	d = reg_driver(svc)
	updated = run(svc.update_driver(d.id, DriverUpdate(phone="+254700000001")))
	assert updated.phone == "+254700000001"


def test_delete_driver():
	svc = make_svc()
	d = reg_driver(svc)
	run(svc.delete_driver(d.id))
	assert run(svc.list_drivers()) == []


# ──────────────────────────────────────────────────────────────────
# Assignment
# ──────────────────────────────────────────────────────────────────

def test_assign_driver():
	svc = make_svc()
	v = reg_vehicle(svc)
	d = reg_driver(svc)
	a = run(svc.assign_driver(VehicleAssignmentCreate(
		tenant_id="t1", vehicle_id=v.id, driver_id=d.id,
	)))
	assert a.vehicle_id == v.id
	assert a.is_active is True


def test_assign_inactive_vehicle_rejected():
	svc = make_svc()
	v = reg_vehicle(svc)
	d = reg_driver(svc)
	run(svc.set_vehicle_status(v.id, VehicleStatus.IN_MAINTENANCE))
	with pytest.raises(Exception):
		run(svc.assign_driver(VehicleAssignmentCreate(
			tenant_id="t1", vehicle_id=v.id, driver_id=d.id,
		)))


# ──────────────────────────────────────────────────────────────────
# Trip lifecycle
# ──────────────────────────────────────────────────────────────────

def test_plan_trip():
	svc = make_svc()
	v = reg_vehicle(svc)
	d = reg_driver(svc)
	t = make_trip(svc, v.id, d.id)
	assert t.status == TripStatus.PLANNED
	assert t.origin == "Nairobi"


def test_trip_full_lifecycle():
	svc = make_svc()
	v = reg_vehicle(svc)
	d = reg_driver(svc)
	t = make_trip(svc, v.id, d.id)

	t = run(svc.dispatch_trip(t.id))
	assert t.status == TripStatus.DISPATCHED

	t = run(svc.start_trip(t.id, Decimal("50000")))
	assert t.status == TripStatus.IN_PROGRESS
	assert t.odometer_start_km == Decimal("50000")

	t = run(svc.complete_trip(t.id, Decimal("50450"), Decimal("80.5")))
	assert t.status == TripStatus.COMPLETED
	assert t.distance_km == Decimal("450.00")


def test_cancel_trip():
	svc = make_svc()
	v = reg_vehicle(svc)
	d = reg_driver(svc)
	t = make_trip(svc, v.id, d.id)
	t = run(svc.cancel_trip(t.id, "Load not ready"))
	assert t.status == TripStatus.CANCELLED
	assert t.delay_reason == "Load not ready"


def test_trip_breakdown():
	svc = make_svc()
	v = reg_vehicle(svc)
	d = reg_driver(svc)
	t = make_trip(svc, v.id, d.id)
	run(svc.dispatch_trip(t.id))
	run(svc.start_trip(t.id, Decimal("50000")))
	t = run(svc.record_trip_breakdown(t.id, "Engine failure"))
	assert t.status == TripStatus.BREAKDOWN
	v_updated = run(svc.get_vehicle(v.id))
	assert v_updated.status == VehicleStatus.BREAKDOWN


def test_change_trip_driver():
	svc = make_svc()
	v = reg_vehicle(svc)
	d1 = reg_driver(svc, name="Driver One", licence="L1")
	d2 = reg_driver(svc, name="Driver Two", licence="L2")
	t = make_trip(svc, v.id, d1.id)
	run(svc.dispatch_trip(t.id))
	t = run(svc.change_trip_driver(t.id, d2.id, "Illness"))
	assert t.driver_id == d2.id


def test_overloaded_trip_rejected():
	svc = make_svc()
	v = reg_vehicle(svc)
	d = reg_driver(svc)
	with pytest.raises(Exception):
		run(svc.plan_trip(TripCreate(
			tenant_id="t1",
			vehicle_id=v.id,
			driver_id=d.id,
			origin="Nairobi", destination="Mombasa",
			planned_departure=datetime.utcnow() + timedelta(hours=1),
			load_kg=Decimal("20000"),  # exceeds 16000 kg capacity
		)))


def test_list_trips_filter():
	svc = make_svc()
	v = reg_vehicle(svc)
	d = reg_driver(svc)
	make_trip(svc, v.id, d.id)
	assert len(run(svc.list_trips(status=TripStatus.PLANNED))) == 1
	assert len(run(svc.list_trips(status=TripStatus.COMPLETED))) == 0


# ──────────────────────────────────────────────────────────────────
# Fuel records
# ──────────────────────────────────────────────────────────────────

def test_record_fuel():
	svc = make_svc()
	v = reg_vehicle(svc)
	f = run(svc.record_fuel_purchase(FuelRecordCreate(
		tenant_id="t1",
		vehicle_id=v.id,
		litres=Decimal("120.5"),
		cost_per_litre=Decimal("185.00"),
		odometer_km=Decimal("50000"),
	)))
	assert f.total_cost == Decimal("22292.50")


def test_fuel_odometer_regression_rejected():
	svc = make_svc()
	v = reg_vehicle(svc)
	run(svc.record_fuel_purchase(FuelRecordCreate(
		tenant_id="t1", vehicle_id=v.id,
		litres=Decimal("100"), cost_per_litre=Decimal("180"),
		odometer_km=Decimal("50000"),
	)))
	with pytest.raises(Exception):
		run(svc.record_fuel_purchase(FuelRecordCreate(
			tenant_id="t1", vehicle_id=v.id,
			litres=Decimal("100"), cost_per_litre=Decimal("180"),
			odometer_km=Decimal("49000"),
		)))


# ──────────────────────────────────────────────────────────────────
# Maintenance
# ──────────────────────────────────────────────────────────────────

def test_schedule_maintenance():
	svc = make_svc()
	v = reg_vehicle(svc)
	m = run(svc.schedule_maintenance(MaintenanceCreate(
		tenant_id="t1", vehicle_id=v.id,
		maintenance_type=MaintenanceType.SCHEDULED,
		description="Annual service",
		scheduled_date=datetime.utcnow() + timedelta(days=30),
		estimated_cost=Decimal("12000"),
	)))
	assert m.status == MaintenanceStatus.SCHEDULED


def test_maintenance_lifecycle():
	svc = make_svc()
	v = reg_vehicle(svc)
	m = run(svc.schedule_maintenance(MaintenanceCreate(
		tenant_id="t1", vehicle_id=v.id,
		maintenance_type=MaintenanceType.CORRECTIVE,
		description="Brake replacement",
		scheduled_date=datetime.utcnow(),
	)))
	m = run(svc.start_maintenance(m.id))
	assert m.status == MaintenanceStatus.IN_PROGRESS
	v_now = run(svc.get_vehicle(v.id))
	assert v_now.status == VehicleStatus.IN_MAINTENANCE

	m = run(svc.complete_maintenance(m.id, Decimal("9500")))
	assert m.status == MaintenanceStatus.COMPLETED
	v_now = run(svc.get_vehicle(v.id))
	assert v_now.status == VehicleStatus.ACTIVE


# ──────────────────────────────────────────────────────────────────
# Inspection failure workflow
# ──────────────────────────────────────────────────────────────────

def test_inspection_failure_workflow():
	svc = make_svc()
	v = reg_vehicle(svc)
	insp = run(svc.record_inspection(InspectionCreate(
		tenant_id="t1",
		vehicle_id=v.id,
		inspection_type=InspectionType.ANNUAL,
		result=InspectionResult.FAIL,
		defects=["Worn brake pads", "Cracked windscreen"],
		inspected_at=datetime.utcnow(),
	)))
	assert insp.result == InspectionResult.FAIL
	v_now = run(svc.get_vehicle(v.id))
	assert v_now.status == VehicleStatus.OUT_OF_SERVICE
	maint = run(svc.list_maintenance(vehicle_id=v.id))
	assert len(maint) == 2  # one per defect


def test_inspection_pass_no_maintenance():
	svc = make_svc()
	v = reg_vehicle(svc)
	run(svc.record_inspection(InspectionCreate(
		tenant_id="t1",
		vehicle_id=v.id,
		inspection_type=InspectionType.PRE_TRIP,
		result=InspectionResult.PASS,
		inspected_at=datetime.utcnow(),
	)))
	maint = run(svc.list_maintenance(vehicle_id=v.id))
	assert len(maint) == 0


# ──────────────────────────────────────────────────────────────────
# COF Inspection
# ──────────────────────────────────────────────────────────────────

def test_cof_inspection():
	svc = make_svc()
	v = reg_vehicle(svc)
	cof = run(svc.record_cof_inspection(COFInspectionCreate(
		tenant_id="t1",
		vehicle_id=v.id,
		inspected_at=datetime.utcnow(),
		result=InspectionResult.PASS,
		cof_number="COF-2024-001",
		issued_at=datetime.utcnow(),
		expires_at=datetime.utcnow() + timedelta(days=365),
		inspection_station="NTSA Bay 1",
	)))
	assert cof.cof_number == "COF-2024-001"
	cofs = run(svc.list_cof_inspections(vehicle_id=v.id))
	assert len(cofs) == 1


# ──────────────────────────────────────────────────────────────────
# Incidents
# ──────────────────────────────────────────────────────────────────

def test_report_incident():
	svc = make_svc()
	v = reg_vehicle(svc)
	inc = run(svc.report_incident(IncidentCreate(
		tenant_id="t1",
		vehicle_id=v.id,
		occurred_at=datetime.utcnow() - timedelta(hours=2),
		severity=IncidentSeverity.MINOR,
		description="Kerb strike at roundabout",
	)))
	assert inc.severity == IncidentSeverity.MINOR


def test_fatal_incident_requires_police_ref():
	svc = make_svc()
	v = reg_vehicle(svc)
	with pytest.raises(Exception, match=".*police.*"):
		run(svc.report_incident(IncidentCreate(
			tenant_id="t1",
			vehicle_id=v.id,
			occurred_at=datetime.utcnow() - timedelta(hours=1),
			severity=IncidentSeverity.FATAL,
			description="Serious accident",
			police_ref="",
		)))


def test_critical_incident_sets_vehicle_oos():
	svc = make_svc()
	v = reg_vehicle(svc)
	run(svc.report_incident(IncidentCreate(
		tenant_id="t1",
		vehicle_id=v.id,
		occurred_at=datetime.utcnow() - timedelta(hours=1),
		severity=IncidentSeverity.CRITICAL,
		description="Serious rollover",
		police_ref="OB-001-2024",
	)))
	v_now = run(svc.get_vehicle(v.id))
	assert v_now.status == VehicleStatus.OUT_OF_SERVICE


def test_close_incident():
	svc = make_svc()
	v = reg_vehicle(svc)
	inc = run(svc.report_incident(IncidentCreate(
		tenant_id="t1",
		vehicle_id=v.id,
		occurred_at=datetime.utcnow() - timedelta(hours=1),
		severity=IncidentSeverity.MINOR,
		description="Minor scrape",
	)))
	closed = run(svc.close_incident(inc.id, "Repaired at depot"))
	from capabilities.transport.fle.models import IncidentStatus
	assert closed.status == IncidentStatus.CLOSED


# ──────────────────────────────────────────────────────────────────
# Insurance & Registration
# ──────────────────────────────────────────────────────────────────

def test_add_insurance():
	svc = make_svc()
	v = reg_vehicle(svc)
	pol = run(svc.add_insurance_policy(InsurancePolicyCreate(
		tenant_id="t1",
		vehicle_id=v.id,
		policy_number="POL-2024-001",
		insurer="Jubilee Insurance",
		policy_type="comprehensive",
		cover_start=datetime.utcnow(),
		cover_end=datetime.utcnow() + timedelta(days=365),
		premium=Decimal("45000"),
	)))
	assert pol.policy_number == "POL-2024-001"


def test_add_registration():
	svc = make_svc()
	v = reg_vehicle(svc)
	reg = run(svc.register_vehicle_docs(RegistrationCreate(
		tenant_id="t1",
		vehicle_id=v.id,
		registration_number="KCA 001T",
		issued_at=datetime.utcnow() - timedelta(days=30),
		expires_at=datetime.utcnow() + timedelta(days=335),
	)))
	assert reg.registration_number == "KCA 001T"
	assert reg.is_current is True


# ──────────────────────────────────────────────────────────────────
# Tachograph / EU HOS
# ──────────────────────────────────────────────────────────────────

def test_tachograph_record():
	svc = make_svc()
	v = reg_vehicle(svc)
	d = reg_driver(svc)
	now = datetime.utcnow()
	rec = run(svc.record_tachograph(TachographRecordCreate(
		tenant_id="t1",
		vehicle_id=v.id,
		driver_id=d.id,
		period_start=now - timedelta(hours=5),
		period_end=now,
		mode=TachographMode.DRIVING,
		driving_minutes=270,
		break_minutes=45,
		distance_km=Decimal("320"),
	)))
	assert rec.driving_minutes == 270


def test_tachograph_continuous_driving_limit():
	svc = make_svc()
	v = reg_vehicle(svc)
	d = reg_driver(svc)
	now = datetime.utcnow()
	with pytest.raises(Exception):
		run(svc.record_tachograph(TachographRecordCreate(
			tenant_id="t1",
			vehicle_id=v.id,
			driver_id=d.id,
			period_start=now - timedelta(hours=6),
			period_end=now,
			mode=TachographMode.DRIVING,
			driving_minutes=350,  # exceeds 270 min EU limit
			break_minutes=0,
			distance_km=Decimal("420"),
		)))


# ──────────────────────────────────────────────────────────────────
# Telematics
# ──────────────────────────────────────────────────────────────────

def test_track_vehicle_realtime():
	svc = make_svc()
	v = reg_vehicle(svc)
	ev = run(svc.track_vehicle_realtime(TelematicsEventCreate(
		tenant_id="t1",
		vehicle_id=v.id,
		event_type="position",
		lat=-1.2921, lon=36.8219,
		speed_kmh=75.0,
	)))
	assert ev.speed_kmh == 75.0


def test_get_last_position():
	svc = make_svc()
	v = reg_vehicle(svc)
	run(svc.track_vehicle_realtime(TelematicsEventCreate(
		tenant_id="t1", vehicle_id=v.id,
		event_type="position", lat=-1.29, lon=36.82, speed_kmh=60.0,
	)))
	pos = run(svc.get_vehicle_last_position(v.id))
	assert pos is not None
	assert pos.vehicle_id == v.id


def test_get_last_position_no_data():
	svc = make_svc()
	v = reg_vehicle(svc)
	assert run(svc.get_vehicle_last_position(v.id)) is None


def test_list_telematics_filter_by_event_type():
	svc = make_svc()
	v = reg_vehicle(svc)
	run(svc.track_vehicle_realtime(TelematicsEventCreate(
		tenant_id="t1", vehicle_id=v.id,
		event_type="position", lat=-1.0, lon=36.0, speed_kmh=60.0,
	)))
	run(svc.track_vehicle_realtime(TelematicsEventCreate(
		tenant_id="t1", vehicle_id=v.id,
		event_type="harsh_braking", lat=-1.0, lon=36.0, speed_kmh=0.0,
	)))
	braking = run(svc.list_telematics_events(vehicle_id=v.id, event_type="harsh_braking"))
	assert len(braking) == 1


# ──────────────────────────────────────────────────────────────────
# TCO
# ──────────────────────────────────────────────────────────────────

def test_calculate_tco():
	svc = make_svc()
	v = reg_vehicle(svc)
	run(svc.record_fuel_purchase(FuelRecordCreate(
		tenant_id="t1", vehicle_id=v.id,
		litres=Decimal("500"), cost_per_litre=Decimal("185"),
		odometer_km=Decimal("50000"),
	)))
	tco = run(svc.calculate_tco(v.id))
	assert tco.fuel_cost == Decimal("92500.00")
	assert tco.total_cost >= tco.fuel_cost


# ──────────────────────────────────────────────────────────────────
# Driver behaviour scoring
# ──────────────────────────────────────────────────────────────────

def test_driver_score_no_events():
	svc = make_svc()
	d = reg_driver(svc)
	score = run(svc.driver_behaviour_scoring(d.id))
	assert score.overall_score == 0.0


def test_driver_score_with_trips_and_events():
	svc = make_svc()
	v = reg_vehicle(svc)
	d = reg_driver(svc)

	t = make_trip(svc, v.id, d.id)
	run(svc.dispatch_trip(t.id))
	run(svc.start_trip(t.id, Decimal("50000")))
	run(svc.complete_trip(t.id, Decimal("50500")))

	run(svc.track_vehicle_realtime(TelematicsEventCreate(
		tenant_id="t1", vehicle_id=v.id, driver_id=d.id,
		event_type="speeding", lat=-1.29, lon=36.82, speed_kmh=130.0,
	)))

	score = run(svc.driver_behaviour_scoring(d.id))
	assert score.trips_count == 1
	assert score.distance_km == Decimal("500.00")
	assert 0 <= score.overall_score <= 100


# ──────────────────────────────────────────────────────────────────
# Compliance calendar
# ──────────────────────────────────────────────────────────────────

def test_compliance_calendar_insurance_soon():
	svc = make_svc()
	v = reg_vehicle(svc)
	run(svc.add_insurance_policy(InsurancePolicyCreate(
		tenant_id="t1",
		vehicle_id=v.id,
		policy_number="POL-001",
		insurer="Jubilee",
		policy_type="comprehensive",
		cover_start=datetime.utcnow() - timedelta(days=300),
		cover_end=datetime.utcnow() + timedelta(days=20),
		premium=Decimal("45000"),
	)))
	calendar = run(svc.compliance_calendar())
	ins_entries = [e for e in calendar if e.event_type == "insurance_renewal"]
	assert len(ins_entries) == 1
	assert ins_entries[0].severity in ("warning", "critical")


def test_compliance_calendar_driver_licence():
	svc = make_svc()
	expiring_soon = datetime.utcnow() + timedelta(days=15)
	run(svc.register_driver(DriverCreate(
		tenant_id="t1",
		name="Expiring Driver",
		licence_number="DLN-EXPIRE",
		licence_class="b",
		licence_expiry=expiring_soon,
	)))
	calendar = run(svc.compliance_calendar())
	lic_entries = [e for e in calendar if e.event_type == "licence_expiry"]
	assert len(lic_entries) == 1
	assert lic_entries[0].days_until_due <= 15


# ──────────────────────────────────────────────────────────────────
# Predictive maintenance
# ──────────────────────────────────────────────────────────────────

def test_predictive_maintenance_overdue():
	svc = make_svc()
	v = reg_vehicle(svc)
	run(svc.schedule_maintenance(MaintenanceCreate(
		tenant_id="t1", vehicle_id=v.id,
		maintenance_type=MaintenanceType.SCHEDULED,
		description="Overdue oil change",
		scheduled_date=datetime.utcnow() - timedelta(days=30),
	)))
	alerts = run(svc.predictive_maintenance_alerts())
	critical = [a for a in alerts if a.urgency == "critical"]
	assert len(critical) >= 1


# ──────────────────────────────────────────────────────────────────
# Dashboard KPIs
# ──────────────────────────────────────────────────────────────────

def test_dashboard_kpis_empty():
	svc = make_svc()
	kpis = run(svc.dashboard_kpis())
	assert kpis.total_vehicles == 0
	assert kpis.total_drivers == 0


def test_dashboard_kpis_with_data():
	svc = make_svc()
	reg_vehicle(svc)
	reg_driver(svc)
	kpis = run(svc.dashboard_kpis())
	assert kpis.total_vehicles == 1
	assert kpis.total_drivers == 1
	assert kpis.active_vehicles == 1


# ──────────────────────────────────────────────────────────────────
# Fleet utilisation
# ──────────────────────────────────────────────────────────────────

def test_fleet_utilisation():
	svc = make_svc()
	reg_vehicle(svc, registration="KCA 001T", vin="VIN000000000001")
	reg_vehicle(svc, registration="KCA 002T", vin="VIN000000000002")
	report = run(svc.fleet_utilisation_analytics())
	assert report.total_vehicles == 2
	assert report.active_vehicles == 2
	assert report.avg_utilisation_pct == 100.0


# ──────────────────────────────────────────────────────────────────
# Domain events emitted
# ──────────────────────────────────────────────────────────────────

def test_events_emitted_on_register():
	svc = make_svc()
	assert len(svc._events) == 0
	reg_vehicle(svc)
	assert any(e["event_type"] == "vehicle.registered" for e in svc._events)


def test_events_emitted_on_trip_lifecycle():
	svc = make_svc()
	v = reg_vehicle(svc)
	d = reg_driver(svc)
	t = make_trip(svc, v.id, d.id)
	run(svc.dispatch_trip(t.id))
	run(svc.start_trip(t.id, Decimal("50000")))
	run(svc.complete_trip(t.id, Decimal("50500")))
	types = [e["event_type"] for e in svc._events]
	assert "trip.planned" in types
	assert "trip.dispatched" in types
	assert "trip.started" in types
	assert "trip.completed" in types
