"""Service tests for Space Planning & Management (spa)."""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
from decimal import Decimal

import pytest

from capabilities.realestate.spa.service import SpaService
from capabilities.realestate.spa.models import (
	FloorPlanCreate,
	SpaceCreate, SpaceUpdate, SpaceType, SpaceStatus,
	SpaceAllocationCreate, AllocationType,
	MoveCreate, MoveType, MoveStatus,
	BookingCreate, BookingType,
	OccupancyDataCreate, SensorType,
	DensityPlanCreate, DensityBand,
)

loop = asyncio.get_event_loop()
T = "test-tenant"


def _svc():
	return SpaService()


def _floor_plan(svc, floor="G"):
	return loop.run_until_complete(svc.upload_floor_plan(FloorPlanCreate(
		tenant_id=T, property_id="prop-1", floor=floor,
		file_format="pdf", file_reference="plans/gf.pdf",
		total_area=Decimal("2000"), created_by="u",
	)))


def _space(svc, fp_id, **kwargs):
	defaults = dict(
		tenant_id=T, property_id="prop-1", floor_plan_id=fp_id,
		space_ref="S-001", space_type=SpaceType.open_plan,
		capacity=20, area=Decimal("200"), created_by="u",
	)
	defaults.update(kwargs)
	return loop.run_until_complete(svc.create_space(SpaceCreate(**defaults)))


# ── Floor Plan ────────────────────────────────────────────────────────────────

def test_upload_floor_plan():
	svc = _svc()
	fp = _floor_plan(svc)
	assert fp.id
	assert fp.version == 1


def test_floor_plan_version_increments():
	svc = _svc()
	fp1 = _floor_plan(svc, floor="1")
	fp2 = _floor_plan(svc, floor="1")
	assert fp2.version == 2


def test_list_floor_plans_by_property():
	svc = _svc()
	_floor_plan(svc, floor="G")
	_floor_plan(svc, floor="1")
	fps = loop.run_until_complete(svc.list_floor_plans(T, property_id="prop-1"))
	assert len(fps) == 2


# ── Space ─────────────────────────────────────────────────────────────────────

def test_create_space():
	svc = _svc()
	fp = _floor_plan(svc)
	s = _space(svc, fp.id)
	assert s.id
	assert s.status == SpaceStatus.available
	# Floor plan should reference the space
	fp_fetched = loop.run_until_complete(svc.get_floor_plan(fp.id, T))
	assert s.id in fp_fetched.space_ids


def test_list_available_spaces_by_capacity():
	svc = _svc()
	fp = _floor_plan(svc)
	_space(svc, fp.id, space_ref="S-1", capacity=5)
	_space(svc, fp.id, space_ref="S-2", capacity=20)
	_space(svc, fp.id, space_ref="S-3", capacity=50)
	large = loop.run_until_complete(svc.get_available_spaces(T, "prop-1", min_capacity=15))
	assert len(large) == 2


def test_update_space_status():
	svc = _svc()
	fp = _floor_plan(svc)
	s = _space(svc, fp.id)
	updated = loop.run_until_complete(svc.update_space(s.id, T, SpaceUpdate(status=SpaceStatus.under_fit_out)))
	assert updated.status == SpaceStatus.under_fit_out


# ── Allocation ────────────────────────────────────────────────────────────────

def test_allocate_and_deallocate_space():
	svc = _svc()
	fp = _floor_plan(svc)
	s = _space(svc, fp.id)
	alloc = loop.run_until_complete(svc.allocate_space(SpaceAllocationCreate(
		tenant_id=T, space_id=s.id, allocation_type=AllocationType.permanent,
		department_id="dept-engineering", occupant_ids=["emp-1", "emp-2"],
		start_date=date(2025, 1, 1), headcount=2, created_by="u",
	)))
	assert alloc.is_active is True
	space = loop.run_until_complete(svc.get_space(s.id, T))
	assert space.status == SpaceStatus.occupied
	# Deallocate
	deallocated = loop.run_until_complete(svc.deallocate_space(alloc.id, T))
	assert deallocated.is_active is False
	space = loop.run_until_complete(svc.get_space(s.id, T))
	assert space.status == SpaceStatus.available


# ── Move ──────────────────────────────────────────────────────────────────────

def test_small_move_auto_approved():
	svc = _svc()
	fp = _floor_plan(svc)
	s1 = _space(svc, fp.id, space_ref="S-A")
	s2 = _space(svc, fp.id, space_ref="S-B")
	move = loop.run_until_complete(svc.create_move(MoveCreate(
		tenant_id=T, move_type=MoveType.internal_move,
		from_space_ids=[s1.id], to_space_ids=[s2.id],
		occupant_ids=["emp-1"], headcount=5,
		scheduled_date=date(2025, 3, 1), created_by="u",
	)))
	assert move.status == MoveStatus.scheduled


def test_large_move_requires_approval():
	svc = _svc()
	fp = _floor_plan(svc)
	s1 = _space(svc, fp.id, space_ref="S-C", capacity=50)
	s2 = _space(svc, fp.id, space_ref="S-D", capacity=50)
	move = loop.run_until_complete(svc.create_move(MoveCreate(
		tenant_id=T, move_type=MoveType.inter_floor_move,
		from_space_ids=[s1.id], to_space_ids=[s2.id],
		occupant_ids=[f"e-{i}" for i in range(25)],
		headcount=25, scheduled_date=date(2025, 4, 1), created_by="u",
	)))
	assert move.status == MoveStatus.planning
	approved = loop.run_until_complete(svc.approve_move(move.id, T, "coo"))
	assert approved.status == MoveStatus.approved


# ── Booking ───────────────────────────────────────────────────────────────────

def test_create_booking():
	svc = _svc()
	fp = _floor_plan(svc)
	s = _space(svc, fp.id, space_type=SpaceType.meeting_room, capacity=10)
	start = datetime.utcnow() + timedelta(hours=1)
	end = start + timedelta(hours=2)
	booking = loop.run_until_complete(svc.create_booking(BookingCreate(
		tenant_id=T, space_id=s.id, booking_type=BookingType.meeting_room,
		booked_by="emp-1", start_datetime=start, end_datetime=end,
		attendees=5, created_by="u",
	)))
	assert booking.status == "confirmed"


def test_double_booking_raises():
	svc = _svc()
	fp = _floor_plan(svc)
	s = _space(svc, fp.id, space_type=SpaceType.meeting_room, capacity=10)
	start = datetime.utcnow() + timedelta(hours=2)
	end = start + timedelta(hours=1)
	loop.run_until_complete(svc.create_booking(BookingCreate(
		tenant_id=T, space_id=s.id, booking_type=BookingType.meeting_room,
		booked_by="emp-1", start_datetime=start, end_datetime=end, created_by="u",
	)))
	with pytest.raises(ValueError):
		loop.run_until_complete(svc.create_booking(BookingCreate(
			tenant_id=T, space_id=s.id, booking_type=BookingType.meeting_room,
			booked_by="emp-2", start_datetime=start, end_datetime=end, created_by="u",
		)))


# ── Occupancy ─────────────────────────────────────────────────────────────────

def test_ingest_occupancy_data():
	svc = _svc()
	fp = _floor_plan(svc)
	s = _space(svc, fp.id)
	rec = loop.run_until_complete(svc.ingest_occupancy_data(OccupancyDataCreate(
		tenant_id=T, space_id=s.id, sensor_type=SensorType.occupancy_sensor,
		recorded_at=datetime.utcnow(), occupant_count=12,
		data_anonymised=True, created_by="u",
	)))
	assert rec.id


def test_ingest_unanonymised_data_raises():
	svc = _svc()
	fp = _floor_plan(svc)
	s = _space(svc, fp.id)
	with pytest.raises(ValueError):
		loop.run_until_complete(svc.ingest_occupancy_data(OccupancyDataCreate(
			tenant_id=T, space_id=s.id, sensor_type=SensorType.wifi_probe,
			recorded_at=datetime.utcnow(), occupant_count=8,
			data_anonymised=False, created_by="u",
		)))


# ── Density ───────────────────────────────────────────────────────────────────

def test_create_density_plan():
	svc = _svc()
	plan = loop.run_until_complete(svc.create_density_plan(DensityPlanCreate(
		tenant_id=T, property_id="prop-1", density_band=DensityBand.standard,
		target_sqm_per_person=Decimal("10"), workplace_strategy="activity_based_working",
		effective_date=date(2025, 1, 1), created_by="u",
	)))
	assert plan.id
