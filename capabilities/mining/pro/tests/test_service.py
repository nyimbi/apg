"""Tests for ProService."""

from __future__ import annotations

import asyncio
from datetime import datetime

import pytest

from capabilities.mining.pro.models import (
	BlastCreate,
	BlastType,
	GradeBoundaryCreate,
	GradeControlMethod,
	MaterialType,
	OreTrackingMethod,
	ProductionActivityCreate,
	ProductionScheduleCreate,
	ScheduleType,
	ShiftReportCreate,
	ShiftReportUpdate,
	ReportStatus,
	StockpileCreate,
	StockpileMovementCreate,
	StockpileType,
	ShiftType,
)
from capabilities.mining.pro.service import ProService

TENANT = "test_mine"


def run(coro):
	return asyncio.get_event_loop().run_until_complete(coro)


def make_service():
	return ProService(tenant_id=TENANT)


def make_shift() -> ShiftReportCreate:
	return ShiftReportCreate(
		tenant_id=TENANT,
		shift_type=ShiftType.DAY,
		shift_date=datetime(2026, 1, 15),
		shift_start=datetime(2026, 1, 15, 6, 0),
		shift_end=datetime(2026, 1, 15, 18, 0),
		mine_area="open_pit_north",
		supervisor_id="super_001",
		operator_count=12,
		activities=[
			ProductionActivityCreate(
				area="pit_bench_3",
				material_type=MaterialType.ORE,
				planned_tonnes=5000.0,
				actual_tonnes=4800.0,
				grade_value=1.5,
				grade_units="g/t",
				tracking_method=OreTrackingMethod.WEIGHBRIDGE,
			)
		],
	)


def test_create_shift_report():
	svc = make_service()
	result = run(svc.create_shift_report(make_shift(), created_by="super_001"))
	assert result.tenant_id == TENANT
	assert result.total_ore_tonnes == 4800.0
	assert result.status == ReportStatus.DRAFT


def test_shift_report_future_date_rejected():
	svc = make_service()
	future_shift = ShiftReportCreate(
		tenant_id=TENANT,
		shift_type=ShiftType.DAY,
		shift_date=datetime(2099, 1, 15),
		shift_start=datetime(2099, 1, 15, 6, 0),
		shift_end=datetime(2099, 1, 15, 18, 0),
		mine_area="open_pit_north",
		supervisor_id="super_001",
		operator_count=5,
	)
	with pytest.raises(ValueError, match="future"):
		run(svc.create_shift_report(future_shift, created_by="super_001"))


def test_submit_and_approve_shift():
	svc = make_service()
	shift = run(svc.create_shift_report(make_shift(), created_by="super_001"))
	submitted = run(svc.submit_shift_report(shift.id, "super_001"))
	assert submitted.status == ReportStatus.SUBMITTED
	approved = run(svc.approve_shift_report(shift.id, "manager_001"))
	assert approved.status == ReportStatus.APPROVED


def test_cannot_modify_approved_shift():
	svc = make_service()
	shift = run(svc.create_shift_report(make_shift(), created_by="super_001"))
	run(svc.submit_shift_report(shift.id, "super_001"))
	run(svc.approve_shift_report(shift.id, "manager_001"))
	with pytest.raises(ValueError, match="approved"):
		run(svc.update_shift_report(shift.id, ShiftReportUpdate(notes="edit attempt")))


def test_create_blast():
	svc = make_service()
	blast = BlastCreate(
		tenant_id=TENANT,
		blast_name="BLT-2026-001",
		blast_type=BlastType.PRODUCTION,
		mine_area="open_pit_north",
		planned_date=datetime(2026, 1, 20),
		planned_material_type=MaterialType.ORE,
		designer_id="blast_eng_001",
	)
	result = run(svc.create_blast(blast, created_by="blast_eng_001"))
	assert result.blast_name == "BLT-2026-001"
	assert result.status.value == "planned"


def test_blast_status_machine():
	svc = make_service()
	blast = BlastCreate(
		tenant_id=TENANT, blast_name="BLT-002", blast_type=BlastType.DEVELOPMENT,
		mine_area="ug_drive", planned_date=datetime(2026, 2, 1),
		planned_material_type=MaterialType.DEVELOPMENT_WASTE,
		designer_id="blast_eng_001",
	)
	b = run(svc.create_blast(blast, created_by="blast_eng_001"))
	# planned -> designed
	from capabilities.mining.pro.models import BlastUpdate, BlastStatus
	b = run(svc.update_blast(b.id, BlastUpdate(status=BlastStatus.DESIGNED)))
	assert b.status == BlastStatus.DESIGNED
	# designed -> drilled
	b = run(svc.update_blast(b.id, BlastUpdate(status=BlastStatus.DRILLED)))
	assert b.status == BlastStatus.DRILLED


def test_blast_invalid_transition_rejected():
	svc = make_service()
	blast = BlastCreate(
		tenant_id=TENANT, blast_name="BLT-003", blast_type=BlastType.PRODUCTION,
		mine_area="pit", planned_date=datetime(2026, 2, 5),
		planned_material_type=MaterialType.ORE, designer_id="eng_001",
	)
	b = run(svc.create_blast(blast, created_by="eng_001"))
	from capabilities.mining.pro.models import BlastUpdate, BlastStatus
	with pytest.raises(ValueError, match="Invalid blast status transition"):
		run(svc.update_blast(b.id, BlastUpdate(status=BlastStatus.FIRED)))


def test_stockpile_add_and_reclaim():
	svc = make_service()
	sp = run(svc.create_stockpile(
		StockpileCreate(tenant_id=TENANT, name="ROM Pad 1", stockpile_type=StockpileType.RUN_OF_MINE, mine_area="main"),
		created_by="ops_user"
	))
	updated = run(svc.record_stockpile_movement(
		StockpileMovementCreate(
			stockpile_id=sp.id, movement_type="add", tonnes=10000.0,
			material_type=MaterialType.ORE, movement_at=datetime(2026, 1, 20),
			operator_id="ops_001",
		),
		created_by="ops_user"
	))
	assert updated.current_tonnes == 10000.0
	# Reclaim 3000t
	updated2 = run(svc.record_stockpile_movement(
		StockpileMovementCreate(
			stockpile_id=sp.id, movement_type="reclaim", tonnes=3000.0,
			material_type=MaterialType.ORE, movement_at=datetime(2026, 1, 21),
			operator_id="ops_001",
		),
		created_by="ops_user"
	))
	assert updated2.current_tonnes == 7000.0


def test_stockpile_reclaim_exceeds_inventory():
	svc = make_service()
	sp = run(svc.create_stockpile(
		StockpileCreate(tenant_id=TENANT, name="ROM 2", stockpile_type=StockpileType.RUN_OF_MINE, mine_area="main"),
		created_by="ops_user"
	))
	with pytest.raises(ValueError, match="Cannot reclaim"):
		run(svc.record_stockpile_movement(
			StockpileMovementCreate(
				stockpile_id=sp.id, movement_type="reclaim", tonnes=5000.0,
				material_type=MaterialType.ORE, movement_at=datetime(2026, 1, 22),
				operator_id="ops_001",
			),
			created_by="ops_user"
		))


def test_grade_boundary_approval_flow():
	svc = make_service()
	gb = run(svc.create_grade_boundary(
		GradeBoundaryCreate(
			tenant_id=TENANT, mine_area="open_pit_north",
			period_start=datetime(2026, 1, 1), period_end=datetime(2026, 1, 31),
			method=GradeControlMethod.BLAST_HOLE_ASSAY,
			commodity="gold", cut_off_grade=0.5, grade_units="g/t",
		),
		created_by="geo_user"
	))
	assert not gb.approved
	approved_gb = run(svc.approve_grade_boundary(gb.id, "mine_manager"))
	assert approved_gb.approved


def test_production_summary():
	svc = make_service()
	shift = run(svc.create_shift_report(make_shift(), created_by="super_001"))
	run(svc.submit_shift_report(shift.id, "super_001"))
	run(svc.approve_shift_report(shift.id, "manager_001"))
	summary = run(svc.get_production_summary())
	assert summary["shifts_counted"] == 1
	assert summary["total_ore_tonnes"] == 4800.0
