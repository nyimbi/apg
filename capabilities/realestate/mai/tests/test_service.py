"""Service tests for Facilities Maintenance (mai)."""

from __future__ import annotations

import asyncio
from datetime import date, timedelta
from decimal import Decimal

import pytest

from capabilities.realestate.mai.service import MaiService
from capabilities.realestate.mai.models import (
	AssetCreate, AssetUpdate, AssetCategory, AssetStatus, LifecyclePhase,
	PpmScheduleCreate,
	WorkOrderCreate, WorkOrderUpdate, WorkOrderType, WorkOrderStatus, Priority,
	MaintenanceContractorCreate,
	InspectionCreate, InspectionType,
	DefectCreate, DefectSeverity,
	SlaCreate, SlaType,
)

loop = asyncio.get_event_loop()
T = "test-tenant"


def _svc():
	return MaiService()


def _asset(svc, **kwargs):
	defaults = dict(
		tenant_id=T, property_id="prop-1", asset_ref="HVAC-001",
		name="Rooftop HVAC Unit", category=AssetCategory.hvac, created_by="u",
	)
	defaults.update(kwargs)
	return loop.run_until_complete(svc.register_asset(AssetCreate(**defaults)))


def _contractor(svc, has_insurance=True):
	expiry = date.today() + timedelta(days=365) if has_insurance else None
	return loop.run_until_complete(svc.register_contractor(MaintenanceContractorCreate(
		tenant_id=T, name="ACME Maintenance", contractor_type="specialist_mechanical",
		email="acme@test.com", phone="+254700000000",
		insurance_expiry=expiry, created_by="u",
	)))


# ── Asset ─────────────────────────────────────────────────────────────────────

def test_register_and_get_asset():
	svc = _svc()
	a = _asset(svc)
	assert a.id
	assert a.status == AssetStatus.active
	fetched = loop.run_until_complete(svc.get_asset(a.id, T))
	assert fetched.name == "Rooftop HVAC Unit"


def test_list_assets_by_category():
	svc = _svc()
	_asset(svc, asset_ref="H1", category=AssetCategory.hvac)
	_asset(svc, asset_ref="E1", category=AssetCategory.electrical)
	hvac = loop.run_until_complete(svc.list_assets(T, category="hvac"))
	assert len(hvac) == 1


def test_update_asset_status():
	svc = _svc()
	a = _asset(svc)
	updated = loop.run_until_complete(svc.update_asset(a.id, T, AssetUpdate(status=AssetStatus.under_maintenance)))
	assert updated.status == AssetStatus.under_maintenance


def test_end_of_life_assets():
	svc = _svc()
	a = _asset(svc)
	loop.run_until_complete(svc.update_asset(a.id, T, AssetUpdate(lifecycle_phase=LifecyclePhase.end_of_life)))
	eol = loop.run_until_complete(svc.get_end_of_life_assets(T))
	assert len(eol) == 1


# ── PPM Schedule ──────────────────────────────────────────────────────────────

def test_create_ppm_schedule():
	svc = _svc()
	a = _asset(svc)
	ppm = loop.run_until_complete(svc.create_ppm_schedule(PpmScheduleCreate(
		tenant_id=T, asset_id=a.id, property_id="prop-1",
		title="Monthly HVAC Service", frequency="monthly",
		next_due=date.today() + timedelta(days=7),
		created_by="u",
	)))
	assert ppm.id
	assert ppm.status.value == "scheduled"


def test_complete_ppm_updates_asset():
	svc = _svc()
	a = _asset(svc)
	ppm = loop.run_until_complete(svc.create_ppm_schedule(PpmScheduleCreate(
		tenant_id=T, asset_id=a.id, property_id="prop-1",
		title="Quarterly Check", frequency="quarterly",
		next_due=date.today(),
		created_by="u",
	)))
	completed = loop.run_until_complete(svc.complete_ppm(ppm.id, T, "tech1"))
	assert completed.completion_count == 1
	assert completed.last_completed is not None
	asset = loop.run_until_complete(svc.get_asset(a.id, T))
	assert asset.last_maintained is not None


# ── Work Order ────────────────────────────────────────────────────────────────

def test_raise_work_order():
	svc = _svc()
	a = _asset(svc)
	wo = loop.run_until_complete(svc.raise_work_order(WorkOrderCreate(
		tenant_id=T, asset_id=a.id, property_id="prop-1",
		work_order_type=WorkOrderType.corrective,
		priority=Priority.p3_medium,
		title="Fix AC Unit", description="AC not cooling",
		reported_by="staff-1", created_by="u",
	)))
	assert wo.id
	assert wo.ref.startswith("WO-")
	assert wo.status == WorkOrderStatus.raised


def test_raise_work_order_for_decommissioned_asset_raises():
	svc = _svc()
	a = _asset(svc)
	loop.run_until_complete(svc.update_asset(a.id, T, AssetUpdate(status=AssetStatus.decommissioned)))
	with pytest.raises(ValueError):
		loop.run_until_complete(svc.raise_work_order(WorkOrderCreate(
			tenant_id=T, asset_id=a.id, property_id="prop-1",
			work_order_type=WorkOrderType.corrective,
			priority=Priority.p3_medium,
			title="Should fail", description="asset decommissioned",
			reported_by="u", created_by="u",
		)))


def test_assign_contractor_with_valid_insurance():
	svc = _svc()
	a = _asset(svc)
	wo = loop.run_until_complete(svc.raise_work_order(WorkOrderCreate(
		tenant_id=T, asset_id=a.id, property_id="prop-1",
		work_order_type=WorkOrderType.preventive, priority=Priority.p4_low,
		title="Annual Service", description="PM", reported_by="u", created_by="u",
	)))
	c = _contractor(svc, has_insurance=True)
	assigned = loop.run_until_complete(svc.assign_work_order(wo.id, T, c.id))
	assert assigned.assigned_contractor_id == c.id
	assert assigned.status == WorkOrderStatus.assigned


def test_assign_contractor_no_insurance_raises():
	svc = _svc()
	a = _asset(svc)
	wo = loop.run_until_complete(svc.raise_work_order(WorkOrderCreate(
		tenant_id=T, asset_id=a.id, property_id="prop-1",
		work_order_type=WorkOrderType.corrective, priority=Priority.p2_high,
		title="Urgent Fix", description="leak", reported_by="u", created_by="u",
	)))
	c = _contractor(svc, has_insurance=False)
	with pytest.raises(ValueError):
		loop.run_until_complete(svc.assign_work_order(wo.id, T, c.id))


def test_close_work_order_without_verification_raises():
	svc = _svc()
	a = _asset(svc)
	wo = loop.run_until_complete(svc.raise_work_order(WorkOrderCreate(
		tenant_id=T, asset_id=a.id, property_id="prop-1",
		work_order_type=WorkOrderType.corrective, priority=Priority.p3_medium,
		title="Fix", description="d", reported_by="u", created_by="u",
	)))
	with pytest.raises(ValueError):
		loop.run_until_complete(svc.close_work_order(wo.id, T, "tech"))


def test_close_verified_work_order():
	svc = _svc()
	a = _asset(svc)
	wo = loop.run_until_complete(svc.raise_work_order(WorkOrderCreate(
		tenant_id=T, asset_id=a.id, property_id="prop-1",
		work_order_type=WorkOrderType.corrective, priority=Priority.p3_medium,
		title="Fix", description="d", reported_by="u", created_by="u",
	)))
	loop.run_until_complete(svc.update_work_order(wo.id, T, WorkOrderUpdate(verification_complete=True)))
	closed = loop.run_until_complete(svc.close_work_order(wo.id, T, "tech"))
	assert closed.status == WorkOrderStatus.closed


# ── Inspection ────────────────────────────────────────────────────────────────

def test_create_and_complete_inspection():
	svc = _svc()
	insp = loop.run_until_complete(svc.create_inspection(InspectionCreate(
		tenant_id=T, property_id="prop-1",
		inspection_type=InspectionType.periodic,
		scheduled_date=date.today(),
		created_by="u",
	)))
	assert insp.status == "scheduled"
	completed = loop.run_until_complete(svc.complete_inspection(insp.id, T, [{"item": "windows", "ok": True}]))
	assert completed.status == "completed"


# ── Defect ────────────────────────────────────────────────────────────────────

def test_raise_and_resolve_defect():
	svc = _svc()
	d = loop.run_until_complete(svc.raise_defect(DefectCreate(
		tenant_id=T, property_id="prop-1",
		severity=DefectSeverity.major,
		description="Crack in ceiling",
		created_by="u",
	)))
	assert d.status == "open"
	resolved = loop.run_until_complete(svc.resolve_defect(d.id, T, "Repaired with epoxy"))
	assert resolved.status == "resolved"


# ── SLA Dashboard ─────────────────────────────────────────────────────────────

def test_sla_dashboard():
	svc = _svc()
	dashboard = loop.run_until_complete(svc.get_sla_dashboard(T))
	assert "total_open_work_orders" in dashboard
	assert "p1_open" in dashboard
