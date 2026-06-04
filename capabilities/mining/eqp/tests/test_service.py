"""Tests for EqpService."""

from __future__ import annotations

import asyncio
from datetime import datetime

import pytest

from capabilities.mining.eqp.models import (
	DispatchStatus,
	EquipmentClass,
	EquipmentCreate,
	EquipmentFaultCreate,
	FaultSeverity,
	FuelDocketCreate,
	FuelType,
	InspectionCreate,
	InspectionItemResult,
	InspectionType,
	LifecycleStatus,
	MaintenanceType,
	MaintenanceWorkOrderCreate,
	OwnershipType,
)
from capabilities.mining.eqp.service import EqpService

TENANT = "test_fleet"


def run(coro):
	return asyncio.get_event_loop().run_until_complete(coro)


def make_service():
	return EqpService(tenant_id=TENANT)


def make_equipment(asset_number: str = "HT-001") -> EquipmentCreate:
	return EquipmentCreate(
		tenant_id=TENANT,
		asset_number=asset_number,
		equipment_class=EquipmentClass.HAUL_TRUCK,
		make="Caterpillar",
		model="793F",
		year=2020,
		serial_number="CAT793F001",
		ownership_type=OwnershipType.OWNED,
		payload_tonnes=227.0,
		fuel_type=FuelType.DIESEL,
	)


def test_register_equipment():
	svc = make_service()
	result = run(svc.register_equipment(make_equipment(), created_by="fleet_admin"))
	assert result.asset_number == "HT-001"
	assert result.lifecycle_status == LifecycleStatus.COMMISSIONED


def test_duplicate_asset_number_rejected():
	svc = make_service()
	run(svc.register_equipment(make_equipment(), created_by="fleet_admin"))
	with pytest.raises(ValueError, match="already registered"):
		run(svc.register_equipment(make_equipment(), created_by="fleet_admin"))


def test_list_equipment_by_class():
	svc = make_service()
	run(svc.register_equipment(make_equipment("HT-001"), created_by="fleet_admin"))
	run(svc.register_equipment(make_equipment("HT-002"), created_by="fleet_admin"))
	run(svc.register_equipment(
		EquipmentCreate(
			tenant_id=TENANT, asset_number="EX-001",
			equipment_class=EquipmentClass.EXCAVATOR,
			make="Komatsu", model="PC4000", year=2019,
			ownership_type=OwnershipType.OWNED, fuel_type=FuelType.DIESEL,
		),
		created_by="fleet_admin"
	))
	trucks = run(svc.list_equipment(equipment_class="haul_truck"))
	assert len(trucks) == 2


def test_decommission_equipment():
	svc = make_service()
	eqp = run(svc.register_equipment(make_equipment(), created_by="fleet_admin"))
	from capabilities.mining.eqp.models import EquipmentUpdate
	run(svc.update_equipment(eqp.id, EquipmentUpdate(lifecycle_status=LifecycleStatus.ACTIVE)))
	decom = run(svc.decommission_equipment(eqp.id, "fleet_admin"))
	assert decom.lifecycle_status == LifecycleStatus.DECOMMISSIONED


def test_report_critical_fault_sets_breakdown():
	svc = make_service()
	eqp = run(svc.register_equipment(make_equipment(), created_by="fleet_admin"))
	fault = EquipmentFaultCreate(
		tenant_id=TENANT,
		equipment_id=eqp.id,
		severity=FaultSeverity.CRITICAL,
		component="engine",
		description="Engine shutdown — loss of oil pressure",
		detected_at=datetime(2026, 1, 20, 10, 0),
		detected_by="operator_001",
	)
	run(svc.report_fault(fault, created_by="operator_001"))
	updated_eqp = run(svc.get_equipment(eqp.id))
	assert updated_eqp.dispatch_status == DispatchStatus.BREAKDOWN


def test_resolve_fault_restores_availability():
	svc = make_service()
	eqp = run(svc.register_equipment(make_equipment(), created_by="fleet_admin"))
	fault = EquipmentFaultCreate(
		tenant_id=TENANT, equipment_id=eqp.id,
		severity=FaultSeverity.CRITICAL, component="hydraulics",
		description="Hydraulic hose burst",
		detected_at=datetime(2026, 1, 21), detected_by="operator_002",
	)
	f = run(svc.report_fault(fault, created_by="operator_002"))
	run(svc.resolve_fault(f.id))
	updated = run(svc.get_equipment(eqp.id))
	assert updated.dispatch_status == DispatchStatus.AVAILABLE


def test_submit_inspection_fail_sets_maintenance():
	svc = make_service()
	eqp = run(svc.register_equipment(make_equipment(), created_by="fleet_admin"))
	inspection = InspectionCreate(
		tenant_id=TENANT,
		equipment_id=eqp.id,
		inspection_type=InspectionType.PRE_SHIFT,
		inspector_id="operator_001",
		inspected_at=datetime(2026, 1, 22, 5, 30),
		items=[InspectionItemResult(item="Brakes", result="fail", notes="No brake pressure")],
		overall_result="fail",
		faults_found=["Brake failure"],
	)
	run(svc.submit_inspection(inspection, created_by="operator_001"))
	updated = run(svc.get_equipment(eqp.id))
	assert updated.dispatch_status == DispatchStatus.MAINTENANCE


def test_record_fuel_docket():
	svc = make_service()
	eqp = run(svc.register_equipment(make_equipment(), created_by="fleet_admin"))
	docket = FuelDocketCreate(
		tenant_id=TENANT, equipment_id=eqp.id,
		fuel_type=FuelType.DIESEL, quantity_litres=500.0,
		fuelled_at=datetime(2026, 1, 22, 6, 0),
		fuelled_by="fuel_tech_001", docket_number="DKT-0001",
		cost_per_litre=1.85,
	)
	result = run(svc.record_fuel_docket(docket, created_by="fuel_tech_001"))
	assert result.quantity_litres == 500.0
	assert result.total_cost == pytest.approx(925.0)


def test_fleet_kpi_summary():
	svc = make_service()
	from capabilities.mining.eqp.models import EquipmentUpdate
	eqp = run(svc.register_equipment(make_equipment(), created_by="fleet_admin"))
	run(svc.update_equipment(eqp.id, EquipmentUpdate(lifecycle_status=LifecycleStatus.ACTIVE)))
	kpis = run(svc.get_fleet_kpi_summary())
	assert kpis["total_active_equipment"] == 1
	assert kpis["physical_availability_pct"] == 100.0
