"""Pydantic v2 models for APG Equipment & Plant Management."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
from uuid6 import uuid7


def uuid7str() -> str:
	return str(uuid7())


# ── Enums ─────────────────────────────────────────────────────────────────────

class EquipmentClass(str, Enum):
	HAUL_TRUCK = "haul_truck"
	EXCAVATOR = "excavator"
	WHEEL_LOADER = "wheel_loader"
	DRILL_RIG = "drill_rig"
	DOZER = "dozer"
	GRADER = "grader"
	WATER_CART = "water_cart"
	SERVICE_TRUCK = "service_truck"
	LHD_LOADER = "lhd_loader"
	UNDERGROUND_TRUCK = "underground_truck"
	CONVEYOR = "conveyor"
	CRUSHER = "crusher"
	MILL = "mill"
	PUMP = "pump"
	COMPRESSOR = "compressor"
	GENERATOR = "generator"
	CRANE = "crane"
	FORKLIFT = "forklift"
	LIGHT_VEHICLE = "light_vehicle"
	BUS = "bus"


class OwnershipType(str, Enum):
	OWNED = "owned"
	LEASED = "leased"
	CONTRACTED = "contracted"
	HIRE = "hire"
	SHARED = "shared"


class LifecycleStatus(str, Enum):
	COMMISSIONED = "commissioned"
	ACTIVE = "active"
	STANDBY = "standby"
	DECOMMISSIONED = "decommissioned"
	DISPOSED = "disposed"
	SOLD = "sold"


class MaintenanceType(str, Enum):
	PREVENTIVE = "preventive"
	CORRECTIVE = "corrective"
	PREDICTIVE = "predictive"
	CONDITION_BASED = "condition_based"
	BREAKDOWN = "breakdown"
	STATUTORY = "statutory"
	REBUILD = "rebuild"


class MaintenanceStatus(str, Enum):
	SCHEDULED = "scheduled"
	IN_PROGRESS = "in_progress"
	AWAITING_PARTS = "awaiting_parts"
	DEFERRED = "deferred"
	COMPLETED = "completed"
	CANCELLED = "cancelled"


class DispatchStatus(str, Enum):
	AVAILABLE = "available"
	OPERATING = "operating"
	STANDBY = "standby"
	MAINTENANCE = "maintenance"
	BREAKDOWN = "breakdown"
	FUELLING = "fuelling"
	PARKED = "parked"
	STANDBY_READY = "standby_ready"


class FaultSeverity(str, Enum):
	CRITICAL = "critical"
	MAJOR = "major"
	MINOR = "minor"
	COSMETIC = "cosmetic"


class InspectionType(str, Enum):
	PRE_SHIFT = "pre_shift"
	POST_SHIFT = "post_shift"
	WEEKLY = "weekly"
	MONTHLY = "monthly"
	ANNUAL = "annual"
	STATUTORY = "statutory"
	AD_HOC = "ad_hoc"


class FuelType(str, Enum):
	DIESEL = "diesel"
	LPG = "lpg"
	PETROL = "petrol"
	ELECTRIC = "electric"
	HYBRID = "hybrid"
	HYDROGEN = "hydrogen"


# ── Base ───────────────────────────────────────────────────────────────────────

class EqpBase(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)


# ── Equipment ─────────────────────────────────────────────────────────────────

class EquipmentCreate(EqpBase):
	tenant_id: str
	asset_number: str = Field(..., description="Unique asset identifier")
	equipment_class: EquipmentClass
	make: str
	model: str
	year: int = Field(..., ge=1950, le=2100)
	serial_number: str | None = None
	ownership_type: OwnershipType
	fleet_number: str | None = None
	mine_area_assignment: str | None = None
	payload_tonnes: float | None = Field(None, ge=0)
	engine_model: str | None = None
	fuel_type: FuelType = FuelType.DIESEL
	commissioned_at: datetime | None = None
	pm_schedule_id: str | None = None
	notes: str | None = None


class EquipmentUpdate(EqpBase):
	lifecycle_status: LifecycleStatus | None = None
	dispatch_status: DispatchStatus | None = None
	mine_area_assignment: str | None = None
	fleet_number: str | None = None
	notes: str | None = None


class EquipmentResponse(EqpBase):
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	asset_number: str
	equipment_class: EquipmentClass
	make: str
	model: str
	year: int
	serial_number: str | None
	ownership_type: OwnershipType
	fleet_number: str | None
	mine_area_assignment: str | None
	payload_tonnes: float | None
	engine_model: str | None
	fuel_type: FuelType
	commissioned_at: datetime | None
	pm_schedule_id: str | None
	lifecycle_status: LifecycleStatus = LifecycleStatus.COMMISSIONED
	dispatch_status: DispatchStatus = DispatchStatus.AVAILABLE
	total_operating_hours: float = 0.0
	notes: str | None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str


# ── Maintenance Work Order ─────────────────────────────────────────────────────

class SparePart(EqpBase):
	part_number: str
	description: str
	quantity: float = Field(..., gt=0)
	unit_cost: float | None = Field(None, ge=0)


class MaintenanceWorkOrderCreate(EqpBase):
	tenant_id: str
	equipment_id: str
	maintenance_type: MaintenanceType
	title: str
	description: str
	planned_start: datetime
	planned_end: datetime
	assigned_technician_id: str | None = None
	spare_parts: list[SparePart] = Field(default_factory=list)
	estimated_hours: float | None = Field(None, ge=0)
	priority: str = Field(..., description="critical, high, medium, low")
	triggered_by: str | None = Field(None, description="pm_schedule, fault_id, inspection_id, or manual")


class MaintenanceWorkOrderUpdate(EqpBase):
	status: MaintenanceStatus | None = None
	actual_start: datetime | None = None
	actual_end: datetime | None = None
	actual_hours: float | None = Field(None, ge=0)
	work_performed: str | None = None
	spare_parts_used: list[SparePart] | None = None
	total_cost: float | None = Field(None, ge=0)


class MaintenanceWorkOrderResponse(EqpBase):
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	equipment_id: str
	maintenance_type: MaintenanceType
	title: str
	description: str
	planned_start: datetime
	planned_end: datetime
	assigned_technician_id: str | None
	spare_parts: list[dict[str, Any]]
	estimated_hours: float | None
	priority: str
	triggered_by: str | None
	status: MaintenanceStatus = MaintenanceStatus.SCHEDULED
	approved_by: str | None = None
	approved_at: datetime | None = None
	actual_start: datetime | None = None
	actual_end: datetime | None = None
	actual_hours: float | None = None
	work_performed: str | None = None
	total_cost: float | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str


# ── Inspection ─────────────────────────────────────────────────────────────────

class InspectionItemResult(EqpBase):
	item: str
	result: str = Field(..., description="pass, fail, n/a")
	notes: str | None = None


class InspectionCreate(EqpBase):
	tenant_id: str
	equipment_id: str
	inspection_type: InspectionType
	inspector_id: str
	inspected_at: datetime
	items: list[InspectionItemResult] = Field(default_factory=list)
	overall_result: str = Field(..., description="pass, fail, conditional_pass")
	faults_found: list[str] = Field(default_factory=list)
	notes: str | None = None


class InspectionResponse(EqpBase):
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	equipment_id: str
	inspection_type: InspectionType
	inspector_id: str
	inspected_at: datetime
	items: list[dict[str, Any]]
	overall_result: str
	faults_found: list[str]
	notes: str | None
	work_order_raised: bool = False
	work_order_id: str | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str


# ── Fuel Docket ────────────────────────────────────────────────────────────────

class FuelDocketCreate(EqpBase):
	tenant_id: str
	equipment_id: str
	fuel_type: FuelType
	quantity_litres: float = Field(..., gt=0)
	odometer_km: float | None = Field(None, ge=0)
	engine_hours: float | None = Field(None, ge=0)
	fuelled_at: datetime
	fuelled_by: str
	docket_number: str
	cost_per_litre: float | None = Field(None, ge=0)


class FuelDocketResponse(EqpBase):
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	equipment_id: str
	fuel_type: FuelType
	quantity_litres: float
	odometer_km: float | None
	engine_hours: float | None
	fuelled_at: datetime
	fuelled_by: str
	docket_number: str
	cost_per_litre: float | None
	total_cost: float | None = None
	variance_flag: bool = False
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str


# ── Equipment Fault ────────────────────────────────────────────────────────────

class EquipmentFaultCreate(EqpBase):
	tenant_id: str
	equipment_id: str
	severity: FaultSeverity
	component: str
	description: str
	detected_at: datetime
	detected_by: str
	impact_on_operations: str | None = None


class EquipmentFaultResponse(EqpBase):
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	equipment_id: str
	severity: FaultSeverity
	component: str
	description: str
	detected_at: datetime
	detected_by: str
	impact_on_operations: str | None
	resolved: bool = False
	resolved_at: datetime | None = None
	work_order_id: str | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str
