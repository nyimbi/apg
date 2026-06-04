"""Pydantic v2 models for Facilities Maintenance (mai)."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from uuid6 import uuid7


def uuid7str() -> str:
	return str(uuid7())


class WorkOrderType(str, Enum):
	preventive = "preventive"
	corrective = "corrective"
	emergency = "emergency"
	predictive = "predictive"
	statutory = "statutory"
	improvement = "improvement"
	inspection = "inspection"
	condition_survey = "condition_survey"


class WorkOrderStatus(str, Enum):
	raised = "raised"
	assigned = "assigned"
	in_progress = "in_progress"
	on_hold = "on_hold"
	pending_parts = "pending_parts"
	completed = "completed"
	verified = "verified"
	closed = "closed"
	cancelled = "cancelled"


class Priority(str, Enum):
	p1_critical = "p1_critical"
	p2_high = "p2_high"
	p3_medium = "p3_medium"
	p4_low = "p4_low"
	p5_planned = "p5_planned"


class AssetCategory(str, Enum):
	hvac = "hvac"
	electrical = "electrical"
	plumbing = "plumbing"
	structural = "structural"
	fire_safety = "fire_safety"
	lifts_escalators = "lifts_escalators"
	access_control = "access_control"
	it_infrastructure = "it_infrastructure"
	landscaping = "landscaping"
	cleaning = "cleaning"
	security = "security"


class AssetStatus(str, Enum):
	active = "active"
	under_maintenance = "under_maintenance"
	decommissioned = "decommissioned"
	condemned = "condemned"
	awaiting_replacement = "awaiting_replacement"
	warranty = "warranty"


class LifecyclePhase(str, Enum):
	new = "new"
	operational = "operational"
	ageing = "ageing"
	end_of_life = "end_of_life"
	replacement_due = "replacement_due"
	decommissioned = "decommissioned"


class PpmStatus(str, Enum):
	scheduled = "scheduled"
	in_progress = "in_progress"
	completed = "completed"
	overdue = "overdue"
	deferred = "deferred"
	cancelled = "cancelled"


class InspectionType(str, Enum):
	statutory = "statutory"
	condition = "condition"
	pre_purchase = "pre_purchase"
	handover = "handover"
	periodic = "periodic"
	post_repair = "post_repair"
	compliance = "compliance"


class DefectSeverity(str, Enum):
	critical = "critical"
	major = "major"
	minor = "minor"
	cosmetic = "cosmetic"


class SlaType(str, Enum):
	response_time = "response_time"
	resolution_time = "resolution_time"
	first_time_fix = "first_time_fix"
	availability = "availability"
	uptime = "uptime"


# ── Asset ─────────────────────────────────────────────────────────────────────

class AssetCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	property_id: str
	asset_ref: str
	name: str
	category: AssetCategory
	make: str | None = None
	model_name: str | None = None
	serial_number: str | None = None
	install_date: date | None = None
	warranty_expiry: date | None = None
	replacement_cost: Decimal | None = None
	currency: str = "KES"
	location_description: str | None = None
	created_by: str


class AssetResponse(AssetCreate):
	id: str = Field(default_factory=uuid7str)
	status: AssetStatus = AssetStatus.active
	lifecycle_phase: LifecyclePhase = LifecyclePhase.operational
	last_maintained: date | None = None
	next_maintenance_due: date | None = None
	open_work_orders: int = 0
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class AssetUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	status: AssetStatus | None = None
	lifecycle_phase: LifecyclePhase | None = None
	warranty_expiry: date | None = None
	replacement_cost: Decimal | None = None
	last_maintained: date | None = None
	next_maintenance_due: date | None = None


# ── PPM Schedule ──────────────────────────────────────────────────────────────

class PpmScheduleCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	asset_id: str
	property_id: str
	title: str
	frequency: str  # from SUPPORTED_MAINTENANCE_FREQUENCIES
	next_due: date
	estimated_duration_hours: Decimal = Decimal("1")
	estimated_cost: Decimal = Decimal("0")
	currency: str = "KES"
	contractor_id: str | None = None
	instructions: str | None = None
	compliance_standard: str | None = None
	created_by: str


class PpmScheduleResponse(PpmScheduleCreate):
	id: str = Field(default_factory=uuid7str)
	status: PpmStatus = PpmStatus.scheduled
	last_completed: date | None = None
	completion_count: int = 0
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── Work Order ────────────────────────────────────────────────────────────────

class WorkOrderCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	asset_id: str
	property_id: str
	work_order_type: WorkOrderType
	priority: Priority
	title: str
	description: str
	ppm_schedule_id: str | None = None
	reported_by: str
	currency: str = "KES"
	created_by: str


class WorkOrderResponse(WorkOrderCreate):
	id: str = Field(default_factory=uuid7str)
	ref: str = ""
	status: WorkOrderStatus = WorkOrderStatus.raised
	assigned_contractor_id: str | None = None
	scheduled_date: date | None = None
	actual_start: datetime | None = None
	actual_end: datetime | None = None
	sla_response_deadline: datetime | None = None
	sla_resolution_deadline: datetime | None = None
	sla_breached: bool = False
	verification_complete: bool = False
	actual_cost: Decimal = Decimal("0")
	cost_lines: list[dict[str, Any]] = Field(default_factory=list)
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class WorkOrderUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	status: WorkOrderStatus | None = None
	assigned_contractor_id: str | None = None
	scheduled_date: date | None = None
	actual_start: datetime | None = None
	actual_end: datetime | None = None
	verification_complete: bool | None = None


# ── Maintenance Contractor ────────────────────────────────────────────────────

class MaintenanceContractorCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	name: str
	contractor_type: str
	registration_number: str | None = None
	email: str
	phone: str
	insurance_expiry: date | None = None
	insurance_policy_ref: str | None = None
	specialisms: list[str] = Field(default_factory=list)
	created_by: str


class MaintenanceContractorResponse(MaintenanceContractorCreate):
	id: str = Field(default_factory=uuid7str)
	has_valid_insurance: bool = False
	active_work_orders: int = 0
	average_response_hours: Decimal | None = None
	first_time_fix_rate: Decimal | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── SLA ───────────────────────────────────────────────────────────────────────

class SlaCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	property_id: str | None = None
	contractor_id: str | None = None
	sla_type: SlaType
	priority: Priority
	target_hours: Decimal

	@field_validator("target_hours")
	@classmethod
	def _positive_hours(cls, v: Decimal) -> Decimal:
		if v <= 0:
			raise ValueError("target_hours must be positive")
		return v


class SlaResponse(SlaCreate):
	id: str = Field(default_factory=uuid7str)
	breach_count_30_days: int = 0
	compliance_rate_pct: Decimal = Decimal("100")
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── Inspection ────────────────────────────────────────────────────────────────

class InspectionCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	property_id: str
	asset_id: str | None = None
	inspection_type: InspectionType
	scheduled_date: date
	inspector_id: str | None = None
	created_by: str


class InspectionResponse(InspectionCreate):
	id: str = Field(default_factory=uuid7str)
	status: str = "scheduled"  # scheduled | in_progress | completed | overdue
	completed_at: datetime | None = None
	findings: list[dict[str, Any]] = Field(default_factory=list)
	defect_ids: list[str] = Field(default_factory=list)
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── Defect ────────────────────────────────────────────────────────────────────

class DefectCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	property_id: str
	asset_id: str | None = None
	inspection_id: str | None = None
	work_order_id: str | None = None
	severity: DefectSeverity
	description: str
	location: str | None = None
	photo_ids: list[str] = Field(default_factory=list)
	created_by: str


class DefectResponse(DefectCreate):
	id: str = Field(default_factory=uuid7str)
	status: str = "open"  # open | in_progress | resolved | closed
	resolved_at: datetime | None = None
	resolution_notes: str | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
