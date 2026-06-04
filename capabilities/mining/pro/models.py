"""Pydantic v2 models for APG Mine Production Operations."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from uuid6 import uuid7


def uuid7str() -> str:
	return str(uuid7())


# ── Enums ─────────────────────────────────────────────────────────────────────

class ShiftType(str, Enum):
	DAY = "day"
	NIGHT = "night"
	AFTERNOON = "afternoon"
	SWING = "swing"
	EXTENDED_DAY = "extended_day"
	EXTENDED_NIGHT = "extended_night"


class MaterialType(str, Enum):
	ORE = "ore"
	WASTE = "waste"
	LOW_GRADE = "low_grade"
	MARGINAL = "marginal"
	MINERALISED_WASTE = "mineralised_waste"
	DEVELOPMENT_WASTE = "development_waste"
	OVERBURDEN = "overburden"
	TOPSOIL = "topsoil"


class BlastStatus(str, Enum):
	PLANNED = "planned"
	DESIGNED = "designed"
	DRILLED = "drilled"
	CHARGED = "charged"
	PRIMED = "primed"
	FIRED = "fired"
	CLEARED = "cleared"
	MUCKED = "mucked"


class BlastType(str, Enum):
	PRODUCTION = "production"
	DEVELOPMENT = "development"
	TRIM = "trim"
	PRE_SPLIT = "pre_split"
	CAST = "cast"
	CONTROLLED_DETONATION = "controlled_detonation"


class OreTrackingMethod(str, Enum):
	SURVEY_VOLUME = "survey_volume"
	TRUCK_COUNT = "truck_count"
	BELT_SCALE = "belt_scale"
	WEIGHBRIDGE = "weighbridge"
	DENSITY_MODEL = "density_model"


class GradeControlMethod(str, Enum):
	FACE_SAMPLING = "face_sampling"
	CHIP_SAMPLING = "chip_sampling"
	BLAST_HOLE_ASSAY = "blast_hole_assay"
	SONIC_DRILL = "sonic_drill"
	GRADE_SCANNER = "grade_scanner"


class StockpileType(str, Enum):
	RUN_OF_MINE = "run_of_mine"
	CRUSHED = "crushed"
	HIGH_GRADE = "high_grade"
	LOW_GRADE = "low_grade"
	BLENDED = "blended"
	PRODUCT = "product"


class ReportStatus(str, Enum):
	DRAFT = "draft"
	SUBMITTED = "submitted"
	APPROVED = "approved"
	REJECTED = "rejected"
	ARCHIVED = "archived"


class ScheduleType(str, Enum):
	SHORT_TERM_WEEKLY = "short_term_weekly"
	MEDIUM_TERM_MONTHLY = "medium_term_monthly"
	LONG_TERM_ANNUAL = "long_term_annual"
	LIFE_OF_MINE = "life_of_mine"


# ── Base ───────────────────────────────────────────────────────────────────────

class ProBase(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)


# ── Shift Report ───────────────────────────────────────────────────────────────

class ProductionActivityCreate(ProBase):
	area: str
	material_type: MaterialType
	planned_tonnes: float = Field(..., ge=0)
	actual_tonnes: float = Field(..., ge=0)
	grade_value: float | None = Field(None, ge=0)
	grade_units: str | None = None
	tracking_method: OreTrackingMethod
	notes: str | None = None


class DelayCreate(ProBase):
	delay_category: str
	duration_minutes: float = Field(..., gt=0)
	equipment_id: str | None = None
	description: str


class ShiftReportCreate(ProBase):
	tenant_id: str
	shift_type: ShiftType
	shift_date: datetime
	shift_start: datetime
	shift_end: datetime
	mine_area: str
	supervisor_id: str
	operator_count: int = Field(..., ge=0)
	activities: list[ProductionActivityCreate] = Field(default_factory=list)
	delays: list[DelayCreate] = Field(default_factory=list)
	safety_observations: list[str] = Field(default_factory=list)
	notes: str | None = None

	@model_validator(mode="after")
	def start_before_end(self) -> "ShiftReportCreate":
		assert self.shift_start < self.shift_end, "shift_start must be before shift_end"
		return self


class ShiftReportUpdate(ProBase):
	activities: list[ProductionActivityCreate] | None = None
	delays: list[DelayCreate] | None = None
	safety_observations: list[str] | None = None
	notes: str | None = None
	status: ReportStatus | None = None
	reviewer_id: str | None = None


class ShiftReportResponse(ProBase):
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	shift_type: ShiftType
	shift_date: datetime
	shift_start: datetime
	shift_end: datetime
	mine_area: str
	supervisor_id: str
	operator_count: int
	activities: list[dict[str, Any]] = Field(default_factory=list)
	delays: list[dict[str, Any]] = Field(default_factory=list)
	safety_observations: list[str]
	notes: str | None
	status: ReportStatus = ReportStatus.DRAFT
	reviewer_id: str | None = None
	total_ore_tonnes: float = 0.0
	total_waste_tonnes: float = 0.0
	total_delay_minutes: float = 0.0
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str


# ── Blast ──────────────────────────────────────────────────────────────────────

class BlastHoleCreate(ProBase):
	hole_id: str
	easting: float
	northing: float
	elevation_m: float
	depth_m: float = Field(..., gt=0)
	diameter_mm: float = Field(..., gt=0)
	explosive_type: str | None = None
	explosive_mass_kg: float | None = Field(None, ge=0)
	stemming_m: float | None = Field(None, ge=0)


class BlastCreate(ProBase):
	tenant_id: str
	blast_name: str
	blast_type: BlastType
	mine_area: str
	bench_level: str | None = None
	pattern_easting: float | None = None
	pattern_northing: float | None = None
	planned_date: datetime
	planned_tonnes: float | None = Field(None, ge=0)
	planned_material_type: MaterialType
	holes: list[BlastHoleCreate] = Field(default_factory=list)
	explosive_total_kg: float | None = Field(None, ge=0)
	powder_factor: float | None = Field(None, ge=0)
	designer_id: str
	notes: str | None = None


class BlastUpdate(ProBase):
	status: BlastStatus | None = None
	fire_authority_id: str | None = None
	fired_at: datetime | None = None
	actual_tonnes: float | None = Field(None, ge=0)
	post_blast_inspection_by: str | None = None
	post_blast_inspection_at: datetime | None = None
	fragmentation_notes: str | None = None
	notes: str | None = None


class BlastResponse(ProBase):
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	blast_name: str
	blast_type: BlastType
	mine_area: str
	bench_level: str | None
	planned_date: datetime
	planned_tonnes: float | None
	actual_tonnes: float | None = None
	planned_material_type: MaterialType
	holes: list[dict[str, Any]]
	explosive_total_kg: float | None = None
	powder_factor: float | None = None
	designer_id: str
	status: BlastStatus = BlastStatus.PLANNED
	design_approved_by: str | None = None
	design_approved_at: datetime | None = None
	fire_authority_id: str | None = None
	fired_at: datetime | None = None
	post_blast_inspection_by: str | None = None
	post_blast_inspection_at: datetime | None = None
	fragmentation_notes: str | None = None
	notes: str | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str


# ── Grade Control ──────────────────────────────────────────────────────────────

class GradeBoundaryCreate(ProBase):
	tenant_id: str
	mine_area: str
	period_start: datetime
	period_end: datetime
	method: GradeControlMethod
	commodity: str
	cut_off_grade: float = Field(..., ge=0)
	grade_units: str
	ore_boundary_coords: list[dict[str, float]] = Field(default_factory=list, description="List of {easting, northing} points")
	approved_by: str | None = None
	notes: str | None = None


class GradeBoundaryResponse(ProBase):
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	mine_area: str
	period_start: datetime
	period_end: datetime
	method: GradeControlMethod
	commodity: str
	cut_off_grade: float
	grade_units: str
	ore_boundary_coords: list[dict[str, float]]
	approved: bool = False
	approved_by: str | None
	approved_at: datetime | None = None
	notes: str | None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str


# ── Stockpile ─────────────────────────────────────────────────────────────────

class StockpileMovementCreate(ProBase):
	stockpile_id: str
	movement_type: str = Field(..., description="add or reclaim")
	tonnes: float = Field(..., gt=0)
	grade_value: float | None = Field(None, ge=0)
	grade_units: str | None = None
	material_type: MaterialType
	source_area: str | None = None
	destination_area: str | None = None
	movement_at: datetime
	operator_id: str

	@field_validator("movement_type")
	@classmethod
	def valid_movement_type(cls, v: str) -> str:
		assert v in ("add", "reclaim"), "movement_type must be 'add' or 'reclaim'"
		return v


class StockpileCreate(ProBase):
	tenant_id: str
	name: str
	stockpile_type: StockpileType
	mine_area: str
	capacity_tonnes: float | None = Field(None, gt=0)
	notes: str | None = None


class StockpileResponse(ProBase):
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	name: str
	stockpile_type: StockpileType
	mine_area: str
	capacity_tonnes: float | None
	current_tonnes: float = 0.0
	average_grade: float | None = None
	notes: str | None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str


# ── Production Schedule ────────────────────────────────────────────────────────

class ProductionScheduleCreate(ProBase):
	tenant_id: str
	schedule_type: ScheduleType
	period_start: datetime
	period_end: datetime
	planned_ore_tonnes: float = Field(..., ge=0)
	planned_waste_tonnes: float = Field(..., ge=0)
	planned_grade: float | None = Field(None, ge=0)
	grade_units: str | None = None
	activities: list[dict[str, Any]] = Field(default_factory=list)
	prepared_by: str
	notes: str | None = None


class ProductionScheduleResponse(ProBase):
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	schedule_type: ScheduleType
	period_start: datetime
	period_end: datetime
	planned_ore_tonnes: float
	planned_waste_tonnes: float
	planned_grade: float | None
	grade_units: str | None
	activities: list[dict[str, Any]]
	prepared_by: str
	approved: bool = False
	approved_by: str | None = None
	approved_at: datetime | None = None
	published: bool = False
	notes: str | None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str
