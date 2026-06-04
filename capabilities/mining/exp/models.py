"""Pydantic v2 models for APG Exploration Data Management."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from uuid6 import uuid7


def uuid7str() -> str:
	return str(uuid7())


# ── Enums ─────────────────────────────────────────────────────────────────────

class HoleType(str, Enum):
	DIAMOND = "diamond"
	ROTARY_AIR_BLAST = "rotary_air_blast"
	REVERSE_CIRCULATION = "reverse_circulation"
	SONIC = "sonic"
	AUGER = "auger"
	PERCUSSION = "percussion"
	CORE = "core"


class SampleType(str, Enum):
	CORE = "core"
	CHIP = "chip"
	CHANNEL = "channel"
	GRAB = "grab"
	SOIL = "soil"
	STREAM_SEDIMENT = "stream_sediment"
	ROCK_CHIP = "rock_chip"


class AssayMethod(str, Enum):
	FIRE_ASSAY = "fire_assay"
	ICP_MS = "icp_ms"
	ICP_OES = "icp_oes"
	XRF = "xrf"
	AAAS = "aaas"
	AQUA_REGIA_DIGEST = "aqua_regia_digest"
	FOUR_ACID_DIGEST = "four_acid_digest"
	NEUTRON_ACTIVATION = "neutron_activation"
	SCREEN_FIRE = "screen_fire"


class ResourceClassification(str, Enum):
	MEASURED = "measured"
	INDICATED = "indicated"
	INFERRED = "inferred"
	EXPLORATION_TARGET = "exploration_target"


class ReserveClassification(str, Enum):
	PROVEN = "proven"
	PROBABLE = "probable"


class ReportingStandard(str, Enum):
	JORC_2012 = "jorc_2012"
	NI_43_101 = "ni_43_101"
	SAMREC = "samrec"
	PERC = "perc"
	KAZRC = "kazrc"


class ReviewStatus(str, Enum):
	PENDING = "pending"
	IN_REVIEW = "in_review"
	APPROVED = "approved"
	REJECTED = "rejected"
	SUPERSEDED = "superseded"


class QAQCType(str, Enum):
	BLANK = "blank"
	STANDARD = "standard"
	DUPLICATE_FIELD = "duplicate_field"
	DUPLICATE_COARSE = "duplicate_coarse"
	DUPLICATE_PULP = "duplicate_pulp"
	CHECK_ASSAY = "check_assay"


class OxidationState(str, Enum):
	FRESH = "fresh"
	TRANSITIONAL = "transitional"
	OXIDISED = "oxidised"
	SUPERGENE = "supergene"


# ── Base ───────────────────────────────────────────────────────────────────────

class ExpBase(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)


# ── Drillhole Collar ───────────────────────────────────────────────────────────

class DrillholeCollarCreate(ExpBase):
	hole_id: str = Field(..., description="Unique hole identifier, e.g. ABDD001")
	tenant_id: str
	hole_type: HoleType
	easting: float
	northing: float
	elevation_m: float
	coordinate_system: str = Field(..., description="e.g. wgs84, mga_zone_54")
	azimuth_deg: float | None = None
	dip_deg: float | None = None
	planned_depth_m: float
	actual_depth_m: float | None = None
	prospect: str | None = None
	project: str | None = None
	drilled_by: str
	drilled_at: datetime
	notes: str | None = None

	@field_validator("planned_depth_m")
	@classmethod
	def depth_positive(cls, v: float) -> float:
		assert v > 0, "planned_depth_m must be positive"
		return v

	@field_validator("azimuth_deg")
	@classmethod
	def azimuth_range(cls, v: float | None) -> float | None:
		if v is not None:
			assert 0 <= v < 360, "azimuth_deg must be in [0, 360)"
		return v

	@field_validator("dip_deg")
	@classmethod
	def dip_range(cls, v: float | None) -> float | None:
		if v is not None:
			assert -90 <= v <= 0, "dip_deg must be in [-90, 0] (downward negative)"
		return v


class DrillholeCollarResponse(ExpBase):
	id: str = Field(default_factory=uuid7str)
	hole_id: str
	tenant_id: str
	hole_type: HoleType
	easting: float
	northing: float
	elevation_m: float
	coordinate_system: str
	azimuth_deg: float | None
	dip_deg: float | None
	planned_depth_m: float
	actual_depth_m: float | None
	prospect: str | None
	project: str | None
	drilled_by: str
	drilled_at: datetime
	notes: str | None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str


# ── Assay Result ───────────────────────────────────────────────────────────────

class AssayResultCreate(ExpBase):
	tenant_id: str
	hole_id: str
	sample_id: str
	from_m: float
	to_m: float
	sample_type: SampleType
	assay_method: AssayMethod
	commodity: str
	grade_value: float
	grade_units: str = Field(..., description="e.g. g/t, %, ppm")
	detection_limit: float
	lab_name: str
	lab_certificate_ref: str
	batch_id: str | None = None
	is_qaqc: bool = False
	qaqc_type: QAQCType | None = None

	@model_validator(mode="after")
	def from_less_than_to(self) -> "AssayResultCreate":
		assert self.from_m < self.to_m, "from_m must be less than to_m"
		return self

	@field_validator("grade_value")
	@classmethod
	def grade_non_negative(cls, v: float) -> float:
		assert v >= 0, "grade_value must be non-negative"
		return v


class AssayResultResponse(ExpBase):
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	hole_id: str
	sample_id: str
	from_m: float
	to_m: float
	sample_type: SampleType
	assay_method: AssayMethod
	commodity: str
	grade_value: float
	grade_units: str
	detection_limit: float
	lab_name: str
	lab_certificate_ref: str
	batch_id: str | None
	is_qaqc: bool
	qaqc_type: QAQCType | None
	qaqc_flag: str | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str


# ── Geology Log ────────────────────────────────────────────────────────────────

class GeologyIntervalCreate(ExpBase):
	tenant_id: str
	hole_id: str
	from_m: float
	to_m: float
	lithology_code: str
	oxidation_state: OxidationState
	mineralisation_style: str | None = None
	rock_quality_designation: float | None = Field(None, ge=0, le=100)
	total_core_recovery_pct: float | None = Field(None, ge=0, le=100)
	colour: str | None = None
	structure: str | None = None
	alteration: str | None = None
	geologist_id: str
	logged_at: datetime
	notes: str | None = None

	@model_validator(mode="after")
	def from_less_than_to(self) -> "GeologyIntervalCreate":
		assert self.from_m < self.to_m, "from_m must be less than to_m"
		return self


class GeologyIntervalResponse(ExpBase):
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	hole_id: str
	from_m: float
	to_m: float
	lithology_code: str
	oxidation_state: OxidationState
	mineralisation_style: str | None
	rock_quality_designation: float | None
	total_core_recovery_pct: float | None
	colour: str | None
	structure: str | None
	alteration: str | None
	geologist_id: str
	logged_at: datetime
	notes: str | None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str


# ── Resource Estimate ──────────────────────────────────────────────────────────

class ResourceEstimateCreate(ExpBase):
	tenant_id: str
	name: str
	commodity: str
	classification: ResourceClassification
	reporting_standard: ReportingStandard
	estimation_method: str = Field(..., description="e.g. ordinary_kriging, inverse_distance, nearest_neighbour")
	tonnes: float = Field(..., gt=0)
	grade_value: float = Field(..., ge=0)
	grade_units: str
	contained_metal: float | None = None
	cut_off_grade: float | None = None
	cut_off_units: str | None = None
	effective_date: datetime
	competent_person_id: str
	competent_person_qualification: str
	notes: str | None = None
	supporting_data_refs: list[str] = Field(default_factory=list)


class ResourceEstimateUpdate(ExpBase):
	classification: ResourceClassification | None = None
	tonnes: float | None = Field(None, gt=0)
	grade_value: float | None = Field(None, ge=0)
	contained_metal: float | None = None
	notes: str | None = None
	review_status: ReviewStatus | None = None
	reviewer_id: str | None = None
	review_notes: str | None = None


class ResourceEstimateResponse(ExpBase):
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	name: str
	commodity: str
	classification: ResourceClassification
	reporting_standard: ReportingStandard
	estimation_method: str
	tonnes: float
	grade_value: float
	grade_units: str
	contained_metal: float | None
	cut_off_grade: float | None
	cut_off_units: str | None
	effective_date: datetime
	competent_person_id: str
	competent_person_qualification: str
	notes: str | None
	supporting_data_refs: list[str]
	review_status: ReviewStatus = ReviewStatus.PENDING
	reviewer_id: str | None = None
	review_notes: str | None = None
	published: bool = False
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str


# ── Compliance Report ──────────────────────────────────────────────────────────

class ComplianceReportCreate(ExpBase):
	tenant_id: str
	title: str
	reporting_standard: ReportingStandard
	reporting_period_start: datetime
	reporting_period_end: datetime
	resource_estimate_ids: list[str] = Field(default_factory=list)
	competent_person_id: str
	notes: str | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)


class ComplianceReportResponse(ExpBase):
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	title: str
	reporting_standard: ReportingStandard
	reporting_period_start: datetime
	reporting_period_end: datetime
	resource_estimate_ids: list[str]
	competent_person_id: str
	competent_person_signed: bool = False
	notes: str | None
	metadata: dict[str, Any]
	review_status: ReviewStatus = ReviewStatus.PENDING
	published: bool = False
	published_at: datetime | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str
