"""Pydantic v2 models for APG Mine Safety & Compliance."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
from uuid6 import uuid7


def uuid7str() -> str:
	return str(uuid7())


# ── Enums ─────────────────────────────────────────────────────────────────────

class IncidentType(str, Enum):
	FATALITY = "fatality"
	LOST_TIME_INJURY = "lost_time_injury"
	MEDICAL_TREATMENT_INJURY = "medical_treatment_injury"
	FIRST_AID_INJURY = "first_aid_injury"
	NEAR_MISS = "near_miss"
	DANGEROUS_OCCURRENCE = "dangerous_occurrence"
	ENVIRONMENTAL_INCIDENT = "environmental_incident"
	PROPERTY_DAMAGE = "property_damage"
	VEHICLE_INCIDENT = "vehicle_incident"
	OCCUPATIONAL_ILLNESS = "occupational_illness"


class HazardCategory(str, Enum):
	MECHANICAL = "mechanical"
	ELECTRICAL = "electrical"
	CHEMICAL = "chemical"
	RADIATION = "radiation"
	GRAVITATIONAL = "gravitational"
	BIOLOGICAL = "biological"
	ERGONOMIC = "ergonomic"
	FIRE_EXPLOSION = "fire_explosion"
	CONFINED_SPACE = "confined_space"
	GROUND_INSTABILITY = "ground_instability"
	DUST_FUMES = "dust_fumes"
	NOISE_VIBRATION = "noise_vibration"


class RiskRating(str, Enum):
	EXTREME = "extreme"
	HIGH = "high"
	MEDIUM = "medium"
	LOW = "low"
	NEGLIGIBLE = "negligible"


class ConsequenceLevel(str, Enum):
	CATASTROPHIC = "catastrophic"
	MAJOR = "major"
	MODERATE = "moderate"
	MINOR = "minor"
	INSIGNIFICANT = "insignificant"


class LikelihoodLevel(str, Enum):
	ALMOST_CERTAIN = "almost_certain"
	LIKELY = "likely"
	POSSIBLE = "possible"
	UNLIKELY = "unlikely"
	RARE = "rare"


class PTWType(str, Enum):
	HOT_WORK = "hot_work"
	CONFINED_SPACE_ENTRY = "confined_space_entry"
	ELECTRICAL_ISOLATION = "electrical_isolation"
	WORKING_AT_HEIGHT = "working_at_height"
	EXCAVATION = "excavation"
	LIFTING_OPERATIONS = "lifting_operations"
	RADIATION_WORK = "radiation_work"
	EXPLOSIVES_HANDLING = "explosives_handling"
	ISOLATION_LOCKOUT = "isolation_lockout"
	GROUND_DISTURBANCE = "ground_disturbance"


class ControlType(str, Enum):
	ELIMINATION = "elimination"
	SUBSTITUTION = "substitution"
	ENGINEERING = "engineering"
	ADMINISTRATIVE = "administrative"
	PPE = "ppe"


class CorrectiveActionStatus(str, Enum):
	OPEN = "open"
	IN_PROGRESS = "in_progress"
	OVERDUE = "overdue"
	CLOSED = "closed"
	VERIFIED = "verified"


class AuditType(str, Enum):
	INTERNAL = "internal"
	EXTERNAL = "external"
	REGULATORY = "regulatory"
	THIRD_PARTY = "third_party"
	SELF_ASSESSMENT = "self_assessment"


class ReviewStatus(str, Enum):
	PENDING = "pending"
	IN_REVIEW = "in_review"
	APPROVED = "approved"
	REJECTED = "rejected"
	CLOSED = "closed"


# ── Base ───────────────────────────────────────────────────────────────────────

class SafBase(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)


# ── Incident ───────────────────────────────────────────────────────────────────

class WitnessCreate(SafBase):
	name: str
	employee_id: str | None = None
	statement: str | None = None


class IncidentCreate(SafBase):
	tenant_id: str
	incident_type: IncidentType
	occurred_at: datetime
	location: str
	mine_area: str
	persons_involved: list[str] = Field(default_factory=list)
	witnesses: list[WitnessCreate] = Field(default_factory=list)
	description: str
	immediate_actions_taken: str | None = None
	equipment_involved: list[str] = Field(default_factory=list)
	reported_by: str
	supervisor_id: str | None = None
	notification_sent: bool = False


class IncidentUpdate(SafBase):
	description: str | None = None
	immediate_actions_taken: str | None = None
	investigation_id: str | None = None
	status: ReviewStatus | None = None
	close_notes: str | None = None


class IncidentResponse(SafBase):
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	incident_type: IncidentType
	occurred_at: datetime
	location: str
	mine_area: str
	persons_involved: list[str]
	witnesses: list[dict[str, Any]]
	description: str
	immediate_actions_taken: str | None
	equipment_involved: list[str]
	reported_by: str
	supervisor_id: str | None
	notification_sent: bool
	regulatory_notification_sent: bool = False
	investigation_id: str | None = None
	status: ReviewStatus = ReviewStatus.PENDING
	close_notes: str | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str


# ── Hazard ─────────────────────────────────────────────────────────────────────

class ControlMeasureCreate(SafBase):
	control_type: ControlType
	description: str
	responsible_person_id: str
	target_date: datetime | None = None


class HazardCreate(SafBase):
	tenant_id: str
	hazard_category: HazardCategory
	location: str
	mine_area: str
	description: str
	potential_consequence: ConsequenceLevel
	likelihood: LikelihoodLevel
	inherent_risk_rating: RiskRating
	control_measures: list[ControlMeasureCreate] = Field(default_factory=list)
	residual_risk_rating: RiskRating | None = None
	identified_by: str
	identified_at: datetime
	review_date: datetime | None = None
	stop_work_invoked: bool = False


class HazardResponse(SafBase):
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	hazard_category: HazardCategory
	location: str
	mine_area: str
	description: str
	potential_consequence: ConsequenceLevel
	likelihood: LikelihoodLevel
	inherent_risk_rating: RiskRating
	control_measures: list[dict[str, Any]]
	residual_risk_rating: RiskRating | None
	identified_by: str
	identified_at: datetime
	review_date: datetime | None
	stop_work_invoked: bool
	status: ReviewStatus = ReviewStatus.PENDING
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str


# ── Risk Register ──────────────────────────────────────────────────────────────

class RiskRegisterEntryCreate(SafBase):
	tenant_id: str
	risk_title: str
	risk_description: str
	hazard_category: HazardCategory
	mine_area: str
	consequence: ConsequenceLevel
	likelihood: LikelihoodLevel
	inherent_risk_rating: RiskRating
	controls: list[ControlMeasureCreate] = Field(default_factory=list)
	residual_consequence: ConsequenceLevel | None = None
	residual_likelihood: LikelihoodLevel | None = None
	residual_risk_rating: RiskRating | None = None
	risk_owner_id: str
	review_date: datetime | None = None


class RiskRegisterEntryResponse(SafBase):
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	risk_title: str
	risk_description: str
	hazard_category: HazardCategory
	mine_area: str
	consequence: ConsequenceLevel
	likelihood: LikelihoodLevel
	inherent_risk_rating: RiskRating
	controls: list[dict[str, Any]]
	residual_consequence: ConsequenceLevel | None
	residual_likelihood: LikelihoodLevel | None
	residual_risk_rating: RiskRating | None
	risk_owner_id: str
	review_date: datetime | None
	status: ReviewStatus = ReviewStatus.PENDING
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str


# ── Permit to Work ─────────────────────────────────────────────────────────────

class PermitToWorkCreate(SafBase):
	tenant_id: str
	ptw_type: PTWType
	location: str
	mine_area: str
	work_description: str
	workers: list[str] = Field(default_factory=list, description="List of employee IDs")
	valid_from: datetime
	valid_to: datetime
	issuer_id: str
	isolation_points: list[str] = Field(default_factory=list)
	isolation_verified_by: str | None = None
	site_inspection_by: str | None = None
	ppe_required: list[str] = Field(default_factory=list)
	conditions: list[str] = Field(default_factory=list)

	@field_validator("valid_to")
	@classmethod
	def valid_to_after_from(cls, v: datetime) -> datetime:
		return v


class PermitToWorkResponse(SafBase):
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	ptw_type: PTWType
	location: str
	mine_area: str
	work_description: str
	workers: list[str]
	valid_from: datetime
	valid_to: datetime
	issuer_id: str
	isolation_points: list[str]
	isolation_verified_by: str | None
	site_inspection_by: str | None
	ppe_required: list[str]
	conditions: list[str]
	status: ReviewStatus = ReviewStatus.PENDING
	closed_at: datetime | None = None
	closed_by: str | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str


# ── Corrective Action ──────────────────────────────────────────────────────────

class CorrectiveActionCreate(SafBase):
	tenant_id: str
	source_type: str = Field(..., description="incident, hazard, audit, observation")
	source_id: str
	description: str
	assignee_id: str
	due_date: datetime
	priority: str = Field(..., description="critical, high, medium, low")
	verification_required: bool = True


class CorrectiveActionResponse(SafBase):
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	source_type: str
	source_id: str
	description: str
	assignee_id: str
	due_date: datetime
	priority: str
	verification_required: bool
	status: CorrectiveActionStatus = CorrectiveActionStatus.OPEN
	closed_at: datetime | None = None
	closed_by: str | None = None
	verified_by: str | None = None
	verified_at: datetime | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str
