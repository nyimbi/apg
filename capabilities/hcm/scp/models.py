"""Pydantic v2 models for Succession Planning capability."""
from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def _uuid() -> str:
	return str(uuid4())


# ── Talent Pool models ────────────────────────────────────────────────────────

class SCPTalentPoolCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	name: str
	description: str | None = None
	target_roles: list[str] = Field(default_factory=list)
	min_readiness_level: str = "developing"  # developing, ready_in_1_year, ready_now


class SCPTalentPoolUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	name: str | None = None
	description: str | None = None
	target_roles: list[str] | None = None
	min_readiness_level: str | None = None
	status: str | None = None  # active, closed, archived


class SCPTalentPoolResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str
	tenant_id: str
	name: str
	description: str | None
	target_roles: list[str]
	min_readiness_level: str
	member_count: int
	status: str
	created_at: str
	updated_at: str | None


# ── Readiness Assessment models ───────────────────────────────────────────────

class SCPReadinessAssessmentCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	employee_id: str
	target_role_id: str
	readiness_level: str  # developing, ready_in_1_year, ready_now
	performance_rating: float  # 1.0 – 5.0
	potential_rating: float  # 1.0 – 5.0
	assessed_by: str
	development_needs: list[str] = Field(default_factory=list)
	risks: list[str] = Field(default_factory=list)
	notes: str | None = None


class SCPReadinessAssessmentUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	readiness_level: str | None = None
	performance_rating: float | None = None
	potential_rating: float | None = None
	development_needs: list[str] | None = None
	risks: list[str] | None = None
	notes: str | None = None
	status: str | None = None  # current, archived


class SCPReadinessAssessmentResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str
	tenant_id: str
	employee_id: str
	target_role_id: str
	readiness_level: str
	performance_rating: float
	potential_rating: float
	nine_box_quadrant: str
	assessed_by: str
	development_needs: list[str]
	risks: list[str]
	notes: str | None
	status: str
	assessed_at: str


# ── Nine-Box Grid models ──────────────────────────────────────────────────────

class SCPNineBoxEntryCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	employee_id: str
	performance_axis: float  # 1.0–3.0 (low/medium/high)
	potential_axis: float    # 1.0–3.0
	review_cycle: str
	reviewer_id: str
	label: str | None = None  # optional descriptive label
	notes: str | None = None


class SCPNineBoxEntryResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str
	tenant_id: str
	employee_id: str
	performance_axis: float
	potential_axis: float
	quadrant: str
	review_cycle: str
	reviewer_id: str
	label: str | None
	notes: str | None
	created_at: str


# ── Succession Scenario models ────────────────────────────────────────────────

class SCPSuccessionScenarioCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	role_id: str
	role_title: str
	incumbent_employee_id: str | None = None
	scenario_type: str = "planned"  # planned, emergency, voluntary
	successors: list[dict[str, Any]] = Field(default_factory=list)
	notes: str | None = None


class SCPSuccessionScenarioUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	successors: list[dict[str, Any]] | None = None
	notes: str | None = None
	status: str | None = None  # draft, active, triggered, completed, archived


class SCPSuccessionScenarioResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str
	tenant_id: str
	role_id: str
	role_title: str
	incumbent_employee_id: str | None
	scenario_type: str
	successors: list[dict[str, Any]]
	notes: str | None
	status: str
	created_at: str
	updated_at: str | None


# ── Critical Role models ──────────────────────────────────────────────────────

class SCPCriticalRoleCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	role_id: str
	role_title: str
	rationale: str
	impact_if_vacant: str  # low, medium, high, critical
	time_to_fill_estimate_days: int = 90
	identified_by: str


class SCPCriticalRoleUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	rationale: str | None = None
	impact_if_vacant: str | None = None
	time_to_fill_estimate_days: int | None = None
	status: str | None = None  # active, under_review, removed


class SCPCriticalRoleResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str
	tenant_id: str
	role_id: str
	role_title: str
	rationale: str
	impact_if_vacant: str
	time_to_fill_estimate_days: int
	successor_count: int
	identified_by: str
	status: str
	created_at: str


# ── Filter & Audit ────────────────────────────────────────────────────────────

class SCPFilter(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	employee_id: str | None = None
	role_id: str | None = None
	readiness_level: str | None = None
	status: str | None = None


class SCPAuditEvent(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str
	tenant_id: str
	event_type: str
	entity_type: str
	entity_id: str
	actor_id: str | None
	payload: dict[str, Any]
	emitted_at: str
