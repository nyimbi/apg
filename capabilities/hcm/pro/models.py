"""Pydantic v2 models for Professional Development capability."""
from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def _uuid() -> str:
	return str(uuid4())


# ── Development Plan models ───────────────────────────────────────────────────

class PRODevelopmentPlanCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	employee_id: str
	plan_year: int
	objectives: list[str] = Field(default_factory=list)
	focus_areas: list[str] = Field(default_factory=list)
	target_role_id: str | None = None
	reviewed_by: str | None = None


class PRODevelopmentPlanUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	objectives: list[str] | None = None
	focus_areas: list[str] | None = None
	target_role_id: str | None = None
	status: str | None = None  # draft, active, completed, cancelled
	reviewed_by: str | None = None
	completion_notes: str | None = None


class PRODevelopmentPlanResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str
	tenant_id: str
	employee_id: str
	plan_year: int
	objectives: list[str]
	focus_areas: list[str]
	target_role_id: str | None
	reviewed_by: str | None
	completion_notes: str | None
	status: str
	progress_pct: float
	created_at: str
	updated_at: str | None


# ── Skill models ──────────────────────────────────────────────────────────────

class PROSkillCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	name: str
	category: str  # technical, leadership, communication, analytical, domain
	description: str | None = None
	proficiency_levels: list[str] = Field(default_factory=lambda: ["beginner", "intermediate", "advanced", "expert"])


class PROSkillAssessmentCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	employee_id: str
	skill_id: str
	current_level: str  # beginner, intermediate, advanced, expert
	target_level: str
	assessed_by: str | None = None
	evidence: str | None = None


class PROSkillAssessmentResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str
	tenant_id: str
	employee_id: str
	skill_id: str
	skill_name: str
	current_level: str
	target_level: str
	gap_exists: bool
	assessed_by: str | None
	evidence: str | None
	status: str
	assessed_at: str


# ── Mentoring models ──────────────────────────────────────────────────────────

class PROMentoringProgrammeCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	mentee_employee_id: str
	mentor_employee_id: str
	programme_name: str
	objectives: list[str] = Field(default_factory=list)
	start_date: str
	end_date: str | None = None
	meeting_frequency: str = "monthly"  # weekly, fortnightly, monthly


class PROMentoringProgrammeUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	objectives: list[str] | None = None
	status: str | None = None  # active, paused, completed, cancelled
	end_date: str | None = None
	completion_notes: str | None = None


class PROMentoringProgrammeResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str
	tenant_id: str
	mentee_employee_id: str
	mentor_employee_id: str
	programme_name: str
	objectives: list[str]
	start_date: str
	end_date: str | None
	meeting_frequency: str
	sessions_completed: int
	status: str
	completion_notes: str | None
	created_at: str


# ── Certification models ──────────────────────────────────────────────────────

class PROCertificationCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	employee_id: str
	certification_name: str
	issuing_body: str
	issue_date: str
	expiry_date: str | None = None
	credential_id: str | None = None
	certificate_url: str | None = None


class PROCertificationUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	expiry_date: str | None = None
	renewal_date: str | None = None
	certificate_url: str | None = None
	status: str | None = None  # active, expired, renewed, revoked


class PROCertificationResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str
	tenant_id: str
	employee_id: str
	certification_name: str
	issuing_body: str
	issue_date: str
	expiry_date: str | None
	renewal_date: str | None
	credential_id: str | None
	certificate_url: str | None
	days_to_expiry: int | None
	status: str
	created_at: str


# ── Career Path models ────────────────────────────────────────────────────────

class PROCareerPathCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	employee_id: str
	current_role: str
	target_role: str
	target_timeline_months: int = 24
	milestones: list[dict[str, Any]] = Field(default_factory=list)
	advisor_employee_id: str | None = None


class PROCareerPathUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	target_role: str | None = None
	target_timeline_months: int | None = None
	milestones: list[dict[str, Any]] | None = None
	status: str | None = None  # active, on_hold, achieved, revised


class PROCareerPathResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str
	tenant_id: str
	employee_id: str
	current_role: str
	target_role: str
	target_timeline_months: int
	milestones: list[dict[str, Any]]
	milestones_completed: int
	advisor_employee_id: str | None
	status: str
	created_at: str
	updated_at: str | None


# ── Filter & Audit models ─────────────────────────────────────────────────────

class PROFilter(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	employee_id: str | None = None
	status: str | None = None
	plan_year: int | None = None


class PROAuditEvent(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str
	tenant_id: str
	event_type: str
	entity_type: str
	entity_id: str
	actor_id: str | None
	payload: dict[str, Any]
	emitted_at: str
