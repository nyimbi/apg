"""Pydantic v2 models for Organizational Management capability."""
from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def _uuid() -> str:
	return str(uuid4())


# ── Org Unit models ───────────────────────────────────────────────────────────

class ORGUnitCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	name: str
	code: str
	unit_type: str  # department, division, team, branch, region, subsidiary
	parent_unit_id: str | None = None
	manager_employee_id: str | None = None
	cost_centre: str | None = None
	location: str | None = None
	description: str | None = None


class ORGUnitUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	name: str | None = None
	code: str | None = None
	manager_employee_id: str | None = None
	cost_centre: str | None = None
	location: str | None = None
	description: str | None = None
	status: str | None = None  # active, inactive, dissolved


class ORGUnitResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str
	tenant_id: str
	name: str
	code: str
	unit_type: str
	parent_unit_id: str | None
	manager_employee_id: str | None
	cost_centre: str | None
	location: str | None
	description: str | None
	headcount: int
	status: str
	created_at: str
	updated_at: str | None


# ── Position models ───────────────────────────────────────────────────────────

class ORGPositionCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	title: str
	code: str
	org_unit_id: str
	job_grade: str | None = None
	reports_to_position_id: str | None = None
	fte_count: float = 1.0
	is_critical: bool = False
	location: str | None = None
	description: str | None = None


class ORGPositionUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	title: str | None = None
	job_grade: str | None = None
	reports_to_position_id: str | None = None
	fte_count: float | None = None
	is_critical: bool | None = None
	location: str | None = None
	status: str | None = None  # open, filled, frozen, abolished


class ORGPositionResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str
	tenant_id: str
	title: str
	code: str
	org_unit_id: str
	job_grade: str | None
	reports_to_position_id: str | None
	incumbent_employee_id: str | None
	fte_count: float
	is_critical: bool
	location: str | None
	description: str | None
	status: str
	created_at: str
	updated_at: str | None


# ── Reporting Line models ─────────────────────────────────────────────────────

class ORGReportingLineCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	employee_id: str
	manager_employee_id: str
	line_type: str = "direct"  # direct, dotted, functional
	effective_date: str
	end_date: str | None = None


class ORGReportingLineResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str
	tenant_id: str
	employee_id: str
	manager_employee_id: str
	line_type: str
	effective_date: str
	end_date: str | None
	status: str
	created_at: str


# ── Restructuring models ──────────────────────────────────────────────────────

class ORGRestructuringCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	name: str
	description: str
	effective_date: str
	units_affected: list[str] = Field(default_factory=list)
	positions_affected: list[str] = Field(default_factory=list)
	initiated_by: str


class ORGRestructuringUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	status: str | None = None  # draft, proposed, approved, in_progress, completed, cancelled
	approved_by: str | None = None
	notes: str | None = None


class ORGRestructuringResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str
	tenant_id: str
	name: str
	description: str
	effective_date: str
	units_affected: list[str]
	positions_affected: list[str]
	initiated_by: str
	status: str
	approved_by: str | None
	notes: str | None
	created_at: str


# ── Filter model ──────────────────────────────────────────────────────────────

class ORGFilter(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	unit_type: str | None = None
	status: str | None = None
	parent_unit_id: str | None = None
	location: str | None = None


# ── Audit model ───────────────────────────────────────────────────────────────

class ORGAuditEvent(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)

	id: str
	tenant_id: str
	event_type: str
	entity_type: str
	entity_id: str
	actor_id: str | None
	payload: dict[str, Any]
	emitted_at: str
