"""Matter Management — Pydantic v2 models."""
from __future__ import annotations

from datetime import date, datetime
from typing import Any
from uuid_extensions import uuid7str
from pydantic import BaseModel, ConfigDict, Field


class MatMatterCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	title: str
	matter_type: str  # litigation, advisory, transactional, regulatory
	client_id: str
	lead_attorney_id: str
	practice_area: str
	jurisdiction: str
	description: str = ""
	priority: str = "normal"  # low, normal, high, urgent
	budget: float | None = None
	opened_date: str = Field(default_factory=lambda: date.today().isoformat())
	tags: list[str] = Field(default_factory=list)
	metadata: dict[str, Any] = Field(default_factory=dict)


class MatMatterUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	title: str | None = None
	status: str | None = None
	lead_attorney_id: str | None = None
	priority: str | None = None
	budget: float | None = None
	description: str | None = None
	closed_date: str | None = None
	tags: list[str] | None = None
	metadata: dict[str, Any] | None = None


class MatMatterResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	title: str
	matter_type: str
	client_id: str
	lead_attorney_id: str
	practice_area: str
	jurisdiction: str
	description: str
	priority: str
	status: str
	budget: float | None
	opened_date: str
	closed_date: str | None = None
	tags: list[str]
	team_ids: list[str]
	task_count: int
	deadline_count: int
	metadata: dict[str, Any]
	created_at: str
	updated_at: str | None = None


class MatMatterListResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	items: list[MatMatterResponse]
	total: int
	page: int = 1
	page_size: int = 50


class MatMatterFilter(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	status: str | None = None
	matter_type: str | None = None
	client_id: str | None = None
	lead_attorney_id: str | None = None
	practice_area: str | None = None
	jurisdiction: str | None = None
	priority: str | None = None
	tags: list[str] | None = None
	opened_after: str | None = None
	opened_before: str | None = None


class MatTaskCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	matter_id: str
	title: str
	description: str = ""
	assigned_to_id: str
	due_date: str
	priority: str = "normal"
	task_type: str = "general"


class MatTaskResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	matter_id: str
	tenant_id: str
	title: str
	description: str
	assigned_to_id: str
	due_date: str
	priority: str
	task_type: str
	status: str
	completed_at: str | None = None
	created_at: str


class MatDeadlineCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	matter_id: str
	title: str
	deadline_date: str
	deadline_type: str  # court, filing, statute_of_limitations, contractual
	description: str = ""
	reminder_days: list[int] = Field(default_factory=lambda: [7, 1])


class MatDeadlineResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	matter_id: str
	tenant_id: str
	title: str
	deadline_date: str
	deadline_type: str
	description: str
	reminder_days: list[int]
	status: str
	created_at: str


class MatDocketEntry(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	matter_id: str
	court: str
	case_number: str
	event_date: str
	event_type: str
	description: str
	judge: str | None = None
	courtroom: str | None = None


class MatDocketEntryResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	matter_id: str
	tenant_id: str
	court: str
	case_number: str
	event_date: str
	event_type: str
	description: str
	judge: str | None
	courtroom: str | None
	status: str
	created_at: str


class MatAuditEvent(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	matter_id: str | None
	event_type: str
	actor_id: str | None
	details: dict[str, Any]
	created_at: str
