"""Legal Compliance Management — Pydantic v2 models."""
from __future__ import annotations

from typing import Any
from uuid_extensions import uuid7str
from pydantic import BaseModel, ConfigDict, Field


class CplRequirementCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	title: str
	description: str
	regulation: str  # GDPR, AML, FATCA, POCAMLA, Companies_Act, etc.
	jurisdiction: str
	category: str  # data_privacy, financial, employment, environmental, corporate
	frequency: str  # one_time, monthly, quarterly, annually, continuous
	due_date: str | None = None
	owner_id: str
	risk_level: str = "medium"  # low, medium, high, critical
	tags: list[str] = Field(default_factory=list)
	metadata: dict[str, Any] = Field(default_factory=dict)


class CplRequirementUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	title: str | None = None
	description: str | None = None
	status: str | None = None
	owner_id: str | None = None
	due_date: str | None = None
	risk_level: str | None = None
	tags: list[str] | None = None
	metadata: dict[str, Any] | None = None


class CplRequirementResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	title: str
	description: str
	regulation: str
	jurisdiction: str
	category: str
	frequency: str
	due_date: str | None
	owner_id: str
	risk_level: str
	status: str  # active, compliant, non_compliant, exempted, archived
	evidence_count: int
	tags: list[str]
	metadata: dict[str, Any]
	created_at: str
	updated_at: str | None = None
	last_assessed_at: str | None = None


class CplRequirementListResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	items: list[CplRequirementResponse]
	total: int
	page: int = 1
	page_size: int = 50


class CplRequirementFilter(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	regulation: str | None = None
	jurisdiction: str | None = None
	category: str | None = None
	status: str | None = None
	risk_level: str | None = None
	owner_id: str | None = None
	due_before: str | None = None


class CplCalendarEntry(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	requirement_id: str
	scheduled_date: str
	title: str
	description: str = ""
	assigned_to_id: str
	reminder_days: list[int] = Field(default_factory=lambda: [14, 7, 1])


class CplCalendarEntryResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	requirement_id: str
	tenant_id: str
	scheduled_date: str
	title: str
	description: str
	assigned_to_id: str
	reminder_days: list[int]
	status: str
	completed_at: str | None = None
	created_at: str


class CplEvidenceCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	requirement_id: str
	evidence_type: str  # document, screenshot, log, certificate, attestation
	title: str
	description: str
	file_reference: str = ""
	collected_by_id: str
	collection_date: str
	valid_until: str | None = None


class CplEvidenceResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	requirement_id: str
	tenant_id: str
	evidence_type: str
	title: str
	description: str
	file_reference: str
	collected_by_id: str
	collection_date: str
	valid_until: str | None
	status: str
	created_at: str


class CplBreachCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	requirement_id: str
	title: str
	description: str
	severity: str  # low, medium, high, critical
	discovered_by_id: str
	discovery_date: str
	affected_records: int = 0
	notification_required: bool = False
	notification_deadline: str | None = None


class CplBreachResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	requirement_id: str
	tenant_id: str
	title: str
	description: str
	severity: str
	discovered_by_id: str
	discovery_date: str
	affected_records: int
	notification_required: bool
	notification_deadline: str | None
	status: str  # open, investigating, remediated, reported, closed
	remediated_at: str | None = None
	reported_at: str | None = None
	created_at: str


class CplAuditEvent(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	requirement_id: str | None
	event_type: str
	actor_id: str | None
	details: dict[str, Any]
	created_at: str
