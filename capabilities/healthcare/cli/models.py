"""Pydantic v2 models for APG Clinical Management."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
from uuid6 import uuid7


def uuid7str() -> str:
	return str(uuid7())


class CarePlanCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	patient_id: str
	title: str
	description: str
	goals: list[str] = Field(default_factory=list)
	care_team_ids: list[str] = Field(default_factory=list)
	icd10_codes: list[str] = Field(default_factory=list)
	created_by: str

	@field_validator("title")
	@classmethod
	def title_not_empty(cls, v: str) -> str:
		if not v.strip():
			raise ValueError("title must not be empty")
		return v.strip()


class CarePlanResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	patient_id: str
	title: str
	description: str
	goals: list[str] = Field(default_factory=list)
	care_team_ids: list[str] = Field(default_factory=list)
	icd10_codes: list[str] = Field(default_factory=list)
	status: str = "draft"
	interventions: list[dict[str, Any]] = Field(default_factory=list)
	adherence_status: str = "not_assessed"
	created_by: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class ProtocolCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	protocol_type: str
	name: str
	description: str
	activation_criteria: str
	steps: list[dict[str, Any]] = Field(default_factory=list)
	evidence_reference: str
	created_by: str


class ProtocolResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	protocol_type: str
	name: str
	description: str
	activation_criteria: str
	steps: list[dict[str, Any]] = Field(default_factory=list)
	evidence_reference: str
	status: str = "draft"
	patient_id: str | None = None
	activated_at: datetime | None = None
	completed_at: datetime | None = None
	created_by: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class ClinicalWorkflowCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	patient_id: str
	care_plan_id: str | None = None
	title: str
	description: str
	assigned_to: str
	due_at: datetime
	created_by: str


class ClinicalWorkflowResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	patient_id: str
	care_plan_id: str | None = None
	title: str
	description: str
	assigned_to: str
	due_at: datetime
	state: str = "pending"
	completed_at: datetime | None = None
	created_by: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class CDSAlertCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	patient_id: str
	cds_type: str
	priority: str
	message: str
	evidence_reference: str
	suggested_action: str
	created_by: str


class CDSAlertResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	patient_id: str
	cds_type: str
	priority: str
	message: str
	evidence_reference: str
	suggested_action: str
	status: str = "active"
	acknowledged_by: str | None = None
	acknowledged_at: datetime | None = None
	created_by: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class HandoffCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	patient_id: str
	handoff_type: str
	from_provider_id: str
	to_provider_id: str
	situation: str
	background: str
	assessment: str
	recommendation: str
	structured_format_used: bool = True
	created_by: str


class HandoffResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	patient_id: str
	handoff_type: str
	from_provider_id: str
	to_provider_id: str
	situation: str
	background: str
	assessment: str
	recommendation: str
	structured_format_used: bool = True
	acknowledged_by: str | None = None
	acknowledged_at: datetime | None = None
	created_by: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
