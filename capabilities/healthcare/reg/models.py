"""Pydantic v2 models for APG Healthcare Regulatory."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
from uuid6 import uuid7


def uuid7str() -> str:
	return str(uuid7())


class LicenseCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	license_type: str
	license_number: str
	issuing_authority: str
	issued_date: datetime
	expiry_date: datetime
	holder_name: str
	scope: str
	created_by: str

	@field_validator("license_number")
	@classmethod
	def license_number_not_empty(cls, v: str) -> str:
		if not v.strip():
			raise ValueError("license_number must not be empty")
		return v.strip()


class LicenseResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	license_type: str
	license_number: str
	issuing_authority: str
	issued_date: datetime
	expiry_date: datetime
	holder_name: str
	scope: str
	status: str = "active"
	days_to_expiry: int = 0
	renewal_initiated: bool = False
	created_by: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class AccreditationCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	accreditation_body: str
	program: str
	award_date: datetime
	expiry_date: datetime
	certificate_reference: str
	scope: str
	created_by: str


class AccreditationResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	accreditation_body: str
	program: str
	award_date: datetime
	expiry_date: datetime
	certificate_reference: str
	scope: str
	status: str = "accredited"
	next_survey_date: datetime | None = None
	created_by: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class IncidentCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	incident_type: str
	severity: str
	description: str
	patient_id: str | None = None
	department: str
	occurred_at: datetime
	reported_by: str
	immediate_actions: str
	witnesses: list[str] = Field(default_factory=list)
	created_by: str


class IncidentResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	incident_type: str
	severity: str
	description: str
	patient_id: str | None = None
	department: str
	occurred_at: datetime
	reported_by: str
	immediate_actions: str
	witnesses: list[str] = Field(default_factory=list)
	status: str = "open"
	rca_completed: bool = False
	rca_reference: str | None = None
	corrective_actions: list[str] = Field(default_factory=list)
	notification_sent: bool = False
	closed_at: datetime | None = None
	created_by: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class RegulatorySubmissionCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	report_type: str
	title: str
	reporting_period_start: datetime
	reporting_period_end: datetime
	submitted_to: str
	prepared_by: str
	data_references: list[str] = Field(default_factory=list)
	created_by: str


class RegulatorySubmissionResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	report_type: str
	title: str
	reporting_period_start: datetime
	reporting_period_end: datetime
	submitted_to: str
	prepared_by: str
	data_references: list[str] = Field(default_factory=list)
	status: str = "draft"
	submission_reference: str | None = None
	submitted_at: datetime | None = None
	decision_at: datetime | None = None
	rejection_reason: str | None = None
	created_by: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class CorrectiveActionCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	incident_id: str | None = None
	source: str
	description: str
	assigned_to: str
	due_date: datetime
	priority: str = "medium"
	created_by: str


class CorrectiveActionResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	incident_id: str | None = None
	source: str
	description: str
	assigned_to: str
	due_date: datetime
	priority: str = "medium"
	status: str = "open"
	completed_at: datetime | None = None
	verified_by: str | None = None
	verified_at: datetime | None = None
	created_by: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
