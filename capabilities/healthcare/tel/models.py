"""Pydantic v2 models for APG Telemedicine."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
from uuid6 import uuid7


def uuid7str() -> str:
	return str(uuid7())


class ConsultationCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	patient_id: str
	provider_id: str
	consultation_type: str
	scheduled_at: datetime
	duration_minutes: int = 30
	chief_complaint: str
	platform: str
	patient_consent_obtained: bool = False
	e911_disclosure_provided: bool = False
	created_by: str

	@field_validator("duration_minutes")
	@classmethod
	def duration_positive(cls, v: int) -> int:
		if v < 5:
			raise ValueError("duration_minutes must be at least 5")
		return v


class ConsultationResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	patient_id: str
	provider_id: str
	consultation_type: str
	scheduled_at: datetime
	duration_minutes: int
	chief_complaint: str
	platform: str
	patient_consent_obtained: bool
	e911_disclosure_provided: bool
	status: str = "scheduled"
	session_id: str | None = None
	started_at: datetime | None = None
	ended_at: datetime | None = None
	billing_code: str | None = None
	notes: str = ""
	created_by: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class TeleSessionCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	consultation_id: str
	patient_id: str
	provider_id: str
	platform: str
	patient_consent_obtained: bool = False
	e911_disclosure_provided: bool = False
	technical_check_completed: bool = False
	created_by: str


class TeleSessionResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	consultation_id: str
	patient_id: str
	provider_id: str
	platform: str
	status: str = "waiting"
	join_url: str = ""
	started_at: datetime | None = None
	ended_at: datetime | None = None
	duration_seconds: int = 0
	recording_consent_obtained: bool = False
	technical_check_completed: bool = False
	created_by: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class RemoteMonitoringEnrollmentCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	patient_id: str
	device_type: str
	device_id: str
	alert_thresholds: dict[str, Any]
	provider_id: str
	alert_threshold_configured: bool = False
	created_by: str


class RemoteMonitoringEnrollmentResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	patient_id: str
	device_type: str
	device_id: str
	alert_thresholds: dict[str, Any]
	provider_id: str
	status: str = "active"
	enrolled_at: datetime = Field(default_factory=datetime.utcnow)
	last_reading_at: datetime | None = None
	created_by: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class PrescriptionTransmitCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	patient_id: str
	consultation_id: str
	drug_name: str
	drug_schedule: str
	dose: str
	route: str
	frequency: str
	quantity: int
	refills: int
	prescriber_id: str
	pharmacy_id: str
	transmission_method: str
	in_person_visit_completed: bool = True
	created_by: str


class PrescriptionTransmitResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	patient_id: str
	consultation_id: str
	drug_name: str
	drug_schedule: str
	dose: str
	route: str
	frequency: str
	quantity: int
	refills: int
	prescriber_id: str
	pharmacy_id: str
	transmission_method: str
	status: str = "transmitted"
	confirmation_number: str | None = None
	transmitted_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class TeleBillingCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	consultation_id: str
	patient_id: str
	provider_id: str
	billing_code: str
	place_of_service: str = "02"
	diagnosis_codes: list[str] = Field(default_factory=list)
	units: int = 1
	created_by: str


class TeleBillingResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	consultation_id: str
	patient_id: str
	provider_id: str
	billing_code: str
	place_of_service: str
	diagnosis_codes: list[str] = Field(default_factory=list)
	units: int
	status: str = "pending"
	created_by: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
