"""Pydantic v2 models for APG School Management."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Annotated
from uuid6 import uuid7

from pydantic import BaseModel, ConfigDict, Field, AfterValidator


def uuid7str() -> str:
	return str(uuid7())


def _non_empty(v: str) -> str:
	assert v and v.strip(), "field must be non-empty"
	return v.strip()


def _non_negative_float(v: float) -> float:
	assert v >= 0.0, f"value must be non-negative, got {v}"
	return v


NonEmptyStr = Annotated[str, AfterValidator(_non_empty)]
NonNegativeFloat = Annotated[float, AfterValidator(_non_negative_float)]

_BASE_CONFIG = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)


# ---------------------------------------------------------------------------
# Student
# ---------------------------------------------------------------------------

class StudentCreate(BaseModel):
	model_config = _BASE_CONFIG

	id: str = Field(default_factory=uuid7str)
	tenant_id: NonEmptyStr
	first_name: NonEmptyStr
	last_name: NonEmptyStr
	date_of_birth: str  # ISO date string
	gender: str | None = None
	national_id: str | None = None
	student_number: NonEmptyStr
	grade_level: NonEmptyStr
	status: str = "active"
	guardian_ids: list[str] = Field(default_factory=list)
	address: dict[str, Any] = Field(default_factory=dict)
	contact_info: dict[str, Any] = Field(default_factory=dict)
	medical_notes: str = ""
	special_needs: str = ""
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: NonEmptyStr


class StudentUpdate(BaseModel):
	model_config = _BASE_CONFIG

	first_name: str | None = None
	last_name: str | None = None
	grade_level: str | None = None
	status: str | None = None
	guardian_ids: list[str] | None = None
	address: dict[str, Any] | None = None
	contact_info: dict[str, Any] | None = None
	medical_notes: str | None = None
	special_needs: str | None = None
	approval_reference: str | None = None
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class StudentResponse(StudentCreate):
	pass


# ---------------------------------------------------------------------------
# AdmissionApplication
# ---------------------------------------------------------------------------

class AdmissionApplicationCreate(BaseModel):
	model_config = _BASE_CONFIG

	id: str = Field(default_factory=uuid7str)
	tenant_id: NonEmptyStr
	applicant_first_name: NonEmptyStr
	applicant_last_name: NonEmptyStr
	date_of_birth: str
	grade_level_applying: NonEmptyStr
	status: str = "draft"
	guardian_name: NonEmptyStr
	guardian_contact: NonEmptyStr
	previous_school: str = ""
	documents: list[str] = Field(default_factory=list)
	notes: str = ""
	reviewer_id: str | None = None
	offer_reference: str | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: NonEmptyStr


class AdmissionApplicationUpdate(BaseModel):
	model_config = _BASE_CONFIG

	status: str | None = None
	reviewer_id: str | None = None
	offer_reference: str | None = None
	notes: str | None = None
	documents: list[str] | None = None
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class AdmissionApplicationResponse(AdmissionApplicationCreate):
	pass


# ---------------------------------------------------------------------------
# FeeInvoice
# ---------------------------------------------------------------------------

class FeeInvoiceCreate(BaseModel):
	model_config = _BASE_CONFIG

	id: str = Field(default_factory=uuid7str)
	tenant_id: NonEmptyStr
	student_id: NonEmptyStr
	fee_type: NonEmptyStr
	amount: NonNegativeFloat
	currency: str = "KES"
	status: str = "pending"
	due_date: str  # ISO date string
	academic_year: NonEmptyStr
	term: NonEmptyStr
	description: str = ""
	payment_reference: str | None = None
	waiver_approval: str | None = None
	refund_approval: str | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: NonEmptyStr


class FeeInvoiceUpdate(BaseModel):
	model_config = _BASE_CONFIG

	status: str | None = None
	payment_reference: str | None = None
	waiver_approval: str | None = None
	refund_approval: str | None = None
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class FeeInvoiceResponse(FeeInvoiceCreate):
	pass


# ---------------------------------------------------------------------------
# StaffRecord
# ---------------------------------------------------------------------------

class StaffRecordCreate(BaseModel):
	model_config = _BASE_CONFIG

	id: str = Field(default_factory=uuid7str)
	tenant_id: NonEmptyStr
	first_name: NonEmptyStr
	last_name: NonEmptyStr
	staff_number: NonEmptyStr
	role: NonEmptyStr
	status: str = "active"
	email: NonEmptyStr
	phone: str | None = None
	subjects: list[str] = Field(default_factory=list)
	qualifications: list[str] = Field(default_factory=list)
	department: str | None = None
	join_date: str  # ISO date string
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: NonEmptyStr


class StaffRecordUpdate(BaseModel):
	model_config = _BASE_CONFIG

	role: str | None = None
	status: str | None = None
	email: str | None = None
	phone: str | None = None
	subjects: list[str] | None = None
	qualifications: list[str] | None = None
	department: str | None = None
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class StaffRecordResponse(StaffRecordCreate):
	pass


# ---------------------------------------------------------------------------
# CalendarEvent
# ---------------------------------------------------------------------------

class CalendarEventCreate(BaseModel):
	model_config = _BASE_CONFIG

	id: str = Field(default_factory=uuid7str)
	tenant_id: NonEmptyStr
	title: NonEmptyStr
	event_type: NonEmptyStr
	start_date: str  # ISO date string
	end_date: str   # ISO date string
	academic_year: NonEmptyStr
	term: NonEmptyStr
	description: str = ""
	is_public: bool = True
	affected_grade_levels: list[str] = Field(default_factory=list)
	location: str | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: NonEmptyStr


class CalendarEventUpdate(BaseModel):
	model_config = _BASE_CONFIG

	title: str | None = None
	start_date: str | None = None
	end_date: str | None = None
	description: str | None = None
	is_public: bool | None = None
	affected_grade_levels: list[str] | None = None
	location: str | None = None
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class CalendarEventResponse(CalendarEventCreate):
	pass


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------

class DocumentCreate(BaseModel):
	model_config = _BASE_CONFIG

	id: str = Field(default_factory=uuid7str)
	tenant_id: NonEmptyStr
	owner_id: NonEmptyStr           # student_id or staff_id
	owner_type: str                 # "student" | "staff"
	document_type: NonEmptyStr
	title: NonEmptyStr
	file_reference: NonEmptyStr
	is_confidential: bool = False
	consent_recorded: bool = False
	expiry_date: str | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: NonEmptyStr


class DocumentResponse(DocumentCreate):
	pass


# ---------------------------------------------------------------------------
# Communication
# ---------------------------------------------------------------------------

class CommunicationCreate(BaseModel):
	model_config = _BASE_CONFIG

	id: str = Field(default_factory=uuid7str)
	tenant_id: NonEmptyStr
	subject: NonEmptyStr
	body: NonEmptyStr
	channel: NonEmptyStr
	sender_id: NonEmptyStr
	recipient_ids: list[str] = Field(default_factory=list)
	recipient_groups: list[str] = Field(default_factory=list)  # e.g. ["grade_1", "parents"]
	sent_at: datetime | None = None
	scheduled_at: datetime | None = None
	is_draft: bool = True
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: NonEmptyStr


class CommunicationUpdate(BaseModel):
	model_config = _BASE_CONFIG

	subject: str | None = None
	body: str | None = None
	is_draft: bool | None = None
	scheduled_at: datetime | None = None
	sent_at: datetime | None = None
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class CommunicationResponse(CommunicationCreate):
	pass


# ---------------------------------------------------------------------------
# SchMgmtAgent
# ---------------------------------------------------------------------------

class SchMgmtAgent(BaseModel):
	model_config = _BASE_CONFIG

	id: str = Field(default_factory=uuid7str)
	tenant_id: NonEmptyStr
	name: NonEmptyStr
	runtime: NonEmptyStr
	role: NonEmptyStr
	scope: str = "school management operations"
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: NonEmptyStr
