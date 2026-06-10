"""Pydantic v2 models for Spa & Activities Management."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def _uid() -> str:
	return uuid4().hex


class TreatmentCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	name: str
	category: str  # massage|facial|body_treatment|manicure|pedicure|hair|activity
	duration_mins: int
	price: float
	therapist_required: int = 1
	description: str | None = None
	is_active: bool = True


class TreatmentUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	name: str | None = None
	price: float | None = None
	duration_mins: int | None = None
	is_active: bool | None = None
	description: str | None = None


class TreatmentResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str
	tenant_id: str
	name: str
	category: str
	duration_mins: int
	price: float
	therapist_required: int
	description: str | None
	is_active: bool
	booking_count: int
	created_at: str


class TherapistCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	first_name: str
	last_name: str
	specialisations: list[str] = Field(default_factory=list)
	employment_type: str = "full_time"
	phone: str | None = None
	email: str | None = None


class TherapistResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str
	tenant_id: str
	first_name: str
	last_name: str
	specialisations: list[str]
	employment_type: str
	status: str
	created_at: str


class AppointmentCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	guest_name: str
	guest_email: str
	treatment_id: str
	therapist_id: str | None = None  # auto-assign if None
	appointment_date: str
	start_time: str  # HH:MM
	reservation_id: str | None = None  # link to room reservation
	special_notes: str | None = None


class AppointmentUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	appointment_date: str | None = None
	start_time: str | None = None
	therapist_id: str | None = None
	status: str | None = None
	special_notes: str | None = None


class AppointmentResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str
	tenant_id: str
	guest_name: str
	guest_email: str
	treatment_id: str
	treatment_name: str
	therapist_id: str | None
	appointment_date: str
	start_time: str
	end_time: str
	duration_mins: int
	price: float
	status: str
	reservation_id: str | None
	created_at: str


class MembershipCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	guest_name: str
	guest_email: str
	membership_type: str  # basic|silver|gold|platinum
	valid_months: int = 12
	price: float
	included_treatments: int = 0
	discount_pct: float = 0.0


class MembershipResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str
	tenant_id: str
	guest_name: str
	guest_email: str
	membership_type: str
	valid_from: str
	valid_to: str
	price: float
	included_treatments: int
	treatments_used: int
	discount_pct: float
	status: str
	created_at: str


class SPAListFilter(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	category: str | None = None
	therapist_id: str | None = None
	date: str | None = None
	status: str | None = None
	limit: int = 100
	offset: int = 0


class AuditEvent(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=_uid)
	tenant_id: str
	event_type: str
	record_id: str
	record_type: str
	details: dict[str, Any] = Field(default_factory=dict)
	created_at: str
