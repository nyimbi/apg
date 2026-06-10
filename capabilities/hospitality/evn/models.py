"""Pydantic v2 models for Events & Venue Management."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def _uid() -> str:
	return uuid4().hex


class VenueCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	name: str
	venue_type: str  # ballroom|conference_room|boardroom|outdoor|banquet_hall|garden
	capacity_seated: int
	capacity_standing: int = 0
	area_sqm: float = 0.0
	rental_rate_per_day: float = 0.0
	av_included: bool = False
	catering_allowed: bool = True
	notes: str | None = None


class VenueUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	name: str | None = None
	capacity_seated: int | None = None
	rental_rate_per_day: float | None = None
	status: str | None = None
	notes: str | None = None


class VenueResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str
	tenant_id: str
	name: str
	venue_type: str
	capacity_seated: int
	capacity_standing: int
	area_sqm: float
	rental_rate_per_day: float
	av_included: bool
	catering_allowed: bool
	status: str
	created_at: str


class EventBookingCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	venue_id: str
	event_name: str
	client_name: str
	client_email: str
	client_phone: str | None = None
	event_type: str  # conference|wedding|gala|birthday|product_launch|training|other
	event_date: str
	start_time: str  # HH:MM
	end_time: str
	expected_attendance: int
	catering_required: bool = False
	av_required: bool = False
	decoration_required: bool = False
	notes: str | None = None


class EventBookingUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	event_name: str | None = None
	event_date: str | None = None
	start_time: str | None = None
	end_time: str | None = None
	expected_attendance: int | None = None
	status: str | None = None
	notes: str | None = None


class EventBookingResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str
	tenant_id: str
	venue_id: str
	event_name: str
	client_name: str
	client_email: str
	event_type: str
	event_date: str
	start_time: str
	end_time: str
	expected_attendance: int
	catering_required: bool
	av_required: bool
	venue_rental: float
	catering_estimate: float
	av_estimate: float
	total_estimate: float
	deposit_paid: float
	balance: float
	status: str
	beo_generated: bool
	created_at: str


class BEOCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	event_booking_id: str
	menu_selections: list[dict[str, Any]] = Field(default_factory=list)
	av_requirements: list[str] = Field(default_factory=list)
	setup_style: str = "theatre"  # theatre|classroom|banquet|u_shape|boardroom|cocktail
	special_requirements: str | None = None


class ContractCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	event_booking_id: str
	deposit_pct: float = 30.0
	payment_terms: str = "50% 30 days before, balance on day"
	cancellation_policy: str = "standard"
	special_clauses: str | None = None


class EVNListFilter(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	venue_id: str | None = None
	event_type: str | None = None
	date_from: str | None = None
	date_to: str | None = None
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
