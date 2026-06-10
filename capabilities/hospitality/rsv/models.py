"""Pydantic v2 models for Reservations & Channel Manager."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def _uid() -> str:
	return uuid4().hex


class ChannelCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	code: str
	name: str
	channel_type: str  # ota|gds|direct|booking_engine|metasearch
	commission_pct: float = 0.0
	api_endpoint: str | None = None
	credentials_ref: str | None = None
	is_active: bool = True


class ChannelResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str
	tenant_id: str
	code: str
	name: str
	channel_type: str
	commission_pct: float
	is_active: bool
	status: str
	created_at: str


class BookingCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	channel_id: str
	external_booking_ref: str | None = None
	guest_name: str
	guest_email: str
	guest_phone: str | None = None
	room_type: str
	check_in_date: str
	check_out_date: str
	adults: int = 1
	children: int = 0
	rate: float
	currency: str = "KES"
	special_requests: str | None = None


class BookingUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	check_in_date: str | None = None
	check_out_date: str | None = None
	adults: int | None = None
	special_requests: str | None = None
	status: str | None = None


class BookingResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str
	tenant_id: str
	channel_id: str
	external_booking_ref: str | None
	guest_name: str
	guest_email: str
	room_type: str
	check_in_date: str
	check_out_date: str
	nights: int
	adults: int
	rate: float
	total_amount: float
	commission: float
	net_revenue: float
	status: str
	created_at: str


class AvailabilityCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	room_type: str
	date: str
	available_count: int
	stop_sell: bool = False


class AvailabilityResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str
	tenant_id: str
	room_type: str
	date: str
	available_count: int
	stop_sell: bool
	updated_at: str


class GDSConnectionCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	gds_provider: str  # amadeus|sabre|travelport
	property_code: str
	chain_code: str | None = None
	credentials_ref: str


class RSVListFilter(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	channel_id: str | None = None
	status: str | None = None
	date_from: str | None = None
	date_to: str | None = None
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
