"""Pydantic v2 models for Property Management System."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def _uid() -> str:
	return uuid4().hex


class RoomCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	room_number: str
	room_type: str  # single|double|suite|deluxe|presidential
	floor: int
	capacity: int = 2
	rate_per_night: float
	amenities: list[str] = Field(default_factory=list)
	notes: str | None = None


class RoomUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	room_type: str | None = None
	capacity: int | None = None
	rate_per_night: float | None = None
	amenities: list[str] | None = None
	status: str | None = None
	notes: str | None = None


class RoomResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str
	tenant_id: str
	room_number: str
	room_type: str
	floor: int
	capacity: int
	rate_per_night: float
	amenities: list[str]
	status: str
	notes: str | None
	created_at: str
	updated_at: str | None = None


class GuestCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	first_name: str
	last_name: str
	email: str
	phone: str | None = None
	nationality: str | None = None
	id_type: str | None = None  # passport|national_id|driver_license
	id_number: str | None = None
	date_of_birth: str | None = None
	vip_level: str = "standard"  # standard|silver|gold|platinum


class GuestUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	phone: str | None = None
	email: str | None = None
	vip_level: str | None = None
	notes: str | None = None


class GuestResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str
	tenant_id: str
	first_name: str
	last_name: str
	email: str
	phone: str | None
	nationality: str | None
	id_type: str | None
	id_number: str | None
	vip_level: str
	status: str
	created_at: str


class ReservationCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	guest_id: str
	room_id: str
	check_in_date: str
	check_out_date: str
	adults: int = 1
	children: int = 0
	rate_plan: str = "standard"
	special_requests: str | None = None
	source: str = "direct"  # direct|ota|gds|phone|walk_in


class ReservationUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	check_in_date: str | None = None
	check_out_date: str | None = None
	adults: int | None = None
	children: int | None = None
	special_requests: str | None = None
	status: str | None = None


class ReservationResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str
	tenant_id: str
	guest_id: str
	room_id: str
	check_in_date: str
	check_out_date: str
	nights: int
	adults: int
	children: int
	rate_plan: str
	total_amount: float
	paid_amount: float
	balance: float
	special_requests: str | None
	source: str
	status: str
	created_at: str


class FolioCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	reservation_id: str
	charge_type: str  # room|food|beverage|spa|laundry|telephone|other
	description: str
	amount: float
	quantity: int = 1


class FolioResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str
	tenant_id: str
	reservation_id: str
	charge_type: str
	description: str
	amount: float
	quantity: int
	total: float
	status: str
	created_at: str


class HousekeepingTaskCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	room_id: str
	task_type: str  # clean|turndown|inspect|maintenance|deep_clean
	priority: str = "normal"  # low|normal|high|urgent
	assigned_to: str | None = None
	notes: str | None = None


class HousekeepingTaskResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str
	tenant_id: str
	room_id: str
	task_type: str
	priority: str
	assigned_to: str | None
	notes: str | None
	status: str
	started_at: str | None
	completed_at: str | None
	created_at: str


class NightAuditReport(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	tenant_id: str
	audit_date: str
	total_rooms: int
	occupied_rooms: int
	occupancy_rate: float
	total_revenue: float
	room_revenue: float
	ancillary_revenue: float
	arrivals: int
	departures: int
	no_shows: int
	walk_ins: int
	generated_at: str


class PMSListFilter(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	status: str | None = None
	room_type: str | None = None
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
	actor: str | None
	details: dict[str, Any] = Field(default_factory=dict)
	created_at: str
