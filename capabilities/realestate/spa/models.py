"""Pydantic v2 models for Space Planning & Management (spa)."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from uuid6 import uuid7


def uuid7str() -> str:
	return str(uuid7())


class SpaceType(str, Enum):
	private_office = "private_office"
	open_plan = "open_plan"
	meeting_room = "meeting_room"
	conference_room = "conference_room"
	hot_desk = "hot_desk"
	collaboration_zone = "collaboration_zone"
	quiet_zone = "quiet_zone"
	reception = "reception"
	amenity = "amenity"
	storage = "storage"
	server_room = "server_room"
	common_area = "common_area"
	balcony = "balcony"
	terrace = "terrace"


class SpaceStatus(str, Enum):
	available = "available"
	occupied = "occupied"
	reserved = "reserved"
	under_fit_out = "under_fit_out"
	decommissioned = "decommissioned"
	mothballed = "mothballed"


class AllocationType(str, Enum):
	permanent = "permanent"
	hot_desk = "hot_desk"
	shared = "shared"
	dedicated = "dedicated"
	project_space = "project_space"
	visitor = "visitor"


class MoveType(str, Enum):
	internal_move = "internal_move"
	inter_floor_move = "inter_floor_move"
	inter_building_move = "inter_building_move"
	consolidation = "consolidation"
	expansion = "expansion"
	decommission = "decommission"


class MoveStatus(str, Enum):
	planning = "planning"
	approved = "approved"
	scheduled = "scheduled"
	in_progress = "in_progress"
	completed = "completed"
	cancelled = "cancelled"


class BookingType(str, Enum):
	desk = "desk"
	meeting_room = "meeting_room"
	parking = "parking"
	locker = "locker"
	visitor_pass = "visitor_pass"


class DensityBand(str, Enum):
	dense = "dense"
	standard = "standard"
	spacious = "spacious"
	executive = "executive"
	social_distancing = "social_distancing"


class SensorType(str, Enum):
	occupancy_sensor = "occupancy_sensor"
	badge_reader = "badge_reader"
	wifi_probe = "wifi_probe"
	camera_ai = "camera_ai"
	desk_sensor = "desk_sensor"
	meeting_room_sensor = "meeting_room_sensor"


# ── Floor Plan ────────────────────────────────────────────────────────────────

class FloorPlanCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	property_id: str
	floor: str
	file_format: str
	file_reference: str  # storage key/URL
	total_area: Decimal
	area_unit: str = "sqm"
	created_by: str

	@field_validator("total_area")
	@classmethod
	def _positive_area(cls, v: Decimal) -> Decimal:
		if v <= 0:
			raise ValueError("total_area must be positive")
		return v


class FloorPlanResponse(FloorPlanCreate):
	id: str = Field(default_factory=uuid7str)
	space_ids: list[str] = Field(default_factory=list)
	version: int = 1
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── Space ─────────────────────────────────────────────────────────────────────

class SpaceCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	property_id: str
	floor_plan_id: str
	space_ref: str
	space_type: SpaceType
	capacity: int
	area: Decimal
	area_unit: str = "sqm"
	description: str | None = None
	amenities: list[str] = Field(default_factory=list)
	created_by: str

	@field_validator("capacity")
	@classmethod
	def _positive_capacity(cls, v: int) -> int:
		if v < 1:
			raise ValueError("capacity must be at least 1")
		return v


class SpaceResponse(SpaceCreate):
	id: str = Field(default_factory=uuid7str)
	status: SpaceStatus = SpaceStatus.available
	current_allocation_id: str | None = None
	current_occupant_ids: list[str] = Field(default_factory=list)
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class SpaceUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	status: SpaceStatus | None = None
	capacity: int | None = None
	description: str | None = None
	amenities: list[str] | None = None


# ── Space Allocation ──────────────────────────────────────────────────────────

class SpaceAllocationCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	space_id: str
	allocation_type: AllocationType
	department_id: str | None = None
	occupant_ids: list[str] = Field(default_factory=list)
	start_date: date
	end_date: date | None = None
	headcount: int = 1
	created_by: str


class SpaceAllocationResponse(SpaceAllocationCreate):
	id: str = Field(default_factory=uuid7str)
	is_active: bool = True
	ended_at: datetime | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── Move ──────────────────────────────────────────────────────────────────────

class MoveCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	move_type: MoveType
	from_space_ids: list[str]
	to_space_ids: list[str]
	occupant_ids: list[str]
	headcount: int
	scheduled_date: date
	reason: str | None = None
	created_by: str

	@field_validator("headcount")
	@classmethod
	def _positive_headcount(cls, v: int) -> int:
		if v < 1:
			raise ValueError("headcount must be at least 1")
		return v


class MoveResponse(MoveCreate):
	id: str = Field(default_factory=uuid7str)
	status: MoveStatus = MoveStatus.planning
	approved_by: str | None = None
	completed_at: datetime | None = None
	churn_reason: str | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── Booking ───────────────────────────────────────────────────────────────────

class BookingCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	space_id: str
	booking_type: BookingType
	booked_by: str
	start_datetime: datetime
	end_datetime: datetime
	attendees: int = 1
	notes: str | None = None
	created_by: str

	@field_validator("end_datetime")
	@classmethod
	def _end_after_start(cls, v: datetime, info: Any) -> datetime:
		start = info.data.get("start_datetime")
		if start and v <= start:
			raise ValueError("end_datetime must be after start_datetime")
		return v


class BookingResponse(BookingCreate):
	id: str = Field(default_factory=uuid7str)
	status: str = "confirmed"  # confirmed | cancelled | no_show | completed
	cancelled_at: datetime | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ── Occupancy Data ────────────────────────────────────────────────────────────

class OccupancyDataCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	space_id: str
	sensor_type: SensorType
	recorded_at: datetime
	occupant_count: int
	data_anonymised: bool = False
	created_by: str


class OccupancyDataResponse(OccupancyDataCreate):
	id: str = Field(default_factory=uuid7str)
	created_at: datetime = Field(default_factory=datetime.utcnow)


# ── Density Plan ──────────────────────────────────────────────────────────────

class DensityPlanCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	property_id: str
	floor: str | None = None
	density_band: DensityBand
	target_sqm_per_person: Decimal
	workplace_strategy: str
	effective_date: date
	created_by: str

	@field_validator("target_sqm_per_person")
	@classmethod
	def _positive_target(cls, v: Decimal) -> Decimal:
		if v <= 0:
			raise ValueError("target_sqm_per_person must be positive")
		return v


class DensityPlanResponse(DensityPlanCreate):
	id: str = Field(default_factory=uuid7str)
	current_sqm_per_person: Decimal | None = None
	optimisation_recommendations: list[str] = Field(default_factory=list)
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
