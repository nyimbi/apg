"""Pydantic v2 models for APG Timetabling & Scheduling."""

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


def _positive_int(v: int) -> int:
	assert v > 0, f"value must be positive, got {v}"
	return v


def _non_negative_int(v: int) -> int:
	assert v >= 0, f"value must be non-negative, got {v}"
	return v


NonEmptyStr = Annotated[str, AfterValidator(_non_empty)]
PositiveInt = Annotated[int, AfterValidator(_positive_int)]
NonNegativeInt = Annotated[int, AfterValidator(_non_negative_int)]

_BASE_CONFIG = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)


# ---------------------------------------------------------------------------
# Timetable
# ---------------------------------------------------------------------------

class TimetableCreate(BaseModel):
	model_config = _BASE_CONFIG

	id: str = Field(default_factory=uuid7str)
	tenant_id: NonEmptyStr
	name: NonEmptyStr
	timetable_type: NonEmptyStr
	academic_year: NonEmptyStr
	term: NonEmptyStr
	status: str = "draft"
	generation_algorithm: str = "constraint_propagation"
	approval_reference: str | None = None
	published_at: datetime | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: NonEmptyStr


class TimetableUpdate(BaseModel):
	model_config = _BASE_CONFIG

	name: str | None = None
	status: str | None = None
	approval_reference: str | None = None
	published_at: datetime | None = None
	metadata: dict[str, Any] | None = None
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class TimetableResponse(TimetableCreate):
	pass


# ---------------------------------------------------------------------------
# Constraint
# ---------------------------------------------------------------------------

class ConstraintCreate(BaseModel):
	model_config = _BASE_CONFIG

	id: str = Field(default_factory=uuid7str)
	tenant_id: NonEmptyStr
	timetable_id: NonEmptyStr
	constraint_type: NonEmptyStr
	description: str = ""
	entity_id: NonEmptyStr        # teacher_id, room_id, subject_id, etc.
	entity_type: NonEmptyStr      # "teacher" | "room" | "subject" | "student_group"
	parameters: dict[str, Any] = Field(default_factory=dict)
	is_hard: bool = True          # hard = must not violate; soft = prefer not to violate
	weight: int = 100             # for soft constraints; 0-100
	removal_approval: str | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: NonEmptyStr


class ConstraintUpdate(BaseModel):
	model_config = _BASE_CONFIG

	description: str | None = None
	parameters: dict[str, Any] | None = None
	is_hard: bool | None = None
	weight: int | None = None
	removal_approval: str | None = None
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class ConstraintResponse(ConstraintCreate):
	pass


# ---------------------------------------------------------------------------
# Room
# ---------------------------------------------------------------------------

class RoomCreate(BaseModel):
	model_config = _BASE_CONFIG

	id: str = Field(default_factory=uuid7str)
	tenant_id: NonEmptyStr
	name: NonEmptyStr
	code: NonEmptyStr
	room_type: NonEmptyStr
	capacity: PositiveInt
	building: str | None = None
	floor: str | None = None
	amenities: list[str] = Field(default_factory=list)
	is_available: bool = True
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: NonEmptyStr


class RoomUpdate(BaseModel):
	model_config = _BASE_CONFIG

	name: str | None = None
	capacity: PositiveInt | None = None
	room_type: str | None = None
	amenities: list[str] | None = None
	is_available: bool | None = None
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class RoomResponse(RoomCreate):
	pass


# ---------------------------------------------------------------------------
# TimeSlot
# ---------------------------------------------------------------------------

class TimeSlotCreate(BaseModel):
	model_config = _BASE_CONFIG

	id: str = Field(default_factory=uuid7str)
	tenant_id: NonEmptyStr
	timetable_id: NonEmptyStr
	day_of_week: NonEmptyStr
	start_time: str   # "HH:MM"
	end_time: str     # "HH:MM"
	duration_minutes: PositiveInt
	period_number: PositiveInt
	is_break: bool = False
	label: str | None = None    # e.g. "Lunch Break", "Period 3"
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: NonEmptyStr


class TimeSlotResponse(TimeSlotCreate):
	pass


# ---------------------------------------------------------------------------
# ScheduleEntry  (the core assignment: teacher + room + subject + student_group + time_slot)
# ---------------------------------------------------------------------------

class ScheduleEntryCreate(BaseModel):
	model_config = _BASE_CONFIG

	id: str = Field(default_factory=uuid7str)
	tenant_id: NonEmptyStr
	timetable_id: NonEmptyStr
	time_slot_id: NonEmptyStr
	room_id: NonEmptyStr
	teacher_id: NonEmptyStr
	subject_id: NonEmptyStr
	student_group_id: NonEmptyStr
	capacity_check_performed: bool = False
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: NonEmptyStr


class ScheduleEntryUpdate(BaseModel):
	model_config = _BASE_CONFIG

	time_slot_id: str | None = None
	room_id: str | None = None
	teacher_id: str | None = None
	capacity_check_performed: bool | None = None
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class ScheduleEntryResponse(ScheduleEntryCreate):
	pass


# ---------------------------------------------------------------------------
# Conflict
# ---------------------------------------------------------------------------

class ConflictCreate(BaseModel):
	model_config = _BASE_CONFIG

	id: str = Field(default_factory=uuid7str)
	tenant_id: NonEmptyStr
	timetable_id: NonEmptyStr
	conflict_type: NonEmptyStr
	entry_ids: list[str] = Field(default_factory=list)   # schedule entries involved
	description: str = ""
	severity: str = "hard"    # "hard" | "soft"
	resolution_type: str | None = None
	resolved_at: datetime | None = None
	resolved_by: str | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: NonEmptyStr


class ConflictUpdate(BaseModel):
	model_config = _BASE_CONFIG

	resolution_type: str | None = None
	resolved_at: datetime | None = None
	resolved_by: str | None = None
	description: str | None = None
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class ConflictResponse(ConflictCreate):
	pass


# ---------------------------------------------------------------------------
# SubstitutionRequest
# ---------------------------------------------------------------------------

class SubstitutionRequestCreate(BaseModel):
	model_config = _BASE_CONFIG

	id: str = Field(default_factory=uuid7str)
	tenant_id: NonEmptyStr
	timetable_id: NonEmptyStr
	original_entry_id: NonEmptyStr
	absent_teacher_id: NonEmptyStr
	substitute_teacher_id: str | None = None
	reason: NonEmptyStr
	status: str = "pending"
	teacher_consent_recorded: bool = False
	date: str   # ISO date string
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: NonEmptyStr


class SubstitutionRequestUpdate(BaseModel):
	model_config = _BASE_CONFIG

	substitute_teacher_id: str | None = None
	status: str | None = None
	teacher_consent_recorded: bool | None = None
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class SubstitutionRequestResponse(SubstitutionRequestCreate):
	pass


# ---------------------------------------------------------------------------
# TtblAgent
# ---------------------------------------------------------------------------

class TtblAgent(BaseModel):
	model_config = _BASE_CONFIG

	id: str = Field(default_factory=uuid7str)
	tenant_id: NonEmptyStr
	name: NonEmptyStr
	runtime: NonEmptyStr
	role: NonEmptyStr
	scope: str = "timetabling operations"
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: NonEmptyStr
