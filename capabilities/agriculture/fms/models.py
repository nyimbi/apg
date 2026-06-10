"""Farm Management System models — Pydantic v2."""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from uuid_extensions import uuid7str


class ParcelStatus(str, Enum):
	ACTIVE = "active"
	FALLOW = "fallow"
	LEASED = "leased"
	DISPUTED = "disputed"
	INACTIVE = "inactive"


class InputCategory(str, Enum):
	SEED = "seed"
	FERTILIZER = "fertilizer"
	PESTICIDE = "pesticide"
	HERBICIDE = "herbicide"
	FUEL = "fuel"
	EQUIPMENT = "equipment"
	OTHER = "other"


class LabourTaskType(str, Enum):
	LAND_PREPARATION = "land_preparation"
	PLANTING = "planting"
	WEEDING = "weeding"
	FERTILIZATION = "fertilization"
	PEST_CONTROL = "pest_control"
	IRRIGATION = "irrigation"
	HARVESTING = "harvesting"
	POST_HARVEST = "post_harvest"
	GENERAL = "general"


# --- Parcel ---

class ParcelCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	name: str
	area_ha: float
	location_lat: float | None = None
	location_lng: float | None = None
	soil_type: str | None = None
	status: ParcelStatus = ParcelStatus.ACTIVE
	owner_id: str | None = None
	notes: str | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)


class ParcelUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	name: str | None = None
	area_ha: float | None = None
	soil_type: str | None = None
	status: ParcelStatus | None = None
	notes: str | None = None
	metadata: dict[str, Any] | None = None


class ParcelResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	name: str
	area_ha: float
	location_lat: float | None = None
	location_lng: float | None = None
	soil_type: str | None = None
	status: ParcelStatus
	owner_id: str | None = None
	notes: str | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)
	created_at: str
	updated_at: str


# --- Input Record ---

class InputRecordCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	farm_parcel_id: str
	crop_id: str | None = None
	category: InputCategory
	product_name: str
	quantity: float
	unit: str
	unit_cost: float
	supplier: str | None = None
	applied_date: str
	notes: str | None = None


class InputRecordResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	farm_parcel_id: str
	crop_id: str | None = None
	category: InputCategory
	product_name: str
	quantity: float
	unit: str
	unit_cost: float
	total_cost: float
	supplier: str | None = None
	applied_date: str
	notes: str | None = None
	created_at: str


# --- Labour Schedule ---

class LabourScheduleCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	farm_parcel_id: str
	task_type: LabourTaskType
	scheduled_date: str
	worker_count: int
	daily_rate: float
	duration_days: float = 1.0
	supervisor_id: str | None = None
	notes: str | None = None


class LabourScheduleUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	scheduled_date: str | None = None
	worker_count: int | None = None
	actual_worker_count: int | None = None
	completed: bool | None = None
	notes: str | None = None


class LabourScheduleResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	farm_parcel_id: str
	task_type: LabourTaskType
	scheduled_date: str
	worker_count: int
	daily_rate: float
	duration_days: float
	total_labour_cost: float
	actual_worker_count: int | None = None
	completed: bool = False
	supervisor_id: str | None = None
	notes: str | None = None
	created_at: str
	updated_at: str


# --- Cost Summary ---

class CostSummaryFilter(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	farm_parcel_id: str | None = None
	season: str | None = None
	category: InputCategory | None = None
	from_date: str | None = None
	to_date: str | None = None


class CostSummaryResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	farm_parcel_id: str | None = None
	season: str | None = None
	total_input_cost: float
	total_labour_cost: float
	total_cost: float
	cost_per_ha: float | None = None
	breakdown: dict[str, float] = Field(default_factory=dict)


# --- Farm Diary ---

class DiaryEntryCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	farm_parcel_id: str | None = None
	entry_date: str
	title: str
	body: str
	tags: list[str] = Field(default_factory=list)
	images: list[str] = Field(default_factory=list)
	author_id: str | None = None


class DiaryEntryResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	farm_parcel_id: str | None = None
	entry_date: str
	title: str
	body: str
	tags: list[str] = Field(default_factory=list)
	images: list[str] = Field(default_factory=list)
	author_id: str | None = None
	created_at: str
	updated_at: str


class AuditEvent(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	event_type: str
	entity_id: str
	entity_type: str
	payload: dict[str, Any] = Field(default_factory=dict)
	occurred_at: str
