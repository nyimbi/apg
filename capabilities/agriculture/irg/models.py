"""Irrigation Management models — Pydantic v2."""
from __future__ import annotations
from enum import Enum
from typing import Any
from pydantic import BaseModel, ConfigDict, Field
from uuid_extensions import uuid7str


class SensorType(str, Enum):
	SOIL_MOISTURE = "soil_moisture"
	FLOW_METER = "flow_meter"
	PRESSURE = "pressure"
	WEATHER = "weather"
	WATER_LEVEL = "water_level"


class IrrigationMethod(str, Enum):
	DRIP = "drip"
	SPRINKLER = "sprinkler"
	FLOOD = "flood"
	FURROW = "furrow"
	CENTRE_PIVOT = "centre_pivot"


class ScheduleStatus(str, Enum):
	SCHEDULED = "scheduled"
	ACTIVE = "active"
	COMPLETED = "completed"
	CANCELLED = "cancelled"
	SKIPPED = "skipped"


class SensorCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	name: str
	sensor_type: SensorType
	farm_parcel_id: str
	location_lat: float | None = None
	location_lng: float | None = None
	unit: str
	min_threshold: float | None = None
	max_threshold: float | None = None
	notes: str | None = None


class SensorUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	name: str | None = None
	min_threshold: float | None = None
	max_threshold: float | None = None
	notes: str | None = None
	active: bool | None = None


class SensorResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	name: str
	sensor_type: SensorType
	farm_parcel_id: str
	location_lat: float | None = None
	location_lng: float | None = None
	unit: str
	min_threshold: float | None = None
	max_threshold: float | None = None
	last_reading: float | None = None
	last_reading_at: str | None = None
	active: bool = True
	notes: str | None = None
	created_at: str
	updated_at: str


class SensorReadingCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	sensor_id: str
	value: float
	recorded_at: str | None = None
	quality_flag: str | None = None


class SensorReadingResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	sensor_id: str
	value: float
	recorded_at: str
	quality_flag: str | None = None
	alert_triggered: bool = False
	created_at: str


class IrrigationScheduleCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	farm_parcel_id: str
	method: IrrigationMethod
	scheduled_start: str
	duration_minutes: int
	volume_m3: float | None = None
	trigger_condition: str | None = None
	notes: str | None = None


class IrrigationScheduleUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	scheduled_start: str | None = None
	duration_minutes: int | None = None
	volume_m3: float | None = None
	status: ScheduleStatus | None = None
	actual_start: str | None = None
	actual_end: str | None = None
	actual_volume_m3: float | None = None
	notes: str | None = None


class IrrigationScheduleResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	farm_parcel_id: str
	method: IrrigationMethod
	scheduled_start: str
	duration_minutes: int
	volume_m3: float | None = None
	trigger_condition: str | None = None
	status: ScheduleStatus = ScheduleStatus.SCHEDULED
	actual_start: str | None = None
	actual_end: str | None = None
	actual_volume_m3: float | None = None
	notes: str | None = None
	created_at: str
	updated_at: str


class CanalCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	name: str
	length_m: float
	capacity_m3_s: float
	served_parcels: list[str] = Field(default_factory=list)
	notes: str | None = None


class CanalResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	name: str
	length_m: float
	capacity_m3_s: float
	served_parcels: list[str]
	maintenance_due: str | None = None
	notes: str | None = None
	created_at: str
	updated_at: str


class WaterAccountEntry(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	farm_parcel_id: str
	period: str
	allocated_m3: float
	used_m3: float
	balance_m3: float
	created_at: str


class AuditEvent(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	event_type: str
	entity_id: str
	entity_type: str
	payload: dict[str, Any] = Field(default_factory=dict)
	occurred_at: str
