"""AgriIoT & Precision Farming models — Pydantic v2."""
from __future__ import annotations
from enum import Enum
from typing import Any
from pydantic import BaseModel, ConfigDict, Field
from uuid_extensions import uuid7str


class DeviceType(str, Enum):
	SOIL_SENSOR = "soil_sensor"
	WEATHER_STATION = "weather_station"
	DRONE = "drone"
	YIELD_MONITOR = "yield_monitor"
	SPRAYER = "sprayer"
	IRRIGATION_CONTROLLER = "irrigation_controller"


class ImageryType(str, Enum):
	RGB = "rgb"
	NDVI = "ndvi"
	THERMAL = "thermal"
	MULTISPECTRAL = "multispectral"


class ZoneStatus(str, Enum):
	OPTIMAL = "optimal"
	STRESSED = "stressed"
	CRITICAL = "critical"
	UNPLANTED = "unplanted"


class DeviceCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	name: str
	device_type: DeviceType
	farm_parcel_id: str
	location_lat: float | None = None
	location_lng: float | None = None
	serial_number: str | None = None
	firmware_version: str | None = None
	calibration_date: str | None = None
	notes: str | None = None


class DeviceUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	name: str | None = None
	firmware_version: str | None = None
	calibration_date: str | None = None
	active: bool | None = None
	notes: str | None = None


class DeviceResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	name: str
	device_type: DeviceType
	farm_parcel_id: str
	location_lat: float | None = None
	location_lng: float | None = None
	serial_number: str | None = None
	firmware_version: str | None = None
	calibration_date: str | None = None
	active: bool = True
	last_telemetry_at: str | None = None
	notes: str | None = None
	created_at: str
	updated_at: str


class TelemetryCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	device_id: str
	readings: dict[str, float]
	recorded_at: str | None = None
	gps_lat: float | None = None
	gps_lng: float | None = None


class TelemetryResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	device_id: str
	readings: dict[str, float]
	recorded_at: str
	gps_lat: float | None = None
	gps_lng: float | None = None
	created_at: str


class DroneImageryCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	farm_parcel_id: str
	drone_id: str | None = None
	imagery_type: ImageryType
	captured_at: str
	file_url: str
	resolution_cm: float | None = None
	coverage_ha: float | None = None
	notes: str | None = None


class DroneImageryResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	farm_parcel_id: str
	drone_id: str | None = None
	imagery_type: ImageryType
	captured_at: str
	file_url: str
	resolution_cm: float | None = None
	coverage_ha: float | None = None
	ndvi_mean: float | None = None
	ndvi_min: float | None = None
	ndvi_max: float | None = None
	zone_analysis: list[dict[str, Any]] = Field(default_factory=list)
	notes: str | None = None
	created_at: str


class YieldMapCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	farm_parcel_id: str
	crop_id: str
	season: str
	harvest_date: str
	zones: list[dict[str, Any]]
	equipment_id: str | None = None
	notes: str | None = None


class YieldMapResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	farm_parcel_id: str
	crop_id: str
	season: str
	harvest_date: str
	zones: list[dict[str, Any]]
	total_yield_kg: float
	avg_yield_kg_ha: float
	equipment_id: str | None = None
	notes: str | None = None
	created_at: str


class PrescriptionCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	farm_parcel_id: str
	crop_id: str | None = None
	application_type: str
	zones: list[dict[str, Any]]
	generated_from: str | None = None
	notes: str | None = None


class PrescriptionResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	farm_parcel_id: str
	crop_id: str | None = None
	application_type: str
	zones: list[dict[str, Any]]
	total_area_ha: float
	generated_from: str | None = None
	applied: bool = False
	applied_at: str | None = None
	notes: str | None = None
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
