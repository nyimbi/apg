"""Pydantic v2 models for APG Store Intelligence."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid6 import uuid7

from pydantic import BaseModel, ConfigDict, Field, AfterValidator
from typing import Annotated


def uuid7str() -> str:
	return str(uuid7())


def _non_empty(v: str) -> str:
	assert v and v.strip(), "must be non-empty"
	return v.strip()


NonEmptyStr = Annotated[str, AfterValidator(_non_empty)]
_CFG = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class SinStoreCreate(BaseModel):
	model_config = _CFG
	tenant_id: NonEmptyStr
	store_code: NonEmptyStr
	name: NonEmptyStr
	store_format: str
	address: dict[str, Any]
	latitude: float
	longitude: float
	sqm_total: float
	sqm_selling: float
	trading_hours: dict[str, Any] = Field(default_factory=dict)
	created_by: NonEmptyStr


class SinStoreResponse(SinStoreCreate):
	id: str = Field(default_factory=uuid7str)
	is_active: bool = True
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Zone
# ---------------------------------------------------------------------------

class SinZoneCreate(BaseModel):
	model_config = _CFG
	tenant_id: NonEmptyStr
	store_id: NonEmptyStr
	zone_code: NonEmptyStr
	zone_name: NonEmptyStr
	zone_type: str
	sqm: float
	floor_level: int = 0
	polygon_coords: list[list[float]] = Field(default_factory=list)
	created_by: NonEmptyStr


class SinZoneResponse(SinZoneCreate):
	id: str = Field(default_factory=uuid7str)
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Sensor
# ---------------------------------------------------------------------------

class SinSensorCreate(BaseModel):
	model_config = _CFG
	tenant_id: NonEmptyStr
	store_id: NonEmptyStr
	zone_id: NonEmptyStr
	sensor_code: NonEmptyStr
	sensor_type: str
	hardware_model: str | None = None
	serial_number: str | None = None
	counting_interval_seconds: int = 60
	is_active: bool = True
	created_by: NonEmptyStr


class SinSensorResponse(SinSensorCreate):
	id: str = Field(default_factory=uuid7str)
	status: str = "offline"
	last_heartbeat_at: datetime | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Traffic Count
# ---------------------------------------------------------------------------

class SinTrafficCountCreate(BaseModel):
	model_config = _CFG
	tenant_id: NonEmptyStr
	store_id: NonEmptyStr
	zone_id: NonEmptyStr
	sensor_id: NonEmptyStr
	period_start: datetime
	period_end: datetime
	entries: int = 0
	exits: int = 0
	occupancy_peak: int = 0
	dwell_avg_seconds: float = 0.0
	created_by: NonEmptyStr


class SinTrafficCountResponse(SinTrafficCountCreate):
	id: str = Field(default_factory=uuid7str)
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Planogram Audit
# ---------------------------------------------------------------------------

class SinPlanogramAuditCreate(BaseModel):
	model_config = _CFG
	tenant_id: NonEmptyStr
	store_id: NonEmptyStr
	zone_id: NonEmptyStr
	planogram_id: NonEmptyStr
	audited_by: NonEmptyStr  # agent or staff id
	audit_method: str  # "image_ai", "manual", "rfid"
	compliance_status: str
	deviation_details: list[dict[str, Any]] = Field(default_factory=list)
	image_references: list[str] = Field(default_factory=list)
	notes: str | None = None
	created_by: NonEmptyStr


class SinPlanogramAuditResponse(SinPlanogramAuditCreate):
	id: str = Field(default_factory=uuid7str)
	compliance_score_pct: float = 100.0
	audited_at: datetime = Field(default_factory=datetime.utcnow)
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Shelf Alert
# ---------------------------------------------------------------------------

class SinShelfAlertCreate(BaseModel):
	model_config = _CFG
	tenant_id: NonEmptyStr
	store_id: NonEmptyStr
	zone_id: NonEmptyStr
	sku: NonEmptyStr
	alert_type: str
	severity: str
	current_stock_level: int | None = None
	threshold_value: int | None = None
	detected_by: str  # sensor_id or agent_id
	created_by: NonEmptyStr


class SinShelfAlertResponse(SinShelfAlertCreate):
	id: str = Field(default_factory=uuid7str)
	status: str = "open"
	assigned_to: str | None = None
	replenishment_triggered: bool = False
	resolved_at: datetime | None = None
	resolution_notes: str | None = None
	detected_at: datetime = Field(default_factory=datetime.utcnow)
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Conversion Event
# ---------------------------------------------------------------------------

class SinConversionEventCreate(BaseModel):
	model_config = _CFG
	tenant_id: NonEmptyStr
	store_id: NonEmptyStr
	session_id: NonEmptyStr
	conversion_metric: str
	from_stage: str
	to_stage: str
	converted: bool
	dwell_seconds: float = 0.0
	created_by: NonEmptyStr


class SinConversionEventResponse(SinConversionEventCreate):
	id: str = Field(default_factory=uuid7str)
	occurred_at: datetime = Field(default_factory=datetime.utcnow)
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# KPI Snapshot
# ---------------------------------------------------------------------------

class SinKpiSnapshotCreate(BaseModel):
	model_config = _CFG
	tenant_id: NonEmptyStr
	store_id: NonEmptyStr
	kpi_category: str
	period_type: str  # "hourly", "daily", "weekly"
	period_start: datetime
	period_end: datetime
	kpi_values: dict[str, float] = Field(default_factory=dict)
	benchmark_type: str | None = None
	benchmark_values: dict[str, float] = Field(default_factory=dict)
	created_by: NonEmptyStr


class SinKpiSnapshotResponse(SinKpiSnapshotCreate):
	id: str = Field(default_factory=uuid7str)
	vs_benchmark_delta: dict[str, float] = Field(default_factory=dict)
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------

class SinHeatmapCreate(BaseModel):
	model_config = _CFG
	tenant_id: NonEmptyStr
	store_id: NonEmptyStr
	floor_level: int = 0
	resolution: str = "2m"
	period_start: datetime
	period_end: datetime
	grid_data: list[list[float]] = Field(default_factory=list)  # 2D intensity grid
	pii_masked: bool = True
	created_by: NonEmptyStr


class SinHeatmapResponse(SinHeatmapCreate):
	id: str = Field(default_factory=uuid7str)
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
