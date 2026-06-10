"""Weather & Climate Analytics models — Pydantic v2."""
from __future__ import annotations
from enum import Enum
from typing import Any
from pydantic import BaseModel, ConfigDict, Field
from uuid_extensions import uuid7str


class AlertSeverity(str, Enum):
	INFO = "info"
	WARNING = "warning"
	WATCH = "watch"
	ADVISORY = "advisory"
	EMERGENCY = "emergency"


class ClimateRiskLevel(str, Enum):
	NEGLIGIBLE = "negligible"
	LOW = "low"
	MODERATE = "moderate"
	HIGH = "high"
	EXTREME = "extreme"


class ForecastCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	region: str
	source: str
	forecast_date: str
	valid_from: str
	valid_to: str
	temperature_min_c: float | None = None
	temperature_max_c: float | None = None
	rainfall_mm: float | None = None
	humidity_pct: float | None = None
	wind_speed_kmh: float | None = None
	wind_direction: str | None = None
	conditions: str | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)


class ForecastResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	region: str
	source: str
	forecast_date: str
	valid_from: str
	valid_to: str
	temperature_min_c: float | None = None
	temperature_max_c: float | None = None
	rainfall_mm: float | None = None
	humidity_pct: float | None = None
	wind_speed_kmh: float | None = None
	wind_direction: str | None = None
	conditions: str | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)
	created_at: str


class AlertThresholdCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	region: str
	parameter: str
	operator: str  # gt, lt, gte, lte
	threshold_value: float
	severity: AlertSeverity
	description: str | None = None
	active: bool = True


class AlertThresholdResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	region: str
	parameter: str
	operator: str
	threshold_value: float
	severity: AlertSeverity
	description: str | None = None
	active: bool = True
	created_at: str
	updated_at: str


class WeatherAlertResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	region: str
	threshold_id: str
	triggered_value: float
	severity: AlertSeverity
	message: str
	forecast_id: str | None = None
	issued_at: str
	acknowledged: bool = False


class HistoricalPatternCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	region: str
	year: int
	month: int
	avg_rainfall_mm: float | None = None
	avg_temp_c: float | None = None
	min_temp_c: float | None = None
	max_temp_c: float | None = None
	drought_days: int | None = None
	frost_days: int | None = None
	source: str | None = None


class HistoricalPatternResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	region: str
	year: int
	month: int
	avg_rainfall_mm: float | None = None
	avg_temp_c: float | None = None
	min_temp_c: float | None = None
	max_temp_c: float | None = None
	drought_days: int | None = None
	frost_days: int | None = None
	source: str | None = None
	created_at: str


class ClimateRiskAssessment(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	region: str
	crop_type: str
	season: str
	risk_level: ClimateRiskLevel
	drought_risk_score: float
	flood_risk_score: float
	frost_risk_score: float
	heat_stress_risk_score: float
	overall_score: float
	recommendations: list[str] = Field(default_factory=list)
	assessed_at: str


class AuditEvent(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True)
	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	event_type: str
	entity_id: str
	entity_type: str
	payload: dict[str, Any] = Field(default_factory=dict)
	occurred_at: str
