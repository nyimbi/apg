"""Pydantic v2 models for APG Time Series Analytics (bia_tsa)."""

from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Any
from pydantic import BaseModel, ConfigDict, Field
from uuid6 import uuid7

def uuid7str() -> str: return str(uuid7())

class StreamState(str, Enum):
	ACTIVE="active"; PAUSED="paused"; ERROR="error"; ARCHIVED="archived"

class AnomalyMethod(str, Enum):
	ZSCORE="zscore"; IQR="iqr"; ISOLATION_FOREST="isolation_forest"
	LSTM_AUTOENCODER="lstm_autoencoder"; PROPHET_RESIDUAL="prophet_residual"
	MAD="mad"; SEASONAL_DECOMPOSITION="seasonal_decomposition"; CUSTOM="custom"

class ForecastModel(str, Enum):
	ARIMA="arima"; SARIMA="sarima"; PROPHET="prophet"
	EXP_SMOOTHING="exponential_smoothing"; LSTM="lstm"
	TRANSFORMER="transformer"; ENSEMBLE="ensemble"

class WindowType(str, Enum):
	TUMBLING="tumbling"; SLIDING="sliding"; SESSION="session"; HOPPING="hopping"

class InterpolationMethod(str, Enum):
	LINEAR="linear"; FORWARD_FILL="forward_fill"; BACKWARD_FILL="backward_fill"
	CUBIC_SPLINE="cubic_spline"; ZERO="zero"; NONE="none"

_CFG = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

class StreamCreate(BaseModel):
	model_config = _CFG
	tenant_id: str; name: str; protocol: str; frequency: str; owner_id: str
	source_identifier: str; data_type: str = "numeric"
	unit_of_measure: str | None = None; description: str | None = None
	tags: list[str] = Field(default_factory=list)

class StreamResponse(BaseModel):
	model_config = _CFG
	id: str = Field(default_factory=uuid7str); tenant_id: str; name: str
	protocol: str; frequency: str; owner_id: str; source_identifier: str
	data_type: str; unit_of_measure: str | None = None
	state: StreamState = StreamState.ACTIVE
	description: str | None = None; tags: list[str] = Field(default_factory=list)
	last_ingested_at: datetime | None = None; point_count: int = 0
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow); created_by: str = "system"

class AnomalyConfigCreate(BaseModel):
	model_config = _CFG
	tenant_id: str; stream_id: str; name: str; method: AnomalyMethod
	sensitivity: float = 0.95; owner_id: str
	config: dict[str, Any] = Field(default_factory=dict)

class AnomalyConfigResponse(BaseModel):
	model_config = _CFG
	id: str = Field(default_factory=uuid7str); tenant_id: str; stream_id: str
	name: str; method: AnomalyMethod; sensitivity: float; owner_id: str
	config: dict[str, Any] = Field(default_factory=dict); active: bool = True
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow); created_by: str = "system"

class AnomalyEvent(BaseModel):
	model_config = _CFG
	id: str = Field(default_factory=uuid7str); tenant_id: str; stream_id: str
	config_id: str; detected_at: str; value: float; score: float
	confirmed: bool = False; severity: str = "medium"
	created_at: datetime = Field(default_factory=datetime.utcnow); created_by: str = "system"

class ForecastCreate(BaseModel):
	model_config = _CFG
	tenant_id: str; stream_id: str; model: ForecastModel
	horizon_periods: int; owner_id: str; confidence_interval: float = 0.95

class ForecastResponse(BaseModel):
	model_config = _CFG
	id: str = Field(default_factory=uuid7str); tenant_id: str; stream_id: str
	model: ForecastModel; horizon_periods: int; confidence_interval: float; owner_id: str
	forecast_data: list[dict[str, Any]] = Field(default_factory=list)
	generated_at: datetime = Field(default_factory=datetime.utcnow); created_by: str = "system"

class WindowCreate(BaseModel):
	model_config = _CFG
	tenant_id: str; stream_id: str; name: str; window_type: WindowType
	size_seconds: int; aggregation_function: str; owner_id: str

class WindowResponse(BaseModel):
	model_config = _CFG
	id: str = Field(default_factory=uuid7str); tenant_id: str; stream_id: str
	name: str; window_type: WindowType; size_seconds: int; aggregation_function: str
	owner_id: str; active: bool = True
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow); created_by: str = "system"

class DecompositionResult(BaseModel):
	model_config = _CFG
	id: str = Field(default_factory=uuid7str); tenant_id: str; stream_id: str
	components: list[str]; trend_data: list[dict[str, Any]] = Field(default_factory=list)
	seasonality_data: list[dict[str, Any]] = Field(default_factory=list)
	residual_data: list[dict[str, Any]] = Field(default_factory=list)
	model_type: str = "additive"
	computed_at: datetime = Field(default_factory=datetime.utcnow); created_by: str = "system"
