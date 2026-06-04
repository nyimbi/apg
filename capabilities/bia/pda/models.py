"""Pydantic v2 models for APG Predictive Analytics (bia_pda)."""

from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Any
from pydantic import BaseModel, ConfigDict, Field
from uuid6 import uuid7

def uuid7str() -> str: return str(uuid7())

class ModelType(str, Enum):
	LINEAR_REGRESSION = "linear_regression"; LOGISTIC_REGRESSION = "logistic_regression"
	RANDOM_FOREST = "random_forest"; GRADIENT_BOOSTING = "gradient_boosting"
	NEURAL_NETWORK = "neural_network"; ARIMA = "arima"; PROPHET = "prophet"
	LSTM = "lstm"; XGBOOST = "xgboost"; ISOLATION_FOREST = "isolation_forest"; CLUSTERING = "clustering"

class ModelState(str, Enum):
	TRAINING = "training"; TRAINED = "trained"; DEPLOYED = "deployed"
	DEPRECATED = "deprecated"; FAILED = "failed"

class ForecastHorizon(str, Enum):
	H1D="1d"; H7D="7d"; H14D="14d"; H30D="30d"; H90D="90d"; H180D="180d"; H365D="365d"; CUSTOM="custom"

class ScenarioType(str, Enum):
	OPTIMISTIC="optimistic"; PESSIMISTIC="pessimistic"; BASE="base"
	STRESS_TEST="stress_test"; CUSTOM="custom"

class OutputType(str, Enum):
	POINT_FORECAST="point_forecast"; INTERVAL_FORECAST="interval_forecast"
	PROBABILITY_DISTRIBUTION="probability_distribution"; CLASSIFICATION="classification"
	ANOMALY_SCORE="anomaly_score"; CLUSTER_LABEL="cluster_label"

_CFG = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

class MLModelCreate(BaseModel):
	model_config = _CFG
	tenant_id: str; name: str; model_type: ModelType; owner_id: str
	training_datasource_id: str; feature_ids: list[str] = Field(default_factory=list)
	target_column: str | None = None; description: str | None = None
	tags: list[str] = Field(default_factory=list)

class MLModelResponse(BaseModel):
	model_config = _CFG
	id: str = Field(default_factory=uuid7str); tenant_id: str; name: str
	model_type: ModelType; state: ModelState = ModelState.TRAINING; version: str = "1.0.0"
	owner_id: str; training_datasource_id: str; feature_ids: list[str] = Field(default_factory=list)
	target_column: str | None = None; accuracy_score: float | None = None
	description: str | None = None; tags: list[str] = Field(default_factory=list)
	trained_at: datetime | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow); created_by: str = "system"

class ForecastCreate(BaseModel):
	model_config = _CFG
	tenant_id: str; model_id: str; horizon: ForecastHorizon
	output_type: OutputType = OutputType.POINT_FORECAST
	confidence_interval: float = 0.95; owner_id: str
	parameters: dict[str, Any] = Field(default_factory=dict)

class ForecastResponse(BaseModel):
	model_config = _CFG
	id: str = Field(default_factory=uuid7str); tenant_id: str; model_id: str
	horizon: ForecastHorizon; output_type: OutputType; confidence_interval: float
	owner_id: str; forecast_data: list[dict[str, Any]] = Field(default_factory=list)
	parameters: dict[str, Any] = Field(default_factory=dict)
	generated_at: datetime = Field(default_factory=datetime.utcnow)
	created_at: datetime = Field(default_factory=datetime.utcnow); created_by: str = "system"

class ScenarioCreate(BaseModel):
	model_config = _CFG
	tenant_id: str; model_id: str; name: str; scenario_type: ScenarioType
	parameters: dict[str, Any]; owner_id: str; description: str | None = None

class ScenarioResponse(BaseModel):
	model_config = _CFG
	id: str = Field(default_factory=uuid7str); tenant_id: str; model_id: str
	name: str; scenario_type: ScenarioType; parameters: dict[str, Any]
	owner_id: str; results: dict[str, Any] = Field(default_factory=dict)
	description: str | None = None
	simulated_at: datetime | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow); created_by: str = "system"

class FeatureCreate(BaseModel):
	model_config = _CFG
	tenant_id: str; name: str; feature_type: str; source_column: str
	datasource_id: str; owner_id: str; description: str | None = None

class FeatureResponse(BaseModel):
	model_config = _CFG
	id: str = Field(default_factory=uuid7str); tenant_id: str; name: str
	feature_type: str; source_column: str; datasource_id: str; owner_id: str
	description: str | None = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow); created_by: str = "system"

class PredictionResponse(BaseModel):
	model_config = _CFG
	id: str = Field(default_factory=uuid7str); tenant_id: str; model_id: str
	input_data: dict[str, Any]; output: Any; confidence: float | None = None
	served_at: datetime = Field(default_factory=datetime.utcnow); created_by: str = "system"
