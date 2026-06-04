"""Pydantic v2 models for APG Clinical Analytics."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
from uuid6 import uuid7


def uuid7str() -> str:
	return str(uuid7())


class CohortCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	name: str
	description: str
	segment: str
	criteria: dict[str, Any] = Field(default_factory=dict)
	icd10_codes: list[str] = Field(default_factory=list)
	created_by: str

	@field_validator("segment")
	@classmethod
	def segment_not_empty(cls, v: str) -> str:
		if not v.strip():
			raise ValueError("segment must not be empty")
		return v.strip()


class CohortUpdate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	name: str | None = None
	description: str | None = None
	criteria: dict[str, Any] | None = None
	icd10_codes: list[str] | None = None
	status: str | None = None


class CohortResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	name: str
	description: str
	segment: str
	criteria: dict[str, Any] = Field(default_factory=dict)
	icd10_codes: list[str] = Field(default_factory=list)
	status: str = "draft"
	patient_count: int = 0
	created_by: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class MetricRecordCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	metric_type: str
	cohort_id: str | None = None
	value: float
	unit: str
	period: str
	period_start: datetime
	period_end: datetime
	data_source: str
	benchmark_value: float | None = None
	benchmark_type: str | None = None
	created_by: str

	@field_validator("value")
	@classmethod
	def value_finite(cls, v: float) -> float:
		import math
		if not math.isfinite(v):
			raise ValueError("metric value must be finite")
		return v


class MetricRecordResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	metric_type: str
	cohort_id: str | None = None
	value: float
	unit: str
	period: str
	period_start: datetime
	period_end: datetime
	data_source: str
	benchmark_value: float | None = None
	benchmark_type: str | None = None
	created_by: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class PredictionModelCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	name: str
	model_type: str
	target_outcome: str
	feature_set: list[str] = Field(default_factory=list)
	auc_score: float
	training_cohort_id: str
	approval_reference: str
	created_by: str

	@field_validator("auc_score")
	@classmethod
	def auc_valid(cls, v: float) -> float:
		if not (0.0 <= v <= 1.0):
			raise ValueError("auc_score must be between 0 and 1")
		return v


class PredictionModelResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	name: str
	model_type: str
	target_outcome: str
	feature_set: list[str] = Field(default_factory=list)
	auc_score: float
	training_cohort_id: str
	approval_reference: str
	status: str = "pending_deployment"
	deployed_at: datetime | None = None
	last_retrained_at: datetime | None = None
	created_by: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class QualityIndicatorCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	indicator_code: str
	indicator_name: str
	value: float
	numerator: int
	denominator: int
	period: str
	data_source: str
	benchmark_type: str | None = None
	benchmark_value: float | None = None
	created_by: str


class QualityIndicatorResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	indicator_code: str
	indicator_name: str
	value: float
	numerator: int
	denominator: int
	period: str
	data_source: str
	benchmark_type: str | None = None
	benchmark_value: float | None = None
	performance_status: str = "pending"
	created_by: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class CareGapCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	patient_id: str
	gap_type: str
	description: str
	severity: str
	evidence_reference: str
	icd10_codes: list[str] = Field(default_factory=list)
	created_by: str


class CareGapResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	patient_id: str
	gap_type: str
	description: str
	severity: str
	evidence_reference: str
	icd10_codes: list[str] = Field(default_factory=list)
	status: str = "open"
	resolved_at: datetime | None = None
	created_by: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class AnalyticsReportCreate(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	report_name: str
	report_type: str
	format: str
	cohort_ids: list[str] = Field(default_factory=list)
	metric_types: list[str] = Field(default_factory=list)
	period: str
	period_start: datetime
	period_end: datetime
	created_by: str


class AnalyticsReportResponse(BaseModel):
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	report_name: str
	report_type: str
	format: str
	cohort_ids: list[str] = Field(default_factory=list)
	metric_types: list[str] = Field(default_factory=list)
	period: str
	period_start: datetime
	period_end: datetime
	status: str = "pending"
	download_url: str | None = None
	created_by: str
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
