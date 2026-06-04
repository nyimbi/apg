"""In-memory models for APG Telecom Analytics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class AnaAnalysisRun:
	id: str
	tenant_id: str
	analysis_type: str
	owner_id: str
	time_granularity: str
	start_time: str
	end_time: str
	evidence_reference: str
	status: str = "completed"

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class AnaMetric:
	id: str
	tenant_id: str
	metric_type: str
	metric_name: str
	value: float
	unit: str
	baseline_value: float
	aggregation_type: str
	recorded_at: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class AnaChurnPrediction:
	id: str
	tenant_id: str
	customer_id: str
	risk_level: str
	confidence_score: float
	model_id: str
	predicted_at: str
	features_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class AnaRevenueEvent:
	id: str
	tenant_id: str
	category: str
	amount: float
	currency: str
	period: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class AnaSegment:
	id: str
	tenant_id: str
	segment_name: str
	segment_type: str
	criteria: str
	customer_count: int
	created_by: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class AnaNetworkAnalytics:
	id: str
	tenant_id: str
	network_layer: str
	metric_name: str
	value: float
	threshold: float
	recorded_at: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class AnaAnomaly:
	id: str
	tenant_id: str
	anomaly_type: str
	confidence_score: float
	description: str
	evidence_reference: str
	detected_at: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class AnaModel:
	id: str
	tenant_id: str
	model_type: str
	model_name: str
	version: str
	validation_reference: str
	registered_by: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class AnaReport:
	id: str
	tenant_id: str
	report_format: str
	analysis_id: str
	approval_reference: str
	generated_by: str
	generated_at: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class AnaAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
