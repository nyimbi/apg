"""In-memory models for APG Predictive Intelligence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class PredictionAuthority:
	id: str
	tenant_id: str
	authority_type: str
	scope_reference: str
	classification: str
	approver_id: str
	expires_at: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class PredictionWorkspace:
	id: str
	tenant_id: str
	workspace_type: str
	name: str
	classification: str
	authority_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class PredictionScenario:
	id: str
	tenant_id: str
	workspace_id: str
	scenario_type: str
	scenario_reference: str
	horizon: str
	owner_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class PredictionIndicator:
	id: str
	tenant_id: str
	scenario_id: str
	indicator_type: str
	indicator_reference: str
	confidence_score: float
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class PredictionModel:
	id: str
	tenant_id: str
	scenario_id: str
	model_type: str
	objective: str
	validation_reference: str
	risk_level: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class PredictionForecast:
	id: str
	tenant_id: str
	model_id: str
	forecast_type: str
	forecast_reference: str
	confidence_score: float
	analyst_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class PredictionProjection:
	id: str
	tenant_id: str
	forecast_id: str
	projection_type: str
	risk_level: str
	probability_score: float
	analyst_id: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class PredictionWarning:
	id: str
	tenant_id: str
	projection_id: str
	warning_type: str
	severity: str
	trigger_reference: str
	approval_reference: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class PredictionRecommendation:
	id: str
	tenant_id: str
	projection_id: str
	recommendation_type: str
	action_reference: str
	approval_reference: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class PredictionReview:
	id: str
	tenant_id: str
	reference_id: str
	reviewer_id: str
	status: str
	evidence_reference: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class PredictionAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
