"""Domain models for APG Predictive Analytics."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> datetime:
	return datetime.now(timezone.utc)


def isoformat(value: datetime) -> str:
	return value.astimezone(timezone.utc).isoformat()


@dataclass
class PredictiveModel:
	id: str
	tenant_id: str
	name: str
	owner: str
	algorithm: str
	target: str
	environment: str = "development"
	approved: bool = False
	explainability_attached: bool = False
	training_history_points: int = 0
	feature_names: tuple[str, ...] = ()
	status: str = "registered"
	metadata: dict[str, Any] = field(default_factory=dict)
	created_at: datetime = field(default_factory=utc_now)
	updated_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"owner": self.owner,
			"algorithm": self.algorithm,
			"target": self.target,
			"environment": self.environment,
			"approved": self.approved,
			"explainability_attached": self.explainability_attached,
			"training_history_points": self.training_history_points,
			"feature_names": list(self.feature_names),
			"status": self.status,
			"metadata": dict(self.metadata),
			"created_at": isoformat(self.created_at),
			"updated_at": isoformat(self.updated_at),
		}


@dataclass
class FeatureSet:
	id: str
	tenant_id: str
	name: str
	owner: str
	feature_names: tuple[str, ...]
	lineage_refs: tuple[str, ...]
	source_system: str
	status: str = "active"
	created_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"owner": self.owner,
			"feature_names": list(self.feature_names),
			"lineage_refs": list(self.lineage_refs),
			"source_system": self.source_system,
			"status": self.status,
			"created_at": isoformat(self.created_at),
		}


@dataclass
class ForecastRun:
	id: str
	tenant_id: str
	model_id: str
	series_name: str
	horizon_days: int
	history_points: int
	confidence_interval: bool
	forecast_values: tuple[float, ...]
	review_recorded: bool = False
	status: str = "forecasted"
	created_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"model_id": self.model_id,
			"series_name": self.series_name,
			"horizon_days": self.horizon_days,
			"history_points": self.history_points,
			"confidence_interval": self.confidence_interval,
			"forecast_values": list(self.forecast_values),
			"review_recorded": self.review_recorded,
			"status": self.status,
			"created_at": isoformat(self.created_at),
		}


@dataclass
class ScoreRun:
	id: str
	tenant_id: str
	model_id: str
	feature_set_id: str
	entity_id: str
	environment: str
	impact: str
	score: float
	explanation_ref: str
	status: str = "scored"
	created_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"model_id": self.model_id,
			"feature_set_id": self.feature_set_id,
			"entity_id": self.entity_id,
			"environment": self.environment,
			"impact": self.impact,
			"score": self.score,
			"explanation_ref": self.explanation_ref,
			"status": self.status,
			"created_at": isoformat(self.created_at),
		}


@dataclass
class ScenarioSimulation:
	id: str
	tenant_id: str
	model_id: str
	name: str
	baseline_score: float
	scenario_score: float
	delta: float
	assumptions: tuple[str, ...]
	status: str = "simulated"
	created_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"model_id": self.model_id,
			"name": self.name,
			"baseline_score": self.baseline_score,
			"scenario_score": self.scenario_score,
			"delta": self.delta,
			"assumptions": list(self.assumptions),
			"status": self.status,
			"created_at": isoformat(self.created_at),
		}


@dataclass
class DriftReport:
	id: str
	tenant_id: str
	model_id: str
	metric_name: str
	drift_score: float
	threshold: float
	status: str
	review_recorded: bool = False
	created_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"model_id": self.model_id,
			"metric_name": self.metric_name,
			"drift_score": self.drift_score,
			"threshold": self.threshold,
			"status": self.status,
			"review_recorded": self.review_recorded,
			"created_at": isoformat(self.created_at),
		}


@dataclass
class PredictionAgentRecord:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str
	owner: str
	purpose: str
	contribution_disclosed: bool
	human_approval_required: bool
	status: str = "active"
	created_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"agent_id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"runtime": self.runtime,
			"role": self.role,
			"scope": self.scope,
			"owner": self.owner,
			"purpose": self.purpose,
			"contribution_disclosed": self.contribution_disclosed,
			"human_approval_required": self.human_approval_required,
			"status": self.status,
			"created_at": isoformat(self.created_at),
		}


@dataclass
class PredLifecycleBatchRecord:
	id: str
	tenant_id: str
	event_stream: str
	mutation_count: int
	operation: str
	accepted: bool
	decision: str
	matched_rules: tuple[str, ...] = ()
	required_processor: str = "bytewax"
	status: str = "accepted"
	created_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"batch_id": self.id,
			"tenant_id": self.tenant_id,
			"event_stream": self.event_stream,
			"mutation_count": self.mutation_count,
			"operation": self.operation,
			"accepted": self.accepted,
			"decision": self.decision,
			"matched_rules": list(self.matched_rules),
			"required_processor": self.required_processor,
			"status": self.status,
			"created_at": isoformat(self.created_at),
		}


@dataclass
class PredAuditEvent:
	id: str
	tenant_id: str
	event_type: str
	subject_id: str
	actor: str
	decision: str
	reasons: tuple[str, ...] = ()
	created_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"event_type": self.event_type,
			"subject_id": self.subject_id,
			"actor": self.actor,
			"decision": self.decision,
			"reasons": list(self.reasons),
			"created_at": isoformat(self.created_at),
		}


PredRecord = PredictiveModel
