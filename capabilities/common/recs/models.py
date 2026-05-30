"""Domain models for APG Recommender Systems."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> datetime:
	return datetime.now(timezone.utc)


def isoformat(value: datetime) -> str:
	return value.astimezone(timezone.utc).isoformat()


@dataclass
class RecommendationDataset:
	id: str
	tenant_id: str
	name: str
	owner: str
	source_ref: str
	schema_fields: tuple[str, ...]
	policy_ref: str
	event_count: int = 0
	status: str = "registered"
	created_at: datetime = field(default_factory=utc_now)
	updated_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"owner": self.owner,
			"source_ref": self.source_ref,
			"schema_fields": list(self.schema_fields),
			"policy_ref": self.policy_ref,
			"event_count": self.event_count,
			"status": self.status,
			"created_at": isoformat(self.created_at),
			"updated_at": isoformat(self.updated_at),
		}


@dataclass
class InteractionEvent:
	id: str
	tenant_id: str
	dataset_id: str
	profile_id: str
	item_id: str
	event_type: str
	occurred_at: str
	weight: float = 1.0
	metadata: dict[str, Any] = field(default_factory=dict)
	created_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"dataset_id": self.dataset_id,
			"profile_id": self.profile_id,
			"item_id": self.item_id,
			"event_type": self.event_type,
			"occurred_at": self.occurred_at,
			"weight": self.weight,
			"metadata": dict(self.metadata),
			"created_at": isoformat(self.created_at),
		}


@dataclass
class RecommendationCatalogItem:
	id: str
	tenant_id: str
	name: str
	item_type: str
	category: str
	features: dict[str, float] = field(default_factory=dict)
	tags: tuple[str, ...] = ()
	sensitive_attributes: tuple[str, ...] = ()
	status: str = "active"
	created_at: datetime = field(default_factory=utc_now)
	updated_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"item_type": self.item_type,
			"category": self.category,
			"features": dict(self.features),
			"tags": list(self.tags),
			"sensitive_attributes": list(self.sensitive_attributes),
			"status": self.status,
			"created_at": isoformat(self.created_at),
			"updated_at": isoformat(self.updated_at),
		}


@dataclass
class RecommendationProfile:
	id: str
	tenant_id: str
	features: dict[str, float] = field(default_factory=dict)
	segments: tuple[str, ...] = ()
	consent_recorded: bool = False
	status: str = "active"
	created_at: datetime = field(default_factory=utc_now)
	updated_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"features": dict(self.features),
			"segments": list(self.segments),
			"consent_recorded": self.consent_recorded,
			"status": self.status,
			"created_at": isoformat(self.created_at),
			"updated_at": isoformat(self.updated_at),
		}


@dataclass
class RankingPolicy:
	id: str
	tenant_id: str
	name: str
	objective: str
	owner: str = "recs"
	minimum_confidence: float = 0.65
	diversity_constraints_enabled: bool = True
	sensitive_attribute_filtering: bool = True
	max_per_category: int = 2
	status: str = "active"
	created_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"objective": self.objective,
			"owner": self.owner,
			"minimum_confidence": self.minimum_confidence,
			"diversity_constraints_enabled": self.diversity_constraints_enabled,
			"sensitive_attribute_filtering": self.sensitive_attribute_filtering,
			"max_per_category": self.max_per_category,
			"status": self.status,
			"created_at": isoformat(self.created_at),
		}


@dataclass
class RecommendationModel:
	id: str
	tenant_id: str
	name: str
	algorithm: str
	owner: str
	training_event_count: int
	feature_names: tuple[str, ...] = ()
	drift_monitoring_enabled: bool = True
	approved: bool = False
	approval_ref: str = ""
	status: str = "trained"
	drift_status: str = "stable"
	created_at: datetime = field(default_factory=utc_now)
	updated_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"algorithm": self.algorithm,
			"owner": self.owner,
			"training_event_count": self.training_event_count,
			"feature_names": list(self.feature_names),
			"drift_monitoring_enabled": self.drift_monitoring_enabled,
			"approved": self.approved,
			"approval_ref": self.approval_ref,
			"status": self.status,
			"drift_status": self.drift_status,
			"created_at": isoformat(self.created_at),
			"updated_at": isoformat(self.updated_at),
		}


@dataclass
class TrainingRun:
	id: str
	tenant_id: str
	model_id: str
	event_count: int
	metric_name: str
	metric_value: float
	status: str = "completed"
	created_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"model_id": self.model_id,
			"event_count": self.event_count,
			"metric_name": self.metric_name,
			"metric_value": self.metric_value,
			"status": self.status,
			"created_at": isoformat(self.created_at),
		}


@dataclass
class Recommendation:
	item_id: str
	rank: int
	score: float
	confidence: float
	reason: str

	def to_dict(self) -> dict[str, Any]:
		return {
			"item_id": self.item_id,
			"rank": self.rank,
			"score": self.score,
			"confidence": self.confidence,
			"reason": self.reason,
		}


@dataclass
class RecommendationSet:
	id: str
	tenant_id: str
	model_id: str
	profile_id: str
	policy_id: str
	impact_level: str
	recommendations: tuple[Recommendation, ...]
	explanation_attached: bool
	status: str = "ready"
	created_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"model_id": self.model_id,
			"profile_id": self.profile_id,
			"policy_id": self.policy_id,
			"impact_level": self.impact_level,
			"recommendations": [item.to_dict() for item in self.recommendations],
			"explanation_attached": self.explanation_attached,
			"status": self.status,
			"created_at": isoformat(self.created_at),
		}


@dataclass
class RecommendationExperiment:
	id: str
	tenant_id: str
	name: str
	model_id: str
	policy_id: str
	experiment_percent: int
	holdout_percent: int
	business_metric: str
	approved: bool
	review_recorded: bool = False
	status: str = "active"
	created_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"model_id": self.model_id,
			"policy_id": self.policy_id,
			"experiment_percent": self.experiment_percent,
			"holdout_percent": self.holdout_percent,
			"business_metric": self.business_metric,
			"approved": self.approved,
			"review_recorded": self.review_recorded,
			"status": self.status,
			"created_at": isoformat(self.created_at),
		}


@dataclass
class ModelDeployment:
	id: str
	tenant_id: str
	model_id: str
	target_runtime: str
	target_ref: str
	approval_recorded: bool
	rollback_plan_ref: str
	approval_ref: str = ""
	status: str = "deployed"
	created_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"model_id": self.model_id,
			"target_runtime": self.target_runtime,
			"target_ref": self.target_ref,
			"approval_recorded": self.approval_recorded,
			"approval_ref": self.approval_ref,
			"rollback_plan_ref": self.rollback_plan_ref,
			"status": self.status,
			"created_at": isoformat(self.created_at),
		}


@dataclass
class RecommendationFeedback:
	id: str
	tenant_id: str
	recommendation_set_id: str
	profile_id: str
	item_id: str
	event_type: str
	value: float = 1.0
	metadata: dict[str, Any] = field(default_factory=dict)
	created_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"recommendation_set_id": self.recommendation_set_id,
			"profile_id": self.profile_id,
			"item_id": self.item_id,
			"event_type": self.event_type,
			"value": self.value,
			"metadata": dict(self.metadata),
			"created_at": isoformat(self.created_at),
		}


@dataclass
class RecommenderAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str
	registered: bool = True
	contribution_disclosed: bool = False
	policy_ref: str = ""
	status: str = "active"
	created_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"runtime": self.runtime,
			"role": self.role,
			"scope": self.scope,
			"registered": self.registered,
			"contribution_disclosed": self.contribution_disclosed,
			"policy_ref": self.policy_ref,
			"status": self.status,
			"created_at": isoformat(self.created_at),
		}


@dataclass
class RecsAuditEvent:
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


RecsRecord = RecommendationModel
