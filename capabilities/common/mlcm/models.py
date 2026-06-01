"""Domain models for the AI Model Lifecycle Management capability."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
	"""Return a stable UTC timestamp string for in-process lifecycle records."""
	return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class ModelArtifact:
	"""Tenant-scoped AI model registered for governed lifecycle operations."""

	id: str
	tenant_id: str
	name: str
	owner: str
	problem_type: str
	risk_level: str = "medium"
	status: str = "registered"
	description: str = ""
	tags: list[str] = field(default_factory=list)
	metadata: dict[str, Any] = field(default_factory=dict)
	created_at: str = field(default_factory=utc_now_iso)
	updated_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"owner": self.owner,
			"problem_type": self.problem_type,
			"risk_level": self.risk_level,
			"status": self.status,
			"description": self.description,
			"tags": list(self.tags),
			"metadata": dict(self.metadata),
			"created_at": self.created_at,
			"updated_at": self.updated_at,
		}


@dataclass
class ModelVersion:
	"""Versioned model artifact with stage, documentation, and evaluation state."""

	id: str
	tenant_id: str
	model_id: str
	version: str
	artifact_uri: str
	stage: str = "dev"
	status: str = "candidate"
	model_card: dict[str, Any] = field(default_factory=dict)
	training_data_ref: str = ""
	baseline_ref: str = ""
	evaluation_score: float | None = None
	evaluation_id: str | None = None
	promoted_at: str | None = None
	decision: str = "allow"
	matched_rules: list[str] = field(default_factory=list)
	review_reasons: list[str] = field(default_factory=list)
	audit_evidence: dict[str, Any] = field(default_factory=dict)
	created_at: str = field(default_factory=utc_now_iso)
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"model_id": self.model_id,
			"version": self.version,
			"artifact_uri": self.artifact_uri,
			"stage": self.stage,
			"status": self.status,
			"model_card": dict(self.model_card),
			"training_data_ref": self.training_data_ref,
			"baseline_ref": self.baseline_ref,
			"evaluation_score": self.evaluation_score,
			"evaluation_id": self.evaluation_id,
			"promoted_at": self.promoted_at,
			"decision": self.decision,
			"matched_rules": list(self.matched_rules),
			"review_reasons": list(self.review_reasons),
			"audit_evidence": dict(self.audit_evidence),
			"created_at": self.created_at,
			"metadata": dict(self.metadata),
		}


@dataclass
class EvaluationRun:
	"""Evaluation evidence attached to a model version."""

	id: str
	tenant_id: str
	model_id: str
	version_id: str
	score: float
	baseline_ref: str
	metrics: dict[str, float] = field(default_factory=dict)
	status: str = "passed"
	evidence_refs: list[str] = field(default_factory=list)
	evaluator: str = ""
	fairness_review_recorded: bool = False
	explainability_recorded: bool = False
	decision: str = "allow"
	matched_rules: list[str] = field(default_factory=list)
	review_reasons: list[str] = field(default_factory=list)
	audit_evidence: dict[str, Any] = field(default_factory=dict)
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"model_id": self.model_id,
			"version_id": self.version_id,
			"score": self.score,
			"baseline_ref": self.baseline_ref,
			"metrics": dict(self.metrics),
			"status": self.status,
			"evidence_refs": list(self.evidence_refs),
			"evaluator": self.evaluator,
			"fairness_review_recorded": self.fairness_review_recorded,
			"explainability_recorded": self.explainability_recorded,
			"decision": self.decision,
			"matched_rules": list(self.matched_rules),
			"review_reasons": list(self.review_reasons),
			"audit_evidence": dict(self.audit_evidence),
			"created_at": self.created_at,
		}


@dataclass
class PromotionRequest:
	"""Governed model-version promotion request."""

	id: str
	tenant_id: str
	model_id: str
	version_id: str
	source_stage: str
	target_stage: str
	requested_by: str
	approval_recorded: bool = False
	approval_ref: str = ""
	status: str = "requested"
	reasons: list[str] = field(default_factory=list)
	created_at: str = field(default_factory=utc_now_iso)
	resolved_at: str | None = None

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"model_id": self.model_id,
			"version_id": self.version_id,
			"source_stage": self.source_stage,
			"target_stage": self.target_stage,
			"requested_by": self.requested_by,
			"approval_recorded": self.approval_recorded,
			"approval_ref": self.approval_ref,
			"status": self.status,
			"reasons": list(self.reasons),
			"created_at": self.created_at,
			"resolved_at": self.resolved_at,
		}


@dataclass
class DeploymentTarget:
	"""Serving target for a model version."""

	id: str
	tenant_id: str
	name: str
	environment: str
	serving_runtime: str
	owner: str
	status: str = "active"
	metadata: dict[str, Any] = field(default_factory=dict)
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"environment": self.environment,
			"serving_runtime": self.serving_runtime,
			"owner": self.owner,
			"status": self.status,
			"metadata": dict(self.metadata),
			"created_at": self.created_at,
		}


@dataclass
class DeploymentRecord:
	"""Concrete deployment of a model version to a target."""

	id: str
	tenant_id: str
	model_id: str
	version_id: str
	target_id: str
	stage: str
	status: str = "serving"
	replicas: int = 1
	canary_percent: int = 0
	approved_by: str = ""
	created_at: str = field(default_factory=utc_now_iso)
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"model_id": self.model_id,
			"version_id": self.version_id,
			"target_id": self.target_id,
			"stage": self.stage,
			"status": self.status,
			"replicas": self.replicas,
			"canary_percent": self.canary_percent,
			"approved_by": self.approved_by,
			"created_at": self.created_at,
			"metadata": dict(self.metadata),
		}


@dataclass
class DriftSignal:
	"""Observed drift signal for a deployed or deployable model version."""

	id: str
	tenant_id: str
	model_id: str
	version_id: str
	metric: str
	score: float
	threshold: float
	drift_detected: bool
	status: str = "review_required"
	review_recorded: bool = False
	review_ref: str = ""
	observed_at: str = field(default_factory=utc_now_iso)
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"model_id": self.model_id,
			"version_id": self.version_id,
			"metric": self.metric,
			"score": self.score,
			"threshold": self.threshold,
			"drift_detected": self.drift_detected,
			"status": self.status,
			"review_recorded": self.review_recorded,
			"review_ref": self.review_ref,
			"observed_at": self.observed_at,
			"metadata": dict(self.metadata),
		}


@dataclass
class RollbackRecord:
	"""Rollback action from one deployed version to another."""

	id: str
	tenant_id: str
	model_id: str
	deployment_id: str
	from_version_id: str
	to_version_id: str
	reason: str
	status: str = "completed"
	requested_by: str = ""
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"model_id": self.model_id,
			"deployment_id": self.deployment_id,
			"from_version_id": self.from_version_id,
			"to_version_id": self.to_version_id,
			"reason": self.reason,
			"status": self.status,
			"requested_by": self.requested_by,
			"created_at": self.created_at,
		}


@dataclass
class RetirementRecord:
	"""Model retirement action with impact-review evidence."""

	id: str
	tenant_id: str
	model_id: str
	impact_review_ref: str
	retired_by: str = ""
	status: str = "completed"
	created_at: str = field(default_factory=utc_now_iso)
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"model_id": self.model_id,
			"impact_review_ref": self.impact_review_ref,
			"retired_by": self.retired_by,
			"status": self.status,
			"created_at": self.created_at,
			"metadata": dict(self.metadata),
		}


@dataclass
class ModelLifecycleAgentRecord:
	"""First-class model lifecycle agent registration."""

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
	decision: str = "allow"
	matched_rules: list[str] = field(default_factory=list)
	review_reasons: list[str] = field(default_factory=list)
	audit_evidence: dict[str, Any] = field(default_factory=dict)
	created_at: str = field(default_factory=utc_now_iso)

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
			"decision": self.decision,
			"matched_rules": list(self.matched_rules),
			"review_reasons": list(self.review_reasons),
			"audit_evidence": dict(self.audit_evidence),
			"created_at": self.created_at,
		}


@dataclass
class MlcmLifecycleBatchRecord:
	"""Bytewax lifecycle-batch validation evidence."""

	id: str
	tenant_id: str
	event_stream: str
	mutation_count: int
	operation: str
	accepted: bool
	decision: str
	matched_rules: list[str] = field(default_factory=list)
	review_reasons: list[str] = field(default_factory=list)
	audit_evidence: dict[str, Any] = field(default_factory=dict)
	required_processor: str = "bytewax"
	status: str = "accepted"
	created_at: str = field(default_factory=utc_now_iso)

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
			"review_reasons": list(self.review_reasons),
			"audit_evidence": dict(self.audit_evidence),
			"required_processor": self.required_processor,
			"status": self.status,
			"created_at": self.created_at,
		}


@dataclass
class MlcmAuditEvent:
	"""Audit event emitted by model lifecycle operations."""

	id: str
	tenant_id: str
	event_type: str
	subject_id: str
	message: str
	severity: str = "info"
	created_at: str = field(default_factory=utc_now_iso)
	metadata: dict[str, Any] = field(default_factory=dict)
	policy_decision: str = "allow"
	matched_rules: list[str] = field(default_factory=list)
	review_reasons: list[str] = field(default_factory=list)
	audit_evidence: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"event_type": self.event_type,
			"subject_id": self.subject_id,
			"message": self.message,
			"severity": self.severity,
			"created_at": self.created_at,
			"metadata": dict(self.metadata),
			"policy_decision": self.policy_decision,
			"matched_rules": list(self.matched_rules),
			"review_reasons": list(self.review_reasons),
			"audit_evidence": dict(self.audit_evidence),
		}


# Compatibility alias for older package callers that import MlcmRecord.
MlcmRecord = ModelArtifact
