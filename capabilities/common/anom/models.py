"""Domain models for APG Anomaly Detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MonitoringSource:
	"""Tenant-scoped stream, metric, or event source for anomaly detection."""

	id: str
	tenant_id: str
	name: str
	kind: str = "metric"
	owner: str = "operations"
	labels: dict[str, Any] = field(default_factory=dict)
	status: str = "active"
	decision: str = "allow"
	matched_rules: tuple[str, ...] = ()
	review_reasons: tuple[str, ...] = ()

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"kind": self.kind,
			"owner": self.owner,
			"labels": dict(self.labels),
			"status": self.status,
			"decision": self.decision,
			"matched_rules": list(self.matched_rules),
			"review_reasons": list(self.review_reasons),
		}


@dataclass(frozen=True)
class BaselineProfile:
	"""Statistical baseline used by deterministic anomaly scoring."""

	id: str
	tenant_id: str
	source_id: str
	metric: str
	mean: float
	stdev: float
	history_points: int
	sensitivity: str = "medium"
	status: str = "active"
	decision: str = "allow"
	matched_rules: tuple[str, ...] = ()
	review_reasons: tuple[str, ...] = ()

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"source_id": self.source_id,
			"metric": self.metric,
			"mean": self.mean,
			"stdev": self.stdev,
			"history_points": self.history_points,
			"sensitivity": self.sensitivity,
			"status": self.status,
			"decision": self.decision,
			"matched_rules": list(self.matched_rules),
			"review_reasons": list(self.review_reasons),
		}


@dataclass(frozen=True)
class Observation:
	"""Single observed metric value or event score."""

	id: str
	tenant_id: str
	source_id: str
	metric: str
	value: float
	timestamp: str | None = None
	context: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"source_id": self.source_id,
			"metric": self.metric,
			"value": self.value,
			"timestamp": self.timestamp,
			"context": dict(self.context),
		}


@dataclass(frozen=True)
class AnomalySignal:
	"""Scored anomaly signal generated from an observation and baseline."""

	id: str
	tenant_id: str
	source_id: str
	baseline_id: str
	observation_id: str
	score: float
	severity: str
	status: str = "open"
	root_cause_hints: tuple[str, ...] = ()
	decision: str = "allow"
	matched_rules: tuple[str, ...] = ()
	review_reasons: tuple[str, ...] = ()

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"source_id": self.source_id,
			"baseline_id": self.baseline_id,
			"observation_id": self.observation_id,
			"score": self.score,
			"severity": self.severity,
			"status": self.status,
			"root_cause_hints": list(self.root_cause_hints),
			"decision": self.decision,
			"matched_rules": list(self.matched_rules),
			"review_reasons": list(self.review_reasons),
		}


@dataclass(frozen=True)
class Investigation:
	"""Governed investigation assigned to an anomaly signal."""

	id: str
	tenant_id: str
	signal_id: str
	owner: str
	status: str = "open"
	resolution: str | None = None
	closed_by: str | None = None
	resolution_evidence: tuple[str, ...] = ()

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"signal_id": self.signal_id,
			"owner": self.owner,
			"status": self.status,
			"resolution": self.resolution,
			"closed_by": self.closed_by,
			"resolution_evidence": list(self.resolution_evidence),
		}


@dataclass(frozen=True)
class DetectionFeedback:
	"""Feedback event used to tune false positives and investigation quality."""

	id: str
	tenant_id: str
	signal_id: str
	label: str
	reviewer: str
	notes: str = ""
	status: str = "recorded"
	decision: str = "allow"
	matched_rules: tuple[str, ...] = ()
	review_reasons: tuple[str, ...] = ()

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"signal_id": self.signal_id,
			"label": self.label,
			"reviewer": self.reviewer,
			"notes": self.notes,
			"status": self.status,
			"decision": self.decision,
			"matched_rules": list(self.matched_rules),
			"review_reasons": list(self.review_reasons),
		}


@dataclass(frozen=True)
class AnomalyAgentRecord:
	"""Provider-neutral AI agent assigned to anomaly lifecycle governance."""

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
		}


@dataclass(frozen=True)
class AnomLifecycleBatchRecord:
	"""Bytewax lifecycle-batch validation evidence."""

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
		}


@dataclass(frozen=True)
class AnomalyAuditEvent:
	"""Tenant-scoped evidence event for anomaly lifecycle changes."""

	id: str
	tenant_id: str
	event_type: str
	subject_id: str
	message: str
	evidence: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"event_type": self.event_type,
			"subject_id": self.subject_id,
			"message": self.message,
			"evidence": dict(self.evidence),
		}
