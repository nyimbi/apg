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

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"kind": self.kind,
			"owner": self.owner,
			"labels": dict(self.labels),
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

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"signal_id": self.signal_id,
			"owner": self.owner,
			"status": self.status,
			"resolution": self.resolution,
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

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"signal_id": self.signal_id,
			"label": self.label,
			"reviewer": self.reviewer,
			"notes": self.notes,
		}
