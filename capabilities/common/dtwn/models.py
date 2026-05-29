"""Domain models for the APG Digital Twin Framework capability."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> datetime:
	"""Return a timezone-aware UTC timestamp."""
	return datetime.now(timezone.utc)


def isoformat(value: datetime | None) -> str | None:
	return value.isoformat() if value is not None else None


@dataclass
class DigitalTwin:
	"""Tenant-scoped virtual representation of a physical or logical asset."""

	id: str
	tenant_id: str
	asset_id: str
	name: str
	owner: str
	twin_type: str
	location: dict[str, Any] = field(default_factory=dict)
	state: dict[str, Any] = field(default_factory=dict)
	topology_refs: list[str] = field(default_factory=list)
	state_version: str = "v000001"
	status: str = "active"
	created_at: datetime = field(default_factory=utc_now)
	updated_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"asset_id": self.asset_id,
			"name": self.name,
			"owner": self.owner,
			"twin_type": self.twin_type,
			"location": dict(self.location),
			"state": dict(self.state),
			"topology_refs": list(self.topology_refs),
			"state_version": self.state_version,
			"status": self.status,
			"created_at": isoformat(self.created_at),
			"updated_at": isoformat(self.updated_at),
		}


@dataclass
class SimulationModel:
	"""Approved model version used by digital-twin simulations."""

	id: str
	tenant_id: str
	name: str
	version: str
	owner: str
	model_type: str
	calibration_evidence: str
	approved_by: str | None = None
	confidence: float = 0.75
	status: str = "approved"
	created_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"version": self.version,
			"owner": self.owner,
			"model_type": self.model_type,
			"calibration_evidence": self.calibration_evidence,
			"approved_by": self.approved_by,
			"confidence": self.confidence,
			"status": self.status,
			"created_at": isoformat(self.created_at),
		}


@dataclass
class TelemetrySample:
	"""Authenticated telemetry fused into twin state."""

	id: str
	tenant_id: str
	twin_id: str
	source_id: str
	source_type: str
	authenticated: bool
	measurements: dict[str, Any]
	geospatial_context: dict[str, Any] = field(default_factory=dict)
	vision_signals: dict[str, Any] = field(default_factory=dict)
	ingested_at: datetime = field(default_factory=utc_now)
	state_version: str = ""

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"twin_id": self.twin_id,
			"source_id": self.source_id,
			"source_type": self.source_type,
			"authenticated": self.authenticated,
			"measurements": dict(self.measurements),
			"geospatial_context": dict(self.geospatial_context),
			"vision_signals": dict(self.vision_signals),
			"ingested_at": isoformat(self.ingested_at),
			"state_version": self.state_version,
		}


@dataclass
class TopologyLink:
	"""Relationship between two twins in the tenant topology graph."""

	id: str
	tenant_id: str
	source_twin_id: str
	target_twin_id: str
	relationship: str
	metadata: dict[str, Any] = field(default_factory=dict)
	created_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"source_twin_id": self.source_twin_id,
			"target_twin_id": self.target_twin_id,
			"relationship": self.relationship,
			"metadata": dict(self.metadata),
			"created_at": isoformat(self.created_at),
		}


@dataclass
class SimulationRun:
	"""Recorded simulation execution for a twin and model version."""

	id: str
	tenant_id: str
	twin_id: str
	model_id: str
	scenario: str
	environment: str
	approved_by: str | None
	status: str
	outputs: dict[str, Any]
	started_at: datetime = field(default_factory=utc_now)
	completed_at: datetime | None = None

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"twin_id": self.twin_id,
			"model_id": self.model_id,
			"scenario": self.scenario,
			"environment": self.environment,
			"approved_by": self.approved_by,
			"status": self.status,
			"outputs": dict(self.outputs),
			"started_at": isoformat(self.started_at),
			"completed_at": isoformat(self.completed_at),
		}


@dataclass
class TwinPrediction:
	"""Prediction produced from a twin simulation or telemetry snapshot."""

	id: str
	tenant_id: str
	twin_id: str
	model_id: str
	risk_score: float
	confidence: float
	horizon: str
	recommendation: str
	review_required: bool = False
	reviewed_by: str | None = None
	status: str = "active"
	created_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"twin_id": self.twin_id,
			"model_id": self.model_id,
			"risk_score": self.risk_score,
			"confidence": self.confidence,
			"horizon": self.horizon,
			"recommendation": self.recommendation,
			"review_required": self.review_required,
			"reviewed_by": self.reviewed_by,
			"status": self.status,
			"created_at": isoformat(self.created_at),
		}


@dataclass
class TwinAuditEvent:
	"""Append-only digital-twin audit event."""

	id: str
	tenant_id: str
	action: str
	resource_id: str
	actor: str
	digest: str
	created_at: datetime = field(default_factory=utc_now)
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"action": self.action,
			"resource_id": self.resource_id,
			"actor": self.actor,
			"digest": self.digest,
			"created_at": isoformat(self.created_at),
			"metadata": dict(self.metadata),
		}
