"""Domain models for the APG EDGE capability."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
	"""Return a stable UTC timestamp for runtime records."""
	return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class EdgeNode:
	"""Tenant-scoped edge node with attestation, policy, health, and transport state."""

	id: str
	tenant_id: str
	name: str
	owner: str
	node_type: str
	location: dict[str, Any]
	location_policy: str
	attested: bool
	health_status: str
	secure_transport: bool
	capacity: dict[str, float] = field(default_factory=dict)
	current_load: dict[str, float] = field(default_factory=dict)
	capabilities: list[str] = field(default_factory=list)
	fleet_id: str | None = None
	status: str = "active"
	created_at: str = field(default_factory=utc_now)
	updated_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"owner": self.owner,
			"node_type": self.node_type,
			"location": dict(self.location),
			"location_policy": self.location_policy,
			"attested": self.attested,
			"health_status": self.health_status,
			"secure_transport": self.secure_transport,
			"capacity": dict(self.capacity),
			"current_load": dict(self.current_load),
			"capabilities": list(self.capabilities),
			"fleet_id": self.fleet_id,
			"status": self.status,
			"created_at": self.created_at,
			"updated_at": self.updated_at,
		}


@dataclass
class EdgeFleet:
	"""Named group of edge nodes managed under one tenant policy."""

	id: str
	tenant_id: str
	name: str
	owner: str
	policy_version: str
	node_ids: list[str] = field(default_factory=list)
	created_at: str = field(default_factory=utc_now)
	updated_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"owner": self.owner,
			"policy_version": self.policy_version,
			"node_ids": list(self.node_ids),
			"created_at": self.created_at,
			"updated_at": self.updated_at,
		}


@dataclass
class EdgeWorkload:
	"""Signed workload artifact and deployment policy for edge execution."""

	id: str
	tenant_id: str
	name: str
	version: str
	owner: str
	artifact_digest: str
	artifact_signed: bool
	deployment_policy: str
	resource_quota: dict[str, float]
	offline_mode_enabled: bool = True
	status: str = "ready"
	created_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"version": self.version,
			"owner": self.owner,
			"artifact_digest": self.artifact_digest,
			"artifact_signed": self.artifact_signed,
			"deployment_policy": self.deployment_policy,
			"resource_quota": dict(self.resource_quota),
			"offline_mode_enabled": self.offline_mode_enabled,
			"status": self.status,
			"created_at": self.created_at,
		}


@dataclass
class EdgeDeployment:
	"""Workload placement on an attested edge node."""

	id: str
	tenant_id: str
	workload_id: str
	node_id: str
	deployed_by: str
	runtime_mode: str
	resource_reservation: dict[str, float]
	status: str = "deployed"
	deployed_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"workload_id": self.workload_id,
			"node_id": self.node_id,
			"deployed_by": self.deployed_by,
			"runtime_mode": self.runtime_mode,
			"resource_reservation": dict(self.resource_reservation),
			"status": self.status,
			"deployed_at": self.deployed_at,
		}


@dataclass
class EdgeSyncSession:
	"""State synchronization session between edge and core runtime."""

	id: str
	tenant_id: str
	node_id: str
	workload_id: str
	conflict_policy: str
	cache_policy: str
	offline_hours: int
	secure_transport: bool
	event_count: int
	conflicts: list[str] = field(default_factory=list)
	status: str = "synced"
	review_required: bool = False
	reviewed_by: str | None = None
	created_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"node_id": self.node_id,
			"workload_id": self.workload_id,
			"conflict_policy": self.conflict_policy,
			"cache_policy": self.cache_policy,
			"offline_hours": self.offline_hours,
			"secure_transport": self.secure_transport,
			"event_count": self.event_count,
			"conflicts": list(self.conflicts),
			"status": self.status,
			"review_required": self.review_required,
			"reviewed_by": self.reviewed_by,
			"created_at": self.created_at,
		}


@dataclass
class EdgeAuditEvent:
	"""Immutable audit event for edge runtime operations."""

	id: str
	tenant_id: str
	action: str
	resource_id: str
	actor: str
	digest: str
	metadata: dict[str, Any] = field(default_factory=dict)
	created_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"action": self.action,
			"resource_id": self.resource_id,
			"actor": self.actor,
			"digest": self.digest,
			"metadata": dict(self.metadata),
			"created_at": self.created_at,
		}
