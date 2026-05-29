"""Domain models for APG Backup and Restore."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BackupPlan:
	"""Tenant-owned backup plan with schedule, sources, RPO, and retention."""

	id: str
	tenant_id: str
	name: str
	owner: str
	schedule: str
	sources: tuple[str, ...]
	retention_days: int
	rpo_minutes: int
	status: str = "active"
	legal_hold: bool = False

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"owner": self.owner,
			"schedule": self.schedule,
			"sources": list(self.sources),
			"retention_days": self.retention_days,
			"rpo_minutes": self.rpo_minutes,
			"status": self.status,
			"legal_hold": self.legal_hold,
		}


@dataclass(frozen=True)
class BackupSnapshot:
	"""Encrypted backup snapshot with lineage and integrity evidence."""

	id: str
	tenant_id: str
	plan_id: str
	source_id: str
	snapshot_hash: str
	size_bytes: int
	encrypted: bool = True
	integrity_status: str = "passed"
	lineage: tuple[str, ...] = ()
	region: str = "primary"
	status: str = "available"

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"plan_id": self.plan_id,
			"source_id": self.source_id,
			"snapshot_hash": self.snapshot_hash,
			"size_bytes": self.size_bytes,
			"encrypted": self.encrypted,
			"integrity_status": self.integrity_status,
			"lineage": list(self.lineage),
			"region": self.region,
			"status": self.status,
		}


@dataclass(frozen=True)
class RestoreRun:
	"""Point-in-time restore execution and approval evidence."""

	id: str
	tenant_id: str
	snapshot_id: str
	target_environment: str
	requested_by: str
	status: str
	integrity_check_passed: bool
	approval_recorded: bool = False
	point_in_time: str | None = None
	review_status: str = "approved"
	rto_minutes: int = 0

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"snapshot_id": self.snapshot_id,
			"target_environment": self.target_environment,
			"requested_by": self.requested_by,
			"status": self.status,
			"integrity_check_passed": self.integrity_check_passed,
			"approval_recorded": self.approval_recorded,
			"point_in_time": self.point_in_time,
			"review_status": self.review_status,
			"rto_minutes": self.rto_minutes,
		}


@dataclass(frozen=True)
class ContinuityReport:
	"""RPO/RTO and restore-test readiness evidence for a plan."""

	id: str
	tenant_id: str
	plan_id: str
	rpo_minutes: int
	rto_minutes: int
	restore_test_status: str
	days_since_restore_test: int
	review_status: str = "current"
	findings: tuple[str, ...] = ()

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"plan_id": self.plan_id,
			"rpo_minutes": self.rpo_minutes,
			"rto_minutes": self.rto_minutes,
			"restore_test_status": self.restore_test_status,
			"days_since_restore_test": self.days_since_restore_test,
			"review_status": self.review_status,
			"findings": list(self.findings),
		}


@dataclass(frozen=True)
class BackupAuditEvent:
	"""Governance event emitted by backup and restore operations."""

	id: str
	tenant_id: str
	subject_id: str
	event_type: str
	actor: str
	decision: str
	reasons: tuple[str, ...] = ()
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"subject_id": self.subject_id,
			"event_type": self.event_type,
			"actor": self.actor,
			"decision": self.decision,
			"reasons": list(self.reasons),
			"metadata": dict(self.metadata),
		}
