"""Deployment-domain models for the APG DEPL capability."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> datetime:
	return datetime.now(timezone.utc)


def isoformat(value: datetime) -> str:
	return value.astimezone(timezone.utc).isoformat()


@dataclass
class DeploymentEnvironment:
	id: str
	tenant_id: str
	name: str
	tier: str
	owner: str
	policy: str
	approvers: tuple[str, ...]
	status: str = "active"
	created_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"tier": self.tier,
			"owner": self.owner,
			"policy": self.policy,
			"approvers": list(self.approvers),
			"status": self.status,
			"created_at": isoformat(self.created_at),
		}


@dataclass
class ReleaseManifest:
	id: str
	tenant_id: str
	version: str
	owner: str
	manifest: dict[str, Any]
	artifact_digest: str
	artifact_signature: str
	change_ticket: str
	created_by: str
	status: str = "ready"
	created_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"version": self.version,
			"owner": self.owner,
			"manifest": dict(self.manifest),
			"artifact_digest": self.artifact_digest,
			"artifact_signature": self.artifact_signature,
			"change_ticket": self.change_ticket,
			"created_by": self.created_by,
			"status": self.status,
			"created_at": isoformat(self.created_at),
		}


@dataclass
class RollbackPlan:
	id: str
	tenant_id: str
	release_id: str
	owner: str
	steps: tuple[str, ...]
	tested: bool
	created_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"release_id": self.release_id,
			"owner": self.owner,
			"steps": list(self.steps),
			"tested": self.tested,
			"created_at": isoformat(self.created_at),
		}


@dataclass
class HealthGate:
	id: str
	tenant_id: str
	release_id: str
	checks: dict[str, bool]
	report_reference: str
	log_trace_link: str
	status: str
	recorded_by: str
	recorded_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"release_id": self.release_id,
			"checks": dict(self.checks),
			"report_reference": self.report_reference,
			"log_trace_link": self.log_trace_link,
			"status": self.status,
			"recorded_by": self.recorded_by,
			"recorded_at": isoformat(self.recorded_at),
		}


@dataclass
class DeploymentPlan:
	id: str
	tenant_id: str
	release_id: str
	environment_id: str
	strategy: str
	requested_by: str
	approval_recorded: bool
	rollback_plan_id: str
	health_gate_id: str
	change_ticket: str
	canary_percent: int = 0
	status: str = "approved"
	review_status: str = "approved"
	created_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"release_id": self.release_id,
			"environment_id": self.environment_id,
			"strategy": self.strategy,
			"requested_by": self.requested_by,
			"approval_recorded": self.approval_recorded,
			"rollback_plan_id": self.rollback_plan_id,
			"health_gate_id": self.health_gate_id,
			"change_ticket": self.change_ticket,
			"canary_percent": self.canary_percent,
			"status": self.status,
			"review_status": self.review_status,
			"created_at": isoformat(self.created_at),
		}


@dataclass
class DeploymentRun:
	id: str
	tenant_id: str
	plan_id: str
	release_id: str
	environment_id: str
	strategy: str
	actor: str
	status: str
	fingerprint: str
	log_trace_link: str
	health_report_reference: str
	started_at: datetime = field(default_factory=utc_now)
	completed_at: datetime | None = None

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"plan_id": self.plan_id,
			"release_id": self.release_id,
			"environment_id": self.environment_id,
			"strategy": self.strategy,
			"actor": self.actor,
			"status": self.status,
			"fingerprint": self.fingerprint,
			"log_trace_link": self.log_trace_link,
			"health_report_reference": self.health_report_reference,
			"started_at": isoformat(self.started_at),
			"completed_at": isoformat(self.completed_at) if self.completed_at else None,
		}


@dataclass
class RollbackEvent:
	id: str
	tenant_id: str
	run_id: str
	plan_id: str
	rollback_plan_id: str
	reason: str
	actor: str
	status: str = "rolled_back"
	created_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"run_id": self.run_id,
			"plan_id": self.plan_id,
			"rollback_plan_id": self.rollback_plan_id,
			"reason": self.reason,
			"actor": self.actor,
			"status": self.status,
			"created_at": isoformat(self.created_at),
		}


@dataclass
class DeploymentAuditEvent:
	id: str
	tenant_id: str
	event_type: str
	subject_id: str
	actor: str
	decision: str
	reasons: tuple[str, ...] = ()
	payload_hash: str = ""
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
			"payload_hash": self.payload_hash,
			"created_at": isoformat(self.created_at),
		}
