"""Domain models for APG Platform Foundation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> datetime:
	return datetime.now(timezone.utc)


def isoformat(value: datetime) -> str:
	return value.astimezone(timezone.utc).isoformat()


@dataclass
class FoundationService:
	id: str
	tenant_id: str
	name: str
	owner: str
	tier: str
	dependencies: tuple[str, ...] = ()
	readiness_score: float = 0.0
	configuration_baseline_present: bool = False
	health_status: str = "unknown"
	monitoring_enabled: bool = False
	rollback_plan_ref: str = ""
	change_window_ref: str = ""
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
			"tier": self.tier,
			"dependencies": list(self.dependencies),
			"readiness_score": self.readiness_score,
			"configuration_baseline_present": self.configuration_baseline_present,
			"health_status": self.health_status,
			"monitoring_enabled": self.monitoring_enabled,
			"rollback_plan_ref": self.rollback_plan_ref,
			"change_window_ref": self.change_window_ref,
			"status": self.status,
			"metadata": dict(self.metadata),
			"created_at": isoformat(self.created_at),
			"updated_at": isoformat(self.updated_at),
		}


@dataclass
class FoundationDependency:
	id: str
	tenant_id: str
	source_service_id: str
	target_service_id: str
	health_status: str
	required: bool = True
	evidence_ref: str = ""
	created_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"source_service_id": self.source_service_id,
			"target_service_id": self.target_service_id,
			"health_status": self.health_status,
			"required": self.required,
			"evidence_ref": self.evidence_ref,
			"created_at": isoformat(self.created_at),
		}


@dataclass
class FoundationBaseline:
	id: str
	tenant_id: str
	service_id: str
	baseline_type: str
	evidence_ref: str
	approved_by: str
	status: str = "approved"
	created_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"service_id": self.service_id,
			"baseline_type": self.baseline_type,
			"evidence_ref": self.evidence_ref,
			"approved_by": self.approved_by,
			"status": self.status,
			"created_at": isoformat(self.created_at),
		}


@dataclass
class ReadinessAssessment:
	id: str
	tenant_id: str
	service_id: str
	score: float
	status: str
	dependencies_healthy: bool
	baselines_complete: bool
	monitoring_ready: bool
	rollback_ready: bool
	change_window_ready: bool
	issues: tuple[str, ...] = ()
	created_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"service_id": self.service_id,
			"score": self.score,
			"status": self.status,
			"dependencies_healthy": self.dependencies_healthy,
			"baselines_complete": self.baselines_complete,
			"monitoring_ready": self.monitoring_ready,
			"rollback_ready": self.rollback_ready,
			"change_window_ready": self.change_window_ready,
			"issues": list(self.issues),
			"created_at": isoformat(self.created_at),
		}


@dataclass
class PlatformChange:
	id: str
	tenant_id: str
	service_id: str
	title: str
	owner: str
	affected_capability_count: int
	dependencies_healthy: bool
	approval_recorded: bool = False
	broad_review_recorded: bool = False
	security_review_recorded: bool = False
	change_window_ref: str = ""
	rollback_plan_ref: str = ""
	status: str = "proposed"
	created_at: datetime = field(default_factory=utc_now)
	approved_at: datetime | None = None

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"service_id": self.service_id,
			"title": self.title,
			"owner": self.owner,
			"affected_capability_count": self.affected_capability_count,
			"dependencies_healthy": self.dependencies_healthy,
			"approval_recorded": self.approval_recorded,
			"broad_review_recorded": self.broad_review_recorded,
			"security_review_recorded": self.security_review_recorded,
			"change_window_ref": self.change_window_ref,
			"rollback_plan_ref": self.rollback_plan_ref,
			"status": self.status,
			"created_at": isoformat(self.created_at),
			"approved_at": isoformat(self.approved_at) if self.approved_at else None,
		}


@dataclass
class PlfdAuditEvent:
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


PlfdRecord = FoundationService


@dataclass
class PlfdAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str
	registered: bool = True
	contribution_disclosed: bool = True
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
			"status": self.status,
			"created_at": isoformat(self.created_at),
		}
