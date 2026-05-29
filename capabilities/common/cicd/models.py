"""Domain models for APG Continuous Integration and Delivery."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PipelineDefinition:
	"""Tenant-scoped pipeline definition with source, worker, and gate policy."""

	id: str
	tenant_id: str
	name: str
	owner: str
	source_ref: str
	worker_pool: str
	stages: tuple[str, ...]
	secret_scope: str
	cache_policy: str
	quality_gate: str
	parallel_job_count: int = 1
	status: str = "active"
	review_status: str = "approved"

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"owner": self.owner,
			"source_ref": self.source_ref,
			"worker_pool": self.worker_pool,
			"stages": list(self.stages),
			"secret_scope": self.secret_scope,
			"cache_policy": self.cache_policy,
			"quality_gate": self.quality_gate,
			"parallel_job_count": self.parallel_job_count,
			"status": self.status,
			"review_status": self.review_status,
		}


@dataclass(frozen=True)
class BuildRun:
	"""Build execution state with trace, log, and status evidence."""

	id: str
	tenant_id: str
	pipeline_id: str
	commit_ref: str
	triggered_by: str
	trace_id: str
	status: str
	log_trace_captured: bool
	secret_scope: str
	cache_policy: str

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"pipeline_id": self.pipeline_id,
			"commit_ref": self.commit_ref,
			"triggered_by": self.triggered_by,
			"trace_id": self.trace_id,
			"status": self.status,
			"log_trace_captured": self.log_trace_captured,
			"secret_scope": self.secret_scope,
			"cache_policy": self.cache_policy,
		}


@dataclass(frozen=True)
class BuildArtifact:
	"""Artifact output from a build with digest and signature state."""

	id: str
	tenant_id: str
	build_id: str
	name: str
	version: str
	digest: str
	signed: bool
	status: str = "available"

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"build_id": self.build_id,
			"name": self.name,
			"version": self.version,
			"digest": self.digest,
			"signed": self.signed,
			"status": self.status,
		}


@dataclass(frozen=True)
class QualityGateResult:
	"""Quality, security, signature, and approval evidence for an artifact."""

	id: str
	tenant_id: str
	artifact_id: str
	status: str
	tests_passed: bool
	security_scan_passed: bool
	artifact_signed: bool
	approval_recorded: bool
	findings: tuple[str, ...] = ()

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"artifact_id": self.artifact_id,
			"status": self.status,
			"tests_passed": self.tests_passed,
			"security_scan_passed": self.security_scan_passed,
			"artifact_signed": self.artifact_signed,
			"approval_recorded": self.approval_recorded,
			"findings": list(self.findings),
		}


@dataclass(frozen=True)
class PromotionRun:
	"""Artifact promotion through an environment with gate and approval state."""

	id: str
	tenant_id: str
	artifact_id: str
	source_environment: str
	target_environment: str
	requested_by: str
	status: str
	quality_gate_id: str
	approval_recorded: bool

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"artifact_id": self.artifact_id,
			"source_environment": self.source_environment,
			"target_environment": self.target_environment,
			"requested_by": self.requested_by,
			"status": self.status,
			"quality_gate_id": self.quality_gate_id,
			"approval_recorded": self.approval_recorded,
		}


@dataclass(frozen=True)
class CicdAuditEvent:
	"""Governance event emitted by CI/CD operations."""

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
