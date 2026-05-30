"""Domain models for APG Quantum Computing."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> datetime:
	return datetime.now(timezone.utc)


def isoformat(value: datetime) -> str:
	return value.astimezone(timezone.utc).isoformat()


@dataclass
class QuantumBackend:
	id: str
	tenant_id: str
	name: str
	provider: str
	backend_type: str
	qubit_count: int
	approved: bool
	credentials_ref: str | None = None
	quota_policy_attached: bool = False
	simulator_fallback: bool = True
	status: str = "registered"
	metadata: dict[str, Any] = field(default_factory=dict)
	created_at: datetime = field(default_factory=utc_now)
	updated_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"provider": self.provider,
			"backend_type": self.backend_type,
			"qubit_count": self.qubit_count,
			"approved": self.approved,
			"credentials_ref": self.credentials_ref,
			"quota_policy_attached": self.quota_policy_attached,
			"simulator_fallback": self.simulator_fallback,
			"status": self.status,
			"metadata": dict(self.metadata),
			"created_at": isoformat(self.created_at),
			"updated_at": isoformat(self.updated_at),
		}


@dataclass
class QuantumCircuit:
	id: str
	tenant_id: str
	name: str
	owner: str
	version: str
	qubits_required: int
	gates: tuple[str, ...]
	sensitive_input_present: bool = False
	encryption_applied: bool = False
	experiment_metadata: dict[str, Any] = field(default_factory=dict)
	status: str = "draft"
	created_at: datetime = field(default_factory=utc_now)
	updated_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"owner": self.owner,
			"version": self.version,
			"qubits_required": self.qubits_required,
			"gates": list(self.gates),
			"sensitive_input_present": self.sensitive_input_present,
			"encryption_applied": self.encryption_applied,
			"experiment_metadata": dict(self.experiment_metadata),
			"status": self.status,
			"created_at": isoformat(self.created_at),
			"updated_at": isoformat(self.updated_at),
		}


@dataclass
class QuantumQuotaPolicy:
	id: str
	tenant_id: str
	backend_id: str
	max_shots_per_job: int
	max_jobs_per_day: int
	cost_limit: float
	retry_policy: str
	created_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"backend_id": self.backend_id,
			"max_shots_per_job": self.max_shots_per_job,
			"max_jobs_per_day": self.max_jobs_per_day,
			"cost_limit": self.cost_limit,
			"retry_policy": self.retry_policy,
			"created_at": isoformat(self.created_at),
		}


@dataclass
class QuantumJob:
	id: str
	tenant_id: str
	backend_id: str
	circuit_id: str
	submitted_by: str
	shot_count: int
	estimated_cost: float
	job_review_recorded: bool = False
	retry_policy_attached: bool = True
	status: str = "submitted"
	created_at: datetime = field(default_factory=utc_now)
	updated_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"backend_id": self.backend_id,
			"circuit_id": self.circuit_id,
			"submitted_by": self.submitted_by,
			"shot_count": self.shot_count,
			"estimated_cost": self.estimated_cost,
			"job_review_recorded": self.job_review_recorded,
			"retry_policy_attached": self.retry_policy_attached,
			"status": self.status,
			"created_at": isoformat(self.created_at),
			"updated_at": isoformat(self.updated_at),
		}


@dataclass
class QuantumResult:
	id: str
	tenant_id: str
	job_id: str
	measurement_counts: dict[str, int]
	confidence: float
	analysis_summary: str
	retained_until_days: int = 90
	created_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"job_id": self.job_id,
			"measurement_counts": dict(self.measurement_counts),
			"confidence": self.confidence,
			"analysis_summary": self.analysis_summary,
			"retained_until_days": self.retained_until_days,
			"created_at": isoformat(self.created_at),
		}


@dataclass
class QuantumExperiment:
	id: str
	tenant_id: str
	name: str
	circuit_id: str
	job_ids: tuple[str, ...]
	hypothesis: str
	post_quantum_review_recorded: bool = False
	status: str = "active"
	created_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"circuit_id": self.circuit_id,
			"job_ids": list(self.job_ids),
			"hypothesis": self.hypothesis,
			"post_quantum_review_recorded": self.post_quantum_review_recorded,
			"status": self.status,
			"created_at": isoformat(self.created_at),
		}


@dataclass
class QuanAuditEvent:
	id: str
	tenant_id: str
	event_type: str
	subject_id: str
	actor: str
	decision: str
	reasons: tuple[str, ...] = ()
	metadata: dict[str, Any] = field(default_factory=dict)
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
			"metadata": dict(self.metadata),
			"created_at": isoformat(self.created_at),
		}


QuanRecord = QuantumBackend


@dataclass
class QuanAgent:
	"""Registered AI quantum governance agent."""

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
