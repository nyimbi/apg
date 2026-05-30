"""Domain models for the APG Sandbox/Testing Environment capability."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class IsolationProfile:
	"""Network, data, and secret controls applied to a sandbox."""

	id: str
	tenant_id: str
	name: str
	level: str
	network_policy_required: bool = True
	secret_redaction_enabled: bool = True
	data_masking_enabled: bool = True
	outbound_network_allowed: bool = False
	network_approval_recorded: bool = False
	approved_by: str | None = None
	created_at: datetime | None = None

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"level": self.level,
			"network_policy_required": self.network_policy_required,
			"secret_redaction_enabled": self.secret_redaction_enabled,
			"data_masking_enabled": self.data_masking_enabled,
			"outbound_network_allowed": self.outbound_network_allowed,
			"network_approval_recorded": self.network_approval_recorded,
			"approved_by": self.approved_by,
			"created_at": self.created_at.isoformat() if self.created_at else None,
		}


@dataclass
class SandboxTemplate:
	"""Reusable sandbox blueprint."""

	id: str
	tenant_id: str
	name: str
	runtime: str
	owner: str
	default_ttl_hours: int
	plugin_test_policy_required: bool = True
	tags: list[str] = field(default_factory=list)
	created_at: datetime | None = None

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"runtime": self.runtime,
			"owner": self.owner,
			"default_ttl_hours": self.default_ttl_hours,
			"plugin_test_policy_required": self.plugin_test_policy_required,
			"tags": list(self.tags),
			"created_at": self.created_at.isoformat() if self.created_at else None,
		}


@dataclass
class SandboxDataset:
	"""Dataset made available to sandbox runs."""

	id: str
	tenant_id: str
	name: str
	dataset_type: str
	owner: str
	lineage: str
	retention_days: int
	production_review_recorded: bool = False
	masked: bool = True
	created_at: datetime | None = None

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"dataset_type": self.dataset_type,
			"owner": self.owner,
			"lineage": self.lineage,
			"retention_days": self.retention_days,
			"production_review_recorded": self.production_review_recorded,
			"masked": self.masked,
			"created_at": self.created_at.isoformat() if self.created_at else None,
		}


@dataclass
class SandboxEnvironment:
	"""Tenant-scoped sandbox environment."""

	id: str
	tenant_id: str
	name: str
	template_id: str
	isolation_profile_id: str
	owner: str
	ttl_hours: int
	dataset_ids: list[str] = field(default_factory=list)
	state: str = "draft"
	lifecycle_review_recorded: bool = False
	secret_access_requested: bool = False
	outbound_network_requested: bool = False
	risk_score: int = 0
	created_at: datetime | None = None
	updated_at: datetime | None = None

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"template_id": self.template_id,
			"isolation_profile_id": self.isolation_profile_id,
			"owner": self.owner,
			"ttl_hours": self.ttl_hours,
			"dataset_ids": list(self.dataset_ids),
			"state": self.state,
			"lifecycle_review_recorded": self.lifecycle_review_recorded,
			"secret_access_requested": self.secret_access_requested,
			"outbound_network_requested": self.outbound_network_requested,
			"risk_score": self.risk_score,
			"created_at": self.created_at.isoformat() if self.created_at else None,
			"updated_at": self.updated_at.isoformat() if self.updated_at else None,
		}


@dataclass
class SandboxRun:
	"""A test or experiment run executed inside a sandbox."""

	id: str
	tenant_id: str
	sandbox_id: str
	run_type: str
	requested_by: str
	status: str = "queued"
	tests_requested: int = 0
	tests_passed: int = 0
	tests_failed: int = 0
	tests_blocked: int = 0
	started_at: datetime | None = None
	completed_at: datetime | None = None
	logs: list[str] = field(default_factory=list)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"sandbox_id": self.sandbox_id,
			"run_type": self.run_type,
			"requested_by": self.requested_by,
			"status": self.status,
			"tests_requested": self.tests_requested,
			"tests_passed": self.tests_passed,
			"tests_failed": self.tests_failed,
			"tests_blocked": self.tests_blocked,
			"started_at": self.started_at.isoformat() if self.started_at else None,
			"completed_at": self.completed_at.isoformat() if self.completed_at else None,
			"logs": list(self.logs),
		}


@dataclass
class SboxAuditEvent:
	"""Audit trail entry for sandbox actions."""

	id: str
	tenant_id: str
	event_type: str
	subject_id: str
	message: str
	actor: str
	severity: str = "info"
	metadata: dict[str, Any] = field(default_factory=dict)
	created_at: datetime | None = None

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"event_type": self.event_type,
			"subject_id": self.subject_id,
			"message": self.message,
			"actor": self.actor,
			"severity": self.severity,
			"metadata": dict(self.metadata),
			"created_at": self.created_at.isoformat() if self.created_at else None,
		}


SboxRecord = SandboxEnvironment


@dataclass
class SboxAgent:
	"""Registered AI sandbox governance agent."""

	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str
	registered: bool = True
	contribution_disclosed: bool = True
	status: str = "active"
	created_at: datetime | None = None

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
			"created_at": self.created_at.isoformat() if self.created_at else None,
		}
