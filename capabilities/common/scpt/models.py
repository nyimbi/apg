"""Domain models for the APG Custom Scripting Engine capability."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ScriptPackagePolicy:
	"""Allowed package, import, secret, filesystem, and network policy."""

	id: str
	tenant_id: str
	name: str
	owner: str
	allowed_packages: list[str] = field(default_factory=list)
	blocked_imports: list[str] = field(default_factory=list)
	secret_access_allowed: bool = False
	filesystem_access_allowed: bool = False
	network_policy_attached: bool = False
	approved_by: str | None = None
	created_at: datetime | None = None

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"owner": self.owner,
			"allowed_packages": list(self.allowed_packages),
			"blocked_imports": list(self.blocked_imports),
			"secret_access_allowed": self.secret_access_allowed,
			"filesystem_access_allowed": self.filesystem_access_allowed,
			"network_policy_attached": self.network_policy_attached,
			"approved_by": self.approved_by,
			"created_at": self.created_at.isoformat() if self.created_at else None,
		}


@dataclass
class ScriptSandbox:
	"""Constrained runtime envelope for script execution."""

	id: str
	tenant_id: str
	name: str
	owner: str
	max_runtime_seconds: int
	max_memory_mb: int
	network_enabled: bool = False
	network_policy_attached: bool = False
	resource_review_recorded: bool = False
	state: str = "ready"
	created_at: datetime | None = None
	updated_at: datetime | None = None

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"owner": self.owner,
			"max_runtime_seconds": self.max_runtime_seconds,
			"max_memory_mb": self.max_memory_mb,
			"network_enabled": self.network_enabled,
			"network_policy_attached": self.network_policy_attached,
			"resource_review_recorded": self.resource_review_recorded,
			"state": self.state,
			"created_at": self.created_at.isoformat() if self.created_at else None,
			"updated_at": self.updated_at.isoformat() if self.updated_at else None,
		}


@dataclass
class ScriptDefinition:
	"""Versioned tenant-owned script definition."""

	id: str
	tenant_id: str
	name: str
	language: str
	source: str
	owner: str
	version: int = 1
	state: str = "draft"
	requested_permissions: list[str] = field(default_factory=list)
	dangerous_permissions: list[str] = field(default_factory=list)
	approval_recorded: bool = False
	package_policy_id: str | None = None
	sandbox_id: str | None = None
	workflow_bindings: list[str] = field(default_factory=list)
	tags: list[str] = field(default_factory=list)
	created_at: datetime | None = None
	updated_at: datetime | None = None
	published_at: datetime | None = None

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"language": self.language,
			"source": self.source,
			"owner": self.owner,
			"version": self.version,
			"state": self.state,
			"requested_permissions": list(self.requested_permissions),
			"dangerous_permissions": list(self.dangerous_permissions),
			"approval_recorded": self.approval_recorded,
			"package_policy_id": self.package_policy_id,
			"sandbox_id": self.sandbox_id,
			"workflow_bindings": list(self.workflow_bindings),
			"tags": list(self.tags),
			"created_at": self.created_at.isoformat() if self.created_at else None,
			"updated_at": self.updated_at.isoformat() if self.updated_at else None,
			"published_at": self.published_at.isoformat() if self.published_at else None,
		}


@dataclass
class ScriptApproval:
	"""Approval record for publication or dangerous permissions."""

	id: str
	tenant_id: str
	script_id: str
	reason: str
	approver: str
	status: str = "approved"
	created_at: datetime | None = None
	decided_at: datetime | None = None

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"script_id": self.script_id,
			"reason": self.reason,
			"approver": self.approver,
			"status": self.status,
			"created_at": self.created_at.isoformat() if self.created_at else None,
			"decided_at": self.decided_at.isoformat() if self.decided_at else None,
		}


@dataclass
class ScriptExecution:
	"""One script execution request and deterministic local result metadata."""

	id: str
	tenant_id: str
	script_id: str
	sandbox_id: str
	requested_by: str
	status: str = "queued"
	input_payload: dict[str, Any] = field(default_factory=dict)
	output: dict[str, Any] = field(default_factory=dict)
	error: str | None = None
	runtime_seconds: float = 0.0
	memory_mb: int = 0
	logs: list[str] = field(default_factory=list)
	started_at: datetime | None = None
	completed_at: datetime | None = None

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"script_id": self.script_id,
			"sandbox_id": self.sandbox_id,
			"requested_by": self.requested_by,
			"status": self.status,
			"input_payload": dict(self.input_payload),
			"output": dict(self.output),
			"error": self.error,
			"runtime_seconds": self.runtime_seconds,
			"memory_mb": self.memory_mb,
			"logs": list(self.logs),
			"started_at": self.started_at.isoformat() if self.started_at else None,
			"completed_at": self.completed_at.isoformat() if self.completed_at else None,
		}


@dataclass
class ScptAuditEvent:
	"""Audit trail entry for scripting operations."""

	id: str
	tenant_id: str
	event_type: str
	subject_id: str
	message: str
	actor: str
	severity: str = "info"
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
			"created_at": self.created_at.isoformat() if self.created_at else None,
		}


ScptRecord = ScriptDefinition
