"""Domain models for APG Logging and Tracing."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
	"""Return a stable UTC timestamp for dependency-light runtime records."""
	return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class IngestionPipeline:
	"""Tenant-owned diagnostic ingestion pipeline."""

	id: str
	tenant_id: str
	name: str
	owner: str
	schema_ref: str
	event_bus_ref: str
	sampling_policy: str
	retention_policy_id: str
	status: str = "active"
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"owner": self.owner,
			"schema_ref": self.schema_ref,
			"event_bus_ref": self.event_bus_ref,
			"sampling_policy": self.sampling_policy,
			"retention_policy_id": self.retention_policy_id,
			"status": self.status,
			"created_at": self.created_at,
		}


@dataclass(frozen=True)
class LogEvent:
	"""Structured log event with redaction and trace correlation metadata."""

	id: str
	tenant_id: str
	pipeline_id: str
	service_name: str
	severity: str
	message: str
	attributes: dict[str, Any]
	trace_id: str
	span_id: str
	sensitive_log_content: bool
	redaction_applied: bool
	timestamp: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"pipeline_id": self.pipeline_id,
			"service_name": self.service_name,
			"severity": self.severity,
			"message": self.message,
			"attributes": dict(self.attributes),
			"trace_id": self.trace_id,
			"span_id": self.span_id,
			"sensitive_log_content": self.sensitive_log_content,
			"redaction_applied": self.redaction_applied,
			"timestamp": self.timestamp,
		}


@dataclass(frozen=True)
class TraceRecord:
	"""Distributed trace root with trace context and service posture."""

	id: str
	tenant_id: str
	pipeline_id: str
	trace_id: str
	root_service: str
	operation: str
	trace_context: dict[str, Any]
	sampling_policy: str
	status: str
	started_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"pipeline_id": self.pipeline_id,
			"trace_id": self.trace_id,
			"root_service": self.root_service,
			"operation": self.operation,
			"trace_context": dict(self.trace_context),
			"sampling_policy": self.sampling_policy,
			"status": self.status,
			"started_at": self.started_at,
		}


@dataclass(frozen=True)
class SpanRecord:
	"""Span attached to a tenant trace."""

	id: str
	tenant_id: str
	trace_id: str
	span_id: str
	parent_span_id: str
	service_name: str
	operation: str
	duration_ms: float
	status: str
	attributes: dict[str, Any] = field(default_factory=dict)
	timestamp: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"trace_id": self.trace_id,
			"span_id": self.span_id,
			"parent_span_id": self.parent_span_id,
			"service_name": self.service_name,
			"operation": self.operation,
			"duration_ms": self.duration_ms,
			"status": self.status,
			"attributes": dict(self.attributes),
			"timestamp": self.timestamp,
		}


@dataclass(frozen=True)
class DiagnosticQuery:
	"""Audited log or trace query request."""

	id: str
	tenant_id: str
	query_text: str
	requested_by: str
	query_window_hours: int
	query_review_recorded: bool
	result_count: int
	status: str
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"query_text": self.query_text,
			"requested_by": self.requested_by,
			"query_window_hours": self.query_window_hours,
			"query_review_recorded": self.query_review_recorded,
			"result_count": self.result_count,
			"status": self.status,
			"created_at": self.created_at,
		}


@dataclass(frozen=True)
class DiagnosticExport:
	"""Approved diagnostic export bundle."""

	id: str
	tenant_id: str
	export_type: str
	requested_by: str
	approval_ref: str
	item_ids: tuple[str, ...]
	status: str
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"export_type": self.export_type,
			"requested_by": self.requested_by,
			"approval_ref": self.approval_ref,
			"item_ids": list(self.item_ids),
			"status": self.status,
			"created_at": self.created_at,
		}


@dataclass(frozen=True)
class RetentionPolicy:
	"""Retention and privacy policy for diagnostic data."""

	id: str
	tenant_id: str
	name: str
	log_retention_days: int
	span_retention_days: int
	redaction_required: bool
	export_approval_required: bool
	status: str = "active"

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"log_retention_days": self.log_retention_days,
			"span_retention_days": self.span_retention_days,
			"redaction_required": self.redaction_required,
			"export_approval_required": self.export_approval_required,
			"status": self.status,
		}


@dataclass(frozen=True)
class LogtAuditEvent:
	"""Governance event emitted by logging and tracing operations."""

	id: str
	tenant_id: str
	subject_id: str
	event_type: str
	actor: str
	decision: str
	reasons: tuple[str, ...] = ()
	metadata: dict[str, Any] = field(default_factory=dict)
	created_at: str = field(default_factory=utc_now_iso)

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
			"created_at": self.created_at,
		}


LogtRecord = LogEvent


@dataclass(frozen=True)
class LogtAgent:
	"""Registered AI observability agent for diagnostic operations."""

	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str
	registered: bool = True
	contribution_disclosed: bool = True
	status: str = "active"
	created_at: str = field(default_factory=utc_now_iso)

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
			"created_at": self.created_at,
		}
