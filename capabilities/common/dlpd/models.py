"""Domain models for the APG Data Loss Prevention capability."""

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
class DlpPolicy:
	"""Tenant DLP policy binding channels, classifiers, and response action."""

	id: str
	tenant_id: str
	name: str
	owner: str
	channels: list[str]
	classifiers: list[str]
	default_action: str = "quarantine"
	egress_policy_attached: bool = True
	large_export_review_required: bool = True
	status: str = "active"
	created_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"owner": self.owner,
			"channels": list(self.channels),
			"classifiers": list(self.classifiers),
			"default_action": self.default_action,
			"egress_policy_attached": self.egress_policy_attached,
			"large_export_review_required": self.large_export_review_required,
			"status": self.status,
			"created_at": isoformat(self.created_at),
		}


@dataclass
class DataClassifier:
	"""Classifier metadata used by the DLP inspection runtime."""

	id: str
	tenant_id: str
	name: str
	classifier_type: str
	sensitivity_label: str
	pattern_keys: list[str]
	reviewed_by: str | None = None
	confidence_threshold: float = 0.82
	status: str = "active"
	created_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"classifier_type": self.classifier_type,
			"sensitivity_label": self.sensitivity_label,
			"pattern_keys": list(self.pattern_keys),
			"reviewed_by": self.reviewed_by,
			"confidence_threshold": self.confidence_threshold,
			"status": self.status,
			"created_at": isoformat(self.created_at),
		}


@dataclass
class EgressInspection:
	"""Inspection decision for one outbound content movement."""

	id: str
	tenant_id: str
	policy_id: str
	channel: str
	subject_id: str
	destination: str
	content_hash: str
	classification_label: str | None
	classifier_hits: list[dict[str, Any]]
	severity: str
	record_count: int
	decision: str
	blocked: bool = False
	quarantined: bool = False
	review_required: bool = False
	reviewed_by: str | None = None
	incident_id: str | None = None
	quarantine_id: str | None = None
	created_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"policy_id": self.policy_id,
			"channel": self.channel,
			"subject_id": self.subject_id,
			"destination": self.destination,
			"content_hash": self.content_hash,
			"classification_label": self.classification_label,
			"classifier_hits": [dict(hit) for hit in self.classifier_hits],
			"severity": self.severity,
			"record_count": self.record_count,
			"decision": self.decision,
			"blocked": self.blocked,
			"quarantined": self.quarantined,
			"review_required": self.review_required,
			"reviewed_by": self.reviewed_by,
			"incident_id": self.incident_id,
			"quarantine_id": self.quarantine_id,
			"created_at": isoformat(self.created_at),
		}


@dataclass
class QuarantineItem:
	"""Encrypted quarantine vault entry for blocked sensitive data."""

	id: str
	tenant_id: str
	inspection_id: str
	content_hash: str
	reason: str
	encrypted: bool
	legal_hold: bool = False
	status: str = "sealed"
	created_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"inspection_id": self.inspection_id,
			"content_hash": self.content_hash,
			"reason": self.reason,
			"encrypted": self.encrypted,
			"legal_hold": self.legal_hold,
			"status": self.status,
			"created_at": isoformat(self.created_at),
		}


@dataclass
class DlpIncident:
	"""DLP incident opened from risky egress activity."""

	id: str
	tenant_id: str
	inspection_id: str
	severity: str
	owner: str
	required_action: str
	notifications_sent: bool
	status: str = "open"
	created_at: datetime = field(default_factory=utc_now)
	resolved_at: datetime | None = None
	resolution: str | None = None

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"inspection_id": self.inspection_id,
			"severity": self.severity,
			"owner": self.owner,
			"required_action": self.required_action,
			"notifications_sent": self.notifications_sent,
			"status": self.status,
			"created_at": isoformat(self.created_at),
			"resolved_at": isoformat(self.resolved_at),
			"resolution": self.resolution,
		}


@dataclass
class DlpAuditEvent:
	"""Append-only DLP audit event."""

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


@dataclass
class DlpAgentRecord:
	"""First-class AI agent assigned to a governed DLP scope."""

	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str
	owner: str
	purpose: str
	contribution_disclosed: bool = True
	human_approval_required: bool = False
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
			"owner": self.owner,
			"purpose": self.purpose,
			"contribution_disclosed": self.contribution_disclosed,
			"human_approval_required": self.human_approval_required,
			"status": self.status,
			"created_at": isoformat(self.created_at),
		}


@dataclass
class DlpdLifecycleBatchRecord:
	"""Bytewax lifecycle batch evidence for DLPD mutations."""

	id: str
	tenant_id: str
	event_stream: str
	mutation_count: int
	operation: str
	accepted: bool
	decision: str
	matched_rules: tuple[str, ...] = ()
	status: str = "accepted"
	created_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"event_stream": self.event_stream,
			"mutation_count": self.mutation_count,
			"operation": self.operation,
			"accepted": self.accepted,
			"decision": self.decision,
			"matched_rules": list(self.matched_rules),
			"status": self.status,
			"created_at": isoformat(self.created_at),
		}
