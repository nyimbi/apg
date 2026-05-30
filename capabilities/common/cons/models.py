"""Privacy-domain models for the APG CONS capability."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> datetime:
	return datetime.now(timezone.utc)


def isoformat(value: datetime) -> str:
	return value.astimezone(timezone.utc).isoformat()


@dataclass
class PrivacyPurpose:
	id: str
	tenant_id: str
	name: str
	owner: str
	legal_basis: str
	retention_policy: str
	notice_id: str
	data_categories: list[str]
	active: bool = True
	created_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"owner": self.owner,
			"legal_basis": self.legal_basis,
			"retention_policy": self.retention_policy,
			"notice_id": self.notice_id,
			"data_categories": list(self.data_categories),
			"active": self.active,
			"created_at": isoformat(self.created_at),
		}


@dataclass
class PrivacyNotice:
	id: str
	tenant_id: str
	version: str
	url: str
	language: str
	purposes: list[str]
	published_by: str
	published_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"version": self.version,
			"url": self.url,
			"language": self.language,
			"purposes": list(self.purposes),
			"published_by": self.published_by,
			"published_at": isoformat(self.published_at),
		}


@dataclass
class ConsentEvent:
	id: str
	tenant_id: str
	subject_id: str
	purpose_id: str
	notice_id: str
	source: str
	captured_by: str
	status: str = "active"
	captured_at: datetime = field(default_factory=utc_now)
	withdrawn_at: datetime | None = None
	provenance_hash: str = ""

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"subject_id": self.subject_id,
			"purpose_id": self.purpose_id,
			"notice_id": self.notice_id,
			"source": self.source,
			"captured_by": self.captured_by,
			"status": self.status,
			"captured_at": isoformat(self.captured_at),
			"withdrawn_at": isoformat(self.withdrawn_at) if self.withdrawn_at else None,
			"provenance_hash": self.provenance_hash,
		}


@dataclass
class PreferenceProfile:
	id: str
	tenant_id: str
	subject_id: str
	channels: dict[str, bool]
	purposes: dict[str, bool]
	updated_by: str
	updated_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"subject_id": self.subject_id,
			"channels": dict(self.channels),
			"purposes": dict(self.purposes),
			"updated_by": self.updated_by,
			"updated_at": isoformat(self.updated_at),
		}


@dataclass
class PrivacyRequest:
	id: str
	tenant_id: str
	subject_id: str
	request_type: str
	submitted_by: str
	identity_verified: bool
	evidence_reference: str
	due_at: datetime
	status: str = "open"
	submitted_at: datetime = field(default_factory=utc_now)
	completed_at: datetime | None = None
	resolution: str = ""

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"subject_id": self.subject_id,
			"request_type": self.request_type,
			"submitted_by": self.submitted_by,
			"identity_verified": self.identity_verified,
			"evidence_reference": self.evidence_reference,
			"due_at": isoformat(self.due_at),
			"status": self.status,
			"submitted_at": isoformat(self.submitted_at),
			"completed_at": isoformat(self.completed_at) if self.completed_at else None,
			"resolution": self.resolution,
		}


@dataclass
class ProcessingDecision:
	id: str
	tenant_id: str
	subject_id: str
	purpose_id: str
	decision: str
	reason: str
	consent_id: str | None
	decided_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"subject_id": self.subject_id,
			"purpose_id": self.purpose_id,
			"decision": self.decision,
			"reason": self.reason,
			"consent_id": self.consent_id,
			"decided_at": isoformat(self.decided_at),
		}


@dataclass
class PrivacyAgent:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str
	registered: bool = True
	contribution_disclosed: bool = True
	policy_ref: str | None = None
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
			"policy_ref": self.policy_ref,
			"status": self.status,
			"created_at": isoformat(self.created_at),
		}


@dataclass
class PrivacyAuditEvent:
	id: str
	tenant_id: str
	event_type: str
	subject_id: str
	actor: str
	payload_hash: str
	created_at: datetime = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"event_type": self.event_type,
			"subject_id": self.subject_id,
			"actor": self.actor,
			"payload_hash": self.payload_hash,
			"created_at": isoformat(self.created_at),
		}
