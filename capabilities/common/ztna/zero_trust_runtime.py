"""Domain runtime records for the Zero Trust Network Access capability."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha1
from typing import Any


IDENTITY_STATUSES = {"pending", "verified", "suspended"}
DEVICE_STATUSES = {"unknown", "trusted", "quarantined", "retired"}
RESOURCE_STATUSES = {"active", "policy_required", "disabled"}
ACCESS_STATUSES = {"approved", "review_required", "denied", "active", "revoked", "closed"}


def utc_now() -> str:
	"""Return a stable UTC timestamp string for dependency-light records."""
	return datetime.now(timezone.utc).isoformat()


def stable_id(prefix: str, *parts: object) -> str:
	"""Build a deterministic short identifier from business-key parts."""
	key = "|".join(str(part) for part in parts)
	digest = sha1(key.encode("utf-8")).hexdigest()[:12]
	return f"{prefix}_{digest}"


def bounded_score(value: float) -> float:
	"""Clamp a trust or risk score into the 0..1 range."""
	return round(max(0.0, min(1.0, float(value))), 4)


@dataclass
class ZeroTrustIdentityRecord:
	"""Tenant identity context used for zero-trust decisions."""

	id: str
	tenant_id: str
	subject_id: str
	display_name: str
	verified: bool = False
	privileged: bool = False
	mfa_completed: bool = False
	status: str = "pending"
	federated_provider: str | None = None
	created_at: str = field(default_factory=utc_now)
	verified_at: str | None = None
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"subject_id": self.subject_id,
			"display_name": self.display_name,
			"verified": self.verified,
			"privileged": self.privileged,
			"mfa_completed": self.mfa_completed,
			"status": self.status,
			"federated_provider": self.federated_provider,
			"created_at": self.created_at,
			"verified_at": self.verified_at,
			"metadata": dict(self.metadata),
		}


@dataclass
class ZeroTrustDeviceRecord:
	"""Device posture and attestation evidence for access decisions."""

	id: str
	tenant_id: str
	identity_id: str
	name: str
	trust_score: float
	posture_present: bool = True
	managed: bool = False
	attested: bool = False
	compliant: bool = True
	status: str = "unknown"
	last_posture_at: str = field(default_factory=utc_now)
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"identity_id": self.identity_id,
			"name": self.name,
			"trust_score": self.trust_score,
			"posture_present": self.posture_present,
			"managed": self.managed,
			"attested": self.attested,
			"compliant": self.compliant,
			"status": self.status,
			"last_posture_at": self.last_posture_at,
			"metadata": dict(self.metadata),
		}


@dataclass
class ZeroTrustResourceRecord:
	"""Protected resource with least-privilege access policy state."""

	id: str
	tenant_id: str
	name: str
	access_level: str = "standard"
	sensitive: bool = False
	policy_attached: bool = False
	policy_id: str | None = None
	network_segment: str = "default"
	status: str = "policy_required"
	created_at: str = field(default_factory=utc_now)
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"access_level": self.access_level,
			"sensitive": self.sensitive,
			"policy_attached": self.policy_attached,
			"policy_id": self.policy_id,
			"network_segment": self.network_segment,
			"status": self.status,
			"created_at": self.created_at,
			"metadata": dict(self.metadata),
		}


@dataclass
class ZeroTrustAccessRequestRecord:
	"""Evaluated access request with deterministic rule evidence."""

	id: str
	tenant_id: str
	identity_id: str
	device_id: str
	resource_id: str
	requested_by: str
	access_level: str
	risk_score: float
	status: str = "approved"
	required_actions: list[str] = field(default_factory=list)
	matched_rules: list[str] = field(default_factory=list)
	decision_reasons: list[str] = field(default_factory=list)
	reviewed_by: str | None = None
	reviewed_at: str | None = None
	created_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"identity_id": self.identity_id,
			"device_id": self.device_id,
			"resource_id": self.resource_id,
			"requested_by": self.requested_by,
			"access_level": self.access_level,
			"risk_score": self.risk_score,
			"status": self.status,
			"required_actions": list(self.required_actions),
			"matched_rules": list(self.matched_rules),
			"decision_reasons": list(self.decision_reasons),
			"reviewed_by": self.reviewed_by,
			"reviewed_at": self.reviewed_at,
			"created_at": self.created_at,
		}


@dataclass
class ZeroTrustSessionRecord:
	"""Active or closed zero-trust resource session."""

	id: str
	tenant_id: str
	access_request_id: str
	identity_id: str
	device_id: str
	resource_id: str
	status: str = "active"
	risk_score: float = 0.0
	started_at: str = field(default_factory=utc_now)
	ended_at: str | None = None
	reauth_required: bool = False

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"access_request_id": self.access_request_id,
			"identity_id": self.identity_id,
			"device_id": self.device_id,
			"resource_id": self.resource_id,
			"status": self.status,
			"risk_score": self.risk_score,
			"started_at": self.started_at,
			"ended_at": self.ended_at,
			"reauth_required": self.reauth_required,
		}


@dataclass
class ZeroTrustAuditEventRecord:
	"""Append-only audit event for access decisions and session changes."""

	id: str
	tenant_id: str
	action: str
	subject_id: str
	actor_id: str
	details: dict[str, Any] = field(default_factory=dict)
	created_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"action": self.action,
			"subject_id": self.subject_id,
			"actor_id": self.actor_id,
			"details": dict(self.details),
			"created_at": self.created_at,
		}


@dataclass
class ZeroTrustAgentRecord:
	"""First-class AI agent assigned to a governed zero-trust scope."""

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
	created_at: str = field(default_factory=utc_now)

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
			"created_at": self.created_at,
		}


@dataclass
class ZtnaLifecycleBatchRecord:
	"""Bytewax lifecycle batch evidence for zero-trust mutations."""

	id: str
	tenant_id: str
	event_stream: str
	mutation_count: int
	operation: str
	accepted: bool
	decision: str
	matched_rules: list[str] = field(default_factory=list)
	status: str = "accepted"
	created_at: str = field(default_factory=utc_now)

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
			"created_at": self.created_at,
		}
