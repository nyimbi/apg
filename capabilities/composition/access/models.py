"""Dependency-light runtime records for composition access control."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any


def utc_now() -> str:
	return datetime.now(timezone.utc).isoformat()


def stable_id(prefix: str, *parts: str) -> str:
	seed = "::".join(str(part) for part in parts)
	return f"{prefix}_{sha256(seed.encode('utf-8')).hexdigest()[:16]}"


@dataclass
class AccessProviderRecord:
	id: str
	tenant_id: str
	name: str
	provider_type: str
	owner_id: str
	status: str = "draft"
	external: bool = True
	metadata_validated: bool = False
	secret_reference: str | None = None
	test_evidence: str | None = None
	created_at: str = field(default_factory=utc_now)
	updated_at: str = field(default_factory=utc_now)
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class AccessResourceRecord:
	id: str
	tenant_id: str
	resource_key: str
	display_name: str
	owner_id: str
	scopes: list[str]
	capability_id: str
	sensitive: bool = False
	status: str = "active"
	created_at: str = field(default_factory=utc_now)
	updated_at: str = field(default_factory=utc_now)
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class AccessPolicyRecord:
	id: str
	tenant_id: str
	name: str
	resource_id: str
	owner_id: str
	effect: str
	conditions: dict[str, Any]
	risk_level: str = "standard"
	status: str = "draft"
	simulation_evidence: str | None = None
	reviewed_by: str | None = None
	created_at: str = field(default_factory=utc_now)
	updated_at: str = field(default_factory=utc_now)
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class AccessGrantRecord:
	id: str
	tenant_id: str
	subject_id: str
	resource_id: str
	scopes: list[str]
	requested_by: str
	justification: str
	privileged: bool = False
	approved_by: str | None = None
	expires_at: str | None = None
	status: str = "active"
	created_at: str = field(default_factory=utc_now)
	updated_at: str = field(default_factory=utc_now)
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class AccessSessionRecord:
	id: str
	tenant_id: str
	subject_id: str
	provider_id: str
	risk_score: int
	status: str
	step_up_completed: bool = False
	evaluated_at: str = field(default_factory=utc_now)
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class AccessDecisionRecord:
	id: str
	tenant_id: str
	subject_id: str
	resource_id: str
	action: str
	decision: str
	reason: str
	policy_ids: list[str]
	event_stream: str = "bytewax"
	created_at: str = field(default_factory=utc_now)
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class AccessAgentRecord:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	instructions: str
	status: str = "active"
	created_at: str = field(default_factory=utc_now)
	updated_at: str = field(default_factory=utc_now)
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass
class AccessAuditEventRecord:
	id: str
	tenant_id: str
	event_type: str
	entity_id: str
	actor_id: str
	created_at: str = field(default_factory=utc_now)
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)
