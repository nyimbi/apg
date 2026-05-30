"""Dependency-light legacy tenant migration runtime primitives."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any


LEGACY_STATES = {"active", "stale", "mapped", "migration_ready", "migrated", "deprecated", "blocked"}
MAPPING_STATES = {"draft", "validated", "invalid"}
MIGRATION_STATES = {"planned", "approved", "executing", "completed", "blocked"}
BOUNDARY_STATES = {"pending", "validated", "failed"}


def utc_now() -> str:
	return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_id(prefix: str, *parts: object) -> str:
	seed = "|".join(str(part).strip().lower() for part in parts if str(part).strip())
	digest = sha256(seed.encode("utf-8")).hexdigest()[:16]
	return f"{prefix}_{digest}"


def tenant_required_actions(rule_result: dict[str, Any]) -> list[str]:
	return [
		str(action["required_action"])
		for action in rule_result.get("actions", [])
		if action.get("required_action")
	]


def serialize(record: object) -> dict[str, Any]:
	return asdict(record)


@dataclass(slots=True)
class LegacyTenantRecord:
	id: str
	tenant_id: str
	legacy_tenant_id: str
	source_system: str
	owner: str
	compatibility_scope: str
	status: str
	days_since_activity: int = 0
	required_actions: list[str] = field(default_factory=list)
	created_at: str = field(default_factory=utc_now)
	updated_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return serialize(self)


@dataclass(slots=True)
class TenantMappingRecord:
	id: str
	tenant_id: str
	legacy_tenant_id: str
	apg_tenant_id: str
	validated_by: str
	status: str
	validation_ref: str
	created_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return serialize(self)


@dataclass(slots=True)
class AccessBoundaryRecord:
	id: str
	tenant_id: str
	legacy_tenant_id: str
	auth_boundary_ref: str
	role_mapping_ref: str
	isolation_validation_ref: str
	privileged_review_ref: str
	status: str
	actor: str
	created_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return serialize(self)


@dataclass(slots=True)
class MigrationPlanRecord:
	id: str
	tenant_id: str
	legacy_tenant_id: str
	mapping_id: str
	owner: str
	approval_ref: str
	rollback_plan_ref: str
	post_migration_validation_ref: str
	status: str
	created_at: str = field(default_factory=utc_now)
	completed_at: str | None = None

	def to_dict(self) -> dict[str, Any]:
		return serialize(self)


@dataclass(slots=True)
class DeprecationPlanRecord:
	id: str
	tenant_id: str
	legacy_tenant_id: str
	owner: str
	deprecation_ref: str
	target_date: str
	status: str = "planned"
	created_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return serialize(self)


@dataclass(slots=True)
class TenantAuditEventRecord:
	id: str
	tenant_id: str
	event_type: str
	subject_id: str
	message: str
	actor: str
	severity: str = "low"
	metadata: dict[str, Any] = field(default_factory=dict)
	created_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return serialize(self)


@dataclass(slots=True)
class TensAgentRecord:
	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str
	owner: str
	status: str = "active"
	human_approval_required: bool = True
	created_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return serialize(self)


__all__ = [
	"BOUNDARY_STATES",
	"LEGACY_STATES",
	"MAPPING_STATES",
	"MIGRATION_STATES",
	"AccessBoundaryRecord",
	"DeprecationPlanRecord",
	"LegacyTenantRecord",
	"MigrationPlanRecord",
	"TensAgentRecord",
	"TenantAuditEventRecord",
	"TenantMappingRecord",
	"stable_id",
	"tenant_required_actions",
	"utc_now",
]
