"""Dependency-light Shutdown and Lifecycle Control runtime primitives."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any


CRITICALITIES = {"low", "normal", "high", "critical"}
TARGET_TYPES = {"service", "worker", "database", "queue", "tenant_app", "integration"}
TARGET_STATES = {
	"running",
	"draining",
	"quiesced",
	"snapshot_ready",
	"stopping",
	"stopped",
	"recovered",
	"failed",
}
PLAN_STATUSES = {"draft", "approved", "scheduled", "executing", "completed", "blocked"}
OPERATION_STATUSES = {"pending", "draining", "quiesced", "completed", "blocked"}


def utc_now() -> str:
	return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_id(prefix: str, *parts: object) -> str:
	seed = "|".join(str(part).strip().lower() for part in parts if str(part).strip())
	digest = sha256(seed.encode("utf-8")).hexdigest()[:16]
	return f"{prefix}_{digest}"


def normalize_target_type(target_type: str) -> str:
	value = str(target_type or "").strip().lower()
	if value in {"app", "application"}:
		value = "tenant_app"
	if value not in TARGET_TYPES:
		raise ValueError(f"unsupported_shutdown_target_type:{target_type}")
	return value


def normalize_criticality(criticality: str) -> str:
	value = str(criticality or "normal").strip().lower()
	if value in {"medium", "standard"}:
		value = "normal"
	if value not in CRITICALITIES:
		raise ValueError(f"unsupported_shutdown_criticality:{criticality}")
	return value


def lifecycle_required_actions(rule_result: dict[str, Any]) -> list[str]:
	return [
		str(action["required_action"])
		for action in rule_result.get("actions", [])
		if action.get("required_action")
	]


def serialize(record: object) -> dict[str, Any]:
	return asdict(record)


@dataclass(slots=True)
class ShutdownTargetRecord:
	id: str
	tenant_id: str
	target_id: str
	target_type: str
	owner: str
	environment: str
	criticality: str
	state: str = "running"
	dependencies: list[str] = field(default_factory=list)
	drain_timeout_seconds: int = 300
	health_gate_ref: str | None = None
	created_at: str = field(default_factory=utc_now)
	updated_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return serialize(self)


@dataclass(slots=True)
class ShutdownPlanRecord:
	id: str
	tenant_id: str
	name: str
	owner: str
	target_ids: list[str]
	reason: str
	status: str
	rollback_plan_ref: str
	restart_sequence: list[str]
	approved_by: str | None = None
	scheduled_for: str | None = None
	maintenance_window_ref: str | None = None
	created_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return serialize(self)


@dataclass(slots=True)
class DrainOperationRecord:
	id: str
	tenant_id: str
	plan_id: str
	target_id: str
	active_sessions: int
	queue_depth: int
	status: str
	started_at: str = field(default_factory=utc_now)
	completed_at: str | None = None

	def to_dict(self) -> dict[str, Any]:
		return serialize(self)


@dataclass(slots=True)
class BackupSnapshotRecord:
	id: str
	tenant_id: str
	plan_id: str
	target_id: str
	evidence_ref: str
	restore_test_ref: str
	verified: bool
	created_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return serialize(self)


@dataclass(slots=True)
class ShutdownExecutionRecord:
	id: str
	tenant_id: str
	plan_id: str
	target_id: str
	actor: str
	status: str
	force_shutdown: bool = False
	required_actions: list[str] = field(default_factory=list)
	matched_rules: list[str] = field(default_factory=list)
	created_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return serialize(self)


@dataclass(slots=True)
class RecoveryRecord:
	id: str
	tenant_id: str
	plan_id: str
	target_id: str
	actor: str
	evidence_ref: str
	post_shutdown_health_check_ref: str
	status: str
	created_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return serialize(self)


@dataclass(slots=True)
class LifecycleAuditEventRecord:
	id: str
	tenant_id: str
	event_type: str
	subject_id: str
	message: str
	actor: str
	severity: str = "low"
	created_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return serialize(self)


__all__ = [
	"CRITICALITIES",
	"OPERATION_STATUSES",
	"PLAN_STATUSES",
	"TARGET_STATES",
	"TARGET_TYPES",
	"BackupSnapshotRecord",
	"DrainOperationRecord",
	"LifecycleAuditEventRecord",
	"RecoveryRecord",
	"ShutdownExecutionRecord",
	"ShutdownPlanRecord",
	"ShutdownTargetRecord",
	"lifecycle_required_actions",
	"normalize_criticality",
	"normalize_target_type",
	"stable_id",
	"utc_now",
]
