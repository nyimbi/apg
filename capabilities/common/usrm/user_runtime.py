"""Dependency-light user lifecycle runtime primitives."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any


USER_STATUSES = {"active", "invited", "suspended", "review_required", "deprovisioned"}
INVITATION_STATUSES = {"sent", "accepted", "expired", "revoked"}
ROLE_STATUSES = {"active", "review_required", "revoked"}
ACCESS_REVIEW_DECISIONS = {"approve", "revoke", "modify", "defer"}


def utc_now() -> str:
	return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_id(prefix: str, *parts: object) -> str:
	seed = "|".join(str(part).strip().lower() for part in parts if str(part).strip())
	digest = sha256(seed.encode("utf-8")).hexdigest()[:16]
	return f"{prefix}_{digest}"


def normalize_user_status(status: str) -> str:
	value = str(status or "active").strip().lower()
	if value not in USER_STATUSES:
		raise ValueError(f"unsupported_user_status:{status}")
	return value


def normalize_access_review_decision(decision: str) -> str:
	value = str(decision or "defer").strip().lower()
	if value not in ACCESS_REVIEW_DECISIONS:
		raise ValueError(f"unsupported_access_review_decision:{decision}")
	return value


def user_required_actions(rule_result: dict[str, Any]) -> list[str]:
	return [
		str(action["required_action"])
		for action in rule_result.get("actions", [])
		if action.get("required_action")
	]


def serialize(record: object) -> dict[str, Any]:
	return asdict(record)


@dataclass(slots=True)
class UserRecord:
	id: str
	tenant_id: str
	identity: str
	display_name: str
	email: str
	owner: str
	status: str = "active"
	profile_validated: bool = True
	privileged_user: bool = False
	mfa_enabled: bool = False
	manager_id: str | None = None
	created_at: str = field(default_factory=utc_now)
	updated_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return serialize(self)


@dataclass(slots=True)
class UserProfileRecord:
	id: str
	tenant_id: str
	user_id: str
	attributes: dict[str, str]
	privacy_preferences: dict[str, str]
	consent_notice_ref: str
	updated_by: str
	updated_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return serialize(self)


@dataclass(slots=True)
class UserInvitationRecord:
	id: str
	tenant_id: str
	user_id: str
	channel: str
	consent_notice_ref: str
	invited_by: str
	status: str = "sent"
	created_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return serialize(self)


@dataclass(slots=True)
class RoleAssignmentRecord:
	id: str
	tenant_id: str
	user_id: str
	role: str
	scope: str
	privileged: bool
	approved_by: str
	status: str = "active"
	created_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return serialize(self)


@dataclass(slots=True)
class AccessReviewRecord:
	id: str
	tenant_id: str
	user_id: str
	reviewer: str
	decision: str
	findings: list[str] = field(default_factory=list)
	status: str = "completed"
	created_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return serialize(self)


@dataclass(slots=True)
class DeprovisionRecord:
	id: str
	tenant_id: str
	user_id: str
	actor: str
	access_revoked: bool
	evidence_ref: str
	status: str
	required_actions: list[str] = field(default_factory=list)
	matched_rules: list[str] = field(default_factory=list)
	created_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return serialize(self)


@dataclass(slots=True)
class BulkUserActionRecord:
	id: str
	tenant_id: str
	action: str
	actor: str
	user_ids: list[str]
	status: str
	required_actions: list[str] = field(default_factory=list)
	matched_rules: list[str] = field(default_factory=list)
	created_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return serialize(self)


@dataclass(slots=True)
class UserAuditEventRecord:
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
class UsrmAgentRecord:
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
	"ACCESS_REVIEW_DECISIONS",
	"INVITATION_STATUSES",
	"ROLE_STATUSES",
	"USER_STATUSES",
	"AccessReviewRecord",
	"BulkUserActionRecord",
	"DeprovisionRecord",
	"RoleAssignmentRecord",
	"UsrmAgentRecord",
	"UserAuditEventRecord",
	"UserInvitationRecord",
	"UserProfileRecord",
	"UserRecord",
	"normalize_access_review_decision",
	"normalize_user_status",
	"stable_id",
	"user_required_actions",
	"utc_now",
]
