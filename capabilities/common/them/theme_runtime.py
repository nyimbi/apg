"""Dependency-light UI/UX theme and brand governance runtime primitives."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any


THEME_STATUSES = {"draft", "preview_ready", "approved", "published", "review_required", "blocked"}
ASSET_STATUSES = {"pending_license", "approved", "blocked"}
TOKEN_GROUPS = {"color", "typography", "spacing", "density", "component"}


def utc_now() -> str:
	return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_id(prefix: str, *parts: object) -> str:
	seed = "|".join(str(part).strip().lower() for part in parts if str(part).strip())
	digest = sha256(seed.encode("utf-8")).hexdigest()[:16]
	return f"{prefix}_{digest}"


def normalize_token_group(group: str) -> str:
	value = str(group or "component").strip().lower()
	if value not in TOKEN_GROUPS:
		raise ValueError(f"unsupported_theme_token_group:{group}")
	return value


def theme_required_actions(rule_result: dict[str, Any]) -> list[str]:
	return [
		str(action["required_action"])
		for action in rule_result.get("actions", [])
		if action.get("required_action")
	]


def serialize(record: object) -> dict[str, Any]:
	return asdict(record)


@dataclass(slots=True)
class ThemeRecord:
	id: str
	tenant_id: str
	name: str
	owner: str
	brand_name: str
	status: str = "draft"
	guidelines_ref: str | None = None
	fallback_theme_id: str | None = None
	token_version: int = 0
	created_at: str = field(default_factory=utc_now)
	updated_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return serialize(self)


@dataclass(slots=True)
class ThemeTokenRecord:
	id: str
	tenant_id: str
	theme_id: str
	group: str
	tokens: dict[str, str]
	version: int
	contrast_validated: bool
	updated_by: str
	updated_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return serialize(self)


@dataclass(slots=True)
class BrandAssetRecord:
	id: str
	tenant_id: str
	theme_id: str
	asset_name: str
	asset_type: str
	license_ref: str
	approved_by: str
	status: str
	created_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return serialize(self)


@dataclass(slots=True)
class ThemePreviewRecord:
	id: str
	tenant_id: str
	theme_id: str
	surface: str
	viewport: str
	preview_ref: str
	contrast_passed: bool
	created_by: str
	created_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return serialize(self)


@dataclass(slots=True)
class ThemePublicationRecord:
	id: str
	tenant_id: str
	theme_id: str
	target_tenant_count: int
	approval_ref: str
	status: str
	published_by: str
	required_actions: list[str] = field(default_factory=list)
	matched_rules: list[str] = field(default_factory=list)
	published_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return serialize(self)


@dataclass(slots=True)
class ThemeAuditEventRecord:
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
class ThemAgentRecord:
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
	"ASSET_STATUSES",
	"THEME_STATUSES",
	"TOKEN_GROUPS",
	"BrandAssetRecord",
	"ThemAgentRecord",
	"ThemeAuditEventRecord",
	"ThemePreviewRecord",
	"ThemePublicationRecord",
	"ThemeRecord",
	"ThemeTokenRecord",
	"normalize_token_group",
	"stable_id",
	"theme_required_actions",
	"utc_now",
]
