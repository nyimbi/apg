"""Executable localization models for the I18N capability."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utc_now_iso() -> str:
	return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class TranslationStatus(str, Enum):
	"""Translation lifecycle state."""

	DRAFT = "draft"
	REVIEWED = "reviewed"
	PUBLISHED = "published"
	ARCHIVED = "archived"


class TranslationSource(str, Enum):
	"""Origin of localized text."""

	HUMAN = "human"
	MACHINE = "machine"
	MEMORY = "memory"


@dataclass(slots=True)
class LocaleDefinition:
	"""Tenant-scoped locale and fallback policy."""

	id: str
	tenant_id: str
	locale_code: str
	display_name: str
	owner_id: str
	fallback_locale: str
	regional_format: dict[str, str] = field(default_factory=dict)
	timezone: str = "UTC"
	enabled: bool = True
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"kind": "locale",
			"tenant_id": self.tenant_id,
			"locale_code": self.locale_code,
			"display_name": self.display_name,
			"owner_id": self.owner_id,
			"fallback_locale": self.fallback_locale,
			"regional_format": dict(self.regional_format),
			"timezone": self.timezone,
			"enabled": self.enabled,
			"created_at": self.created_at,
		}


@dataclass(slots=True)
class GlossaryTerm:
	"""Tenant-scoped glossary term with localized variants."""

	id: str
	tenant_id: str
	source_term: str
	localized_terms: dict[str, str] = field(default_factory=dict)
	description: str = ""
	owner_id: str = ""
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"kind": "glossary_term",
			"tenant_id": self.tenant_id,
			"source_term": self.source_term,
			"localized_terms": dict(self.localized_terms),
			"description": self.description,
			"owner_id": self.owner_id,
			"created_at": self.created_at,
		}


@dataclass(slots=True)
class TranslationEntry:
	"""Tenant-scoped localized string."""

	id: str
	tenant_id: str
	key: str
	locale_code: str
	source_text: str
	translated_text: str
	status: TranslationStatus = TranslationStatus.DRAFT
	source: TranslationSource = TranslationSource.HUMAN
	reviewer_id: str | None = None
	restricted: bool = False
	version: int = 1
	created_at: str = field(default_factory=utc_now_iso)
	updated_at: str = field(default_factory=utc_now_iso)
	published_at: str | None = None

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"kind": "translation",
			"tenant_id": self.tenant_id,
			"key": self.key,
			"locale_code": self.locale_code,
			"source_text": self.source_text,
			"translated_text": self.translated_text,
			"status": self.status.value,
			"source": self.source.value,
			"reviewer_id": self.reviewer_id,
			"restricted": self.restricted,
			"version": self.version,
			"created_at": self.created_at,
			"updated_at": self.updated_at,
			"published_at": self.published_at,
		}


@dataclass(slots=True)
class CoverageReport:
	"""Coverage report for a locale and required translation keys."""

	id: str
	tenant_id: str
	locale_code: str
	total_key_count: int
	published_key_count: int
	missing_keys: list[str]
	coverage_percent: float
	requires_review: bool
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"kind": "coverage_report",
			"tenant_id": self.tenant_id,
			"locale_code": self.locale_code,
			"total_key_count": self.total_key_count,
			"published_key_count": self.published_key_count,
			"missing_keys": list(self.missing_keys),
			"coverage_percent": self.coverage_percent,
			"requires_review": self.requires_review,
			"created_at": self.created_at,
		}


@dataclass(slots=True)
class PublishBatch:
	"""Approved publication batch for a locale."""

	id: str
	tenant_id: str
	locale_code: str
	translation_ids: list[str]
	approver_id: str
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"kind": "publish_batch",
			"tenant_id": self.tenant_id,
			"locale_code": self.locale_code,
			"translation_ids": list(self.translation_ids),
			"approver_id": self.approver_id,
			"created_at": self.created_at,
		}


@dataclass(slots=True)
class I18nAgent:
	"""Registered AI agent allowed to assist localization work."""

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
			"kind": "i18n_agent",
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


@dataclass(slots=True)
class I18nAuditEvent:
	"""Audit event emitted by localization lifecycle operations."""

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
			"kind": "audit_event",
			"tenant_id": self.tenant_id,
			"subject_id": self.subject_id,
			"event_type": self.event_type,
			"actor": self.actor,
			"decision": self.decision,
			"reasons": list(self.reasons),
			"metadata": dict(self.metadata),
			"created_at": self.created_at,
		}
