"""Executable help and knowledge-base models for the HELP capability."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utc_now_iso() -> str:
	return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class ArticleStatus(str, Enum):
	"""Knowledge article lifecycle states."""

	DRAFT = "draft"
	REVIEW = "review"
	PUBLISHED = "published"
	ARCHIVED = "archived"


class ContentVisibility(str, Enum):
	"""Help article visibility classes."""

	PUBLIC = "public"
	INTERNAL = "internal"
	RESTRICTED = "restricted"


@dataclass(slots=True)
class HelpArticle:
	"""Tenant-scoped knowledge article."""

	id: str
	tenant_id: str
	title: str
	body: str
	owner_id: str
	topics: list[str] = field(default_factory=list)
	locale: str = "en"
	visibility: ContentVisibility = ContentVisibility.INTERNAL
	status: ArticleStatus = ArticleStatus.DRAFT
	source_ids: list[str] = field(default_factory=list)
	created_at: str = field(default_factory=utc_now_iso)
	updated_at: str = field(default_factory=utc_now_iso)
	published_at: str | None = None
	last_reviewed_at: str | None = None

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"kind": "article",
			"tenant_id": self.tenant_id,
			"title": self.title,
			"body": self.body,
			"owner_id": self.owner_id,
			"topics": list(self.topics),
			"locale": self.locale,
			"visibility": self.visibility.value,
			"status": self.status.value,
			"source_ids": list(self.source_ids),
			"created_at": self.created_at,
			"updated_at": self.updated_at,
			"published_at": self.published_at,
			"last_reviewed_at": self.last_reviewed_at,
		}


@dataclass(slots=True)
class HelpSource:
	"""Approved source reference for help content."""

	id: str
	tenant_id: str
	title: str
	uri: str
	owner_id: str
	approved: bool = False
	approved_by: str | None = None
	visibility: ContentVisibility = ContentVisibility.INTERNAL
	created_at: str = field(default_factory=utc_now_iso)
	approved_at: str | None = None

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"kind": "source",
			"tenant_id": self.tenant_id,
			"title": self.title,
			"uri": self.uri,
			"owner_id": self.owner_id,
			"approved": self.approved,
			"approved_by": self.approved_by,
			"visibility": self.visibility.value,
			"created_at": self.created_at,
			"approved_at": self.approved_at,
		}


@dataclass(slots=True)
class HelpCitation:
	"""Citation linking an answer back to an approved article."""

	article_id: str
	title: str
	excerpt: str

	def to_dict(self) -> dict[str, str]:
		return {
			"article_id": self.article_id,
			"title": self.title,
			"excerpt": self.excerpt,
		}


@dataclass(slots=True)
class HelpAnswer:
	"""Generated or curated answer with mandatory citations."""

	id: str
	tenant_id: str
	query: str
	answer: str
	confidence: float
	citations: list[HelpCitation]
	blocked: bool = False
	block_reason: str | None = None
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"kind": "answer",
			"tenant_id": self.tenant_id,
			"query": self.query,
			"answer": self.answer,
			"confidence": self.confidence,
			"citations": [citation.to_dict() for citation in self.citations],
			"blocked": self.blocked,
			"block_reason": self.block_reason,
			"created_at": self.created_at,
		}


@dataclass(slots=True)
class HelpLocalization:
	"""Localized article variant with translation ownership."""

	id: str
	tenant_id: str
	article_id: str
	locale: str
	source_locale: str
	title: str
	body: str
	translator_id: str
	fallback_locale: str
	status: str = "draft"
	created_at: str = field(default_factory=utc_now_iso)
	updated_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"kind": "localization",
			"tenant_id": self.tenant_id,
			"article_id": self.article_id,
			"locale": self.locale,
			"source_locale": self.source_locale,
			"title": self.title,
			"body": self.body,
			"translator_id": self.translator_id,
			"fallback_locale": self.fallback_locale,
			"status": self.status,
			"created_at": self.created_at,
			"updated_at": self.updated_at,
		}


@dataclass(slots=True)
class HelpFeedback:
	"""User feedback for an article or answer."""

	id: str
	tenant_id: str
	user_id: str
	rating: int
	comment: str = ""
	article_id: str | None = None
	answer_id: str | None = None
	requires_review: bool = False
	status: str = "open"
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"kind": "feedback",
			"tenant_id": self.tenant_id,
			"user_id": self.user_id,
			"rating": self.rating,
			"comment": self.comment,
			"article_id": self.article_id,
			"answer_id": self.answer_id,
			"requires_review": self.requires_review,
			"status": self.status,
			"created_at": self.created_at,
		}


@dataclass(slots=True)
class HelpCurationItem:
	"""Governance task for article quality, approval, or freshness."""

	id: str
	tenant_id: str
	article_id: str
	reason: str
	status: str = "open"
	reviewer_id: str | None = None
	evidence: list[str] = field(default_factory=list)
	created_at: str = field(default_factory=utc_now_iso)
	closed_at: str | None = None

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"kind": "curation_item",
			"tenant_id": self.tenant_id,
			"article_id": self.article_id,
			"reason": self.reason,
			"status": self.status,
			"reviewer_id": self.reviewer_id,
			"evidence": list(self.evidence),
			"created_at": self.created_at,
			"closed_at": self.closed_at,
		}


@dataclass(slots=True)
class HelpAuditEvent:
	"""Audit event emitted by dependency-light HELP runtime operations."""

	id: str
	tenant_id: str
	event_type: str
	subject_id: str
	message: str
	actor: str
	severity: str = "low"
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"kind": "audit_event",
			"tenant_id": self.tenant_id,
			"event_type": self.event_type,
			"subject_id": self.subject_id,
			"message": self.message,
			"actor": self.actor,
			"severity": self.severity,
			"created_at": self.created_at,
		}


@dataclass(slots=True)
class HelpAgentRecord:
	"""First-class provider-neutral help governance agent."""

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
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"kind": "help_agent",
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


@dataclass(slots=True)
class HelpLifecycleBatchRecord:
	"""Bytewax lifecycle-batch validation evidence for help mutations."""

	id: str
	tenant_id: str
	event_stream: str
	mutation_count: int
	operation: str
	accepted: bool
	decision: str
	matched_rules: list[str] = field(default_factory=list)
	status: str = "accepted"
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"kind": "help_lifecycle_batch",
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
