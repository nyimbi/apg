"""Dependency-light Search Engine runtime primitives."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any


CLASSIFICATIONS = {"public", "internal", "confidential", "restricted"}
INDEX_STATES = {"creating", "ready", "embedding_pending", "embedding_ready", "degraded", "retired"}
QUERY_TYPES = {"keyword", "semantic", "hybrid"}


def utc_now() -> str:
	return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_id(prefix: str, *parts: object) -> str:
	seed = "|".join(str(part).strip().lower() for part in parts if str(part).strip())
	digest = sha256(seed.encode("utf-8")).hexdigest()[:16]
	return f"{prefix}_{digest}"


def normalize_classification(classification: str) -> str:
	value = str(classification or "internal").strip().lower()
	if value in {"private", "sensitive"}:
		value = "confidential"
	if value not in CLASSIFICATIONS:
		raise ValueError(f"unsupported_content_classification:{classification}")
	return value


def normalize_query_type(query_type: str) -> str:
	value = str(query_type or "keyword").strip().lower()
	if value not in QUERY_TYPES:
		raise ValueError(f"unsupported_query_type:{query_type}")
	return value


def search_required_actions(rule_result: dict[str, Any]) -> list[str]:
	return [
		str(action["required_action"])
		for action in rule_result.get("actions", [])
		if action.get("required_action")
	]


def serialize(record: object) -> dict[str, Any]:
	return asdict(record)


@dataclass(slots=True)
class SearchIndexRecord:
	id: str
	tenant_id: str
	name: str
	owner: str
	content_type: str
	classification: str
	status: str = "ready"
	source_lineage_ref: str | None = None
	embedding_index_ready: bool = False
	document_count: int = 0
	created_at: str = field(default_factory=utc_now)
	updated_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return serialize(self)


@dataclass(slots=True)
class SearchDocumentRecord:
	id: str
	tenant_id: str
	index_id: str
	document_id: str
	title: str
	body: str
	classification: str
	facets: dict[str, str] = field(default_factory=dict)
	metadata: dict[str, Any] = field(default_factory=dict)
	indexed_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return serialize(self)


@dataclass(slots=True)
class QueryRecord:
	id: str
	tenant_id: str
	query_text: str
	query_type: str
	index_ids: list[str]
	result_window: int
	rbac_filter_applied: bool
	review_recorded: bool
	status: str
	result_count: int
	required_actions: list[str] = field(default_factory=list)
	matched_rules: list[str] = field(default_factory=list)
	created_at: str = field(default_factory=utc_now)

	def to_dict(self) -> dict[str, Any]:
		return serialize(self)


@dataclass(slots=True)
class SearchAuditEventRecord:
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
	"CLASSIFICATIONS",
	"INDEX_STATES",
	"QUERY_TYPES",
	"QueryRecord",
	"SearchAuditEventRecord",
	"SearchDocumentRecord",
	"SearchIndexRecord",
	"normalize_classification",
	"normalize_query_type",
	"search_required_actions",
	"stable_id",
	"utc_now",
]
