"""Domain models for APG Knowledge Graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
	"""Return a stable UTC timestamp for dependency-light runtime records."""
	return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class KnowledgeSource:
	"""Tenant-scoped source used as evidence for graph facts."""

	id: str
	tenant_id: str
	name: str
	source_uri: str
	owner: str
	connector: str
	evidence_refs: tuple[str, ...]
	confidence_score: float
	status: str = "active"
	decision: str = "allow"
	matched_rules: tuple[str, ...] = ()
	review_reasons: tuple[str, ...] = ()
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"source_uri": self.source_uri,
			"owner": self.owner,
			"connector": self.connector,
			"evidence_refs": list(self.evidence_refs),
			"confidence_score": self.confidence_score,
			"status": self.status,
			"decision": self.decision,
			"matched_rules": list(self.matched_rules),
			"review_reasons": list(self.review_reasons),
			"created_at": self.created_at,
		}


@dataclass(frozen=True)
class KnowledgeEntity:
	"""Curatable semantic entity resolved from one or more source assets."""

	id: str
	tenant_id: str
	canonical_label: str
	entity_type: str
	source_id: str
	source_evidence_refs: tuple[str, ...]
	aliases: tuple[str, ...] = ()
	attributes: dict[str, Any] = field(default_factory=dict)
	confidence_score: float = 1.0
	curation_status: str = "draft"
	status: str = "active"
	decision: str = "allow"
	matched_rules: tuple[str, ...] = ()
	review_reasons: tuple[str, ...] = ()
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"canonical_label": self.canonical_label,
			"entity_type": self.entity_type,
			"source_id": self.source_id,
			"source_evidence_refs": list(self.source_evidence_refs),
			"aliases": list(self.aliases),
			"attributes": dict(self.attributes),
			"confidence_score": self.confidence_score,
			"curation_status": self.curation_status,
			"status": self.status,
			"decision": self.decision,
			"matched_rules": list(self.matched_rules),
			"review_reasons": list(self.review_reasons),
			"created_at": self.created_at,
		}


@dataclass(frozen=True)
class KnowledgeRelationship:
	"""Evidence-backed directed semantic relationship between two entities."""

	id: str
	tenant_id: str
	subject_entity_id: str
	predicate: str
	object_entity_id: str
	source_id: str
	evidence_links: tuple[str, ...]
	confidence_score: float
	status: str
	decision: str = "allow"
	matched_rules: tuple[str, ...] = ()
	review_reasons: tuple[str, ...] = ()
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"subject_entity_id": self.subject_entity_id,
			"predicate": self.predicate,
			"object_entity_id": self.object_entity_id,
			"source_id": self.source_id,
			"evidence_links": list(self.evidence_links),
			"confidence_score": self.confidence_score,
			"status": self.status,
			"decision": self.decision,
			"matched_rules": list(self.matched_rules),
			"review_reasons": list(self.review_reasons),
			"created_at": self.created_at,
		}


@dataclass(frozen=True)
class SemanticEnrichment:
	"""Semantic labels and attributes attached to a curated entity."""

	id: str
	tenant_id: str
	entity_id: str
	semantic_labels: tuple[str, ...]
	attributes: dict[str, Any]
	evidence_links: tuple[str, ...]
	confidence_score: float
	review_recorded: bool
	status: str
	decision: str = "allow"
	matched_rules: tuple[str, ...] = ()
	review_reasons: tuple[str, ...] = ()
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"entity_id": self.entity_id,
			"semantic_labels": list(self.semantic_labels),
			"attributes": dict(self.attributes),
			"evidence_links": list(self.evidence_links),
			"confidence_score": self.confidence_score,
			"review_recorded": self.review_recorded,
			"status": self.status,
			"decision": self.decision,
			"matched_rules": list(self.matched_rules),
			"review_reasons": list(self.review_reasons),
			"created_at": self.created_at,
		}


@dataclass(frozen=True)
class ReasoningPath:
	"""Bounded evidence path returned by graph reasoning."""

	id: str
	tenant_id: str
	query: str
	start_entity_id: str
	end_entity_id: str
	relationship_ids: tuple[str, ...]
	evidence_links: tuple[str, ...]
	reasoning_depth: int
	review_recorded: bool
	status: str
	decision: str = "allow"
	matched_rules: tuple[str, ...] = ()
	review_reasons: tuple[str, ...] = ()
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"query": self.query,
			"start_entity_id": self.start_entity_id,
			"end_entity_id": self.end_entity_id,
			"relationship_ids": list(self.relationship_ids),
			"evidence_links": list(self.evidence_links),
			"reasoning_depth": self.reasoning_depth,
			"review_recorded": self.review_recorded,
			"status": self.status,
			"decision": self.decision,
			"matched_rules": list(self.matched_rules),
			"review_reasons": list(self.review_reasons),
			"created_at": self.created_at,
		}


@dataclass(frozen=True)
class CurationRecord:
	"""Human or governed curation decision for graph publication."""

	id: str
	tenant_id: str
	entity_id: str
	curator: str
	decision: str
	evidence_links: tuple[str, ...]
	notes: str = ""
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"entity_id": self.entity_id,
			"curator": self.curator,
			"decision": self.decision,
			"evidence_links": list(self.evidence_links),
			"notes": self.notes,
			"created_at": self.created_at,
		}


@dataclass(frozen=True)
class GraphPublication:
	"""Publication record for a curated tenant graph snapshot."""

	id: str
	tenant_id: str
	name: str
	entity_ids: tuple[str, ...]
	relationship_ids: tuple[str, ...]
	published_by: str
	status: str
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"entity_ids": list(self.entity_ids),
			"relationship_ids": list(self.relationship_ids),
			"published_by": self.published_by,
			"status": self.status,
			"created_at": self.created_at,
		}


@dataclass(frozen=True)
class KnowledgeAgentRecord:
	"""Provider-neutral AI agent registered for knowledge-graph governance."""

	id: str
	tenant_id: str
	name: str
	runtime: str
	role: str
	scope: str
	owner: str
	purpose: str
	contribution_disclosed: bool
	human_approval_required: bool
	status: str = "active"
	decision: str = "allow"
	matched_rules: tuple[str, ...] = ()
	review_reasons: tuple[str, ...] = ()
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"agent_id": self.id,
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
			"decision": self.decision,
			"matched_rules": list(self.matched_rules),
			"review_reasons": list(self.review_reasons),
			"created_at": self.created_at,
		}


@dataclass(frozen=True)
class KngrLifecycleBatchRecord:
	"""Bytewax lifecycle batch validation evidence for knowledge graph changes."""

	id: str
	tenant_id: str
	event_stream: str
	mutation_count: int
	operation: str
	accepted: bool
	decision: str
	matched_rules: tuple[str, ...] = ()
	review_reasons: tuple[str, ...] = ()
	required_processor: str = "bytewax"
	status: str = "accepted"
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"batch_id": self.id,
			"tenant_id": self.tenant_id,
			"event_stream": self.event_stream,
			"mutation_count": self.mutation_count,
			"operation": self.operation,
			"accepted": self.accepted,
			"decision": self.decision,
			"matched_rules": list(self.matched_rules),
			"review_reasons": list(self.review_reasons),
			"required_processor": self.required_processor,
			"status": self.status,
			"created_at": self.created_at,
		}


@dataclass(frozen=True)
class KngrAuditEvent:
	"""Governance event emitted by knowledge graph operations."""

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
			"tenant_id": self.tenant_id,
			"subject_id": self.subject_id,
			"event_type": self.event_type,
			"actor": self.actor,
			"decision": self.decision,
			"reasons": list(self.reasons),
			"metadata": dict(self.metadata),
			"created_at": self.created_at,
		}


KngrRecord = KnowledgeEntity
