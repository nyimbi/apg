"""Domain models for the Ontology Management capability."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
	"""Return a compact UTC timestamp string for in-process ontology records."""
	return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class Ontology:
	"""Tenant-scoped ontology registry record."""

	id: str
	tenant_id: str
	name: str
	owner: str
	domain: str
	version: str = "0.1.0"
	status: str = "draft"
	description: str = ""
	metadata: dict[str, Any] = field(default_factory=dict)
	decision: str = "allow"
	matched_rules: tuple[str, ...] = ()
	review_reasons: tuple[str, ...] = ()
	audit_evidence: dict[str, Any] = field(default_factory=dict)
	created_at: str = field(default_factory=utc_now_iso)
	updated_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"name": self.name,
			"owner": self.owner,
			"domain": self.domain,
			"version": self.version,
			"status": self.status,
			"description": self.description,
			"metadata": dict(self.metadata),
			"decision": self.decision,
			"matched_rules": list(self.matched_rules),
			"review_reasons": list(self.review_reasons),
			"audit_evidence": dict(self.audit_evidence),
			"created_at": self.created_at,
			"updated_at": self.updated_at,
		}


@dataclass
class OntologyTerm:
	"""Controlled vocabulary term within an ontology."""

	id: str
	tenant_id: str
	ontology_id: str
	label: str
	owner: str
	definition: str = ""
	status: str = "draft"
	synonyms: list[str] = field(default_factory=list)
	external_refs: list[str] = field(default_factory=list)
	metadata: dict[str, Any] = field(default_factory=dict)
	decision: str = "allow"
	matched_rules: tuple[str, ...] = ()
	review_reasons: tuple[str, ...] = ()
	audit_evidence: dict[str, Any] = field(default_factory=dict)
	created_at: str = field(default_factory=utc_now_iso)
	updated_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"ontology_id": self.ontology_id,
			"label": self.label,
			"owner": self.owner,
			"definition": self.definition,
			"status": self.status,
			"synonyms": list(self.synonyms),
			"external_refs": list(self.external_refs),
			"metadata": dict(self.metadata),
			"decision": self.decision,
			"matched_rules": list(self.matched_rules),
			"review_reasons": list(self.review_reasons),
			"audit_evidence": dict(self.audit_evidence),
			"created_at": self.created_at,
			"updated_at": self.updated_at,
		}


@dataclass
class OntologyNamespace:
	"""Namespace prefix and URI bound to an ontology."""

	id: str
	tenant_id: str
	ontology_id: str
	prefix: str
	uri: str
	owner: str
	status: str = "active"
	metadata: dict[str, Any] = field(default_factory=dict)
	decision: str = "allow"
	matched_rules: tuple[str, ...] = ()
	review_reasons: tuple[str, ...] = ()
	audit_evidence: dict[str, Any] = field(default_factory=dict)
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"ontology_id": self.ontology_id,
			"prefix": self.prefix,
			"uri": self.uri,
			"owner": self.owner,
			"status": self.status,
			"metadata": dict(self.metadata),
			"decision": self.decision,
			"matched_rules": list(self.matched_rules),
			"review_reasons": list(self.review_reasons),
			"audit_evidence": dict(self.audit_evidence),
			"created_at": self.created_at,
		}


@dataclass
class TaxonomyEdge:
	"""Parent-child relationship between ontology terms."""

	id: str
	tenant_id: str
	ontology_id: str
	parent_term_id: str
	child_term_id: str
	relationship_type: str = "broader_than"
	status: str = "active"
	decision: str = "allow"
	matched_rules: tuple[str, ...] = ()
	review_reasons: tuple[str, ...] = ()
	audit_evidence: dict[str, Any] = field(default_factory=dict)
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"ontology_id": self.ontology_id,
			"parent_term_id": self.parent_term_id,
			"child_term_id": self.child_term_id,
			"relationship_type": self.relationship_type,
			"status": self.status,
			"decision": self.decision,
			"matched_rules": list(self.matched_rules),
			"review_reasons": list(self.review_reasons),
			"audit_evidence": dict(self.audit_evidence),
			"created_at": self.created_at,
		}


@dataclass
class SemanticMapping:
	"""Mapping between an ontology term and an external semantic concept."""

	id: str
	tenant_id: str
	ontology_id: str
	term_id: str
	target_ref: str
	mapping_type: str = "exact"
	confidence: float = 1.0
	review_recorded: bool = False
	review_ref: str = ""
	status: str = "active"
	created_at: str = field(default_factory=utc_now_iso)
	metadata: dict[str, Any] = field(default_factory=dict)
	decision: str = "allow"
	matched_rules: tuple[str, ...] = ()
	review_reasons: tuple[str, ...] = ()
	audit_evidence: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"ontology_id": self.ontology_id,
			"term_id": self.term_id,
			"target_ref": self.target_ref,
			"mapping_type": self.mapping_type,
			"confidence": self.confidence,
			"review_recorded": self.review_recorded,
			"review_ref": self.review_ref,
			"status": self.status,
			"created_at": self.created_at,
			"metadata": dict(self.metadata),
			"decision": self.decision,
			"matched_rules": list(self.matched_rules),
			"review_reasons": list(self.review_reasons),
			"audit_evidence": dict(self.audit_evidence),
		}


@dataclass
class CurationReview:
	"""Governance review for breaking changes, mapping reviews, or term curation."""

	id: str
	tenant_id: str
	ontology_id: str
	subject_id: str
	review_type: str
	reviewer: str
	status: str = "approved"
	notes: str = ""
	decision: str = "allow"
	matched_rules: tuple[str, ...] = ()
	review_reasons: tuple[str, ...] = ()
	audit_evidence: dict[str, Any] = field(default_factory=dict)
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"ontology_id": self.ontology_id,
			"subject_id": self.subject_id,
			"review_type": self.review_type,
			"reviewer": self.reviewer,
			"status": self.status,
			"notes": self.notes,
			"decision": self.decision,
			"matched_rules": list(self.matched_rules),
			"review_reasons": list(self.review_reasons),
			"audit_evidence": dict(self.audit_evidence),
			"created_at": self.created_at,
		}


@dataclass
class OntologyPublication:
	"""Publication event for an ontology version."""

	id: str
	tenant_id: str
	ontology_id: str
	version: str
	approval_recorded: bool
	approval_ref: str = ""
	status: str = "published"
	duplicate_count: int = 0
	term_count: int = 0
	mapping_count: int = 0
	decision: str = "allow"
	matched_rules: tuple[str, ...] = ()
	review_reasons: tuple[str, ...] = ()
	audit_evidence: dict[str, Any] = field(default_factory=dict)
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"ontology_id": self.ontology_id,
			"version": self.version,
			"approval_recorded": self.approval_recorded,
			"approval_ref": self.approval_ref,
			"status": self.status,
			"duplicate_count": self.duplicate_count,
			"term_count": self.term_count,
			"mapping_count": self.mapping_count,
			"decision": self.decision,
			"matched_rules": list(self.matched_rules),
			"review_reasons": list(self.review_reasons),
			"audit_evidence": dict(self.audit_evidence),
			"created_at": self.created_at,
		}


@dataclass
class ValidationReport:
	"""Validation result for ontology publication readiness."""

	id: str
	tenant_id: str
	ontology_id: str
	issue_count: int
	issues: list[str] = field(default_factory=list)
	status: str = "passed"
	review_recorded: bool = False
	review_ref: str = ""
	decision: str = "allow"
	matched_rules: tuple[str, ...] = ()
	review_reasons: tuple[str, ...] = ()
	audit_evidence: dict[str, Any] = field(default_factory=dict)
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"ontology_id": self.ontology_id,
			"issue_count": self.issue_count,
			"issues": list(self.issues),
			"status": self.status,
			"review_recorded": self.review_recorded,
			"review_ref": self.review_ref,
			"decision": self.decision,
			"matched_rules": list(self.matched_rules),
			"review_reasons": list(self.review_reasons),
			"audit_evidence": dict(self.audit_evidence),
			"created_at": self.created_at,
		}


@dataclass
class OntologyExport:
	"""Export event for an ontology in an interchange format."""

	id: str
	tenant_id: str
	ontology_id: str
	format: str
	version: str
	status: str = "ready"
	artifact_ref: str = ""
	decision: str = "allow"
	matched_rules: tuple[str, ...] = ()
	review_reasons: tuple[str, ...] = ()
	audit_evidence: dict[str, Any] = field(default_factory=dict)
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"ontology_id": self.ontology_id,
			"format": self.format,
			"version": self.version,
			"status": self.status,
			"artifact_ref": self.artifact_ref,
			"decision": self.decision,
			"matched_rules": list(self.matched_rules),
			"review_reasons": list(self.review_reasons),
			"audit_evidence": dict(self.audit_evidence),
			"created_at": self.created_at,
		}


@dataclass
class OntologyAgentRecord:
	"""Provider-neutral AI agent registered for ontology governance."""

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
	audit_evidence: dict[str, Any] = field(default_factory=dict)
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
			"audit_evidence": dict(self.audit_evidence),
			"created_at": self.created_at,
		}


@dataclass
class OntoLifecycleBatchRecord:
	"""Bytewax lifecycle batch validation evidence for ontology changes."""

	id: str
	tenant_id: str
	event_stream: str
	mutation_count: int
	operation: str
	accepted: bool
	decision: str
	matched_rules: list[str] = field(default_factory=list)
	review_reasons: tuple[str, ...] = ()
	audit_evidence: dict[str, Any] = field(default_factory=dict)
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
			"audit_evidence": dict(self.audit_evidence),
			"required_processor": self.required_processor,
			"status": self.status,
			"created_at": self.created_at,
		}


@dataclass
class OntoAuditEvent:
	"""Audit event emitted by ontology operations."""

	id: str
	tenant_id: str
	event_type: str
	subject_id: str
	message: str
	severity: str = "info"
	created_at: str = field(default_factory=utc_now_iso)
	metadata: dict[str, Any] = field(default_factory=dict)
	decision: str = "allow"
	matched_rules: tuple[str, ...] = ()
	review_reasons: tuple[str, ...] = ()
	audit_evidence: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"tenant_id": self.tenant_id,
			"event_type": self.event_type,
			"subject_id": self.subject_id,
			"message": self.message,
			"severity": self.severity,
			"created_at": self.created_at,
			"metadata": dict(self.metadata),
			"decision": self.decision,
			"matched_rules": list(self.matched_rules),
			"review_reasons": list(self.review_reasons),
			"audit_evidence": dict(self.audit_evidence),
		}


# Compatibility alias for older package callers that import OntoRecord.
OntoRecord = Ontology
