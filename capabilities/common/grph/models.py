"""Executable graph data models for the GRPH capability."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utc_now_iso() -> str:
	return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class GraphKind(str, Enum):
	"""Supported graph usage modes."""

	PROPERTY = "property"
	LINEAGE = "lineage"
	KNOWLEDGE = "knowledge"
	DEPENDENCY = "dependency"


class RelationshipClassification(str, Enum):
	"""Governance classification for edges."""

	PUBLIC = "public"
	INTERNAL = "internal"
	CONFIDENTIAL = "confidential"
	RESTRICTED = "restricted"


@dataclass(slots=True)
class GraphSchema:
	"""Tenant-scoped graph schema definition."""

	id: str
	tenant_id: str
	name: str
	graph_kind: GraphKind = GraphKind.PROPERTY
	node_types: dict[str, list[str]] = field(default_factory=dict)
	edge_types: dict[str, dict[str, Any]] = field(default_factory=dict)
	source_asset_id: str | None = None
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"kind": "schema",
			"tenant_id": self.tenant_id,
			"name": self.name,
			"graph_kind": self.graph_kind.value,
			"node_types": {key: list(value) for key, value in self.node_types.items()},
			"edge_types": {key: dict(value) for key, value in self.edge_types.items()},
			"source_asset_id": self.source_asset_id,
			"created_at": self.created_at,
		}


@dataclass(slots=True)
class GraphNode:
	"""Tenant-scoped graph node."""

	id: str
	tenant_id: str
	schema_id: str
	node_type: str
	owner_id: str
	labels: list[str] = field(default_factory=list)
	properties: dict[str, Any] = field(default_factory=dict)
	source_asset_id: str | None = None
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"kind": "node",
			"tenant_id": self.tenant_id,
			"schema_id": self.schema_id,
			"node_type": self.node_type,
			"owner_id": self.owner_id,
			"labels": list(self.labels),
			"properties": dict(self.properties),
			"source_asset_id": self.source_asset_id,
			"created_at": self.created_at,
		}


@dataclass(slots=True)
class GraphEdge:
	"""Tenant-scoped typed relationship between graph nodes."""

	id: str
	tenant_id: str
	schema_id: str
	from_node_id: str
	to_node_id: str
	edge_type: str
	owner_id: str
	classification: RelationshipClassification = RelationshipClassification.INTERNAL
	properties: dict[str, Any] = field(default_factory=dict)
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"kind": "edge",
			"tenant_id": self.tenant_id,
			"schema_id": self.schema_id,
			"from_node_id": self.from_node_id,
			"to_node_id": self.to_node_id,
			"edge_type": self.edge_type,
			"owner_id": self.owner_id,
			"classification": self.classification.value,
			"properties": dict(self.properties),
			"created_at": self.created_at,
		}


@dataclass(slots=True)
class GraphTraversalResult:
	"""Bounded traversal result."""

	id: str
	tenant_id: str
	start_node_id: str
	max_depth: int
	node_ids: list[str]
	edge_ids: list[str]
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"kind": "traversal",
			"tenant_id": self.tenant_id,
			"start_node_id": self.start_node_id,
			"max_depth": self.max_depth,
			"node_ids": list(self.node_ids),
			"edge_ids": list(self.edge_ids),
			"created_at": self.created_at,
		}


@dataclass(slots=True)
class GraphQualityReport:
	"""Graph data quality summary."""

	id: str
	tenant_id: str
	schema_id: str
	orphan_node_count: int
	missing_owner_count: int
	restricted_edge_count: int
	created_at: str = field(default_factory=utc_now_iso)

	@property
	def status(self) -> str:
		return "attention_required" if self.orphan_node_count or self.missing_owner_count else "healthy"

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"kind": "quality_report",
			"tenant_id": self.tenant_id,
			"schema_id": self.schema_id,
			"orphan_node_count": self.orphan_node_count,
			"missing_owner_count": self.missing_owner_count,
			"restricted_edge_count": self.restricted_edge_count,
			"status": self.status,
			"created_at": self.created_at,
		}


@dataclass(slots=True)
class GraphAgentRecord:
	"""Provider-neutral AI agent registered for graph governance work."""

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
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"agent_id": self.id,
			"kind": "graph_agent",
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
class GrphLifecycleBatchRecord:
	"""Bytewax lifecycle batch validation evidence for graph operations."""

	id: str
	tenant_id: str
	event_stream: str
	mutation_count: int
	operation: str
	accepted: bool
	decision: str
	matched_rules: list[str] = field(default_factory=list)
	required_processor: str = "bytewax"
	status: str = "accepted"
	created_at: str = field(default_factory=utc_now_iso)

	def to_dict(self) -> dict[str, Any]:
		return {
			"id": self.id,
			"batch_id": self.id,
			"kind": "lifecycle_batch",
			"tenant_id": self.tenant_id,
			"event_stream": self.event_stream,
			"mutation_count": self.mutation_count,
			"operation": self.operation,
			"accepted": self.accepted,
			"decision": self.decision,
			"matched_rules": list(self.matched_rules),
			"required_processor": self.required_processor,
			"status": self.status,
			"created_at": self.created_at,
		}


@dataclass(slots=True)
class GraphAuditEventRecord:
	"""Tenant-scoped graph audit event."""

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
