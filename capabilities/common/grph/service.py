"""Executable graph service for APG graph composition."""

from __future__ import annotations

from itertools import count
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract
from .graph_runtime import GraphQualityInspector, GraphTraversalPlanner
from .models import (
	GraphEdge,
	GraphKind,
	GraphNode,
	GraphQualityReport,
	GraphSchema,
	GraphTraversalResult,
	RelationshipClassification,
)


class GrphService:
	"""Tenant-aware graph schema, node, edge, traversal, and quality runtime."""

	def __init__(self) -> None:
		self._schemas: dict[str, GraphSchema] = {}
		self._nodes: dict[str, GraphNode] = {}
		self._edges: dict[str, GraphEdge] = {}
		self._traversals: dict[str, GraphTraversalResult] = {}
		self._quality_reports: dict[str, GraphQualityReport] = {}
		self._counter = count(1)
		self._traversal_planner = GraphTraversalPlanner()
		self._quality_inspector = GraphQualityInspector()

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def create_schema(
		self,
		schema_id: str,
		tenant_id: str,
		name: str,
		graph_kind: str = GraphKind.PROPERTY.value,
		node_types: dict[str, list[str]] | None = None,
		edge_types: dict[str, dict[str, Any]] | None = None,
		source_asset_id: str | None = None,
	) -> dict[str, Any]:
		self._enforce_graph_policy(
			tenant_id=tenant_id,
			operation="write_schema",
			graph_type=graph_kind,
			source_asset_present=bool(source_asset_id),
		)
		schema = GraphSchema(
			id=schema_id,
			tenant_id=tenant_id,
			name=name,
			graph_kind=GraphKind(graph_kind),
			node_types={key: list(value) for key, value in (node_types or {}).items()},
			edge_types={key: dict(value) for key, value in (edge_types or {}).items()},
			source_asset_id=source_asset_id,
		)
		self._schemas[schema_id] = schema
		return schema.to_dict()

	def create_node(
		self,
		node_id: str,
		tenant_id: str,
		schema_id: str,
		node_type: str,
		owner_id: str,
		labels: list[str] | None = None,
		properties: dict[str, Any] | None = None,
		source_asset_id: str | None = None,
	) -> dict[str, Any]:
		self._require_schema(schema_id, tenant_id)
		self._enforce_graph_policy(
			tenant_id=tenant_id,
			operation="write_node",
			owner_assigned=bool(owner_id),
		)
		schema = self._schemas[schema_id]
		if schema.node_types and node_type not in schema.node_types:
			raise ValueError("node_type_not_in_schema")
		node = GraphNode(
			id=node_id,
			tenant_id=tenant_id,
			schema_id=schema_id,
			node_type=node_type,
			owner_id=owner_id,
			labels=list(labels or []),
			properties=dict(properties or {}),
			source_asset_id=source_asset_id,
		)
		self._nodes[node_id] = node
		return node.to_dict()

	def create_edge(
		self,
		edge_id: str,
		tenant_id: str,
		schema_id: str,
		from_node_id: str,
		to_node_id: str,
		edge_type: str,
		owner_id: str,
		classification: str = RelationshipClassification.INTERNAL.value,
		properties: dict[str, Any] | None = None,
		review_recorded: bool = True,
	) -> dict[str, Any]:
		self._require_schema(schema_id, tenant_id)
		self._require_node(from_node_id, tenant_id)
		self._require_node(to_node_id, tenant_id)
		self._enforce_graph_policy(
			tenant_id=tenant_id,
			operation="write_edge",
			owner_assigned=bool(owner_id),
			edge_type_present=bool(edge_type),
			relationship_classification=classification,
			review_recorded=review_recorded,
		)
		schema = self._schemas[schema_id]
		if schema.edge_types and edge_type not in schema.edge_types:
			raise ValueError("edge_type_not_in_schema")
		edge = GraphEdge(
			id=edge_id,
			tenant_id=tenant_id,
			schema_id=schema_id,
			from_node_id=from_node_id,
			to_node_id=to_node_id,
			edge_type=edge_type,
			owner_id=owner_id,
			classification=RelationshipClassification(classification),
			properties=dict(properties or {}),
		)
		self._edges[edge_id] = edge
		return edge.to_dict()

	def traverse(
		self,
		traversal_id: str,
		tenant_id: str,
		start_node_id: str,
		max_depth: int = 1,
		review_recorded: bool = True,
	) -> dict[str, Any]:
		self._require_node(start_node_id, tenant_id)
		self._enforce_graph_policy(
			tenant_id=tenant_id,
			operation="traverse",
			traversal_depth=max_depth,
			review_recorded=review_recorded,
		)
		node_ids, edge_ids = self._traversal_planner.traverse(
			tenant_id=tenant_id,
			start_node_id=start_node_id,
			max_depth=max_depth,
			edges=list(self._edges.values()),
		)
		result = GraphTraversalResult(
			id=traversal_id,
			tenant_id=tenant_id,
			start_node_id=start_node_id,
			max_depth=max_depth,
			node_ids=node_ids,
			edge_ids=edge_ids,
		)
		self._traversals[traversal_id] = result
		return result.to_dict()

	def lineage_path(
		self,
		traversal_id: str,
		tenant_id: str,
		source_asset_id: str,
		start_node_id: str,
		max_depth: int = 2,
		review_recorded: bool = True,
	) -> dict[str, Any]:
		self._enforce_graph_policy(
			tenant_id=tenant_id,
			operation="lineage_query",
			graph_type=GraphKind.LINEAGE.value,
			source_asset_present=bool(source_asset_id),
			traversal_depth=max_depth,
			review_recorded=review_recorded,
		)
		return self.traverse(traversal_id, tenant_id, start_node_id, max_depth, review_recorded)

	def quality_report(self, report_id: str, tenant_id: str, schema_id: str) -> dict[str, Any]:
		self._require_schema(schema_id, tenant_id)
		schema_nodes = [node for node in self._nodes.values() if node.tenant_id == tenant_id and node.schema_id == schema_id]
		schema_edges = [edge for edge in self._edges.values() if edge.tenant_id == tenant_id and edge.schema_id == schema_id]
		metrics = self._quality_inspector.inspect(schema_nodes, schema_edges)
		report = GraphQualityReport(
			id=report_id,
			tenant_id=tenant_id,
			schema_id=schema_id,
			orphan_node_count=metrics["orphan_node_count"],
			missing_owner_count=metrics["missing_owner_count"],
			restricted_edge_count=metrics["restricted_edge_count"],
		)
		self._quality_reports[report_id] = report
		return report.to_dict()

	def create_record(
		self,
		record_id: str,
		tenant_id: str,
		metadata: dict[str, Any] | None = None,
		status: str = "active",
	) -> dict[str, Any]:
		"""Compatibility helper for generated package probes."""
		data = dict(metadata or {})
		schema_id = str(data.get("schema_id") or f"{record_id}-schema")
		if schema_id not in self._schemas:
			self.create_schema(
				schema_id=schema_id,
				tenant_id=tenant_id,
				name=str(data.get("schema_name") or "Compatibility Graph"),
				node_types={str(data.get("node_type") or "Entity"): []},
				edge_types={str(data.get("edge_type") or "RELATED_TO"): {}},
			)
		return self.create_node(
			node_id=record_id,
			tenant_id=tenant_id,
			schema_id=schema_id,
			node_type=str(data.get("node_type") or "Entity"),
			owner_id=str(data.get("owner_id") or "system"),
			labels=list(data.get("labels") or [status]),
			properties=data,
		)

	def list_schemas(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._schemas, tenant_id)

	def list_nodes(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._nodes, tenant_id)

	def list_edges(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._edges, tenant_id)

	def list_traversals(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._traversals, tenant_id)

	def list_quality_reports(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._quality_reports, tenant_id)

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		records: list[dict[str, Any]] = []
		for store in (self._schemas, self._nodes, self._edges, self._traversals, self._quality_reports):
			records.extend(self._list(store, tenant_id))
		return sorted(records, key=lambda item: (item["kind"], item["id"]))

	def dashboard_summary(self, tenant_id: str | None = None) -> dict[str, Any]:
		nodes = self.list_nodes(tenant_id)
		edges = self.list_edges(tenant_id)
		restricted_edges = [edge for edge in edges if edge["classification"] == RelationshipClassification.RESTRICTED.value]
		return {
			"tenant_id": tenant_id,
			"schema_count": len(self.list_schemas(tenant_id)),
			"node_count": len(nodes),
			"edge_count": len(edges),
			"restricted_edge_count": len(restricted_edges),
			"traversal_count": len(self.list_traversals(tenant_id)),
			"quality_report_count": len(self.list_quality_reports(tenant_id)),
		}

	def _enforce_graph_policy(
		self,
		tenant_id: str,
		operation: str,
		owner_assigned: bool = True,
		edge_type_present: bool = True,
		relationship_classification: str = RelationshipClassification.INTERNAL.value,
		review_recorded: bool = True,
		traversal_depth: int = 0,
		graph_type: str = GraphKind.PROPERTY.value,
		source_asset_present: bool = True,
	) -> None:
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": operation,
			"owner_assigned": owner_assigned,
			"edge_type_present": edge_type_present,
			"relationship_classification": relationship_classification,
			"review_recorded": review_recorded,
			"traversal_depth": traversal_depth,
			"graph_type": graph_type,
			"source_asset_present": source_asset_present,
		})
		if result["decision"] != "allow":
			reasons = ", ".join(action.get("reason", "graph_policy_blocked") for action in result["actions"])
			raise PermissionError(reasons or "graph_policy_blocked")

	def _require_schema(self, schema_id: str, tenant_id: str) -> None:
		schema = self._schemas.get(schema_id)
		if schema is None or schema.tenant_id != tenant_id:
			raise PermissionError("schema_missing")

	def _require_node(self, node_id: str, tenant_id: str) -> None:
		node = self._nodes.get(node_id)
		if node is None or node.tenant_id != tenant_id:
			raise PermissionError("node_missing")

	def _list(self, store: dict[str, Any], tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = list(store.values())
		if tenant_id is not None:
			items = [item for item in items if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(items, key=lambda item: item.id)]
