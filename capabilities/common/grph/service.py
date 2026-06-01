"""Executable graph service for APG graph composition."""

from __future__ import annotations

from itertools import count
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract
from .graph_runtime import GraphQualityInspector, GraphTraversalPlanner
from .models import (
	GraphAuditEventRecord,
	GraphAgentRecord,
	GraphEdge,
	GraphKind,
	GrphLifecycleBatchRecord,
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
		self._graph_agents: dict[str, GraphAgentRecord] = {}
		self._lifecycle_batches: dict[str, GrphLifecycleBatchRecord] = {}
		self._audit_events: dict[str, GraphAuditEventRecord] = {}
		self._counter = count(1)
		self._traversal_planner = GraphTraversalPlanner()
		self._quality_inspector = GraphQualityInspector()
		contract = get_capability_contract()
		self._agent_runtimes = set(contract["agents"]["supported_runtimes"])
		self._agent_roles = set(contract["agents"]["supported_roles"])
		self._privileged_agent_roles = set(contract["agents"]["privileged_roles"])
		self._lifecycle_operations = set(contract["streaming"]["required_operations"])

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
		review_recorded: bool = False,
	) -> dict[str, Any]:
		config = self.describe(tenant_id or "default")["configuration"]
		kind_value = str(graph_kind or "").strip().lower()
		kind_known = kind_value in config["schemas"]["allowed_graph_kinds"]
		stored_kind = kind_value if kind_known else GraphKind.PROPERTY.value
		node_type_count = len(node_types or {})
		edge_type_count = len(edge_types or {})
		result = self.evaluate({
			"tenant_context_present": bool(str(tenant_id or "").strip()),
			"operation": "write_schema",
			"schema_id_present": bool(str(schema_id or "").strip()),
			"schema_name_present": bool(str(name or "").strip()),
			"graph_kind_present": bool(kind_value),
			"graph_kind_known": kind_known,
			"graph_type": stored_kind,
			"source_asset_present": bool(str(source_asset_id or "").strip()),
			"node_type_count": node_type_count,
			"edge_type_count": edge_type_count,
			"review_recorded": bool(review_recorded),
		})
		self._raise_if_denied(result)
		status = "pending_review" if result["decision"] == "require_review" else "active"
		schema = GraphSchema(
			id=schema_id,
			tenant_id=tenant_id,
			name=name,
			graph_kind=GraphKind(stored_kind),
			node_types={key: list(value) for key, value in (node_types or {}).items()},
			edge_types={key: dict(value) for key, value in (edge_types or {}).items()},
			source_asset_id=source_asset_id,
			status=status,
			decision=result["decision"],
			matched_rules=list(result["matched_rules"]),
			review_reasons=list(_review_reasons(result)),
		)
		self._schemas[schema_id] = schema
		self._record_event(
			tenant_id,
			"schema_created",
			schema.id,
			f"Graph schema created: {name}",
			"system",
			"medium" if status == "pending_review" else "low",
			_rule_evidence(result),
		)
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
		review_recorded: bool = False,
	) -> dict[str, Any]:
		schema = self._schemas.get(schema_id)
		config = self.describe(tenant_id or "default")["configuration"]
		label_values = [str(label) for label in (labels or [])]
		allowed_prefixes = tuple(config["nodes"]["allowed_label_prefixes"])
		labels_allowed = all(label.startswith(allowed_prefixes) for label in label_values) if label_values else True
		node_type_allowed = bool(schema and (not schema.node_types or node_type in schema.node_types))
		result = self.evaluate({
			"tenant_context_present": bool(str(tenant_id or "").strip()),
			"operation": "write_node",
			"schema_present": bool(schema and schema.tenant_id == tenant_id),
			"node_id_present": bool(str(node_id or "").strip()),
			"node_type_present": bool(str(node_type or "").strip()),
			"owner_assigned": bool(str(owner_id or "").strip()),
			"node_type_allowed": node_type_allowed,
			"labels_allowed": labels_allowed,
			"review_recorded": bool(review_recorded),
		})
		self._raise_if_denied(result)
		assert schema is not None
		status = "pending_review" if result["decision"] == "require_review" else "active"
		node = GraphNode(
			id=node_id,
			tenant_id=tenant_id,
			schema_id=schema_id,
			node_type=node_type,
			owner_id=owner_id,
			labels=label_values,
			properties=dict(properties or {}),
			source_asset_id=source_asset_id,
			status=status,
			decision=result["decision"],
			matched_rules=list(result["matched_rules"]),
			review_reasons=list(_review_reasons(result)),
		)
		self._nodes[node_id] = node
		self._record_event(
			tenant_id,
			"node_created",
			node.id,
			f"Graph node created: {node_id}",
			owner_id,
			"medium" if status == "pending_review" else "low",
			_rule_evidence(result),
		)
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
		review_recorded: bool = False,
	) -> dict[str, Any]:
		schema = self._schemas.get(schema_id)
		source_any = self._nodes.get(from_node_id)
		target_any = self._nodes.get(to_node_id)
		source = source_any if source_any and source_any.tenant_id == tenant_id else None
		target = target_any if target_any and target_any.tenant_id == tenant_id else None
		edge_type_allowed = bool(schema and (not schema.edge_types or edge_type in schema.edge_types))
		cross_tenant_edge = bool(
			source_any
			and target_any
			and (source_any.tenant_id != tenant_id or target_any.tenant_id != tenant_id or source_any.tenant_id != target_any.tenant_id)
		)
		classification_value = str(classification or "").strip().lower()
		allowed_classifications = self.describe(tenant_id or "default")["configuration"]["edges"]["allowed_classifications"]
		classification_known = classification_value in allowed_classifications
		stored_classification = classification_value if classification_known else RelationshipClassification.RESTRICTED.value
		result = self.evaluate({
			"tenant_context_present": bool(str(tenant_id or "").strip()),
			"operation": "write_edge",
			"schema_present": bool(schema and schema.tenant_id == tenant_id),
			"edge_id_present": bool(str(edge_id or "").strip()),
			"source_node_present": bool(source),
			"target_node_present": bool(target),
			"edge_type_present": bool(str(edge_type or "").strip()),
			"owner_assigned": bool(str(owner_id or "").strip()),
			"classification_present": bool(classification_value),
			"classification_known": classification_known,
			"edge_type_allowed": edge_type_allowed,
			"cross_tenant_edge": cross_tenant_edge,
			"relationship_classification": stored_classification,
			"self_edge": bool(from_node_id and from_node_id == to_node_id),
			"review_recorded": bool(review_recorded),
		})
		self._raise_if_denied(result)
		assert schema is not None
		status = "pending_review" if result["decision"] == "require_review" else "active"
		edge = GraphEdge(
			id=edge_id,
			tenant_id=tenant_id,
			schema_id=schema_id,
			from_node_id=from_node_id,
			to_node_id=to_node_id,
			edge_type=edge_type,
			owner_id=owner_id,
			classification=RelationshipClassification(stored_classification),
			properties=dict(properties or {}),
			status=status,
			decision=result["decision"],
			matched_rules=list(result["matched_rules"]),
			review_reasons=list(_review_reasons(result)),
		)
		self._edges[edge_id] = edge
		self._record_event(
			tenant_id,
			"edge_created",
			edge.id,
			f"Graph edge created: {edge_type}",
			owner_id,
			"medium" if status == "pending_review" else "low",
			_rule_evidence(result),
		)
		return edge.to_dict()

	def traverse(
		self,
		traversal_id: str,
		tenant_id: str,
		start_node_id: str,
		max_depth: int = 1,
		review_recorded: bool = False,
		rbac_filter_applied: bool = True,
	) -> dict[str, Any]:
		start_node = self._nodes.get(start_node_id)
		restricted_in_scope = any(
			edge.tenant_id == tenant_id and edge.classification == RelationshipClassification.RESTRICTED
			for edge in self._edges.values()
		)
		result = self.evaluate({
			"tenant_context_present": bool(str(tenant_id or "").strip()),
			"operation": "traverse",
			"start_node_present": bool(start_node and start_node.tenant_id == tenant_id),
			"traversal_depth": int(max_depth),
			"review_recorded": bool(review_recorded),
			"restricted_relationships_in_scope": restricted_in_scope,
			"rbac_filter_applied": bool(rbac_filter_applied),
		})
		self._raise_if_denied(result)
		node_ids, edge_ids = self._traversal_planner.traverse(
			tenant_id=tenant_id,
			start_node_id=start_node_id,
			max_depth=max_depth,
			edges=list(self._edges.values()),
		)
		traversal = GraphTraversalResult(
			id=traversal_id,
			tenant_id=tenant_id,
			start_node_id=start_node_id,
			max_depth=max_depth,
			node_ids=node_ids,
			edge_ids=edge_ids,
			status="pending_review" if result["decision"] == "require_review" else "completed",
			decision=result["decision"],
			matched_rules=list(result["matched_rules"]),
			review_reasons=list(_review_reasons(result)),
		)
		self._traversals[traversal_id] = traversal
		self._record_event(
			tenant_id,
			"traversal_completed",
			traversal.id,
			f"Graph traversal completed: {traversal_id}",
			"system",
			"medium" if traversal.status == "pending_review" else "low",
			_rule_evidence(result),
		)
		return traversal.to_dict()

	def lineage_path(
		self,
		traversal_id: str,
		tenant_id: str,
		source_asset_id: str,
		start_node_id: str,
		max_depth: int = 2,
		review_recorded: bool = False,
		rbac_filter_applied: bool = True,
	) -> dict[str, Any]:
		result = self.evaluate({
			"tenant_context_present": bool(str(tenant_id or "").strip()),
			"operation": "lineage_query",
			"source_asset_present": bool(str(source_asset_id or "").strip()),
		})
		self._raise_if_blocked(result)
		return self.traverse(traversal_id, tenant_id, start_node_id, max_depth, review_recorded, rbac_filter_applied)

	def impact_analysis(
		self,
		traversal_id: str,
		tenant_id: str,
		start_node_id: str,
		max_depth: int = 3,
		review_recorded: bool = False,
		rbac_filter_applied: bool = True,
	) -> dict[str, Any]:
		return self.traverse(traversal_id, tenant_id, start_node_id, max_depth, review_recorded, rbac_filter_applied)

	def neighborhood(
		self,
		traversal_id: str,
		tenant_id: str,
		start_node_id: str,
		review_recorded: bool = False,
		rbac_filter_applied: bool = True,
	) -> dict[str, Any]:
		return self.traverse(traversal_id, tenant_id, start_node_id, 1, review_recorded, rbac_filter_applied)

	def quality_report(
		self,
		report_id: str,
		tenant_id: str,
		schema_id: str,
		review_recorded: bool = False,
	) -> dict[str, Any]:
		schema = self._schemas.get(schema_id)
		if schema is None or schema.tenant_id != tenant_id:
			raise PermissionError("schema_missing")
		schema_nodes = [node for node in self._nodes.values() if node.tenant_id == tenant_id and node.schema_id == schema_id]
		schema_edges = [edge for edge in self._edges.values() if edge.tenant_id == tenant_id and edge.schema_id == schema_id]
		metrics = self._quality_inspector.inspect(schema_nodes, schema_edges)
		quality_issue_count = metrics["orphan_node_count"] + metrics["missing_owner_count"]
		result = self.evaluate({
			"tenant_context_present": bool(str(tenant_id or "").strip()),
			"operation": "quality_report",
			"quality_issue_count": quality_issue_count,
			"review_recorded": bool(review_recorded),
		})
		self._raise_if_denied(result)
		status = "pending_review" if result["decision"] == "require_review" else (
			"attention_required" if metrics["orphan_node_count"] or metrics["missing_owner_count"] else "healthy"
		)
		report = GraphQualityReport(
			id=report_id,
			tenant_id=tenant_id,
			schema_id=schema_id,
			orphan_node_count=metrics["orphan_node_count"],
			missing_owner_count=metrics["missing_owner_count"],
			restricted_edge_count=metrics["restricted_edge_count"],
			status=status,
			decision=result["decision"],
			matched_rules=list(result["matched_rules"]),
			review_reasons=list(_review_reasons(result)),
		)
		self._quality_reports[report_id] = report
		self._record_event(
			tenant_id,
			"quality_report_created",
			report.id,
			f"Graph quality report created: {report_id}",
			"system",
			"medium" if status == "pending_review" else "low",
			_rule_evidence(result),
		)
		return report.to_dict()

	def retire_schema(self, tenant_id: str, schema_id: str, review_recorded: bool = False) -> dict[str, Any]:
		self._raise_if_blocked(self.evaluate({
			"tenant_context_present": bool(str(tenant_id or "").strip()),
			"operation": "retire_schema",
			"review_recorded": bool(review_recorded),
		}))
		schema = self._schemas.pop(schema_id, None)
		if schema is None or schema.tenant_id != tenant_id:
			raise PermissionError("schema_missing")
		self._record_event(tenant_id, "schema_retired", schema_id, f"Graph schema retired: {schema_id}", "system", "medium")
		return schema.to_dict() | {"status": "retired"}

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
		node_type = str(data.get("node_type") or "Entity")
		edge_type = str(data.get("edge_type") or "RELATED_TO")
		if schema_id not in self._schemas:
			self.create_schema(
				schema_id=schema_id,
				tenant_id=tenant_id,
				name=str(data.get("schema_name") or "Compatibility Graph"),
				node_types={node_type: []},
				edge_types={edge_type: {}},
				review_recorded=True,
			)
		return self.create_node(
			node_id=record_id,
			tenant_id=tenant_id,
			schema_id=schema_id,
			node_type=node_type,
			owner_id=str(data.get("owner_id") or "system"),
			labels=list(data.get("labels") or [f"entity-{status}"]),
			properties=data,
			review_recorded=True,
		)

	def register_graph_agent(
		self,
		agent_id: str,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
		owner: str,
		purpose: str,
		contribution_disclosed: bool = True,
		human_approval_required: bool = False,
	) -> dict[str, Any]:
		runtime_value = _normalize_token(runtime)
		role_value = _normalize_token(role)
		result = self.evaluate({
			"tenant_context_present": bool(str(tenant_id or "").strip()),
			"operation": "register_graph_agent",
			"agent_runtime_supported": runtime_value in self._agent_runtimes,
			"agent_role_supported": role_value in self._agent_roles,
			"scope_present": bool(str(scope or "").strip()),
			"owner_present": bool(str(owner or "").strip()),
			"purpose_present": bool(str(purpose or "").strip()),
			"contribution_disclosed": bool(contribution_disclosed),
			"privileged_role": role_value in self._privileged_agent_roles,
			"human_approval_required": bool(human_approval_required),
		})
		self._raise_if_denied(result)
		if not str(agent_id or "").strip():
			raise ValueError("graph_agent_id_required")
		if not str(name or "").strip():
			raise ValueError("graph_agent_name_required")
		status = "pending_review" if result["decision"] == "require_review" else "active"
		record = GraphAgentRecord(
			id=str(agent_id).strip(),
			tenant_id=tenant_id,
			name=str(name).strip(),
			runtime=runtime_value,
			role=role_value,
			scope=str(scope).strip(),
			owner=str(owner).strip(),
			purpose=str(purpose).strip(),
			contribution_disclosed=bool(contribution_disclosed),
			human_approval_required=bool(human_approval_required),
			status=status,
			decision=result["decision"],
			matched_rules=list(result["matched_rules"]),
			review_reasons=list(_review_reasons(result)),
		)
		self._graph_agents[self._tenant_record_key(tenant_id, record.id)] = record
		self._record_event(
			tenant_id,
			"graph_agent_registered",
			record.id,
			f"Graph agent registered: {name}",
			owner,
			"medium" if status == "pending_review" else "low",
			_rule_evidence(result),
		)
		return record.to_dict()

	def validate_grph_lifecycle_batch(
		self,
		tenant_id: str,
		event_stream: str,
		mutation_count: int,
		operation: str = "graph_agent_batch",
		batch_id: str | None = None,
	) -> dict[str, Any]:
		mutation_count = int(mutation_count)
		if mutation_count <= 0:
			raise ValueError("grph_lifecycle_batch_empty")
		stream_value = _normalize_token(event_stream)
		operation_value = _normalize_token(operation)
		if operation_value not in self._lifecycle_operations:
			raise ValueError(f"unsupported_grph_lifecycle_operation:{operation_value}")
		result = self.evaluate({
			"tenant_context_present": bool(str(tenant_id or "").strip()),
			"operation": "validate_grph_lifecycle_batch",
			"event_stream": stream_value,
		})
		accepted = result["decision"] == "allow"
		record = GrphLifecycleBatchRecord(
			id=batch_id or f"grphbatch:{len(self._lifecycle_batches) + 1:06d}",
			tenant_id=tenant_id,
			event_stream=stream_value,
			mutation_count=mutation_count,
			operation=operation_value,
			accepted=accepted,
			decision=result["decision"],
			matched_rules=list(result["matched_rules"]),
			review_reasons=list(_review_reasons(result)),
			status="accepted" if accepted else "denied",
		)
		self._lifecycle_batches[self._tenant_record_key(tenant_id, record.id)] = record
		self._record_event(
			tenant_id,
			f"grph_lifecycle_batch_{record.status}",
			record.id,
			f"Validated GRPH lifecycle batch: {record.id}",
			"grph",
			"medium" if not accepted else "low",
			_rule_evidence(result),
		)
		if not accepted:
			self._raise_if_denied(result)
		return record.to_dict()

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

	def list_graph_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._graph_agents, tenant_id)

	def list_lifecycle_batches(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._lifecycle_batches, tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._audit_events, tenant_id)

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		records: list[dict[str, Any]] = []
		for store in (self._schemas, self._nodes, self._edges, self._traversals, self._quality_reports, self._graph_agents, self._lifecycle_batches):
			records.extend(self._list(store, tenant_id))
		return sorted(records, key=lambda item: (item["kind"], item["id"]))

	def dashboard_summary(self, tenant_id: str | None = None) -> dict[str, Any]:
		schemas = self.list_schemas(tenant_id)
		nodes = self.list_nodes(tenant_id)
		edges = self.list_edges(tenant_id)
		traversals = self.list_traversals(tenant_id)
		quality_reports = self.list_quality_reports(tenant_id)
		agents = self.list_graph_agents(tenant_id)
		lifecycle_batches = self.list_lifecycle_batches(tenant_id)
		restricted_edges = [edge for edge in edges if edge["classification"] == RelationshipClassification.RESTRICTED.value]
		return {
			"tenant_id": tenant_id,
			"schema_count": len(schemas),
			"node_count": len(nodes),
			"edge_count": len(edges),
			"restricted_edge_count": len(restricted_edges),
			"traversal_count": len(traversals),
			"quality_report_count": len(quality_reports),
			"graph_agent_count": len(agents),
			"pending_schema_review_count": len([item for item in schemas if item["status"] == "pending_review"]),
			"pending_node_review_count": len([item for item in nodes if item["status"] == "pending_review"]),
			"pending_edge_review_count": len([item for item in edges if item["status"] == "pending_review"]),
			"pending_traversal_review_count": len([item for item in traversals if item["status"] == "pending_review"]),
			"pending_quality_review_count": len([item for item in quality_reports if item["status"] == "pending_review"]),
			"pending_agent_review_count": len([item for item in agents if item["status"] == "pending_review"]),
			"lifecycle_batch_count": len(lifecycle_batches),
			"denied_lifecycle_batch_count": len([item for item in lifecycle_batches if item["status"] == "denied"]),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
		}

	def _raise_if_blocked(self, result: dict[str, Any]) -> None:
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", "graph_policy_blocked") for action in result["actions"])
		raise PermissionError(reasons or "graph_policy_blocked")

	def _raise_if_denied(self, result: dict[str, Any]) -> None:
		if result["decision"] == "deny":
			reasons = ", ".join(action.get("reason", "graph_policy_blocked") for action in result["actions"])
			raise PermissionError(reasons or "graph_policy_blocked")

	def _record_event(
		self,
		tenant_id: str,
		event_type: str,
		subject_id: str,
		message: str,
		actor: str,
		severity: str = "low",
		evidence: dict[str, Any] | None = None,
	) -> None:
		record = GraphAuditEventRecord(
			id=f"grph_audit_{next(self._counter)}",
			tenant_id=tenant_id,
			event_type=event_type,
			subject_id=subject_id,
			message=message,
			actor=actor,
			severity=severity,
			evidence=dict(evidence or {}),
		)
		self._audit_events[record.id] = record

	def _list(self, store: dict[str, Any], tenant_id: str | None = None) -> list[dict[str, Any]]:
		items = list(store.values())
		if tenant_id is not None:
			items = [item for item in items if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(items, key=lambda item: item.id)]

	def _tenant_record_key(self, tenant_id: str, record_id: str) -> str:
		return f"{tenant_id}:{record_id}"


def _normalize_token(value: str) -> str:
	return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _review_reasons(result: dict[str, Any]) -> tuple[str, ...]:
	return tuple(
		str(action["reason"])
		for action in result.get("actions", [])
		if action.get("decision") == "require_review" and action.get("reason")
	)


def _rule_evidence(result: dict[str, Any]) -> dict[str, Any]:
	return {
		"decision": result["decision"],
		"matched_rules": list(result["matched_rules"]),
		"review_reasons": list(_review_reasons(result)),
	}
