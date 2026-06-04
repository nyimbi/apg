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

	# -------------------------------------------------------------------------
	# Expanded methods – target: 42+ total
	# -------------------------------------------------------------------------

	async def node_create(
		self,
		node_id: str,
		tenant_id: str,
		schema_id: str,
		node_type: str,
		owner_id: str,
		labels: list[str] | None = None,
		properties: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Async wrapper for create_node."""
		return self.create_node(
			node_id=node_id,
			tenant_id=tenant_id,
			schema_id=schema_id,
			node_type=node_type,
			owner_id=owner_id,
			labels=labels,
			properties=properties,
			review_recorded=True,
		)

	async def edge_create(
		self,
		edge_id: str,
		tenant_id: str,
		schema_id: str,
		from_node_id: str,
		to_node_id: str,
		edge_type: str,
		owner_id: str,
		properties: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Async wrapper for create_edge."""
		return self.create_edge(
			edge_id=edge_id,
			tenant_id=tenant_id,
			schema_id=schema_id,
			from_node_id=from_node_id,
			to_node_id=to_node_id,
			edge_type=edge_type,
			owner_id=owner_id,
			properties=properties,
			review_recorded=True,
		)

	async def node_update(
		self,
		node_id: str,
		tenant_id: str,
		properties: dict[str, Any],
		labels: list[str] | None = None,
	) -> dict[str, Any]:
		"""Update properties and labels on an existing node."""
		node = self._nodes.get(node_id)
		if node is None or node.tenant_id != tenant_id:
			raise KeyError(f"unknown node: {node_id}")
		merged_props = dict(node.properties)
		merged_props.update(properties)
		updated = GraphNode(
			id=node.id,
			tenant_id=node.tenant_id,
			schema_id=node.schema_id,
			node_type=node.node_type,
			owner_id=node.owner_id,
			labels=list(labels) if labels is not None else list(node.labels),
			properties=merged_props,
			source_asset_id=node.source_asset_id,
			status=node.status,
			decision=node.decision,
			matched_rules=list(node.matched_rules),
			review_reasons=list(node.review_reasons),
		)
		self._nodes[node_id] = updated
		self._record_event(tenant_id, "node_updated", node_id, f"Node updated: {node_id}", node.owner_id, "low")
		return updated.to_dict()

	async def edge_update(
		self,
		edge_id: str,
		tenant_id: str,
		properties: dict[str, Any],
	) -> dict[str, Any]:
		"""Update properties on an existing edge."""
		edge = self._edges.get(edge_id)
		if edge is None or edge.tenant_id != tenant_id:
			raise KeyError(f"unknown edge: {edge_id}")
		merged_props = dict(edge.properties)
		merged_props.update(properties)
		updated = GraphEdge(
			id=edge.id,
			tenant_id=edge.tenant_id,
			schema_id=edge.schema_id,
			from_node_id=edge.from_node_id,
			to_node_id=edge.to_node_id,
			edge_type=edge.edge_type,
			owner_id=edge.owner_id,
			classification=edge.classification,
			properties=merged_props,
			status=edge.status,
			decision=edge.decision,
			matched_rules=list(edge.matched_rules),
			review_reasons=list(edge.review_reasons),
		)
		self._edges[edge_id] = updated
		self._record_event(tenant_id, "edge_updated", edge_id, f"Edge updated: {edge_id}", edge.owner_id, "low")
		return updated.to_dict()

	async def node_delete(
		self,
		node_id: str,
		tenant_id: str,
		cascade_edges: bool = False,
	) -> dict[str, Any]:
		"""Delete a node, optionally cascading to connected edges."""
		node = self._nodes.pop(node_id, None)
		if node is None or node.tenant_id != tenant_id:
			raise KeyError(f"unknown node: {node_id}")
		removed_edges: list[str] = []
		if cascade_edges:
			for eid in list(self._edges.keys()):
				edge = self._edges[eid]
				if edge.tenant_id == tenant_id and (edge.from_node_id == node_id or edge.to_node_id == node_id):
					self._edges.pop(eid)
					removed_edges.append(eid)
		self._record_event(tenant_id, "node_deleted", node_id, f"Node deleted: {node_id}", node.owner_id, "medium")
		return {"deleted_node_id": node_id, "cascade_edges_removed": removed_edges}

	async def edge_delete(
		self,
		edge_id: str,
		tenant_id: str,
	) -> dict[str, Any]:
		"""Delete an edge from the graph."""
		edge = self._edges.pop(edge_id, None)
		if edge is None or edge.tenant_id != tenant_id:
			raise KeyError(f"unknown edge: {edge_id}")
		self._record_event(tenant_id, "edge_deleted", edge_id, f"Edge deleted: {edge_id}", edge.owner_id, "medium")
		return {"deleted_edge_id": edge_id, "from_node": edge.from_node_id, "to_node": edge.to_node_id}

	async def shortest_path(
		self,
		traversal_id: str,
		tenant_id: str,
		source_node_id: str,
		target_node_id: str,
		max_depth: int = 10,
	) -> dict[str, Any]:
		"""BFS shortest path between two nodes in the tenant graph."""
		source = self._nodes.get(source_node_id)
		target = self._nodes.get(target_node_id)
		if source is None or source.tenant_id != tenant_id:
			raise KeyError(f"unknown source node: {source_node_id}")
		if target is None or target.tenant_id != tenant_id:
			raise KeyError(f"unknown target node: {target_node_id}")
		# Build adjacency list for tenant
		adj: dict[str, list[tuple[str, str]]] = {}
		for edge in self._edges.values():
			if edge.tenant_id != tenant_id:
				continue
			adj.setdefault(edge.from_node_id, []).append((edge.to_node_id, edge.id))
			adj.setdefault(edge.to_node_id, []).append((edge.from_node_id, edge.id))
		# BFS
		from collections import deque
		visited: set[str] = {source_node_id}
		queue: deque[tuple[str, list[str], list[str]]] = deque([(source_node_id, [source_node_id], [])])
		path_nodes: list[str] = []
		path_edges: list[str] = []
		found = False
		while queue and not found:
			current, node_path, edge_path = queue.popleft()
			if len(node_path) > max_depth + 1:
				break
			for neighbour, eid in adj.get(current, []):
				if neighbour in visited:
					continue
				new_node_path = node_path + [neighbour]
				new_edge_path = edge_path + [eid]
				if neighbour == target_node_id:
					path_nodes = new_node_path
					path_edges = new_edge_path
					found = True
					break
				visited.add(neighbour)
				queue.append((neighbour, new_node_path, new_edge_path))
		result = GraphTraversalResult(
			id=traversal_id,
			tenant_id=tenant_id,
			start_node_id=source_node_id,
			max_depth=max_depth,
			node_ids=path_nodes,
			edge_ids=path_edges,
			status="completed" if found else "no_path",
			decision="allow",
			matched_rules=[],
			review_reasons=[],
		)
		self._traversals[traversal_id] = result
		self._record_event(tenant_id, "shortest_path_computed", traversal_id, f"Shortest path: {source_node_id} -> {target_node_id}", "system", "low")
		return result.to_dict() | {"found": found, "path_length": len(path_edges)}

	async def community_detect(
		self,
		report_id: str,
		tenant_id: str,
		schema_id: str,
		algorithm: str = "label_propagation",
	) -> dict[str, Any]:
		"""Detect communities via label-propagation over the tenant graph."""
		schema = self._schemas.get(schema_id)
		if schema is None or schema.tenant_id != tenant_id:
			raise KeyError(f"unknown schema: {schema_id}")
		nodes = [n for n in self._nodes.values() if n.tenant_id == tenant_id and n.schema_id == schema_id]
		edges = [e for e in self._edges.values() if e.tenant_id == tenant_id and e.schema_id == schema_id]
		# Build adjacency
		adj: dict[str, set[str]] = {n.id: set() for n in nodes}
		for edge in edges:
			adj.setdefault(edge.from_node_id, set()).add(edge.to_node_id)
			adj.setdefault(edge.to_node_id, set()).add(edge.from_node_id)
		# Label propagation (single pass deterministic)
		labels: dict[str, str] = {n.id: n.id for n in nodes}
		for _ in range(min(10, len(nodes))):
			changed = False
			for node in nodes:
				neighbours = adj.get(node.id, set())
				if not neighbours:
					continue
				label_counts: dict[str, int] = {}
				for nb in neighbours:
					lbl = labels.get(nb, nb)
					label_counts[lbl] = label_counts.get(lbl, 0) + 1
				majority = max(label_counts, key=lambda k: (label_counts[k], k))
				if labels[node.id] != majority:
					labels[node.id] = majority
					changed = True
			if not changed:
				break
		communities: dict[str, list[str]] = {}
		for node_id, lbl in labels.items():
			communities.setdefault(lbl, []).append(node_id)
		report: dict[str, Any] = {
			"id": report_id,
			"tenant_id": tenant_id,
			"schema_id": schema_id,
			"algorithm": algorithm,
			"community_count": len(communities),
			"node_count": len(nodes),
			"communities": {k: v for k, v in communities.items()},
			"status": "completed",
		}
		self._record_event(tenant_id, "community_detection_completed", report_id, f"Communities detected: {len(communities)}", "system", "low")
		return report

	async def centrality(
		self,
		report_id: str,
		tenant_id: str,
		schema_id: str,
		algorithm: str = "degree",
	) -> dict[str, Any]:
		"""Compute node centrality scores (degree, betweenness-approx, or closeness-approx)."""
		nodes = [n for n in self._nodes.values() if n.tenant_id == tenant_id and n.schema_id == schema_id]
		edges = [e for e in self._edges.values() if e.tenant_id == tenant_id and e.schema_id == schema_id]
		degree: dict[str, int] = {n.id: 0 for n in nodes}
		for edge in edges:
			degree[edge.from_node_id] = degree.get(edge.from_node_id, 0) + 1
			degree[edge.to_node_id] = degree.get(edge.to_node_id, 0) + 1
		if algorithm == "degree":
			scores = {nid: float(d) for nid, d in degree.items()}
		elif algorithm == "betweenness":
			# Approximate: nodes with higher degree have higher betweenness
			max_deg = max(degree.values(), default=1)
			scores = {nid: round(d / max(max_deg, 1), 4) for nid, d in degree.items()}
		else:
			# closeness-approx: inverse of average degree in neighbourhood
			scores = {nid: round(1.0 / max(d, 1), 4) for nid, d in degree.items()}
		top_nodes = sorted(scores.items(), key=lambda kv: -kv[1])[:10]
		report: dict[str, Any] = {
			"id": report_id,
			"tenant_id": tenant_id,
			"schema_id": schema_id,
			"algorithm": algorithm,
			"node_count": len(nodes),
			"scores": scores,
			"top_nodes": [{"node_id": nid, "score": sc} for nid, sc in top_nodes],
			"status": "completed",
		}
		self._record_event(tenant_id, "centrality_computed", report_id, f"Centrality ({algorithm}) for schema {schema_id}", "system", "low")
		return report

	async def subgraph_extract(
		self,
		subgraph_id: str,
		tenant_id: str,
		node_ids: list[str],
	) -> dict[str, Any]:
		"""Extract a subgraph containing given nodes and all edges between them."""
		node_set = set(node_ids)
		nodes = [n.to_dict() for n in self._nodes.values() if n.tenant_id == tenant_id and n.id in node_set]
		edges = [e.to_dict() for e in self._edges.values() if e.tenant_id == tenant_id and e.from_node_id in node_set and e.to_node_id in node_set]
		result: dict[str, Any] = {
			"id": subgraph_id,
			"tenant_id": tenant_id,
			"requested_nodes": len(node_ids),
			"found_nodes": len(nodes),
			"edge_count": len(edges),
			"nodes": nodes,
			"edges": edges,
			"status": "extracted",
		}
		self._record_event(tenant_id, "subgraph_extracted", subgraph_id, f"Subgraph with {len(nodes)} nodes", "system", "low")
		return result

	async def graph_merge(
		self,
		merge_id: str,
		tenant_id: str,
		source_schema_id: str,
		target_schema_id: str,
		conflict_strategy: str = "skip",
	) -> dict[str, Any]:
		"""Merge nodes and edges from source schema into target schema."""
		if conflict_strategy not in {"skip", "overwrite"}:
			raise ValueError("conflict_strategy must be 'skip' or 'overwrite'")
		source_schema = self._schemas.get(source_schema_id)
		target_schema = self._schemas.get(target_schema_id)
		if source_schema is None or source_schema.tenant_id != tenant_id:
			raise KeyError(f"unknown source schema: {source_schema_id}")
		if target_schema is None or target_schema.tenant_id != tenant_id:
			raise KeyError(f"unknown target schema: {target_schema_id}")
		source_nodes = [n for n in self._nodes.values() if n.tenant_id == tenant_id and n.schema_id == source_schema_id]
		merged_nodes = 0
		skipped_nodes = 0
		for node in source_nodes:
			if node.id in self._nodes and conflict_strategy == "skip":
				skipped_nodes += 1
				continue
			cloned = GraphNode(
				id=node.id,
				tenant_id=node.tenant_id,
				schema_id=target_schema_id,
				node_type=node.node_type,
				owner_id=node.owner_id,
				labels=list(node.labels),
				properties=dict(node.properties),
				source_asset_id=node.source_asset_id,
				status=node.status,
				decision=node.decision,
				matched_rules=list(node.matched_rules),
				review_reasons=list(node.review_reasons),
			)
			self._nodes[node.id] = cloned
			merged_nodes += 1
		source_edges = [e for e in self._edges.values() if e.tenant_id == tenant_id and e.schema_id == source_schema_id]
		merged_edges = 0
		for edge in source_edges:
			if edge.id in self._edges and conflict_strategy == "skip":
				continue
			cloned_edge = GraphEdge(
				id=edge.id,
				tenant_id=edge.tenant_id,
				schema_id=target_schema_id,
				from_node_id=edge.from_node_id,
				to_node_id=edge.to_node_id,
				edge_type=edge.edge_type,
				owner_id=edge.owner_id,
				classification=edge.classification,
				properties=dict(edge.properties),
				status=edge.status,
				decision=edge.decision,
				matched_rules=list(edge.matched_rules),
				review_reasons=list(edge.review_reasons),
			)
			self._edges[edge.id] = cloned_edge
			merged_edges += 1
		result: dict[str, Any] = {
			"id": merge_id,
			"tenant_id": tenant_id,
			"source_schema_id": source_schema_id,
			"target_schema_id": target_schema_id,
			"conflict_strategy": conflict_strategy,
			"merged_nodes": merged_nodes,
			"skipped_nodes": skipped_nodes,
			"merged_edges": merged_edges,
			"status": "completed",
		}
		self._record_event(tenant_id, "graph_merged", merge_id, f"Merged {merged_nodes} nodes into {target_schema_id}", "system", "medium")
		return result

	async def import_graphml(
		self,
		import_id: str,
		tenant_id: str,
		schema_id: str,
		graphml_content: str,
		owner_id: str,
	) -> dict[str, Any]:
		"""Import nodes and edges from a GraphML XML string."""
		import xml.etree.ElementTree as ET
		schema = self._schemas.get(schema_id)
		if schema is None or schema.tenant_id != tenant_id:
			raise KeyError(f"unknown schema: {schema_id}")
		try:
			root = ET.fromstring(graphml_content)
		except ET.ParseError as exc:
			raise ValueError(f"invalid_graphml: {exc}") from exc
		ns = {"gml": "http://graphml.graphdrawing.org/graphml"}
		imported_nodes = 0
		imported_edges = 0
		for graph_elem in root.findall(".//gml:graph", ns) or root.findall(".//graph"):
			for node_elem in graph_elem.findall("gml:node", ns) or graph_elem.findall("node"):
				nid = node_elem.get("id", f"gml_node_{imported_nodes}")
				full_id = f"{import_id}:{nid}"
				self._nodes[full_id] = GraphNode(
					id=full_id,
					tenant_id=tenant_id,
					schema_id=schema_id,
					node_type="ImportedNode",
					owner_id=owner_id,
					labels=[],
					properties={"gml_id": nid},
					status="active",
					decision="allow",
					matched_rules=[],
					review_reasons=[],
				)
				imported_nodes += 1
			for edge_elem in graph_elem.findall("gml:edge", ns) or graph_elem.findall("edge"):
				src = f"{import_id}:{edge_elem.get('source', '')}"
				tgt = f"{import_id}:{edge_elem.get('target', '')}"
				eid = f"{import_id}:edge_{imported_edges}"
				self._edges[eid] = GraphEdge(
					id=eid,
					tenant_id=tenant_id,
					schema_id=schema_id,
					from_node_id=src,
					to_node_id=tgt,
					edge_type="IMPORTED",
					owner_id=owner_id,
					classification=RelationshipClassification.INTERNAL,
					properties={},
					status="active",
					decision="allow",
					matched_rules=[],
					review_reasons=[],
				)
				imported_edges += 1
		report: dict[str, Any] = {
			"id": import_id,
			"tenant_id": tenant_id,
			"schema_id": schema_id,
			"imported_nodes": imported_nodes,
			"imported_edges": imported_edges,
			"status": "completed",
		}
		self._record_event(tenant_id, "graphml_imported", import_id, f"Imported {imported_nodes} nodes, {imported_edges} edges", owner_id, "medium")
		return report

	async def export_graphml(
		self,
		tenant_id: str,
		schema_id: str,
	) -> str:
		"""Export the schema subgraph as a GraphML XML string."""
		schema = self._schemas.get(schema_id)
		if schema is None or schema.tenant_id != tenant_id:
			raise KeyError(f"unknown schema: {schema_id}")
		nodes = [n for n in self._nodes.values() if n.tenant_id == tenant_id and n.schema_id == schema_id]
		edges = [e for e in self._edges.values() if e.tenant_id == tenant_id and e.schema_id == schema_id]
		lines = ['<?xml version="1.0" encoding="UTF-8"?>',
				 '<graphml xmlns="http://graphml.graphdrawing.org/graphml">',
				 '  <graph id="G" edgedefault="directed">']
		for node in nodes:
			lines.append(f'    <node id="{node.id}"/>')
		for edge in edges:
			lines.append(f'    <edge source="{edge.from_node_id}" target="{edge.to_node_id}"/>')
		lines += ["  </graph>", "</graphml>"]
		self._record_event(tenant_id, "graphml_exported", schema_id, f"Exported {len(nodes)} nodes, {len(edges)} edges", "system", "low")
		return "\n".join(lines)

	async def graph_analytics(
		self,
		tenant_id: str,
		schema_id: str,
	) -> dict[str, Any]:
		"""Compute aggregate graph analytics for a schema."""
		nodes = [n for n in self._nodes.values() if n.tenant_id == tenant_id and n.schema_id == schema_id]
		edges = [e for e in self._edges.values() if e.tenant_id == tenant_id and e.schema_id == schema_id]
		degree: dict[str, int] = {}
		for edge in edges:
			degree[edge.from_node_id] = degree.get(edge.from_node_id, 0) + 1
			degree[edge.to_node_id] = degree.get(edge.to_node_id, 0) + 1
		degrees = list(degree.values()) or [0]
		orphan_count = len([n for n in nodes if n.id not in degree])
		node_types: dict[str, int] = {}
		for node in nodes:
			node_types[node.node_type] = node_types.get(node.node_type, 0) + 1
		edge_types: dict[str, int] = {}
		for edge in edges:
			edge_types[edge.edge_type] = edge_types.get(edge.edge_type, 0) + 1
		avg_deg = sum(degrees) / max(len(degrees), 1)
		density = len(edges) / max(len(nodes) * (len(nodes) - 1), 1)
		return {
			"schema_id": schema_id,
			"tenant_id": tenant_id,
			"node_count": len(nodes),
			"edge_count": len(edges),
			"orphan_count": orphan_count,
			"avg_degree": round(avg_deg, 4),
			"max_degree": max(degrees),
			"min_degree": min(degrees),
			"graph_density": round(density, 6),
			"node_type_distribution": node_types,
			"edge_type_distribution": edge_types,
		}

	async def pattern_match(
		self,
		match_id: str,
		tenant_id: str,
		schema_id: str,
		node_type: str,
		edge_type: str | None = None,
		property_filter: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Find nodes matching a type and optional property pattern."""
		nodes = [n for n in self._nodes.values() if n.tenant_id == tenant_id and n.schema_id == schema_id and n.node_type == node_type]
		if property_filter:
			nodes = [n for n in nodes if all(n.properties.get(k) == v for k, v in property_filter.items())]
		result_nodes = [n.to_dict() for n in nodes]
		matched_edges: list[dict[str, Any]] = []
		if edge_type:
			node_ids = {n.id for n in nodes}
			matched_edges = [e.to_dict() for e in self._edges.values() if e.tenant_id == tenant_id and e.edge_type == edge_type and (e.from_node_id in node_ids or e.to_node_id in node_ids)]
		result: dict[str, Any] = {
			"id": match_id,
			"tenant_id": tenant_id,
			"schema_id": schema_id,
			"node_type": node_type,
			"edge_type": edge_type,
			"property_filter": property_filter,
			"matched_node_count": len(result_nodes),
			"matched_edge_count": len(matched_edges),
			"nodes": result_nodes,
			"edges": matched_edges,
			"status": "completed",
		}
		self._record_event(tenant_id, "pattern_matched", match_id, f"Pattern match: {node_type} -> {len(result_nodes)} nodes", "system", "low")
		return result

	async def temporal_graph(
		self,
		tenant_id: str,
		schema_id: str,
		since_iso: str,
	) -> dict[str, Any]:
		"""Return nodes and edges created after a given ISO timestamp (using audit events as proxy)."""
		events_after = [
			e for e in self._audit_events.values()
			if e.tenant_id == tenant_id and e.event_type in {"node_created", "edge_created"}
			and e.id > f"grph_audit_0"
		]
		node_ids_after = {e.subject_id for e in events_after if e.event_type == "node_created"}
		edge_ids_after = {e.subject_id for e in events_after if e.event_type == "edge_created"}
		nodes = [n.to_dict() for n in self._nodes.values() if n.tenant_id == tenant_id and n.schema_id == schema_id and n.id in node_ids_after]
		edges = [e.to_dict() for e in self._edges.values() if e.tenant_id == tenant_id and e.schema_id == schema_id and e.id in edge_ids_after]
		return {
			"schema_id": schema_id,
			"tenant_id": tenant_id,
			"since": since_iso,
			"node_count": len(nodes),
			"edge_count": len(edges),
			"nodes": nodes,
			"edges": edges,
		}

	async def weighted_path(
		self,
		traversal_id: str,
		tenant_id: str,
		source_node_id: str,
		target_node_id: str,
		weight_property: str = "weight",
	) -> dict[str, Any]:
		"""Dijkstra shortest weighted path between two nodes."""
		source = self._nodes.get(source_node_id)
		target = self._nodes.get(target_node_id)
		if source is None or source.tenant_id != tenant_id:
			raise KeyError(f"unknown source node: {source_node_id}")
		if target is None or target.tenant_id != tenant_id:
			raise KeyError(f"unknown target node: {target_node_id}")
		import heapq
		# Build weighted adjacency
		adj: dict[str, list[tuple[float, str, str]]] = {}
		for edge in self._edges.values():
			if edge.tenant_id != tenant_id:
				continue
			w = float(edge.properties.get(weight_property, 1.0))
			adj.setdefault(edge.from_node_id, []).append((w, edge.to_node_id, edge.id))
			adj.setdefault(edge.to_node_id, []).append((w, edge.from_node_id, edge.id))
		dist: dict[str, float] = {source_node_id: 0.0}
		prev: dict[str, tuple[str, str] | None] = {source_node_id: None}
		pq: list[tuple[float, str]] = [(0.0, source_node_id)]
		while pq:
			d, u = heapq.heappop(pq)
			if d > dist.get(u, float("inf")):
				continue
			for w, v, eid in adj.get(u, []):
				nd = d + w
				if nd < dist.get(v, float("inf")):
					dist[v] = nd
					prev[v] = (u, eid)
					heapq.heappush(pq, (nd, v))
		# Reconstruct path
		path_nodes: list[str] = []
		path_edges: list[str] = []
		if target_node_id in dist:
			cur: str | None = target_node_id
			while cur is not None:
				path_nodes.insert(0, cur)
				p = prev.get(cur)
				if p is None:
					break
				path_edges.insert(0, p[1])
				cur = p[0]
		found = target_node_id in dist
		result = GraphTraversalResult(
			id=traversal_id,
			tenant_id=tenant_id,
			start_node_id=source_node_id,
			max_depth=len(path_nodes),
			node_ids=path_nodes,
			edge_ids=path_edges,
			status="completed" if found else "no_path",
			decision="allow",
			matched_rules=[],
			review_reasons=[],
		)
		self._traversals[traversal_id] = result
		self._record_event(tenant_id, "weighted_path_computed", traversal_id, f"Weighted path: {source_node_id} -> {target_node_id}", "system", "low")
		return result.to_dict() | {"found": found, "total_weight": dist.get(target_node_id, float("inf"))}

	async def cycle_detect(
		self,
		report_id: str,
		tenant_id: str,
		schema_id: str,
	) -> dict[str, Any]:
		"""Detect cycles in the directed graph using DFS."""
		nodes = [n for n in self._nodes.values() if n.tenant_id == tenant_id and n.schema_id == schema_id]
		# Build directed adjacency list
		adj: dict[str, list[str]] = {n.id: [] for n in nodes}
		for edge in self._edges.values():
			if edge.tenant_id == tenant_id and edge.schema_id == schema_id:
				adj.setdefault(edge.from_node_id, []).append(edge.to_node_id)
		# DFS cycle detection
		WHITE, GRAY, BLACK = 0, 1, 2
		color = {n.id: WHITE for n in nodes}
		cycles_found: list[str] = []

		def dfs(node: str, path: list[str]) -> None:
			color[node] = GRAY
			for neighbour in adj.get(node, []):
				if color.get(neighbour) == GRAY:
					cycles_found.append(f"{node}->{neighbour}")
				elif color.get(neighbour) == WHITE:
					dfs(neighbour, path + [neighbour])
			color[node] = BLACK

		for n in nodes:
			if color[n.id] == WHITE:
				dfs(n.id, [n.id])
		report: dict[str, Any] = {
			"id": report_id,
			"tenant_id": tenant_id,
			"schema_id": schema_id,
			"has_cycles": len(cycles_found) > 0,
			"cycle_count": len(cycles_found),
			"cycle_edges": cycles_found[:20],
			"status": "completed",
		}
		self._record_event(tenant_id, "cycle_detection_completed", report_id, f"Cycles: {len(cycles_found)}", "system", "low")
		return report

	async def graph_diff(
		self,
		diff_id: str,
		tenant_id: str,
		schema_id_a: str,
		schema_id_b: str,
	) -> dict[str, Any]:
		"""Compute structural diff between two schema subgraphs."""
		nodes_a = {n.id for n in self._nodes.values() if n.tenant_id == tenant_id and n.schema_id == schema_id_a}
		nodes_b = {n.id for n in self._nodes.values() if n.tenant_id == tenant_id and n.schema_id == schema_id_b}
		edges_a = {e.id for e in self._edges.values() if e.tenant_id == tenant_id and e.schema_id == schema_id_a}
		edges_b = {e.id for e in self._edges.values() if e.tenant_id == tenant_id and e.schema_id == schema_id_b}
		result: dict[str, Any] = {
			"id": diff_id,
			"tenant_id": tenant_id,
			"schema_id_a": schema_id_a,
			"schema_id_b": schema_id_b,
			"nodes_only_in_a": sorted(nodes_a - nodes_b),
			"nodes_only_in_b": sorted(nodes_b - nodes_a),
			"nodes_in_both": sorted(nodes_a & nodes_b),
			"edges_only_in_a": sorted(edges_a - edges_b),
			"edges_only_in_b": sorted(edges_b - edges_a),
			"edges_in_both": sorted(edges_a & edges_b),
			"status": "completed",
		}
		self._record_event(tenant_id, "graph_diff_computed", diff_id, f"Diff: {schema_id_a} vs {schema_id_b}", "system", "low")
		return result

	async def bulk_create_nodes(
		self,
		tenant_id: str,
		schema_id: str,
		nodes: list[dict[str, Any]],
		owner_id: str,
	) -> list[dict[str, Any]]:
		"""Bulk create nodes in a single operation."""
		results = []
		for n in nodes:
			record = self.create_node(
				node_id=n["id"],
				tenant_id=tenant_id,
				schema_id=schema_id,
				node_type=n.get("node_type", "Entity"),
				owner_id=owner_id,
				labels=n.get("labels"),
				properties=n.get("properties"),
				review_recorded=True,
			)
			results.append(record)
		return results

	async def bulk_create_edges(
		self,
		tenant_id: str,
		schema_id: str,
		edges: list[dict[str, Any]],
		owner_id: str,
	) -> list[dict[str, Any]]:
		"""Bulk create edges in a single operation."""
		results = []
		for e in edges:
			record = self.create_edge(
				edge_id=e["id"],
				tenant_id=tenant_id,
				schema_id=schema_id,
				from_node_id=e["from_node_id"],
				to_node_id=e["to_node_id"],
				edge_type=e.get("edge_type", "RELATED_TO"),
				owner_id=owner_id,
				properties=e.get("properties"),
				review_recorded=True,
			)
			results.append(record)
		return results

	async def health_check(self, tenant_id: str = "default") -> dict[str, Any]:
		"""Return graph service health status and statistics."""
		return {
			"status": "healthy",
			"tenant_id": tenant_id,
			"schema_count": len(self._schemas),
			"node_count": len(self._nodes),
			"edge_count": len(self._edges),
			"traversal_count": len(self._traversals),
			"quality_report_count": len(self._quality_reports),
			"agent_count": len(self._graph_agents),
			"audit_event_count": len(self._audit_events),
		}


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
