"""APG graph extraction and rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from compiler.ast_builder import (
	ASTBuilder,
	AIAgentDeclaration,
	AgentTeamDeclaration,
	ApplicationDeclaration,
	CapabilityDeclaration,
	DatabaseDeclaration,
	EntityDeclaration,
	EntityType,
	ModuleDeclaration,
)
from compiler.parser import APGParser


@dataclass(frozen=True)
class GraphNode:
	id: str
	kind: str
	label: str
	metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphEdge:
	source: str
	target: str
	kind: str
	label: str
	metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class APGGraph:
	kind: str
	source: str
	nodes: list[GraphNode]
	edges: list[GraphEdge]

	def to_dict(self) -> dict[str, Any]:
		return {
			"format": "apg.graph.v1",
			"kind": self.kind,
			"source": self.source,
			"nodes": [
				{
					"id": node.id,
					"kind": node.kind,
					"label": node.label,
					"metadata": node.metadata,
				}
				for node in self.nodes
			],
			"edges": [
				{
					"source": edge.source,
					"target": edge.target,
					"kind": edge.kind,
					"label": edge.label,
					"metadata": edge.metadata,
				}
				for edge in self.edges
			],
		}


def parse_apg_module(path: Path) -> ModuleDeclaration:
	parser = APGParser()
	parse_result = parser.parse_file(str(path))
	if not parse_result.get("success"):
		message = "; ".join(str(error) for error in parse_result.get("errors", []))
		raise ValueError(message or f"Failed to parse {path}")
	ast = parse_result.get("ast") or ASTBuilder().build_ast(parse_result["parse_tree"], str(path))
	if ast is None:
		raise ValueError(f"Failed to build APG AST for {path}")
	return ast


def build_graph(path: Path, kind: str) -> APGGraph:
	module = parse_apg_module(path)
	normalized = kind.lower().replace("_", "-")
	if normalized in {"er", "entity-relationship"}:
		return _entity_relationship_graph(module, path)
	if normalized == "agent":
		return _agent_graph(module, path)
	if normalized == "capability":
		return _capability_graph(module, path)
	if normalized in {"handler", "workflow", "lookup", "security", "deployment", "package"}:
		return _generic_entity_graph(module, path, normalized)
	raise ValueError(f"Unsupported graph kind: {kind}")


def render_mermaid(graph: APGGraph) -> str:
	lines = ["graph TD"]
	for node in graph.nodes:
		lines.append(f"  {_mermaid_id(node.id)}[\"{_escape_label(node.label)}\"]")
	for edge in graph.edges:
		lines.append(
			f"  {_mermaid_id(edge.source)} -->|{_escape_label(edge.label)}| {_mermaid_id(edge.target)}"
		)
	return "\n".join(lines) + "\n"


def render_dot(graph: APGGraph) -> str:
	lines = [f"digraph {_dot_id(graph.kind)} {{"]
	for node in graph.nodes:
		lines.append(f'  "{node.id}" [label="{_escape_dot(node.label)}", kind="{node.kind}"];')
	for edge in graph.edges:
		lines.append(
			f'  "{edge.source}" -> "{edge.target}" '
			f'[label="{_escape_dot(edge.label)}", kind="{edge.kind}"];'
		)
	lines.append("}")
	return "\n".join(lines) + "\n"


def _entity_relationship_graph(module: ModuleDeclaration, path: Path) -> APGGraph:
	nodes: dict[str, GraphNode] = {}
	edges: list[GraphEdge] = []
	entity_names = {entity.name for entity in module.entities}
	for entity in module.entities:
		if entity.entity_type not in {EntityType.ENTITY, EntityType.DATABASE}:
			continue
		node_id = f"entity:{entity.name}"
		nodes[node_id] = GraphNode(
			id=node_id,
			kind="entity",
			label=entity.name,
			metadata={"property_count": len(entity.properties)},
		)
		for prop in entity.properties:
			type_name = prop.type_annotation.type_name
			field_id = f"field:{entity.name}.{prop.name}"
			nodes[field_id] = GraphNode(
				id=field_id,
				kind="field",
				label=f"{entity.name}.{prop.name}: {type_name}",
				metadata={"type": type_name},
			)
			edges.append(GraphEdge(node_id, field_id, "owns_field", "field"))
			if type_name in entity_names:
				edges.append(GraphEdge(field_id, f"entity:{type_name}", "references", "references"))
			inferred_target = _infer_reference_from_field(prop.name, entity_names)
			if inferred_target:
				edges.append(
					GraphEdge(
						field_id,
						f"entity:{inferred_target}",
						"inferred_reference",
						"references",
						{"strategy": "foreign_key_name"},
					)
				)

		if isinstance(entity, DatabaseDeclaration):
			for schema in entity.schemas:
				for table in schema.tables:
					table_id = f"table:{table.name}"
					nodes[table_id] = GraphNode(
						id=table_id,
						kind="table",
						label=table.name,
						metadata={"schema": schema.name},
					)
					edges.append(GraphEdge(node_id, table_id, "contains_table", "table"))
					for column in table.columns:
						column_id = f"column:{table.name}.{column.name}"
						nodes[column_id] = GraphNode(
							id=column_id,
							kind="column",
							label=f"{table.name}.{column.name}: {column.data_type}",
							metadata={"primary_key": column.is_primary_key},
						)
						edges.append(GraphEdge(table_id, column_id, "owns_column", "column"))
						if column.reference:
							target = f"table:{column.reference.split('.', 1)[0]}"
							edges.append(GraphEdge(column_id, target, "references", "references"))

	return APGGraph("er", str(path), _sorted_nodes(nodes), _sorted_edges(edges))


def _agent_graph(module: ModuleDeclaration, path: Path) -> APGGraph:
	nodes: dict[str, GraphNode] = {}
	edges: list[GraphEdge] = []
	for entity in module.entities:
		if isinstance(entity, AIAgentDeclaration) or entity.entity_type in {
			EntityType.AGENT,
			EntityType.AI_AGENT,
		}:
			node_id = f"agent:{entity.name}"
			nodes[node_id] = GraphNode(
				id=node_id,
				kind="agent",
				label=entity.name,
				metadata={
					"runtime": getattr(entity, "runtime", None),
					"model": getattr(entity, "model", None),
				},
			)
			for handoff in getattr(entity, "handoffs", []):
				edges.append(
					GraphEdge(
						f"agent:{handoff.source}",
						f"agent:{handoff.target}",
						"handoff",
						handoff.condition,
					)
				)
		if isinstance(entity, AgentTeamDeclaration) or entity.entity_type in {
			EntityType.AGENT_TEAM,
			EntityType.SWARM,
		}:
			team_id = f"team:{entity.name}"
			nodes[team_id] = GraphNode(team_id, "agent_team", entity.name)
			for agent_name in getattr(entity, "agents", []):
				agent_id = f"agent:{agent_name}"
				nodes.setdefault(agent_id, GraphNode(agent_id, "agent", agent_name))
				edges.append(GraphEdge(team_id, agent_id, "contains_agent", "agent"))
	return APGGraph("agent", str(path), _sorted_nodes(nodes), _sorted_edges(edges))


def _capability_graph(module: ModuleDeclaration, path: Path) -> APGGraph:
	nodes: dict[str, GraphNode] = {}
	edges: list[GraphEdge] = []
	for entity in module.entities:
		if isinstance(entity, CapabilityDeclaration) or entity.entity_type == EntityType.CAPABILITY:
			cap_id = f"capability:{entity.name}"
			nodes[cap_id] = GraphNode(cap_id, "capability", entity.name)
			for required in getattr(entity, "requires", []):
				req_id = f"capability:{required}"
				nodes.setdefault(req_id, GraphNode(req_id, "capability", required))
				edges.append(GraphEdge(cap_id, req_id, "requires", "requires"))
		if isinstance(entity, ApplicationDeclaration):
			app_id = f"application:{entity.name}"
			nodes[app_id] = GraphNode(app_id, "application", entity.name)
			for capability in entity.capabilities:
				cap_id = f"capability:{capability}"
				nodes.setdefault(cap_id, GraphNode(cap_id, "capability", capability))
				edges.append(GraphEdge(app_id, cap_id, "uses_capability", "uses"))
	return APGGraph("capability", str(path), _sorted_nodes(nodes), _sorted_edges(edges))


def _generic_entity_graph(module: ModuleDeclaration, path: Path, kind: str) -> APGGraph:
	nodes: dict[str, GraphNode] = {
		f"module:{module.name}": GraphNode(f"module:{module.name}", "module", module.name)
	}
	edges: list[GraphEdge] = []
	for entity in module.entities:
		entity_id = f"{entity.entity_type.value}:{entity.name}"
		nodes[entity_id] = GraphNode(entity_id, entity.entity_type.value, entity.name)
		edges.append(GraphEdge(f"module:{module.name}", entity_id, "declares", "declares"))
	return APGGraph(kind, str(path), _sorted_nodes(nodes), _sorted_edges(edges))


def _sorted_nodes(nodes: dict[str, GraphNode]) -> list[GraphNode]:
	return [nodes[key] for key in sorted(nodes)]


def _sorted_edges(edges: list[GraphEdge]) -> list[GraphEdge]:
	return sorted(edges, key=lambda edge: (edge.source, edge.kind, edge.target, edge.label))


def _infer_reference_from_field(field_name: str, entity_names: set[str]) -> str | None:
	if not field_name.endswith("_id"):
		return None
	prefix = field_name[:-3]
	candidates = {
		prefix,
		prefix.title().replace("_", ""),
		"".join(part.capitalize() for part in prefix.split("_")),
	}
	for entity_name in sorted(entity_names):
		if entity_name in candidates or entity_name.lower() == prefix.replace("_", "").lower():
			return entity_name
	return None


def _mermaid_id(value: str) -> str:
	return "n_" + "".join(char if char.isalnum() else "_" for char in value)


def _dot_id(value: str) -> str:
	return "".join(char if char.isalnum() else "_" for char in value)


def _escape_label(value: str) -> str:
	return value.replace('"', "'").replace("|", "/")


def _escape_dot(value: str) -> str:
	return value.replace("\\", "\\\\").replace('"', '\\"')
