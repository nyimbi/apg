"""APG graph extraction and rendering."""

from __future__ import annotations

import json
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


SUPPORTED_GRAPH_KINDS = (
	"er",
	"lookup",
	"workflow",
	"handler",
	"capability",
	"security",
	"agent",
	"deployment",
	"package",
)
GRAPH_FIXTURE_AUDIT_FORMAT = "apg.graph-fixture-audit.v1"
DEFAULT_GRAPH_FIXTURE_CATALOG = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "graphs" / "catalog.json"


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
	return build_graph_from_module(module, path, kind)


def build_graph_from_module(module: ModuleDeclaration, path: Path, kind: str) -> APGGraph:
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


def build_graph_suite(path: Path) -> dict[str, Any]:
	module = parse_apg_module(path)
	graphs: dict[str, dict[str, Any]] = {}
	for kind in SUPPORTED_GRAPH_KINDS:
		graph = build_graph_from_module(module, path, kind)
		graphs[kind] = {
			"json": graph.to_dict(),
			"mermaid": render_mermaid(graph),
			"dot": render_dot(graph),
		}
	return {
		"format": "apg.graph-suite-report.v1",
		"ok": True,
		"source": str(path),
		"graph_kinds": list(SUPPORTED_GRAPH_KINDS),
		"graphs": graphs,
		"summary": {
			kind: {
				"nodes": len(rendered["json"]["nodes"]),
				"edges": len(rendered["json"]["edges"]),
			}
			for kind, rendered in graphs.items()
		},
	}


def audit_graph_fixtures(catalog_path: Path | None = None) -> dict[str, Any]:
	"""Run the checked-in graph-suite fixture catalog."""
	catalog_file = Path(catalog_path or DEFAULT_GRAPH_FIXTURE_CATALOG)
	catalog_root = catalog_file.parent
	catalog = json.loads(catalog_file.read_text(encoding="utf-8"))
	required_graph_kinds = sorted(str(kind) for kind in catalog.get("graph_kinds_required", SUPPORTED_GRAPH_KINDS))
	required_tags = sorted(str(tag) for tag in catalog.get("tags_required", []))
	fixture_reports: list[dict[str, Any]] = []
	blocking_gaps: list[dict[str, Any]] = []
	observed_graph_kinds: set[str] = set()
	covered_tags: set[str] = set()

	for fixture in catalog.get("fixtures", []):
		report = _audit_graph_fixture(catalog_root, fixture, required_graph_kinds)
		fixture_reports.append(report)
		observed_graph_kinds.update(report["graph_kinds"])
		if report["ok"]:
			covered_tags.update(report["tags"])
		else:
			blocking_gaps.append({
				"id": report["id"],
				"source": report["source"],
				"errors": report["errors"],
			})

	missing_tags = sorted(set(required_tags).difference(covered_tags))
	for tag in missing_tags:
		blocking_gaps.append({
			"id": f"missing_tag:{tag}",
			"source": str(catalog_file),
			"errors": [f"required graph fixture tag {tag!r} is not covered by a passing fixture"],
		})

	missing_graph_kinds = sorted(set(required_graph_kinds).difference(observed_graph_kinds))
	for kind in missing_graph_kinds:
		blocking_gaps.append({
			"id": f"missing_graph_kind:{kind}",
			"source": str(catalog_file),
			"errors": [f"required graph kind {kind!r} was not emitted by any graph-suite fixture"],
		})

	return {
		"format": GRAPH_FIXTURE_AUDIT_FORMAT,
		"ok": not blocking_gaps,
		"fixture_catalog": str(catalog_file),
		"graph_kinds_required": required_graph_kinds,
		"graph_kinds_observed": sorted(observed_graph_kinds),
		"missing_graph_kinds": missing_graph_kinds,
		"tags_required": required_tags,
		"tags_covered": sorted(covered_tags),
		"missing_tags": missing_tags,
		"fixtures": fixture_reports,
		"summary": {
			"fixture_count": len(fixture_reports),
			"passing_fixture_count": sum(1 for report in fixture_reports if report["ok"]),
			"failing_fixture_count": sum(1 for report in fixture_reports if not report["ok"]),
			"blocking_gap_count": len(blocking_gaps),
		},
		"blocking_gaps": blocking_gaps,
	}


def _audit_graph_fixture(
	catalog_root: Path,
	fixture: dict[str, Any],
	required_graph_kinds: list[str],
) -> dict[str, Any]:
	fixture_id = str(fixture["id"])
	source = (catalog_root / str(fixture["source"])).resolve()
	tags = sorted(str(tag) for tag in fixture.get("tags", []))
	errors: list[str] = []
	report: dict[str, Any] | None = None

	try:
		report = build_graph_suite(source)
	except Exception as error:
		errors.append(str(error))

	if report is None:
		return {
			"id": fixture_id,
			"source": str(source),
			"tags": tags,
			"graph_kinds": [],
			"expectations_checked": [],
			"ok": False,
			"errors": errors,
		}

	graphs = report.get("graphs", {})
	graph_kinds = sorted(str(kind) for kind in report.get("graph_kinds", []))
	for kind in required_graph_kinds:
		if kind not in graphs:
			errors.append(f"missing graph kind {kind!r}")
			continue
		graph_report = graphs[kind]
		json_graph = graph_report.get("json", {})
		if json_graph.get("format") != "apg.graph.v1":
			errors.append(f"{kind} graph JSON format mismatch")
		if json_graph.get("kind") != kind:
			errors.append(f"{kind} graph JSON kind mismatch")
		mermaid = str(graph_report.get("mermaid", ""))
		dot = str(graph_report.get("dot", ""))
		if not mermaid.startswith("graph TD\n"):
			errors.append(f"{kind} Mermaid rendering does not start with graph TD")
		if not dot.startswith(f"digraph {kind} "):
			errors.append(f"{kind} DOT rendering does not start with digraph {kind}")

	expectations_checked: list[str] = []
	for kind, expectation in dict(fixture.get("expectations", {})).items():
		kind = str(kind)
		graph = graphs.get(kind, {}).get("json", {})
		nodes = {str(node.get("id")) for node in graph.get("nodes", [])}
		edges = {
			(
				str(edge.get("source")),
				str(edge.get("target")),
				str(edge.get("kind")),
			)
			for edge in graph.get("edges", [])
		}
		min_nodes = int(expectation.get("min_nodes", 0))
		min_edges = int(expectation.get("min_edges", 0))
		if len(nodes) < min_nodes:
			errors.append(f"{kind} expected at least {min_nodes} nodes, got {len(nodes)}")
		if len(edges) < min_edges:
			errors.append(f"{kind} expected at least {min_edges} edges, got {len(edges)}")
		for node_id in expectation.get("nodes", []):
			if str(node_id) not in nodes:
				errors.append(f"{kind} missing node {node_id}")
		for expected_edge in expectation.get("edges", []):
			edge_key = (
				str(expected_edge.get("source")),
				str(expected_edge.get("target")),
				str(expected_edge.get("kind")),
			)
			if edge_key not in edges:
				errors.append(f"{kind} missing edge {edge_key[0]} -> {edge_key[1]} ({edge_key[2]})")
		expectations_checked.append(kind)

	return {
		"id": fixture_id,
		"source": str(source),
		"tags": tags,
		"graph_kinds": graph_kinds,
		"expectations_checked": sorted(expectations_checked),
		"ok": not errors,
		"errors": errors,
	}


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
