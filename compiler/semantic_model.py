"""Serializable APG semantic model builder.

This module turns parsed APG source into the stable JSON contract used by CLI,
IDE, agents, tests, and generators. It is intentionally dependency-light and
does not write generated application files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# ── file-level mtime cache ──────────────────────────────────────────────────
# Keyed by (resolved_path_str, mtime_ns) — avoids re-parsing unchanged files.
_MODEL_CACHE: dict[tuple[str, int], dict[str, Any]] = {}

from .ast_builder import (
	AIAgentDeclaration,
	AgentTeamDeclaration,
	ApplicationDeclaration,
	ASTBuilder,
	CapabilityDeclaration,
	DatabaseDeclaration,
	EntityDeclaration,
	EntityType,
	ModuleDeclaration,
	PropertyDeclaration,
)

# New entity types introduced in language design pass — map to stable semantic kind strings
_NEW_ENTITY_KIND_MAP: dict[EntityType, str] = {
	EntityType.ENUM: "enum",
	EntityType.STATEMACHINE: "statemachine",
	EntityType.MIGRATION: "migration",
	EntityType.DEPLOYMENT: "deployment",
	EntityType.MARKETPLACE: "marketplace",
	EntityType.EVENT_STORE: "event_store",
}
from .graphs import SUPPORTED_GRAPH_KINDS, build_graph_from_module
from .parser import APGParser, APGSyntaxError
from .semantic_analyzer import SemanticAnalyzer, SemanticError

SEMANTIC_MODEL_FIXTURE_AUDIT_FORMAT = "apg.semantic-model-fixture-audit.v1"
DEFAULT_SEMANTIC_MODEL_CATALOG = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "semantic_model" / "catalog.json"


def build_semantic_model(path: Path) -> dict[str, Any]:
	"""Build an ``apg.semantic-model.v1`` report for one APG source file.

	Results are cached by (resolved_path, mtime_ns); callers can invalidate by
	calling ``invalidate_semantic_model_cache(path)``.
	"""
	resolved = str(path.resolve())
	try:
		mtime_ns = path.stat().st_mtime_ns
	except OSError:
		mtime_ns = 0
	cache_key = (resolved, mtime_ns)
	if cache_key in _MODEL_CACHE:
		return _MODEL_CACHE[cache_key]
	source = path.read_text(encoding="utf-8")
	model = build_semantic_model_from_source(source, display_path=path)
	_MODEL_CACHE[cache_key] = model
	return model


def invalidate_semantic_model_cache(path: Path | None = None) -> None:
	"""Evict one path (or all entries) from the mtime cache."""
	if path is None:
		_MODEL_CACHE.clear()
	else:
		resolved = str(path.resolve())
		for key in [k for k in _MODEL_CACHE if k[0] == resolved]:
			del _MODEL_CACHE[key]


def build_semantic_model_from_source(
	source: str,
	display_path: Path | None = None,
) -> dict[str, Any]:
	"""Build a semantic model directly from source text without touching disk."""
	label = str(display_path) if display_path else "<string>"
	path_for_model = display_path or Path(label)

	parser = _shared_parser()
	parse_result = parser.parse_string(source, label)
	diagnostics = [
		_diagnostic_from_error(error, path_for_model, "error")
		for error in parse_result.get("errors", [])
	]

	module = parse_result.get("ast")
	if module is None and parse_result.get("success"):
		module = _shared_builder().build_ast(parse_result["parse_tree"], label)

	if module is None:
		return _empty_model(path_for_model, diagnostics)

	analysis = SemanticAnalyzer().analyze(module)
	for error in analysis.get("errors", []):
		diagnostics.append(_diagnostic_from_error(error, path_for_model, "error"))
	for warning in analysis.get("warnings", []):
		diagnostics.append(_diagnostic_from_error(warning, path_for_model, "warning"))

	model = _model_from_module(module, path_for_model)
	model["diagnostics"] = diagnostics
	model["diagnostics"].extend(_database_backed_view_diagnostics(model, path_for_model))
	model["ok"] = not any(d["severity"] == "error" for d in model["diagnostics"])
	return model


# ── module-level singleton parser and builder (construction is ~5ms each) ──
_SHARED_PARSER: APGParser | None = None
_SHARED_BUILDER: ASTBuilder | None = None


def _shared_parser() -> APGParser:
	global _SHARED_PARSER
	if _SHARED_PARSER is None:
		_SHARED_PARSER = APGParser()
	return _SHARED_PARSER


def _shared_builder() -> ASTBuilder:
	global _SHARED_BUILDER
	if _SHARED_BUILDER is None:
		_SHARED_BUILDER = ASTBuilder()
	return _SHARED_BUILDER


def audit_semantic_model_fixtures(catalog_path: Path | None = None) -> dict[str, Any]:
	"""Run checked-in semantic-model fixtures against ``apg.semantic-model.v1``."""
	catalog_file = Path(catalog_path or DEFAULT_SEMANTIC_MODEL_CATALOG)
	catalog_root = catalog_file.parent
	catalog = json.loads(catalog_file.read_text(encoding="utf-8"))
	required_tags = sorted(str(tag) for tag in catalog.get("tags_required", []))
	covered_tags: set[str] = set()
	fixture_reports: list[dict[str, Any]] = []
	blocking_gaps: list[dict[str, Any]] = []

	for fixture in catalog.get("fixtures", []):
		report = _audit_semantic_model_fixture(catalog_root, fixture)
		fixture_reports.append(report)
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
			"errors": [f"required semantic-model fixture tag {tag!r} is not covered by a passing fixture"],
		})

	return {
		"format": SEMANTIC_MODEL_FIXTURE_AUDIT_FORMAT,
		"ok": not blocking_gaps,
		"fixture_catalog": str(catalog_file),
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


def build_semantic_model_from_module(module: ModuleDeclaration, source: str | Path) -> dict[str, Any]:
	"""Build an ``apg.semantic-model.v1`` report from an existing AST module."""
	path = Path(source)
	analyzer = SemanticAnalyzer()
	analysis = analyzer.analyze(module)
	diagnostics: list[dict[str, Any]] = []
	for error in analysis.get("errors", []):
		diagnostics.append(_diagnostic_from_error(error, path, "error"))
	for warning in analysis.get("warnings", []):
		diagnostics.append(_diagnostic_from_error(warning, path, "warning"))
	model = _model_from_module(module, path)
	model["diagnostics"] = diagnostics
	model["diagnostics"].extend(_database_backed_view_diagnostics(model, path))
	model["ok"] = not any(diagnostic["severity"] == "error" for diagnostic in model["diagnostics"])
	return model


def _empty_model(path: Path, diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
	return {
		"format": "apg.semantic-model.v1",
		"ok": False,
		"source_files": [str(path)],
		"app": {},
		"symbols": {},
		"tables": {},
		"views": {},
		"flows": {},
		"operations": {},
		"rules": {},
		"roles": {},
		"security": {},
		"agents": {},
		"llms": {},
		"capabilities": {},
		"composition": {},
		"contracts": {},
		"deployment": {},
		"packages": {},
		"graphs": {},
		"diagnostics": diagnostics,
	}


def _model_from_module(module: ModuleDeclaration, path: Path) -> dict[str, Any]:
	symbols: dict[str, dict[str, Any]] = {}
	tables: dict[str, dict[str, Any]] = {}
	views: dict[str, dict[str, Any]] = {}
	flows: dict[str, dict[str, Any]] = {}
	rules: dict[str, dict[str, Any]] = {}
	agents: dict[str, dict[str, Any]] = {}
	capabilities: dict[str, dict[str, Any]] = {}
	composition: dict[str, Any] = {
		"applications": {},
		"agent_teams": {},
		"capability_dependencies": {},
	}
	contracts: dict[str, dict[str, Any]] = {}
	deployment: dict[str, Any] = {"target": "python", "source": str(path)}

	symbols[_symbol_id("module", module.name)] = _symbol("module", module.name, module, path)

	for entity in module.entities:
		kind = _entity_kind(entity)
		symbols[_symbol_id(kind, entity.name)] = _symbol(kind, entity.name, entity, path)

		if isinstance(entity, AIAgentDeclaration):
			agents[entity.name] = _agent_model(entity)
			if entity.model:
				symbols[_symbol_id("llm", entity.model)] = _symbol("llm", entity.model, entity, path)
			continue

		# EntityDeclarations with agent entity_type (platform examples use generic parsing)
		if entity.entity_type in {EntityType.AGENT, EntityType.AI_AGENT, EntityType.SWARM}:
			agents[entity.name] = {
				"name": entity.name, "role": "", "model": "", "runtime": "",
				"capabilities": [], "tools": [], "memory": None,
				"inputs": [], "outputs": [], "handoffs": [],
				"configuration": {}, "rules": [], "ui": {}, "theme": {}, "system": "",
			}
			continue

		if isinstance(entity, AgentTeamDeclaration):
			composition["agent_teams"][entity.name] = {
				"agents": list(entity.agents),
				"flow": [
					{"from": handoff.source, "to": handoff.target, "condition": handoff.condition}
					for handoff in entity.flow
				],
				"capabilities": list(entity.capabilities),
				"configuration": dict(entity.configuration),
				"rules": [dict(rule) for rule in entity.rules],
				"ui": dict(entity.ui),
				"theme": dict(entity.theme),
			}
			continue

		if isinstance(entity, CapabilityDeclaration):
			capabilities[entity.name] = _capability_model(entity)
			contracts[entity.name] = dict(entity.contract)
			composition["capability_dependencies"][entity.name] = list(entity.requires)
			for rule in [*entity.rules, *entity.business_rules, *entity.rule_engine.get("rules", [])]:
				if isinstance(rule, dict):
					rule_name = str(rule.get("name") or f"{entity.name}.rule")
					rules[f"{entity.name}.{rule_name}"] = dict(rule)
			continue

		if isinstance(entity, ApplicationDeclaration):
			composition["applications"][entity.name] = _application_model(entity)
			continue

		if _is_table_like(entity):
			tables[entity.name] = _table_model(entity)
			for field in entity.properties:
				symbols[_symbol_id("field", f"{entity.name}.{field.name}")] = _symbol(
					"field",
					f"{entity.name}.{field.name}",
					field,
					path,
				)
			continue

		if entity.entity_type in {EntityType.SCREEN, EntityType.FORM, EntityType.UI_COMPONENT}:
			views[entity.name] = _view_model(entity)
		elif entity.entity_type in {EntityType.WORKFLOW, EntityType.FLOW}:
			flows[entity.name] = _flow_model(entity)
		elif entity.entity_type in {EntityType.RULE, EntityType.RULE_SET, EntityType.POLICY}:
			rules[entity.name] = _generic_entity_model(entity)

	for database in [entity for entity in module.entities if isinstance(entity, DatabaseDeclaration)]:
		for schema in database.schemas:
			for table in schema.tables:
				table_id = f"{schema.name}.{table.name}" if schema.name else table.name
				tables[table_id] = {
					"name": table.name,
					"schema": schema.name,
					"database": database.name,
					"fields": {
						column.name: {
							"type": column.data_type,
							"required": not column.is_nullable,
							"primary_key": column.is_primary_key,
							"default": column.default_value,
							"constraints": list(column.constraints),
							"relationship": column.reference,
						}
						for column in table.columns
					},
					"indexes": [
						{
							"name": index.name,
							"columns": list(index.columns),
							"unique": index.is_unique,
							"type": index.index_type,
						}
						for index in table.indexes
					],
				}

	return {
		"format": "apg.semantic-model.v1",
		"ok": True,
		"source_files": [str(path)],
		"app": {
			"name": module.name,
			"version": module.version,
			"description": module.description,
			"entity_count": len(module.entities),
		},
		"symbols": symbols,
		"tables": tables,
		"views": views,
		"flows": flows,
		"operations": {},
		"rules": rules,
		"roles": {},
		"security": {},
		"agents": agents,
		"llms": {
			agent_name: {"model": agent["model"], "runtime": agent["runtime"]}
			for agent_name, agent in agents.items()
			if agent.get("model")
		},
		"capabilities": capabilities,
		"composition": composition,
		"contracts": contracts,
		"deployment": deployment,
		"packages": {},
		"graphs": _graph_summaries(module, path),
		"diagnostics": [],
	}


def _audit_semantic_model_fixture(catalog_root: Path, fixture: dict[str, Any]) -> dict[str, Any]:
	fixture_id = str(fixture["id"])
	source = (catalog_root / str(fixture["source"])).resolve()
	tags = sorted(str(tag) for tag in fixture.get("tags", []))
	expected_ok = bool(fixture.get("expected_ok", True))
	errors: list[str] = []
	model: dict[str, Any] | None = None

	try:
		model = build_semantic_model(source)
	except Exception as error:
		errors.append(str(error))

	if model is None:
		return {
			"id": fixture_id,
			"source": str(source),
			"tags": tags,
			"ok": False,
			"format": "",
			"expected_ok": expected_ok,
			"actual_ok": False,
			"errors": errors or ["semantic model was not produced"],
		}

	if model.get("format") != "apg.semantic-model.v1":
		errors.append(f"expected apg.semantic-model.v1, got {model.get('format')}")
	if bool(model.get("ok")) != expected_ok:
		errors.append(f"expected ok={expected_ok}, got ok={model.get('ok')}")

	diagnostic_codes = [str(diagnostic.get("code")) for diagnostic in model.get("diagnostics", [])]
	for code in fixture.get("diagnostic_codes", []):
		if str(code) not in diagnostic_codes:
			errors.append(f"expected diagnostic {code} was not emitted")

	for symbol_id in fixture.get("symbols", []):
		if symbol_id not in model.get("symbols", {}):
			errors.append(f"expected symbol {symbol_id} is missing")

	for table_name, field_names in fixture.get("table_fields", {}).items():
		table = model.get("tables", {}).get(table_name)
		if table is None:
			errors.append(f"expected table {table_name} is missing")
			continue
		for field_name in field_names:
			if field_name not in table.get("fields", {}):
				errors.append(f"expected field {table_name}.{field_name} is missing")

	for view_name, expected_bindings in fixture.get("view_bindings", {}).items():
		view = model.get("views", {}).get(view_name)
		if view is None:
			errors.append(f"expected view {view_name} is missing")
			continue
		actual_bindings = list(view.get("bindings", []))
		if actual_bindings != list(expected_bindings):
			errors.append(f"expected bindings for {view_name}={list(expected_bindings)}, got {actual_bindings}")

	for capability_name, expected in fixture.get("capabilities", {}).items():
		capability = model.get("capabilities", {}).get(capability_name)
		if capability is None:
			errors.append(f"expected capability {capability_name} is missing")
			continue
		for key in ["provides", "requires"]:
			if key in expected and list(capability.get(key, [])) != list(expected[key]):
				errors.append(
					f"expected capability {capability_name}.{key}={list(expected[key])}, got {list(capability.get(key, []))}"
				)

	for graph_kind in fixture.get("graph_kinds", []):
		if graph_kind not in model.get("graphs", {}):
			errors.append(f"expected graph kind {graph_kind} is missing")

	return {
		"id": fixture_id,
		"source": str(source),
		"tags": tags,
		"ok": not errors,
		"format": model.get("format"),
		"expected_ok": expected_ok,
		"actual_ok": bool(model.get("ok")),
		"diagnostic_codes": diagnostic_codes,
		"errors": errors,
	}


def _symbol(kind: str, name: str, node: Any, path: Path) -> dict[str, Any]:
	line = max(0, int(getattr(node, "line", 0) or 0))
	column = max(0, int(getattr(node, "column", 0) or 0))
	return {
		"id": _symbol_id(kind, name),
		"kind": kind,
		"name": name,
		"file": str(path),
		"range": {
			"start": {"line": line, "character": column},
			"end": {"line": line, "character": column + 1},
		},
		"references": [],
	}


def _symbol_id(kind: str, name: str) -> str:
	return f"{kind}.{name}"


def _entity_kind(entity: EntityDeclaration) -> str:
	if _is_table_like(entity):
		return "table"
	if isinstance(entity, AIAgentDeclaration):
		return "agent"
	if isinstance(entity, AgentTeamDeclaration):
		return "composition"
	if isinstance(entity, CapabilityDeclaration):
		return "capability"
	if isinstance(entity, ApplicationDeclaration):
		return "app"
	if isinstance(entity, DatabaseDeclaration):
		return "database"
	# New entity types added in language design pass
	if entity.entity_type in _NEW_ENTITY_KIND_MAP:
		return _NEW_ENTITY_KIND_MAP[entity.entity_type]
	return entity.entity_type.value


def _is_table_like(entity: EntityDeclaration) -> bool:
	return entity.entity_type == EntityType.ENTITY and bool(entity.properties)


def _table_model(entity: EntityDeclaration) -> dict[str, Any]:
	return {
		"name": entity.name,
		"fields": {
			field.name: {
				"type": _field_type(field),
				"required": field.is_required,
				"relationship": _field_relationship(field),
			}
			for field in entity.properties
		},
		"lookup_paths": {
			f"{field.name}.id": {
				"chain": [f"{entity.name}.{field.name}", f"{_field_type(field)}.id"],
				"valid": True,
			}
			for field in entity.properties
			if _field_type(field) not in {"str", "int", "float", "bool", "any", "list", "dict"}
		},
	}


def _field_type(field: PropertyDeclaration) -> str:
	return field.type_annotation.type_name if field.type_annotation else "any"


def _field_relationship(field: PropertyDeclaration) -> dict[str, Any] | None:
	field_type = _field_type(field)
	if field_type not in {"str", "int", "float", "bool", "any", "list", "dict"}:
		return {"target_table": field_type, "target_field": "id", "cardinality": "many-to-one"}
	if field.name.endswith("_id"):
		target = field.name[:-3]
		if target:
			return {
				"target_table": "".join(part.capitalize() for part in target.split("_")),
				"target_field": "id",
				"cardinality": "many-to-one",
				"alias": target,
			}
	return None


def _agent_model(agent: AIAgentDeclaration) -> dict[str, Any]:
	return {
		"name": agent.name,
		"role": agent.role,
		"model": agent.model,
		"runtime": agent.runtime,
		"system": agent.system_prompt,
		"capabilities": list(agent.capabilities),
		"tools": list(agent.tools),
		"memory": (
			{"kind": agent.memory.kind, "name": agent.memory.name}
			if agent.memory else None
		),
		"inputs": list(agent.inputs),
		"outputs": list(agent.outputs),
		"handoffs": [
			{"from": handoff.source, "to": handoff.target, "condition": handoff.condition}
			for handoff in agent.handoffs
		],
		"configuration": dict(agent.configuration),
		"rules": [dict(rule) for rule in agent.rules],
		"ui": dict(agent.ui),
		"theme": dict(agent.theme),
	}


def _capability_model(capability: CapabilityDeclaration) -> dict[str, Any]:
	return {
		"name": capability.name,
		"provides": list(capability.provides),
		"requires": list(capability.requires),
		"configuration": dict(capability.configuration),
		"rules": [dict(rule) for rule in capability.rules],
		"rule_engine": dict(capability.rule_engine),
		"ui": dict(capability.ui),
		"theme": dict(capability.theme),
		"runtime": dict(capability.runtime),
		"erp_modules": list(capability.erp_modules),
		"components": capability.components,
		"business_rules": [dict(rule) for rule in capability.business_rules],
		"approvals": capability.approvals,
		"master_data": capability.master_data,
		"i18n": dict(capability.i18n),
		"streaming": dict(capability.streaming),
		"screens": capability.screens,
	}


def _application_model(application: ApplicationDeclaration) -> dict[str, Any]:
	return {
		"name": application.name,
		"description": application.description,
		"capabilities": list(application.capabilities),
		"agents": list(application.agents),
		"agent_teams": list(application.agent_teams),
		"components": application.components,
		"screens": application.screens,
		"routes": list(application.routes),
		"workflows": list(application.workflows),
		"policies": application.policies,
		"configuration": dict(application.configuration),
		"theme": dict(application.theme),
		"runtime": dict(application.runtime),
		"integrations": application.integrations,
		"deployments": application.deployments,
	}


def _view_model(entity: EntityDeclaration) -> dict[str, Any]:
	return {**_generic_entity_model(entity), "bindings": [field.name for field in entity.properties]}


def _flow_model(entity: EntityDeclaration) -> dict[str, Any]:
	return {**_generic_entity_model(entity), "states": [], "transitions": []}


def _generic_entity_model(entity: EntityDeclaration) -> dict[str, Any]:
	return {
		"name": entity.name,
		"type": entity.entity_type.value,
		"properties": {
			field.name: {"type": _field_type(field), "required": field.is_required}
			for field in entity.properties
		},
		"methods": [method.name for method in entity.methods],
	}


def _graph_summaries(module: ModuleDeclaration, path: Path) -> dict[str, dict[str, Any]]:
	graphs: dict[str, dict[str, Any]] = {}
	for kind in SUPPORTED_GRAPH_KINDS:
		graph = build_graph_from_module(module, path, kind)
		graph_dict = graph.to_dict()
		graphs[kind] = {
			"kind": kind,
			"nodes": len(graph_dict["nodes"]),
			"edges": len(graph_dict["edges"]),
		}
	return graphs


VIEW_METADATA_BINDINGS = {
	"table",
	"subject",
	"entity",
	"model",
	"title",
	"route",
	"layout",
	"description",
}


def _database_backed_view_diagnostics(model: dict[str, Any], path: Path) -> list[dict[str, Any]]:
	diagnostics: list[dict[str, Any]] = []
	tables = model.get("tables", {})
	symbols = model.get("symbols", {})
	for view_name, view in model.get("views", {}).items():
		if view.get("type") != "form":
			continue
		subject = _view_subject_table(view_name, view, tables)
		if subject is None:
			continue
		symbol = symbols.get(_symbol_id("form", view_name))
		table = tables.get(subject)
		if table is None:
			diagnostics.append(_semantic_diagnostic(
				"APG0401",
				"Unknown view subject table",
				"error",
				f"Form '{view_name}' is backed by unknown table '{subject}'.",
				path,
				symbol,
			))
			continue
		valid_bindings = set(table.get("fields", {})) | set(table.get("lookup_paths", {}))
		for binding in view.get("bindings", []):
			if binding in VIEW_METADATA_BINDINGS:
				continue
			if binding not in valid_bindings:
				diagnostics.append(_semantic_diagnostic(
					"APG0402",
					"Invalid view binding",
					"error",
					f"Form '{view_name}' binds '{binding}', but table '{subject}' has no such field or lookup path.",
					path,
					symbol,
				))
	return diagnostics


def _view_subject_table(view_name: str, view: dict[str, Any], tables: dict[str, Any]) -> str | None:
	properties = view.get("properties", {})
	for key in ["table", "subject", "entity", "model"]:
		value = properties.get(key, {}).get("type")
		if isinstance(value, str) and value:
			return value
	if view_name in tables:
		return view_name
	if view_name.endswith("Form"):
		candidate = view_name[:-4]
		if candidate:
			return candidate
	return None


def _semantic_diagnostic(
	code: str,
	title: str,
	severity: str,
	message: str,
	path: Path,
	symbol: dict[str, Any] | None,
) -> dict[str, Any]:
	start = (
		symbol.get("range", {}).get("start", {"line": 0, "character": 0})
		if symbol else {"line": 0, "character": 0}
	)
	return {
		"code": code,
		"title": title,
		"severity": severity,
		"message": message,
		"file": str(path),
		"range": {
			"start": start,
			"end": {"line": start["line"], "character": start["character"] + 1},
		},
		"related_locations": [],
		"fixes": [],
		"docs_url": "docs/tooling.md#semantic-model-contract",
	}


def _diagnostic_from_error(error: APGSyntaxError | SemanticError | Exception, path: Path, severity: str) -> dict[str, Any]:
	if isinstance(error, APGSyntaxError):
		line = error.line
		column = error.column
		message = error.message
		code = "APG0001"
		title = "Syntax error"
	elif isinstance(error, SemanticError):
		node = error.node
		line = getattr(node, "line", 1)
		column = getattr(node, "column", 0)
		message = error.message
		code = "APG0100"
		title = "Semantic warning" if severity == "warning" else "Semantic error"
	else:
		line = 1
		column = 0
		message = str(error)
		code = "APG9000"
		title = "Internal tooling error"

	start = {
		"line": max(0, int(line or 1) - 1),
		"character": max(0, int(column or 0)),
	}
	return {
		"code": code,
		"title": title,
		"severity": severity,
		"message": message,
		"file": str(path),
		"range": {
			"start": start,
			"end": {"line": start["line"], "character": start["character"] + 1},
		},
		"related_locations": [],
		"fixes": [],
		"docs_url": "docs/tooling.md#semantic-model-contract",
	}
