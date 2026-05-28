"""Dependency-light APG language service over the shared semantic model.

This module is the executable core for LSP-facing behavior.  It deliberately
does not import pygls so CLI checks, tests, VS Code glue, and the actual LSP
server can all consume the same semantic snapshot.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from compiler.ast_builder import ASTBuilder
from compiler.diagnostics import diagnostic_registry, explain_diagnostic
from compiler.formatter import format_apg_source
from compiler.parser import APGParser, APGSyntaxError
from compiler.semantic_analyzer import SemanticError
from compiler.semantic_model import build_semantic_model_from_module


TOP_LEVEL_COMPLETIONS = [
	("module", "Keyword", "Declare an APG module", "module ${1:name} version 1.0.0 {\n  $0\n}"),
	("table", "Keyword", "Declare a database-backed entity", "table ${1:Name} {\n  ${2:id}: int;\n}"),
	("screen", "Keyword", "Declare a user-facing screen", "screen ${1:Name} {\n  $0\n}"),
	("form", "Keyword", "Declare a database-backed form", "form ${1:Name} {\n  $0\n}"),
	("flow", "Keyword", "Declare a workflow flow", "flow ${1:Name} {\n  $0\n}"),
	("operation", "Keyword", "Declare an executable operation", "operation ${1:Name} {\n  $0\n}"),
	("rule", "Keyword", "Declare a business rule", "rule ${1:Name} {\n  $0\n}"),
	("capability", "Keyword", "Declare a capability contract", "capability ${1:Name} {\n  $0\n}"),
	("composition", "Keyword", "Declare capability composition", "composition ${1:Name} {\n  $0\n}"),
	("agent", "Keyword", "Declare a first-class AI agent", "agent ${1:Name} {\n  role: \"${2:role}\";\n}"),
	("agent_team", "Keyword", "Declare an AI agent team", "agent_team ${1:Name} {\n  $0\n}"),
	("application", "Keyword", "Declare a composable APG application", "application ${1:Name} {\n  $0\n}"),
	("package", "Keyword", "Declare packaging metadata", "package ${1:Name} {\n  $0\n}"),
	("deploy", "Keyword", "Declare deployment topology", "deploy ${1:Name} {\n  $0\n}"),
]

TYPE_COMPLETIONS = [
	("str", "TypeParameter", "String field type", "str"),
	("int", "TypeParameter", "Integer field type", "int"),
	("float", "TypeParameter", "Floating-point field type", "float"),
	("bool", "TypeParameter", "Boolean field type", "bool"),
	("date", "TypeParameter", "Date field type", "date"),
	("datetime", "TypeParameter", "Timestamp field type", "datetime"),
	("json", "TypeParameter", "Structured JSON field type", "json"),
]

AGENT_COMPLETIONS = [
	("role", "Property", "Agent responsibility", "role: \"${1:assistant}\";"),
	("model", "Property", "LLM provider and model", "model: \"${1:openai:gpt-4.1-mini}\";"),
	("runtime", "Property", "Agent runtime provider", "runtime: ${1:codex};"),
	("system", "Property", "Agent system prompt", "system: \"${1:Instructions}\";"),
	("capabilities", "Property", "Capability keys this agent can use", "capabilities: [${1:capability_key}];"),
	("tools", "Property", "External tools this agent may call", "tools: [${1:tool_name}];"),
	("handoff", "Property", "Agent handoff rule", "handoff ${1:TargetAgent} when \"${2:condition}\";"),
]


def build_language_service_snapshot(source: str, source_name: str | Path = "<memory>") -> dict[str, Any]:
	"""Build an IDE-ready snapshot from APG source text."""
	source_path = Path(source_name)
	parse_result = APGParser().parse_string(source, str(source_path))
	diagnostics = [
		_diagnostic_from_error(error, source_path, "error")
		for error in parse_result.get("errors", [])
	]

	module = parse_result.get("ast")
	if module is None and parse_result.get("success") and parse_result.get("parse_tree"):
		module = ASTBuilder().build_ast(parse_result["parse_tree"], str(source_path))

	if module is None:
		model = _empty_model(source_path, diagnostics)
	else:
		model = build_semantic_model_from_module(module, source_path)
		model["diagnostics"] = [*diagnostics, *model.get("diagnostics", [])]
		model["ok"] = not any(item.get("severity") == "error" for item in model["diagnostics"])

	semantic_symbols = _sorted_symbols(model)
	return {
		"format": "apg.language-service-snapshot.v1",
		"ok": bool(model.get("ok")),
		"source_file": str(source_path),
		"semantic_model": model,
		"diagnostics": model.get("diagnostics", []),
		"completions": completion_items(model, source=source),
		"document_symbols": document_symbols(model),
		"definitions": {
			symbol["name"]: _symbol_location(symbol)
			for symbol in semantic_symbols
		},
		"capabilities": [
			"textDocument/didOpen",
			"textDocument/didChange",
			"textDocument/completion",
			"textDocument/hover",
			"textDocument/definition",
			"textDocument/references",
			"textDocument/documentSymbol",
			"textDocument/codeAction",
			"textDocument/formatting",
			"workspace/symbol",
		],
	}


def build_language_server_check(path: Path) -> dict[str, Any]:
	"""Build a machine-readable language-server check report for one APG file."""
	source = path.read_text(encoding="utf-8")
	snapshot = build_language_service_snapshot(source, path)
	formatted = format_apg_source(source)
	report = {
		"format": "apg.language-server-check.v1",
		"ok": snapshot["ok"],
		"file": str(path),
		"semantic_model_format": snapshot["semantic_model"].get("format"),
		"semantic_model_ok": snapshot["semantic_model"].get("ok"),
		"diagnostic_count": len(snapshot["diagnostics"]),
		"completion_count": len(snapshot["completions"]),
		"document_symbol_count": len(snapshot["document_symbols"]),
		"code_action_count": len(code_actions(snapshot["diagnostics"])),
		"formatting": {
			"changed": formatted.changed,
			"idempotent": formatted.idempotent,
		},
		"capabilities": snapshot["capabilities"],
		"diagnostics": snapshot["diagnostics"],
		"sample_completions": snapshot["completions"][:25],
		"document_symbols": snapshot["document_symbols"],
	}
	return report


def completion_items(
	model: dict[str, Any],
	source: str = "",
	line: int | None = None,
	character: int | None = None,
) -> list[dict[str, Any]]:
	"""Return context-aware completion items from the semantic model."""
	context = _line_prefix(source, line, character)
	items = [_completion(*item) for item in TOP_LEVEL_COMPLETIONS]

	if _looks_like_field_context(context):
		items.extend(_completion(*item) for item in TYPE_COMPLETIONS)
		for table_name in sorted(model.get("tables", {})):
			items.append(_completion(table_name, "Class", f"Relationship target table {table_name}", table_name))

	if _looks_like_agent_context(context):
		items.extend(_completion(*item) for item in AGENT_COMPLETIONS)

	for symbol in _sorted_symbols(model):
		items.append(_completion(symbol["name"], _completion_kind(symbol["kind"]), symbol["id"], symbol["name"]))

	for capability_name, capability in sorted(model.get("capabilities", {}).items()):
		for provided in capability.get("provides", []):
			items.append(_completion(str(provided), "Interface", f"Provided by capability {capability_name}", str(provided)))

	return _dedupe_completions(items)


def hover(model: dict[str, Any], word: str) -> dict[str, Any] | None:
	"""Return hover content for a word from symbols or diagnostics."""
	if not word:
		return None
	if word.upper() in diagnostic_registry():
		diagnostic = explain_diagnostic(word)
		return {
			"kind": "markdown",
			"value": f"**{word.upper()}: {diagnostic['title']}**\n\n{diagnostic['meaning']}\n\n{diagnostic['next_step']}",
		}

	symbol = _find_symbol(model, word)
	if symbol:
		return {
			"kind": "markdown",
			"value": f"**{symbol['name']}** ({symbol['kind']})\n\nDefined in `{symbol['file']}`.",
		}

	return None


def definition(model: dict[str, Any], word: str) -> dict[str, Any] | None:
	"""Return a source location for the symbol matching ``word``."""
	symbol = _find_symbol(model, word)
	if not symbol:
		return None
	return _symbol_location(symbol)


def references(source: str, word: str, source_name: str | Path = "<memory>") -> list[dict[str, Any]]:
	"""Find whole-word references in source text."""
	if not word:
		return []
	pattern = re.compile(rf"\b{re.escape(word)}\b")
	locations: list[dict[str, Any]] = []
	for line_number, line in enumerate(source.splitlines()):
		for match in pattern.finditer(line):
			locations.append({
				"file": str(source_name),
				"range": {
					"start": {"line": line_number, "character": match.start()},
					"end": {"line": line_number, "character": match.end()},
				},
			})
	return locations


def document_symbols(model: dict[str, Any]) -> list[dict[str, Any]]:
	"""Return a hierarchical-ish outline from semantic-model symbols."""
	symbols: list[dict[str, Any]] = []
	for symbol in _sorted_symbols(model):
		if symbol["kind"] == "field":
			continue
		children = [
			{
				"name": field["name"].split(".", 1)[1],
				"kind": "field",
				"range": field["range"],
				"selection_range": field["range"],
				"children": [],
			}
			for field in _sorted_symbols(model)
			if field["kind"] == "field" and field["name"].startswith(f"{symbol['name']}.")
		]
		symbols.append({
			"name": symbol["name"],
			"kind": symbol["kind"],
			"range": symbol["range"],
			"selection_range": symbol["range"],
			"children": children,
		})
	return symbols


def code_actions(diagnostics: list[dict[str, Any]]) -> list[dict[str, Any]]:
	"""Return deterministic quick-fix suggestions for diagnostics."""
	actions: list[dict[str, Any]] = []
	for diagnostic in diagnostics:
		code = str(diagnostic.get("code") or "")
		message = str(diagnostic.get("message") or "")
		if code == "APG0001":
			actions.append({
				"title": "Inspect syntax and run APG formatter",
				"kind": "quickfix",
				"diagnostic": code,
				"command": "apg.format.check",
			})
		elif "unknown" in message.lower() or "missing" in message.lower():
			actions.append({
				"title": "Create missing APG declaration",
				"kind": "quickfix",
				"diagnostic": code,
				"command": "apg.createDeclaration",
			})
		else:
			actions.append({
				"title": "Explain APG diagnostic",
				"kind": "quickfix",
				"diagnostic": code,
				"command": "apg.explainDiagnostic",
			})
	return actions


def formatting(source: str) -> dict[str, Any]:
	"""Return formatted text using the shared APG formatter."""
	return format_apg_source(source).to_dict(include_text=True)


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


def _diagnostic_from_error(error: APGSyntaxError | SemanticError | Exception, path: Path, severity: str) -> dict[str, Any]:
	if isinstance(error, APGSyntaxError):
		line = max(0, int(error.line or 1) - 1)
		column = max(0, int(error.column or 0))
		return {
			"code": "APG0001",
			"title": "Syntax error",
			"severity": severity,
			"message": error.message,
			"file": str(path),
			"range": {
				"start": {"line": line, "character": column},
				"end": {"line": line, "character": column + 1},
			},
			"related_locations": [],
			"fixes": [],
			"docs_url": "docs/tooling.md#diagnostic-specification",
		}
	return {
		"code": "APG9000",
		"title": "Language service error",
		"severity": severity,
		"message": str(error),
		"file": str(path),
		"range": {
			"start": {"line": 0, "character": 0},
			"end": {"line": 0, "character": 1},
		},
		"related_locations": [],
		"fixes": [],
		"docs_url": "docs/tooling.md#language-server-specification",
	}


def _completion(label: str, kind: str, detail: str, insert_text: str) -> dict[str, Any]:
	return {
		"label": label,
		"kind": kind,
		"detail": detail,
		"insert_text": insert_text,
	}


def _completion_kind(symbol_kind: str) -> str:
	return {
		"table": "Class",
		"field": "Property",
		"agent": "Class",
		"capability": "Interface",
		"composition": "Module",
		"flow": "Event",
		"app": "Module",
		"module": "Module",
	}.get(symbol_kind, "Variable")


def _dedupe_completions(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
	seen: set[tuple[str, str]] = set()
	unique: list[dict[str, Any]] = []
	for item in items:
		key = (item["label"], item["kind"])
		if key in seen:
			continue
		seen.add(key)
		unique.append(item)
	return unique


def _sorted_symbols(model: dict[str, Any]) -> list[dict[str, Any]]:
	return sorted(
		model.get("symbols", {}).values(),
		key=lambda symbol: (str(symbol.get("kind")), str(symbol.get("name"))),
	)


def _find_symbol(model: dict[str, Any], word: str) -> dict[str, Any] | None:
	for symbol in _sorted_symbols(model):
		if symbol.get("name") == word or str(symbol.get("name", "")).split(".")[-1] == word:
			return symbol
	return None


def _symbol_location(symbol: dict[str, Any]) -> dict[str, Any]:
	return {
		"file": symbol["file"],
		"range": symbol["range"],
	}


def _line_prefix(source: str, line: int | None, character: int | None) -> str:
	if line is None or character is None:
		return ""
	lines = source.splitlines()
	if line < 0 or line >= len(lines):
		return ""
	return lines[line][:character]


def _looks_like_field_context(context: str) -> bool:
	return ":" in context or context.strip().endswith(("{", ","))


def _looks_like_agent_context(context: str) -> bool:
	return "agent" in context or any(token in context for token in ("role", "model", "runtime", "system"))
