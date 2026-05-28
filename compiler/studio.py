"""Dependency-light APG Studio and visual-designer round-trip service."""

from __future__ import annotations

import json
import re
from difflib import unified_diff
from pathlib import Path
from typing import Any

from compiler.semantic_model import build_semantic_model


STUDIO_SNAPSHOT_FORMAT = "apg.studio-snapshot.v1"
STUDIO_EDIT_PLAN_FORMAT = "apg.studio-edit-plan.v1"
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SCALAR_TYPES = {"str", "int", "float", "bool", "date", "datetime", "json", "any"}


def build_studio_snapshot(path: Path) -> dict[str, Any]:
	"""Build a Studio-ready designer snapshot from one APG source file."""
	source = path.read_text(encoding="utf-8")
	model = build_semantic_model(path)
	return {
		"format": STUDIO_SNAPSHOT_FORMAT,
		"ok": bool(model.get("ok")),
		"file": str(path),
		"semantic_model_format": model.get("format"),
		"diagnostics": model.get("diagnostics", []),
		"panels": {
			"dsl_editor": _dsl_editor_panel(path, source),
			"component_palette": _component_palette(),
			"database_designer": _database_designer(model),
			"form_designer": _form_designer(model),
			"workflow_designer": _workflow_designer(model),
			"capability_composition_designer": _capability_designer(model),
			"package_deployment_designer": _package_deployment_designer(model),
			"graph_explain_panel": _graph_explain_panel(model),
		},
		"round_trip": {
			"source_of_truth": "apg_dsl",
			"supported_edit_operations": [
				"add_table",
				"add_field",
				"add_agent",
				"add_capability",
				"add_screen",
			],
			"invalid_visual_edits_are_rejected": True,
		},
	}


def build_studio_edit_plan(
	path: Path,
	edit: dict[str, Any],
	write: bool = False,
) -> dict[str, Any]:
	"""Plan and optionally apply a visual-designer edit as APG DSL."""
	source = path.read_text(encoding="utf-8")
	current = build_studio_snapshot(path)
	errors = _validate_edit(edit, current)
	new_source = source
	if not errors:
		new_source = _apply_visual_edit(source, edit)

	post_snapshot = None
	post_errors: list[str] = []
	if not errors:
		post_snapshot = _snapshot_from_source(path, new_source)
		for diagnostic in post_snapshot.get("diagnostics", []):
			if diagnostic.get("severity") == "error":
				post_errors.append(str(diagnostic.get("message", "semantic error")))

	errors.extend(post_errors)
	ok = not errors
	written = False
	if ok and write and new_source != source:
		path.write_text(new_source, encoding="utf-8")
		written = True

	return {
		"format": STUDIO_EDIT_PLAN_FORMAT,
		"ok": ok,
		"file": str(path),
		"operation": edit.get("operation"),
		"errors": errors,
		"changed": ok and new_source != source,
		"written": written,
		"requires_review": True,
		"review_reasons": _review_reasons(edit),
		"diff": _source_diff(source, new_source, str(path)) if ok and new_source != source else "",
		"edit": edit,
		"pre_snapshot_summary": _snapshot_summary(current),
		"post_snapshot_summary": _snapshot_summary(post_snapshot) if post_snapshot else None,
		"new_source": new_source,
	}


def _dsl_editor_panel(path: Path, source: str) -> dict[str, Any]:
	return {
		"language": "apg",
		"file": str(path),
		"source_length": len(source),
		"line_count": len(source.splitlines()),
	}


def _component_palette() -> dict[str, Any]:
	return {
		"components": [
			{"kind": "field", "label": "Text field", "creates": "add_field", "type": "str"},
			{"kind": "field", "label": "Number field", "creates": "add_field", "type": "float"},
			{"kind": "field", "label": "Boolean toggle", "creates": "add_field", "type": "bool"},
			{"kind": "table", "label": "Data table", "creates": "add_table"},
			{"kind": "screen", "label": "Screen", "creates": "add_screen"},
			{"kind": "agent", "label": "AI agent", "creates": "add_agent"},
			{"kind": "capability", "label": "Capability contract", "creates": "add_capability"},
		],
		"theme_tokens": ["primary", "surface", "accent", "density", "radius"],
	}


def _database_designer(model: dict[str, Any]) -> dict[str, Any]:
	tables = []
	for table_name, table in sorted(model.get("tables", {}).items()):
		fields = []
		for field_name, field in sorted(table.get("fields", {}).items()):
			fields.append({
				"name": field_name,
				"type": field.get("type"),
				"required": bool(field.get("required")),
				"relationship": field.get("relationship"),
			})
		tables.append({
			"name": table_name,
			"fields": fields,
			"lookup_paths": table.get("lookup_paths", {}),
		})
	return {"tables": tables}


def _form_designer(model: dict[str, Any]) -> dict[str, Any]:
	forms = []
	for view_name, view in sorted(model.get("views", {}).items()):
		forms.append({
			"name": view_name,
			"type": view.get("type"),
			"bindings": view.get("bindings", []),
			"properties": view.get("properties", {}),
		})
	if not forms:
		for table_name, table in sorted(model.get("tables", {}).items()):
			forms.append({
				"name": f"{table_name}Form",
				"type": "generated_form_projection",
				"table": table_name,
				"bindings": sorted(table.get("fields", {}).keys()),
			})
	return {"forms": forms}


def _workflow_designer(model: dict[str, Any]) -> dict[str, Any]:
	return {
		"flows": [
			{"name": name, **flow}
			for name, flow in sorted(model.get("flows", {}).items())
		]
	}


def _capability_designer(model: dict[str, Any]) -> dict[str, Any]:
	return {
		"capabilities": [
			{"name": name, **capability}
			for name, capability in sorted(model.get("capabilities", {}).items())
		],
		"composition": model.get("composition", {}),
		"contracts": model.get("contracts", {}),
	}


def _package_deployment_designer(model: dict[str, Any]) -> dict[str, Any]:
	return {
		"packages": model.get("packages", {}),
		"deployment": model.get("deployment", {}),
	}


def _graph_explain_panel(model: dict[str, Any]) -> dict[str, Any]:
	return {
		"graphs": model.get("graphs", {}),
		"symbol_count": len(model.get("symbols", {})),
	}


def _validate_edit(edit: dict[str, Any], snapshot: dict[str, Any]) -> list[str]:
	errors: list[str] = []
	operation = edit.get("operation")
	if operation not in {"add_table", "add_field", "add_agent", "add_capability", "add_screen"}:
		return [f"Unsupported Studio edit operation: {operation}"]

	name = str(edit.get("name", ""))
	if operation != "add_field" and not _valid_identifier(name):
		errors.append(f"Invalid APG identifier for name: {name}")

	tables = {
		table["name"]
		for table in snapshot["panels"]["database_designer"]["tables"]
	}

	if operation == "add_table" and name in tables:
		errors.append(f"Table already exists: {name}")
	elif operation == "add_field":
		table_name = str(edit.get("table", ""))
		field_name = str(edit.get("name", ""))
		field_type = str(edit.get("type", "str"))
		if table_name not in tables:
			errors.append(f"Cannot add field to unknown table: {table_name}")
		if not _valid_identifier(field_name):
			errors.append(f"Invalid APG identifier for field: {field_name}")
		if not _valid_type(field_type, tables):
			errors.append(f"Unknown field type for visual edit: {field_type}")
		for table in snapshot["panels"]["database_designer"]["tables"]:
			if table["name"] == table_name and any(field["name"] == field_name for field in table["fields"]):
				errors.append(f"Field already exists: {table_name}.{field_name}")
	elif operation == "add_screen":
		table_name = str(edit.get("table", ""))
		if table_name and table_name not in tables:
			errors.append(f"Cannot bind screen to unknown table: {table_name}")
	return errors


def _apply_visual_edit(source: str, edit: dict[str, Any]) -> str:
	operation = edit["operation"]
	if operation == "add_table":
		return _append_block(source, _table_block(edit))
	if operation == "add_field":
		return _insert_field(source, str(edit["table"]), _field_line(edit))
	if operation == "add_agent":
		return _append_block(source, _agent_block(edit))
	if operation == "add_capability":
		return _append_block(source, _capability_block(edit))
	if operation == "add_screen":
		return _append_block(source, _screen_block(edit))
	return source


def _table_block(edit: dict[str, Any]) -> str:
	fields = edit.get("fields") or [{"name": "id", "type": "int"}]
	lines = [f"table {edit['name']} {{"]
	for field in fields:
		lines.append(_field_line(field))
	lines.append("}")
	return "\n".join(lines) + "\n"


def _field_line(edit: dict[str, Any]) -> str:
	required = " [required]" if edit.get("required") else ""
	return f"  {edit['name']}: {edit.get('type', 'str')}{required};"


def _insert_field(source: str, table_name: str, field_line: str) -> str:
	pattern = re.compile(rf"(table\s+{re.escape(table_name)}\s*\{{)(.*?)(\n\}})", re.DOTALL)
	match = pattern.search(source)
	if not match:
		return source
	body = match.group(2).rstrip()
	indent_match = re.search(r"\n([ \t]+)[A-Za-z_][A-Za-z0-9_]*\s*:", body)
	if indent_match:
		field_line = re.sub(r"^[ \t]+", indent_match.group(1), field_line)
	new_body = f"{body}\n{field_line}" if body else f"\n{field_line}"
	return source[:match.start()] + match.group(1) + new_body + match.group(3) + source[match.end():]


def _agent_block(edit: dict[str, Any]) -> str:
	return (
		f"agent {edit['name']} {{\n"
		f"  role: \"{edit.get('role', 'assistant')}\";\n"
		f"  model: \"{edit.get('model', 'openai:gpt-4.1-mini')}\";\n"
		f"  runtime: {edit.get('runtime', 'codex')};\n"
		"}\n"
	)


def _capability_block(edit: dict[str, Any]) -> str:
	name = edit["name"]
	provides = edit.get("provides") or [name.lower()]
	theme = edit.get("theme") or {"primary": "#2563eb", "surface": "#ffffff"}
	return (
		f"capability {name} {{\n"
		"  contract: {\n"
		f"    id: {name.lower()},\n"
		f"    provides: [{', '.join(provides)}]\n"
		"  };\n"
		f"  theme: {_literal_object(theme)};\n"
		"}\n"
	)


def _screen_block(edit: dict[str, Any]) -> str:
	name = edit["name"]
	table = edit.get("table")
	fields = edit.get("fields") or []
	binding = f"\n  table: {table};" if table else ""
	field_line = f"\n  fields: [{', '.join(fields)}];" if fields else ""
	return f"screen {name} {{{binding}{field_line}\n}}\n"


def _literal_object(value: dict[str, Any]) -> str:
	parts = []
	for key, item in sorted(value.items()):
		rendered = f'"{item}"' if isinstance(item, str) and item.startswith("#") else json.dumps(item)
		parts.append(f"{key}: {rendered}")
	return "{" + ", ".join(parts) + "}"


def _append_block(source: str, block: str) -> str:
	return source.rstrip() + "\n\n" + block


def _snapshot_from_source(path: Path, source: str) -> dict[str, Any]:
	temporary = path.with_suffix(path.suffix + ".studio.tmp")
	try:
		temporary.write_text(source, encoding="utf-8")
		return build_studio_snapshot(temporary)
	finally:
		if temporary.exists():
			temporary.unlink()


def _snapshot_summary(snapshot: dict[str, Any] | None) -> dict[str, Any]:
	if not snapshot:
		return {}
	panels = snapshot.get("panels", {})
	return {
		"tables": len(panels.get("database_designer", {}).get("tables", [])),
		"forms": len(panels.get("form_designer", {}).get("forms", [])),
		"flows": len(panels.get("workflow_designer", {}).get("flows", [])),
		"capabilities": len(panels.get("capability_composition_designer", {}).get("capabilities", [])),
		"diagnostics": len(snapshot.get("diagnostics", [])),
	}


def _review_reasons(edit: dict[str, Any]) -> list[str]:
	return {
		"add_table": ["database designer edit changes schema and migration surface"],
		"add_field": ["form/database designer edit changes schema and generated APIs"],
		"add_agent": ["agent designer edit changes runtime, model, and permission surface"],
		"add_capability": ["capability designer edit changes composition contract surface"],
		"add_screen": ["screen designer edit changes UI bindings and generated routes"],
	}.get(str(edit.get("operation")), ["visual edit requires review"])


def _valid_identifier(value: str) -> bool:
	return bool(IDENTIFIER_PATTERN.fullmatch(value))


def _valid_type(value: str, tables: set[str]) -> bool:
	return value in SCALAR_TYPES or value in tables


def _source_diff(source: str, new_source: str, source_name: str) -> str:
	return "\n".join(unified_diff(
		source.splitlines(),
		new_source.splitlines(),
		fromfile=source_name,
		tofile=source_name,
		lineterm="",
	)) + "\n"
