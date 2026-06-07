"""Dependency-light APG Studio and visual-designer round-trip service."""

from __future__ import annotations

import json
import re
from difflib import unified_diff
from pathlib import Path
from typing import Any

from compiler.semantic_model import (
	build_semantic_model,
	build_semantic_model_from_source,
	invalidate_semantic_model_cache,
)

# ── snapshot cache ──────────────────────────────────────────────────────────
# Keyed by (resolved_path, mtime_ns) — avoids rebuilding snapshot on repeated
# calls to build_studio_snapshot for the same unchanged file.
_SNAPSHOT_CACHE: dict[tuple[str, int], dict[str, Any]] = {}


STUDIO_SNAPSHOT_FORMAT = "apg.studio-snapshot.v1"
STUDIO_EDIT_PLAN_FORMAT = "apg.studio-edit-plan.v1"
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SCALAR_TYPES = {"str", "int", "float", "bool", "date", "datetime", "json", "any"}


def build_studio_snapshot(path: Path) -> dict[str, Any]:
	"""Build a Studio-ready designer snapshot from one APG source file.

	Cached by (resolved_path, mtime_ns); invalidated automatically when the
	file changes.  Call ``invalidate_studio_snapshot_cache(path)`` to evict
	explicitly (e.g. after a write).
	"""
	resolved = str(path.resolve())
	try:
		mtime_ns = path.stat().st_mtime_ns
	except OSError:
		mtime_ns = 0
	cache_key = (resolved, mtime_ns)
	if cache_key in _SNAPSHOT_CACHE:
		return _SNAPSHOT_CACHE[cache_key]

	source = path.read_text(encoding="utf-8")
	model = build_semantic_model(path)
	snapshot = _build_snapshot_from_model(model, path, source)
	_SNAPSHOT_CACHE[cache_key] = snapshot
	return snapshot


def invalidate_studio_snapshot_cache(path: Path | None = None) -> None:
	"""Evict one path (or all entries) from the snapshot cache."""
	if path is None:
		_SNAPSHOT_CACHE.clear()
	else:
		resolved = str(path.resolve())
		for key in [k for k in _SNAPSHOT_CACHE if k[0] == resolved]:
			del _SNAPSHOT_CACHE[key]


def _build_snapshot_from_model(model: dict[str, Any], path: Path, source: str) -> dict[str, Any]:
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
				"add_rule",
				"add_workflow_state",
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
		# Invalidate both caches so next read sees the updated source.
		invalidate_studio_snapshot_cache(path)
		invalidate_semantic_model_cache(path)

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
	if operation not in {
		"add_table", "add_field", "add_agent", "add_capability", "add_screen",
		"add_rule", "add_workflow_state",
	}:
		return [f"Unsupported Studio edit operation: {operation}"]

	name = str(edit.get("name", ""))
	if operation not in {"add_field", "add_rule", "add_workflow_state"} and not _valid_identifier(name):
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
	elif operation == "add_rule":
		cap_name = str(edit.get("capability", ""))
		rule_name = str(edit.get("name", ""))
		when = str(edit.get("when", ""))
		action = str(edit.get("action", ""))
		if not cap_name:
			errors.append("add_rule requires 'capability'")
		if not rule_name:
			errors.append("add_rule requires 'name'")
		if not when:
			errors.append("add_rule requires 'when'")
		if not action:
			errors.append("add_rule requires 'action'")
	elif operation == "add_workflow_state":
		workflow_name = str(edit.get("workflow", ""))
		state = str(edit.get("state", ""))
		if not workflow_name:
			errors.append("add_workflow_state requires 'workflow'")
		if not state:
			errors.append("add_workflow_state requires 'state'")
		if not _valid_identifier(state):
			errors.append(f"Invalid APG identifier for state: {state!r}")
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
	if operation == "add_rule":
		return _add_rule_to_capability(source, edit)
	if operation == "add_workflow_state":
		return _add_state_to_workflow(source, edit)
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
	"""Build a snapshot from in-memory source — no temp file written."""
	model = build_semantic_model_from_source(source, display_path=path)
	return _build_snapshot_from_model(model, path, source)


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


def _add_rule_to_capability(source: str, edit: dict[str, Any]) -> str:
	"""Append a rule entry to the rules list inside a named capability's contract block."""
	cap_name = edit["capability"]
	rule_name = edit["name"]
	when = edit["when"]
	action = edit["action"]
	rule_text = f'{{name: "{rule_name}", when: "{when}", action: {action}}}'

	# Find 'rules: [' inside the capability block and append before closing ']'
	# Strategy: locate the capability block, then find the rules list within it.
	cap_pattern = re.compile(
		rf'(capability\s+{re.escape(cap_name)}\s*\{{.*?rules\s*:\s*\[)(.*?)(\])',
		re.DOTALL,
	)
	match = cap_pattern.search(source)
	if not match:
		# No rules list found — append the capability block unchanged, add rules key
		# Find end of capability contract block and insert a rules list
		cap_block_pattern = re.compile(
			rf'(capability\s+{re.escape(cap_name)}\s*\{{.*?contract\s*:\s*\{{)(.*?)(\}}\s*;)',
			re.DOTALL,
		)
		block_match = cap_block_pattern.search(source)
		if not block_match:
			return source
		inner = block_match.group(2).rstrip()
		indent = "        "
		new_inner = f"{inner}\n{indent}rules: [{rule_text}]"
		return (
			source[: block_match.start()]
			+ block_match.group(1)
			+ new_inner
			+ "\n    "
			+ block_match.group(3)
			+ source[block_match.end() :]
		)

	existing = match.group(2).rstrip()
	separator = ",\n            " if existing.strip() else ""
	if existing.strip():
		new_rules_body = f"{existing}{separator}{rule_text}"
	else:
		new_rules_body = f"\n            {rule_text}\n        "
	return (
		source[: match.start()]
		+ match.group(1)
		+ new_rules_body
		+ match.group(3)
		+ source[match.end() :]
	)


def _add_state_to_workflow(source: str, edit: dict[str, Any]) -> str:
	"""Insert a new state into a workflow's steps string, after the specified state."""
	workflow_name = edit["workflow"]
	new_state = edit["state"]
	after_state = edit.get("after", "")

	# Find 'steps: str = "..."' inside the named workflow block
	steps_pattern = re.compile(
		rf'(workflow\s+{re.escape(workflow_name)}\s*\{{.*?steps\s*:\s*str\s*=\s*")(.*?)(")',
		re.DOTALL,
	)
	match = steps_pattern.search(source)
	if not match:
		# Fallback: append a steps line to the workflow block
		wf_block_pattern = re.compile(
			rf'(workflow\s+{re.escape(workflow_name)}\s*\{{)(.*?)(\}})',
			re.DOTALL,
		)
		block_match = wf_block_pattern.search(source)
		if not block_match:
			return source
		inner = block_match.group(2).rstrip()
		new_inner = f'{inner}\n    steps: str = "{new_state}";\n'
		return (
			source[: block_match.start()]
			+ block_match.group(1)
			+ new_inner
			+ block_match.group(3)
			+ source[block_match.end() :]
		)

	steps_str = match.group(2)
	states = [s.strip() for s in re.split(r'\s*->\s*', steps_str)]

	if after_state and after_state in states:
		idx = states.index(after_state)
		states.insert(idx + 1, new_state)
	else:
		states.append(new_state)

	new_steps = " -> ".join(states)
	return (
		source[: match.start()]
		+ match.group(1)
		+ new_steps
		+ match.group(3)
		+ source[match.end() :]
	)


def _review_reasons(edit: dict[str, Any]) -> list[str]:
	return {
		"add_table": ["database designer edit changes schema and migration surface"],
		"add_field": ["form/database designer edit changes schema and generated APIs"],
		"add_agent": ["agent designer edit changes runtime, model, and permission surface"],
		"add_capability": ["capability designer edit changes composition contract surface"],
		"add_screen": ["screen designer edit changes UI bindings and generated routes"],
		"add_rule": ["capability rule edit changes enforcement logic and access control surface"],
		"add_workflow_state": ["workflow state edit changes lifecycle and human-task assignment surface"],
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
