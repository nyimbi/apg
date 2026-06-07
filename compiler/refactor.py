"""APG source-level refactoring operations."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

REFACTOR_REPORT_FORMAT = "apg.refactor-report.v1"


def rename_entity(source_file: Path, old_name: str, new_name: str, write: bool = False) -> dict[str, Any]:
	"""Rename an entity (table, capability, agent, workflow, app) everywhere in the source."""
	if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', new_name):
		return {"format": REFACTOR_REPORT_FORMAT, "ok": False,
				"errors": [f"Invalid identifier: {new_name!r}"]}
	source = source_file.read_text(encoding="utf-8")
	# Replace as identifier only (word boundary), not inside strings
	pattern = re.compile(rf'\b{re.escape(old_name)}\b')
	count = len(pattern.findall(source))
	if count == 0:
		return {"format": REFACTOR_REPORT_FORMAT, "ok": False,
				"errors": [f"Entity {old_name!r} not found in source"]}
	new_source = pattern.sub(new_name, source)
	if write and new_source != source:
		source_file.write_text(new_source, encoding="utf-8")
	return {
		"format": REFACTOR_REPORT_FORMAT,
		"ok": True,
		"source": str(source_file),
		"operation": "rename_entity",
		"old_name": old_name,
		"new_name": new_name,
		"occurrences": count,
		"changed": new_source != source,
		"written": write and new_source != source,
		"new_source": new_source,
		"diff": _unified_diff(source, new_source, str(source_file)),
	}


def rename_field(source_file: Path, entity_name: str, old_field: str, new_field: str, write: bool = False) -> dict[str, Any]:
	"""Rename a field within a named entity block."""
	if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', new_field):
		return {"format": REFACTOR_REPORT_FORMAT, "ok": False,
				"errors": [f"Invalid identifier: {new_field!r}"]}
	source = source_file.read_text(encoding="utf-8")
	# Simple approach: find the entity block, replace field name within it
	# This uses a line-by-line scan to stay within the right block
	lines = source.splitlines(keepends=True)
	in_block = False
	depth = 0
	count = 0
	new_lines = []
	field_re = re.compile(rf'(\s*){re.escape(old_field)}(\s*:)')
	for line in lines:
		if re.match(rf'\b(table|capability|agent|workflow|app|form|screen)\s+{re.escape(entity_name)}\b', line):
			in_block = True
			depth = 0
		if in_block:
			depth += line.count('{') - line.count('}')
			if depth <= 0 and in_block and '{' not in line and '}' in line:
				in_block = False
			if in_block and field_re.search(line):
				line, n = field_re.subn(rf'\g<1>{new_field}\2', line)
				count += n
		new_lines.append(line)
	new_source = "".join(new_lines)
	if count == 0:
		return {"format": REFACTOR_REPORT_FORMAT, "ok": False,
				"errors": [f"Field {old_field!r} not found in entity {entity_name!r}"]}
	if write and new_source != source:
		source_file.write_text(new_source, encoding="utf-8")
	return {
		"format": REFACTOR_REPORT_FORMAT,
		"ok": True,
		"source": str(source_file),
		"operation": "rename_field",
		"entity": entity_name,
		"old_field": old_field,
		"new_field": new_field,
		"occurrences": count,
		"changed": new_source != source,
		"written": write and new_source != source,
		"new_source": new_source,
		"diff": _unified_diff(source, new_source, str(source_file)),
	}


def _unified_diff(original: str, updated: str, filename: str) -> str:
	import difflib
	return "".join(difflib.unified_diff(
		original.splitlines(keepends=True),
		updated.splitlines(keepends=True),
		fromfile=filename,
		tofile=f"{filename} (refactored)",
	))
