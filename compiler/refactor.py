"""APG source-level refactoring operations."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

REFACTOR_REPORT_FORMAT = "apg.refactor-report.v1"


def _mask_strings(source: str) -> tuple[str, list[tuple[int, int, str]]]:
	"""Replace string literal contents with placeholders, return holes list."""
	result = list(source)
	holes: list[tuple[int, int, str]] = []  # (start, end, original)
	i = 0
	in_str: str | None = None
	while i < len(source):
		c = source[i]
		if in_str is None:
			if c in ('"', "'"):
				in_str = c
				start = i
		else:
			if c == '\\' and i + 1 < len(source):
				i += 2
				continue
			if c == in_str:
				# found closing quote — mask the content (not the quotes)
				end = i + 1
				holes.append((start, end, source[start:end]))
				placeholder = c + '\x00' * (end - start - 2) + c
				for j, ch in enumerate(placeholder):
					result[start + j] = ch
				in_str = None
		i += 1
	return "".join(result), holes


def _unmask_strings(masked: str, holes: list[tuple[int, int, str]]) -> str:
	"""Restore masked string literals."""
	result = list(masked)
	for start, end, original in holes:
		for j, ch in enumerate(original):
			result[start + j] = ch
	return "".join(result)


def rename_entity(source_file: Path, old_name: str, new_name: str, write: bool = False) -> dict[str, Any]:
	"""Rename an entity (table, capability, agent, workflow, app) everywhere in the source."""
	if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', new_name):
		return {"format": REFACTOR_REPORT_FORMAT, "ok": False,
				"errors": [f"Invalid identifier: {new_name!r}"]}
	source = source_file.read_text(encoding="utf-8")

	# Protect string literal content from replacement
	masked, holes = _mask_strings(source)
	pattern = re.compile(rf'\b{re.escape(old_name)}\b')
	count = len(pattern.findall(masked))
	if count == 0:
		# also check if it appears in string literals (informational)
		total = len(re.compile(rf'\b{re.escape(old_name)}\b').findall(source))
		msg = f"Entity {old_name!r} not found in source"
		if total > count:
			msg += f" (appears {total} time(s) inside string literals — not renamed)"
		return {"format": REFACTOR_REPORT_FORMAT, "ok": False,
				"errors": [msg]}

	new_masked = pattern.sub(new_name, masked)
	new_source = _unmask_strings(new_masked, holes)
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
			if in_block and depth == 1 and field_re.search(line):
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
