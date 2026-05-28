"""Deterministic APG source formatter."""

from __future__ import annotations

import re
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


TOP_LEVEL_DECLARATION = re.compile(
	r"^(module|app|application|table|db|database|view|screen|form|flow|workflow|"
	r"operation|rule|role|permission|agent|agent_team|team|swarm|capability|"
	r"api|event|job|report|menu|component|package|deploy|deployment)\b"
)
FIELD_MODIFIER_DECLARATION = re.compile(
	r"^(?P<prefix>[^\W\d]\w*\s*:\s*[^\[\]{};]+?)\s*"
	r"\[(?P<modifiers>.+?)\](?P<suffix>\s*;?\s*(?://.*)?)$",
	re.UNICODE,
)
DEFAULT_FORMATTER_CATALOG = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "formatter" / "catalog.json"
FORMATTER_AUDIT_FORMAT = "apg.formatter-audit.v1"


@dataclass(frozen=True)
class FormatResult:
	"""Result of formatting one APG source string."""

	changed: bool
	text: str
	diagnostics: list[dict[str, object]]
	idempotent: bool

	def to_dict(self, include_text: bool = True) -> dict[str, object]:
		payload: dict[str, object] = {
			"format": "apg.format-result.v1",
			"changed": self.changed,
			"idempotent": self.idempotent,
			"diagnostics": self.diagnostics,
		}
		if include_text:
			payload["text"] = self.text
		return payload


def _brace_delta(text: str) -> int:
	"""Count braces outside strings and line comments."""
	delta = 0
	quote: str | None = None
	escaped = False
	index = 0
	while index < len(text):
		char = text[index]
		next_char = text[index + 1] if index + 1 < len(text) else ""
		if quote is None and char == "/" and next_char == "/":
			break
		if quote is not None:
			if escaped:
				escaped = False
			elif char == "\\":
				escaped = True
			elif char == quote:
				quote = None
			index += 1
			continue
		if char in {'"', "'"}:
			quote = char
		elif char == "{":
			delta += 1
		elif char == "}":
			delta -= 1
		index += 1
	return delta


def _leading_close_count(text: str) -> int:
	count = 0
	for char in text:
		if char == "}":
			count += 1
			continue
		break
	return count


def format_apg_source(source: str) -> FormatResult:
	"""Format APG source using stable, conservative whitespace rules."""
	formatted = _format_text(source)
	idempotent = _format_text(formatted) == formatted
	return FormatResult(
		changed=formatted != source,
		text=formatted,
		diagnostics=[],
		idempotent=idempotent,
	)


def _format_text(source: str) -> str:
	lines: list[str] = []
	indent_level = 0

	for raw_line in source.splitlines():
		stripped = raw_line.expandtabs(2).strip()
		if not stripped:
			if lines and lines[-1] != "":
				lines.append("")
			continue

		if (
			indent_level == 0
			and TOP_LEVEL_DECLARATION.match(stripped)
			and lines
			and lines[-1] != ""
			and not lines[-1].lstrip().startswith("//")
		):
			lines.append("")

		stripped = _normalize_field_modifiers(stripped)
		line_indent = max(indent_level - _leading_close_count(stripped), 0)
		lines.append(f"{'  ' * line_indent}{stripped}")
		indent_level = max(indent_level + _brace_delta(stripped), 0)

	return "\n".join(lines).rstrip() + "\n"


def _normalize_field_modifiers(line: str) -> str:
	"""Canonicalize typed field modifier order without touching list literals."""
	match = FIELD_MODIFIER_DECLARATION.match(line)
	if not match:
		return line

	modifiers = _split_modifier_items(match.group("modifiers"))
	if len(modifiers) < 2:
		return line

	ordered = sorted(
		enumerate(modifiers),
		key=lambda item: (_modifier_rank(item[1]), item[0]),
	)
	normalized = ", ".join(modifier for _index, modifier in ordered)
	return f"{match.group('prefix')} [{normalized}]{match.group('suffix')}"


def _split_modifier_items(text: str) -> list[str]:
	items: list[str] = []
	start = 0
	quote: str | None = None
	escaped = False
	depth = 0
	for index, char in enumerate(text):
		if quote is not None:
			if escaped:
				escaped = False
			elif char == "\\":
				escaped = True
			elif char == quote:
				quote = None
			continue
		if char in {'"', "'"}:
			quote = char
		elif char in "([{":
			depth += 1
		elif char in ")]}" and depth:
			depth -= 1
		elif char == "," and depth == 0:
			items.append(text[start:index].strip())
			start = index + 1
	items.append(text[start:].strip())
	return [item for item in items if item]


def _modifier_rank(modifier: str) -> int:
	normalized = modifier.strip().lower()
	if normalized in {"pk", "primary key"}:
		return 0
	if normalized == "required" or normalized.startswith("required:"):
		return 1
	if normalized == "unique" or normalized.startswith("unique:"):
		return 2
	if normalized == "hidden" or normalized.startswith("hidden:"):
		return 3
	if normalized in {"search", "searchable"} or normalized.startswith(("search:", "searchable:")):
		return 4
	if normalized.startswith("default"):
		return 5
	if normalized.startswith("ref ") or "->" in normalized:
		return 6
	return 7


def audit_formatter_fixtures(catalog_path: Path | None = None) -> dict[str, Any]:
	"""Run the checked-in formatter fixture catalog."""
	catalog_file = Path(catalog_path or DEFAULT_FORMATTER_CATALOG)
	catalog_root = catalog_file.parent
	catalog = json.loads(catalog_file.read_text(encoding="utf-8"))
	required_tags = sorted(str(tag) for tag in catalog.get("tags_required", []))
	covered_tags: set[str] = set()
	fixture_reports: list[dict[str, Any]] = []
	blocking_gaps: list[dict[str, Any]] = []

	for fixture in catalog.get("fixtures", []):
		report = _audit_formatter_fixture(catalog_root, fixture)
		fixture_reports.append(report)
		if report["ok"]:
			covered_tags.update(report["tags"])
		else:
			blocking_gaps.append({
				"id": report["id"],
				"source": report["source"],
				"expected": report["expected"],
				"errors": report["errors"],
			})

	missing_tags = sorted(set(required_tags).difference(covered_tags))
	for tag in missing_tags:
		blocking_gaps.append({
			"id": f"missing_tag:{tag}",
			"source": str(catalog_file),
			"expected": str(catalog_file),
			"errors": [f"required formatter fixture tag {tag!r} is not covered by a passing fixture"],
		})

	return {
		"format": FORMATTER_AUDIT_FORMAT,
		"ok": not blocking_gaps,
		"catalog": str(catalog_file),
		"tags_required": required_tags,
		"tags_covered": sorted(covered_tags),
		"missing_tags": missing_tags,
		"fixtures": fixture_reports,
		"summary": {
			"fixture_count": len(fixture_reports),
			"passing_fixture_count": sum(1 for report in fixture_reports if report["ok"]),
			"changed_fixture_count": sum(1 for report in fixture_reports if report["changed"]),
			"blocking_gap_count": len(blocking_gaps),
		},
		"blocking_gaps": blocking_gaps,
	}


def _audit_formatter_fixture(catalog_root: Path, fixture: dict[str, Any]) -> dict[str, Any]:
	fixture_id = str(fixture["id"])
	source_path = (catalog_root / str(fixture["source"])).resolve()
	expected_path = (catalog_root / str(fixture["expected"])).resolve()
	tags = sorted(str(tag) for tag in fixture.get("tags", []))
	errors: list[str] = []

	source = source_path.read_text(encoding="utf-8")
	expected = expected_path.read_text(encoding="utf-8")
	result = format_apg_source(source)
	second_pass = format_apg_source(result.text)

	if result.text != expected:
		errors.append("formatted output differs from expected fixture")
	if not result.idempotent or second_pass.text != result.text:
		errors.append("formatter output is not idempotent")

	return {
		"id": fixture_id,
		"source": str(source_path),
		"expected": str(expected_path),
		"tags": tags,
		"changed": result.changed,
		"idempotent": result.idempotent and second_pass.text == result.text,
		"ok": not errors,
		"errors": errors,
	}
