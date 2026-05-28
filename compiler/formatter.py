"""Deterministic APG source formatter."""

from __future__ import annotations

import re
from dataclasses import dataclass


TOP_LEVEL_DECLARATION = re.compile(
	r"^(module|app|application|table|db|database|view|screen|form|flow|workflow|"
	r"operation|rule|role|permission|agent|agent_team|team|swarm|capability|"
	r"api|event|job|report|menu|component|package|deploy|deployment)\b"
)


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
		):
			lines.append("")

		line_indent = max(indent_level - _leading_close_count(stripped), 0)
		lines.append(f"{'  ' * line_indent}{stripped}")
		indent_level = max(indent_level + _brace_delta(stripped), 0)

	return "\n".join(lines).rstrip() + "\n"
