"""Source preparation and statement splitting (spec 1.2, 1.4).

Two jobs, in the order the spec gives them:

1. ``prepare_source`` — fence extraction, comment stripping, trimming.
2. ``split_statements`` — cut the text into statements, honouring the three
   continuation rules and reporting whether the text ended mid-construct.

Both are used by the streaming parser, which re-runs them on every chunk over
the pending tail only.
"""

from __future__ import annotations

from dataclasses import dataclass

OPENUI_MARKER = "]]>openui:"


@dataclass(frozen=True)
class Prepared:
	text: str
	# True when the source ended inside an unclosed string or bracket, or on an
	# open ternary. The caller closes it implicitly (spec 4.2).
	incomplete: bool


def strip_markers(text: str) -> str:
	"""Drop ``]]>openui:kind`` marker lines (spec 7.1).

	A marker counts only at the start of a line, except a `content` marker,
	which may start a new line mid-line.
	"""
	lines = text.split("\n")
	kept: list[str] = []
	for line in lines:
		if line.startswith(OPENUI_MARKER):
			continue
		index = line.find(OPENUI_MARKER)
		if index > 0 and _is_content_marker_at(line, index):
			line = line[index:]
			if line.startswith(OPENUI_MARKER):
				continue
		kept.append(line)
	return "\n".join(kept)


def _is_content_marker_at(line: str, index: int) -> bool:
	rest = line[index + len(OPENUI_MARKER):]
	return rest.startswith("content") and (len(rest) == 7 or rest[7] in "?\r")


def extract_fences(text: str) -> str:
	"""Join the contents of every fenced block; prose outside is discarded.

	An unterminated fence runs to the end of the text (spec 1.2).
	"""
	lines = text.split("\n")
	fences: list[str] = []
	current: list[str] | None = None
	for line in lines:
		if line.startswith("```"):
			if current is None:
				current = []
			else:
				fences.append("\n".join(current))
				current = None
			continue
		if current is not None:
			current.append(line)
	if current is not None:
		fences.append("\n".join(current))
	if not fences:
		return text
	return "\n".join(fences)


def strip_comments(text: str) -> str:
	"""Remove `//` and `#` comments, respecting string literals."""
	output: list[str] = []
	index = 0
	length = len(text)
	quote = ""
	while index < length:
		char = text[index]
		if quote:
			output.append(char)
			if char == "\\" and index + 1 < length:
				output.append(text[index + 1])
				index += 2
				continue
			if char == quote:
				quote = ""
			index += 1
			continue
		if char in "\"'":
			quote = char
			output.append(char)
			index += 1
			continue
		if char == "#" or text.startswith("//", index):
			while index < length and text[index] != "\n":
				index += 1
			continue
		output.append(char)
		index += 1
	return "".join(output)


def prepare_source(text: str) -> Prepared:
	"""Spec 1.2, in order: markers, fences, comments, trimming."""
	without_markers = strip_markers(text)
	fenced = extract_fences(without_markers)
	uncommented = strip_comments(fenced)
	return Prepared(uncommented.strip(), _ends_incomplete(uncommented))


def _ends_incomplete(text: str) -> bool:
	"""True when the text ends inside a string, bracket, or open ternary."""
	depth = 0
	quote = ""
	index = 0
	length = len(text)
	ternary_open = False
	ternary_seen_colon = False
	while index < length:
		char = text[index]
		if quote:
			if char == "\\":
				index += 2
				continue
			if char == quote:
				quote = ""
			index += 1
			continue
		if char in "\"'":
			quote = char
			index += 1
			continue
		if char in "([{":
			depth += 1
		elif char in ")]}":
			depth = max(0, depth - 1)
		elif char == "?" and depth == 0:
			ternary_open = True
			ternary_seen_colon = False
		elif char == ":" and depth == 0 and ternary_open:
			ternary_seen_colon = True
		elif char == "\n" and depth == 0 and ternary_open and ternary_seen_colon:
			ternary_open = False
		index += 1
	return bool(quote) or depth > 0 or (ternary_open and not ternary_seen_colon)


@dataclass(frozen=True)
class StatementText:
	name: str
	text: str
	complete: bool
	needed_closing: bool


def split_statements(text: str, *, final: bool = True) -> list[StatementText]:
	"""Cut *text* into statements at bracket depth zero (spec 1.4).

	* Statements continue across a newline inside an unclosed bracket or string.
	* A ternary open at depth zero continues the statement.
	* A line whose first non-space token is `?` continues the previous one.

	With ``final=False`` the last statement may be pending: it is closed
	implicitly and marked incomplete, matching the streaming rule that a
	completed statement never changes.
	"""
	lines = text.split("\n")
	statements: list[StatementText] = []
	buffer: list[str] = []
	depth = 0
	quote = ""
	ternary_open = False
	ternary_colon = False
	# A line ended with the statement structurally closed, but the ternary
	# lookahead is undecided until the next line's first token arrives (4.1).
	candidate = False

	def flush(complete: bool, needed_closing: bool = False) -> None:
		raw = "\n".join(buffer).strip()
		if raw:
			name = _statement_name(raw)
			if name:
				statements.append(StatementText(name, raw, complete, needed_closing))
		buffer.clear()

	def reset_state() -> tuple[int, str, bool, bool]:
		return 0, "", False, False

	for line in lines:
		stripped = line.lstrip()
		if buffer and candidate:
			if stripped.startswith("?"):
				# Rule 3: a line whose first token is `?` continues the previous
				# statement, so the statement was not complete after all.
				candidate = False
			elif _statement_name("\n".join(buffer).strip()):
				flush(complete=True)
				depth, quote, ternary_open, ternary_colon = reset_state()
			else:
				candidate = False
		if not buffer and not stripped:
			continue
		buffer.append(line)
		depth, quote, ternary_open, ternary_colon = _scan(
			line, depth, quote, ternary_open, ternary_colon
		)
		if _inside(quote, depth) or (ternary_open and not ternary_colon):
			candidate = False
			continue
		candidate = True

	if buffer:
		raw = "\n".join(buffer).strip()
		if raw:
			name = _statement_name(raw)
			if name:
				statements.append(
					StatementText(name, raw, complete=final and candidate, needed_closing=not candidate)
				)
	return statements


def _inside(quote: str, depth: int) -> bool:
	return bool(quote) or depth > 0


def _scan(line: str, depth: int, quote: str, ternary_open: bool, ternary_colon: bool):
	index = 0
	while index < len(line):
		char = line[index]
		if quote:
			if char == "\\":
				# An escape consumes the next character, so `\"` never closes.
				index += 2
				continue
			if char == quote:
				quote = ""
			index += 1
			continue
		if char in "\"'":
			quote = char
		elif char in "([{":
			depth += 1
		elif char in ")]}":
			depth = max(0, depth - 1)
		elif char == "?" and depth == 0:
			ternary_open = True
			ternary_colon = False
		elif char == ":" and depth == 0 and ternary_open:
			ternary_colon = True
		index += 1
	if depth == 0 and ternary_open and ternary_colon:
		ternary_open = False
		ternary_colon = False
	return depth, quote, ternary_open, ternary_colon


def _statement_name(raw: str) -> str:
	first_line = raw.split("\n", 1)[0].strip()
	if "=" not in first_line:
		# A statement is `name = expression`; anything else is skipped (spec 1.4).
		return ""
	head = first_line.split("=", 1)[0].strip()
	if not head:
		return ""
	if head.startswith("$"):
		body = head[1:]
		if body and (body[0].isalpha() or body[0] == "_") and body.replace("_", "").isalnum():
			return head
		return ""
	if head[0].isalpha() or head[0] == "_":
		if head.replace("_", "").isalnum():
			return head
	return ""


def close_implicitly(text: str) -> str:
	"""Close an unterminated string, then unclosed brackets in reverse (spec 4.2)."""
	depth_stack: list[str] = []
	quote = ""
	index = 0
	length = len(text)
	ends_mid_escape = False
	while index < length:
		char = text[index]
		if quote:
			if char == "\\":
				ends_mid_escape = index + 1 >= length
				index += 2
				continue
			ends_mid_escape = False
			if char == quote:
				quote = ""
			index += 1
			continue
		ends_mid_escape = False
		if char in "\"'":
			quote = char
		elif char in "([{":
			depth_stack.append(char)
		elif char in ")]}":
			expected = {")": "(", "]": "[", "}": "{"}[char]
			if depth_stack and depth_stack[-1] == expected:
				depth_stack.pop()
			# A mismatched closer is skipped and leaves the stack alone.
		index += 1

	result = text
	if quote:
		if ends_mid_escape:
			result += "\\"
		result += quote
	for opener in reversed(depth_stack):
		result += _closer(opener)
	return result


def _closer(opener: str) -> str:
	return {"(": ")", "[": "]", "{": "}"}[opener]
