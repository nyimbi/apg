"""
AI Agent Composition Parser
===========================

Small front-end for the first-class AI agent composition syntax.
It gives the new agentic surface an executable path while the legacy
ANTLR grammar is being reduced and regenerated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from .ast_builder import (
	AIAgentDeclaration,
	AgentHandoff,
	AgentMemory,
	AgentTeamDeclaration,
	EntityType,
	ModuleDeclaration,
)


@dataclass
class AIAgentParseError(Exception):
	message: str
	line: int = 0
	column: int = 0

	def __str__(self) -> str:
		return f"{self.line}:{self.column}: {self.message}"


def looks_like_ai_agent_composition(source: str) -> bool:
	"""Return true when the source uses the first-class agentic surface."""
	return bool(
		re.search(r"\b(agent_team|team|swarm)\s+[A-Za-z_][\w]*\s*\{", source)
		or re.search(r"\bagent\s+[A-Za-z_][\w]*\s*\{", source)
	)


def parse_ai_agent_composition(source: str, source_name: str = "<string>") -> ModuleDeclaration:
	"""Parse terse AI agent composition syntax into APG AST nodes."""
	cleaned = _strip_comments(source)
	module = _parse_module(cleaned, source_name)

	for kind, name, body, line, column in _iter_blocks(cleaned):
		if kind == "module":
			continue
		if kind == "agent":
			module.entities.append(_parse_agent(name, body, source_name, line, column))
		elif kind in {"agent_team", "team", "swarm"}:
			nested_agents = [
				_parse_agent(nested_name, nested_body, source_name, nested_line, nested_column)
				for nested_kind, nested_name, nested_body, nested_line, nested_column in _iter_blocks(body)
				if nested_kind == "agent"
			]
			module.entities.extend(nested_agents)
			module.entities.append(_parse_team(
				name,
				body,
				source_name,
				line,
				column,
				additional_agents=[agent.name for agent in nested_agents],
			))

	if not module.entities and looks_like_ai_agent_composition(source):
		raise AIAgentParseError("expected at least one agent or team declaration")

	return module


def _strip_comments(source: str) -> str:
	source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
	source = re.sub(r"//[^\n]*", "", source)
	source = re.sub(r"#[^\n]*", "", source)
	return source


def _parse_module(source: str, source_name: str) -> ModuleDeclaration:
	match = re.search(
		r"\bmodule\s+(?P<name>[A-Za-z_][\w.]*)(?:\s+version\s+(?P<version>[0-9A-Za-z_.+-]+))?",
		source,
	)
	name = match.group("name") if match else "main"
	version = match.group("version") if match and match.group("version") else "1.0.0"
	description = None

	module_body = _block_body_for(source, "module", name) if match else None
	if module_body:
		props = _parse_properties(module_body)
		description_value = props.get("description")
		if isinstance(description_value, str):
			description = description_value

	return ModuleDeclaration(
		name=name,
		version=version,
		description=description,
		source_file=source_name,
	)


def _iter_blocks(source: str) -> Iterable[tuple[str, str, str, int, int]]:
	header = re.compile(r"\b(module|agent|agent_team|team|swarm)\s+([A-Za-z_][\w.]*)\s*(?:version\s+[0-9A-Za-z_.+-]+\s*)?\{")
	pos = 0
	while True:
		match = header.search(source, pos)
		if not match:
			return
		open_brace = source.find("{", match.end() - 1)
		close_brace = _find_matching_brace(source, open_brace)
		if close_brace < 0:
			line, column = _line_column(source, open_brace)
			raise AIAgentParseError("unclosed block", line, column)
		line, column = _line_column(source, match.start())
		yield match.group(1), match.group(2), source[open_brace + 1:close_brace], line, column
		pos = close_brace + 1


def _block_body_for(source: str, kind: str, name: str) -> Optional[str]:
	for block_kind, block_name, body, _line, _column in _iter_blocks(source):
		if block_kind == kind and block_name == name:
			return body
	return None


def _find_matching_brace(source: str, open_brace: int) -> int:
	depth = 0
	quote: Optional[str] = None
	escaped = False
	for index in range(open_brace, len(source)):
		char = source[index]
		if quote:
			if escaped:
				escaped = False
			elif char == "\\":
				escaped = True
			elif char == quote:
				quote = None
			continue
		if char in {"'", '"'}:
			quote = char
		elif char == "{":
			depth += 1
		elif char == "}":
			depth -= 1
			if depth == 0:
				return index
	return -1


def _parse_agent(name: str, body: str, source_name: str, line: int, column: int) -> AIAgentDeclaration:
	props = _parse_properties(body)
	tools = _string_list(props.get("tools"))
	inputs = _string_list(props.get("input", props.get("inputs")))
	outputs = _string_list(props.get("output", props.get("outputs")))
	memory = _parse_memory(props.get("memory"))
	handoffs = _parse_handoffs(name, props.get("handoff", props.get("handoffs")))

	return AIAgentDeclaration(
		entity_type=EntityType.AI_AGENT,
		name=name,
		role=_optional_string(props.get("role")),
		model=_optional_string(props.get("model")),
		runtime=_optional_string(props.get("runtime", props.get("runner"))),
		system_prompt=_optional_string(props.get("system")),
		tools=tools,
		memory=memory,
		inputs=inputs,
		outputs=outputs,
		handoffs=handoffs,
		line=line,
		column=column,
		source_file=source_name,
	)


def _parse_team(
	name: str,
	body: str,
	source_name: str,
	line: int,
	column: int,
	additional_agents: Optional[List[str]] = None,
) -> AgentTeamDeclaration:
	team_body = _remove_nested_blocks(body)
	props = _parse_properties(team_body)
	agents = _string_list(props.get("agents"))
	flow = _parse_handoffs("", props.get("flow"))

	for statement in _split_statements(team_body):
		if "->" in statement and ":" not in statement:
			flow.extend(_parse_handoffs("", statement))

	if not agents:
		agents = _unique([agent for edge in flow for agent in (edge.source, edge.target)])
	agents = _unique([*(additional_agents or []), *agents])

	policy = {
		key: value for key, value in props.items()
		if key not in {"agents", "flow"} and isinstance(value, (str, int, float, bool, list))
	}

	return AgentTeamDeclaration(
		entity_type=EntityType.AGENT_TEAM,
		name=name,
		agents=agents,
		flow=flow,
		policy=policy,
		line=line,
		column=column,
		source_file=source_name,
	)


def _remove_nested_blocks(body: str) -> str:
	"""Remove nested declarations before parsing team-level statements."""
	ranges: List[tuple[int, int]] = []
	header = re.compile(r"\b(agent|agent_team|team|swarm)\s+[A-Za-z_][\w.]*\s*(?:version\s+[0-9A-Za-z_.+-]+\s*)?\{")
	for match in header.finditer(body):
		open_brace = body.find("{", match.end() - 1)
		close_brace = _find_matching_brace(body, open_brace)
		if close_brace >= 0:
			ranges.append((match.start(), close_brace + 1))
	if not ranges:
		return body
	result = []
	start = 0
	for range_start, range_end in ranges:
		result.append(body[start:range_start])
		start = range_end
	result.append(body[start:])
	return "".join(result)


def _parse_properties(body: str) -> Dict[str, Any]:
	props: Dict[str, Any] = {}
	for statement in _split_statements(body):
		if ":" not in statement:
			continue
		key, value = statement.split(":", 1)
		props[key.strip()] = _parse_value(value.strip())
	return props


def _split_statements(body: str) -> List[str]:
	statements: List[str] = []
	start = 0
	depth = 0
	quote: Optional[str] = None
	escaped = False
	for index, char in enumerate(body):
		if quote:
			if escaped:
				escaped = False
			elif char == "\\":
				escaped = True
			elif char == quote:
				quote = None
			continue
		if char in {"'", '"'}:
			quote = char
		elif char in "[{(":
			depth += 1
		elif char in "]})":
			depth -= 1
		elif char == ";" and depth == 0:
			statement = body[start:index].strip()
			if statement:
				statements.append(statement)
			start = index + 1
	remainder = body[start:].strip()
	if remainder:
		statements.append(remainder)
	return statements


def _parse_value(value: str) -> Any:
	value = value.strip()
	if not value:
		return ""
	if value[0:1] in {"'", '"'} and value[-1:] == value[0]:
		return value[1:-1]
	if value.startswith("[") and value.endswith("]"):
		inner = value[1:-1].strip()
		return [_parse_value(part.strip()) for part in _split_commas(inner)] if inner else []
	if value.lower() in {"true", "false"}:
		return value.lower() == "true"
	if re.fullmatch(r"-?\d+", value):
		return int(value)
	if re.fullmatch(r"-?\d+\.\d+", value):
		return float(value)
	return value


def _split_commas(value: str) -> List[str]:
	items: List[str] = []
	start = 0
	depth = 0
	quote: Optional[str] = None
	for index, char in enumerate(value):
		if quote:
			if char == quote and value[index - 1:index] != "\\":
				quote = None
			continue
		if char in {"'", '"'}:
			quote = char
		elif char in "[{(":
			depth += 1
		elif char in "]})":
			depth -= 1
		elif char == "," and depth == 0:
			items.append(value[start:index])
			start = index + 1
	items.append(value[start:])
	return items


def _parse_memory(value: Any) -> Optional[AgentMemory]:
	if value is None:
		return None
	if isinstance(value, str):
		parts = value.split(None, 1)
		kind = parts[0]
		name = parts[1].strip("'\"") if len(parts) > 1 else None
		return AgentMemory(kind=kind, name=name)
	return None


def _parse_handoffs(default_source: str, value: Any) -> List[AgentHandoff]:
	if value is None:
		return []
	if isinstance(value, list):
		edges: List[AgentHandoff] = []
		for item in value:
			edges.extend(_parse_handoffs(default_source, item))
		return edges
	text = str(value).strip()
	if not text:
		return []

	condition = "done"
	if " when " in text:
		text, condition = text.split(" when ", 1)
		condition = condition.strip()

	names = [part.strip() for part in text.split("->") if part.strip()]
	if len(names) == 1 and default_source:
		names = [default_source, names[0]]
	return [
		AgentHandoff(source=source, target=target, condition=condition)
		for source, target in zip(names, names[1:])
	]


def _string_list(value: Any) -> List[str]:
	if value is None:
		return []
	if isinstance(value, list):
		return [str(item) for item in value]
	return [str(value)]


def _optional_string(value: Any) -> Optional[str]:
	if value is None:
		return None
	return str(value)


def _unique(values: Iterable[str]) -> List[str]:
	seen = set()
	result = []
	for value in values:
		if value not in seen:
			seen.add(value)
			result.append(value)
	return result


def _line_column(source: str, offset: int) -> tuple[int, int]:
	line = source.count("\n", 0, offset) + 1
	line_start = source.rfind("\n", 0, offset)
	column = offset if line_start < 0 else offset - line_start - 1
	return line, column
