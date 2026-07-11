#!/usr/bin/env python3
"""Stdlib-only APG Language Server Protocol server over stdio."""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import unquote, urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from compiler.compiler import APGCompiler
from compiler.parser import APGSyntaxError
from compiler.semantic_analyzer import SemanticError


TEXT_DOCUMENT_SYNC_INCREMENTAL = 2

COMPLETION_KIND_TEXT = 1
COMPLETION_KIND_FIELD = 5
COMPLETION_KIND_CLASS = 7
COMPLETION_KIND_PROPERTY = 10
COMPLETION_KIND_KEYWORD = 14
COMPLETION_KIND_SNIPPET = 15
COMPLETION_KIND_TYPE_PARAMETER = 25

DIAGNOSTIC_SEVERITY_ERROR = 1

TYPE_ORDER = ["str", "int", "float", "bool", "date", "datetime", "text", "uuid", "json", "file"]
TYPE_DOCS = {
	"str": ("String value for short text.", "name: str;"),
	"int": ("Integer number.", "quantity: int;"),
	"float": ("Floating-point number.", "price: float;"),
	"bool": ("Boolean true or false value.", "active: bool;"),
	"date": ("Calendar date without a time component.", "due_date: date;"),
	"datetime": ("Timestamp with date and time.", "created_at: datetime;"),
	"text": ("Long-form text content.", "notes: text;"),
	"uuid": ("Universally unique identifier.", "record_id: uuid;"),
	"json": ("Structured JSON object or array.", "metadata: json;"),
	"file": ("Uploaded or referenced file value.", "attachment: file;"),
}

KEYWORD_DOCS = {
	"module": "Declares an APG module and version boundary.",
	"entity": "Declares a named APG entity with fields and relationships.",
	"table": "Declares a database-backed APG entity.",
	"security": "Declares security configuration for an APG module or capability.",
	"authentication": "Configures whether requests require authentication.",
	"required": "Marks a setting or field as required.",
	"has_many": "Declares a one-to-many relationship to another entity.",
	"belongs_to": "Declares ownership or foreign-key style relationship to another entity.",
	"has_one": "Declares a one-to-one relationship to another entity.",
	"through": "Names the junction entity used by a has_many relationship.",
}

VALIDATORS = ["@min_length", "@max_length", "@min", "@max", "@email", "@pattern", "@optional", "@required"]
RELATIONSHIP_KEYWORDS = [
	("has_many", "has_many ${1:Entity};"),
	("belongs_to", "belongs_to ${1:Entity};"),
	("has_one", "has_one ${1:Entity};"),
]


@dataclass
class FieldInfo:
	name: str
	type_name: str
	line: int
	character: int


@dataclass
class RelationshipInfo:
	kind: str
	target: str
	line: int
	character: int


@dataclass
class EntityInfo:
	kind: str
	name: str
	line: int
	character: int
	body_start: int
	body_end: int
	fields: list[FieldInfo] = field(default_factory=list)
	relationships: list[RelationshipInfo] = field(default_factory=list)

	@property
	def range(self) -> dict[str, Any]:
		return {
			"start": {"line": self.line, "character": self.character},
			"end": {"line": self.line, "character": self.character + len(self.name)},
		}


@dataclass
class DocumentState:
	uri: str
	text: str
	version: int | None = None
	path: Path | None = None
	entities: list[EntityInfo] = field(default_factory=list)

	def refresh(self) -> None:
		self.entities = parse_entities(self.text)

	def entity_names(self) -> list[str]:
		return sorted({entity.name for entity in self.entities})

	def entity_by_name(self, name: str) -> EntityInfo | None:
		for entity in self.entities:
			if entity.name == name:
				return entity
		return None


def _line_starts(text: str) -> list[int]:
	starts = [0]
	for index, char in enumerate(text):
		if char == "\n":
			starts.append(index + 1)
	return starts


def offset_to_position(text: str, offset: int) -> tuple[int, int]:
	offset = max(0, min(offset, len(text)))
	starts = _line_starts(text)
	low = 0
	high = len(starts) - 1
	while low <= high:
		mid = (low + high) // 2
		if starts[mid] <= offset:
			low = mid + 1
		else:
			high = mid - 1
	line = max(0, high)
	return line, offset - starts[line]


def position_to_offset(text: str, position: dict[str, Any]) -> int:
	line = max(0, int(position.get("line", 0)))
	character = max(0, int(position.get("character", 0)))
	starts = _line_starts(text)
	if line >= len(starts):
		return len(text)
	line_start = starts[line]
	line_end = starts[line + 1] - 1 if line + 1 < len(starts) else len(text)
	return min(line_start + character, line_end)


def line_prefix(text: str, position: dict[str, Any]) -> str:
	line = max(0, int(position.get("line", 0)))
	character = max(0, int(position.get("character", 0)))
	lines = text.splitlines()
	if line >= len(lines):
		return ""
	return lines[line][:character]


def _matching_brace(text: str, open_offset: int) -> int:
	depth = 0
	index = open_offset
	quote: str | None = None
	escaped = False
	line_comment = False
	block_comment = False
	while index < len(text):
		char = text[index]
		next_char = text[index + 1] if index + 1 < len(text) else ""
		if line_comment:
			if char == "\n":
				line_comment = False
			index += 1
			continue
		if block_comment:
			if char == "*" and next_char == "/":
				block_comment = False
				index += 2
			else:
				index += 1
			continue
		if quote:
			if escaped:
				escaped = False
			elif char == "\\":
				escaped = True
			elif char == quote:
				quote = None
			index += 1
			continue
		if char == "/" and next_char == "/":
			line_comment = True
			index += 2
			continue
		if char == "/" and next_char == "*":
			block_comment = True
			index += 2
			continue
		if char in {"'", '"'}:
			quote = char
			index += 1
			continue
		if char == "{":
			depth += 1
		elif char == "}":
			depth -= 1
			if depth == 0:
				return index
		index += 1
	return len(text)


def parse_entities(text: str) -> list[EntityInfo]:
	entities: list[EntityInfo] = []
	pattern = re.compile(r"\b(entity|table)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{")
	for match in pattern.finditer(text):
		open_brace = text.find("{", match.start())
		if open_brace < 0:
			continue
		close_brace = _matching_brace(text, open_brace)
		line, character = offset_to_position(text, match.start(2))
		entity = EntityInfo(
			kind=match.group(1),
			name=match.group(2),
			line=line,
			character=character,
			body_start=open_brace + 1,
			body_end=close_brace,
		)
		body = text[entity.body_start:entity.body_end]
		entity.fields = _parse_fields(body, entity.body_start, text)
		entity.relationships = _parse_relationships(body, entity.body_start, text)
		entities.append(entity)
	return entities


def _parse_fields(body: str, body_offset: int, source: str) -> list[FieldInfo]:
	fields: list[FieldInfo] = []
	pattern = re.compile(
		r"(?:^|(?<=;))\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([^=;{\n]+?)(?:\s*=.*?)?\s*;",
		re.MULTILINE | re.DOTALL,
	)
	for match in pattern.finditer(body):
		type_text = match.group(2).strip()
		if type_text.startswith("(") or type_text.startswith("async"):
			continue
		type_name = type_text.split("@", 1)[0].strip()
		line, character = offset_to_position(source, body_offset + match.start(1))
		fields.append(FieldInfo(match.group(1), type_name, line, character))
	return fields


def _parse_relationships(body: str, body_offset: int, source: str) -> list[RelationshipInfo]:
	relationships: list[RelationshipInfo] = []
	pattern = re.compile(
		r"(?:^|;)\s*(has_many|belongs_to|has_one)\s+([A-Za-z_][A-Za-z0-9_]*)(?:\s+through\s+[A-Za-z_][A-Za-z0-9_]*)?\s*;",
		re.MULTILINE,
	)
	for match in pattern.finditer(body):
		line, character = offset_to_position(source, body_offset + match.start(1))
		relationships.append(RelationshipInfo(match.group(1), match.group(2), line, character))
	return relationships


def word_at_position(text: str, position: dict[str, Any]) -> tuple[str, dict[str, Any]] | tuple[None, None]:
	offset = position_to_offset(text, position)
	if offset == len(text) or (offset < len(text) and not _is_word_char(text[offset])):
		if offset > 0 and _is_word_char(text[offset - 1]):
			offset -= 1
	start = offset
	end = offset
	while start > 0 and _is_word_char(text[start - 1]):
		start -= 1
	while end < len(text) and _is_word_char(text[end]):
		end += 1
	if start == end:
		return None, None
	start_line, start_char = offset_to_position(text, start)
	end_line, end_char = offset_to_position(text, end)
	return text[start:end], {
		"start": {"line": start_line, "character": start_char},
		"end": {"line": end_line, "character": end_char},
	}


def _is_word_char(char: str) -> bool:
	return char.isalnum() or char in {"_", "@"}


def _completion_item(
	label: str,
	kind: int = COMPLETION_KIND_TEXT,
	detail: str | None = None,
	insert_text: str | None = None,
) -> dict[str, Any]:
	item: dict[str, Any] = {"label": label, "kind": kind}
	if detail:
		item["detail"] = detail
	if insert_text:
		item["insertText"] = insert_text
	return item


def _dedupe_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
	seen: set[str] = set()
	unique: list[dict[str, Any]] = []
	for item in items:
		label = str(item.get("label", ""))
		if label in seen:
			continue
		seen.add(label)
		unique.append(item)
	return unique


class APGLanguageServer:
	def __init__(self, stdin: BinaryIO | None = None, stdout: BinaryIO | None = None):
		self.stdin = stdin or sys.stdin.buffer
		self.stdout = stdout or sys.stdout.buffer
		self.documents: dict[str, DocumentState] = {}
		self._shutdown_requested = False
		self._exit_requested = False

	def run(self) -> None:
		while not self._exit_requested:
			message = self._read_message()
			if message is None:
				break
			self._handle_message(message)

	def _read_message(self) -> dict[str, Any] | None:
		headers: dict[str, str] = {}
		while True:
			line = self.stdin.readline()
			if line == b"":
				return None
			if line in {b"\r\n", b"\n"}:
				break
			name, _, value = line.decode("ascii", errors="replace").partition(":")
			if name:
				headers[name.lower()] = value.strip()
		try:
			length = int(headers.get("content-length", "0"))
		except ValueError:
			return None
		if length <= 0:
			return None
		body = self.stdin.read(length)
		if not body:
			return None
		return json.loads(body.decode("utf-8"))

	def _write_message(self, payload: dict[str, Any]) -> None:
		body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
		header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
		self.stdout.write(header + body)
		self.stdout.flush()

	def _handle_message(self, message: dict[str, Any]) -> None:
		if "method" not in message:
			return
		method = str(message["method"])
		request_id = message.get("id")
		params = message.get("params") or {}
		try:
			if request_id is None:
				self._handle_notification(method, params)
			else:
				result = self._handle_request(method, params)
				self._write_message({"jsonrpc": "2.0", "id": request_id, "result": result})
		except JsonRpcError as error:
			if request_id is not None:
				self._write_message({
					"jsonrpc": "2.0",
					"id": request_id,
					"error": {"code": error.code, "message": error.message},
				})
			else:
				self._log_exception(method)
		except Exception:
			self._log_exception(method)
			if request_id is not None:
				self._write_message({
					"jsonrpc": "2.0",
					"id": request_id,
					"error": {"code": -32603, "message": "Internal error"},
				})

	def _handle_request(self, method: str, params: dict[str, Any]) -> Any:
		if method == "initialize":
			return create_initialize_result()
		if method == "shutdown":
			self._shutdown_requested = True
			return None
		if method == "textDocument/completion":
			return self._completion(params)
		if method == "textDocument/hover":
			return self._hover(params)
		if method == "textDocument/definition":
			return self._definition(params)
		raise JsonRpcError(-32601, f"Method not found: {method}")

	def _handle_notification(self, method: str, params: dict[str, Any]) -> None:
		if method == "initialized":
			return
		if method == "exit":
			self._exit_requested = True
			return
		if method == "textDocument/didOpen":
			doc = params.get("textDocument", {})
			uri = str(doc.get("uri", ""))
			if not uri:
				return
			state = DocumentState(
				uri=uri,
				text=str(doc.get("text", "")),
				version=doc.get("version"),
				path=uri_to_path(uri),
			)
			state.refresh()
			self.documents[uri] = state
			self._publish_diagnostics(uri)
			return
		if method == "textDocument/didChange":
			doc = params.get("textDocument", {})
			uri = str(doc.get("uri", ""))
			state = self.documents.get(uri)
			if state is None:
				state = DocumentState(uri=uri, text="", path=uri_to_path(uri))
				self.documents[uri] = state
			state.version = doc.get("version", state.version)
			for change in params.get("contentChanges", []):
				state.text = apply_text_change(state.text, change)
			state.refresh()
			self._publish_diagnostics(uri)
			return
		if method == "textDocument/didClose":
			doc = params.get("textDocument", {})
			self.documents.pop(str(doc.get("uri", "")), None)

	def _completion(self, params: dict[str, Any]) -> list[dict[str, Any]]:
		uri = str(params.get("textDocument", {}).get("uri", ""))
		position = params.get("position", {})
		state = self._document_state(uri)
		prefix = line_prefix(state.text, position)
		offset = position_to_offset(state.text, position)
		entity = entity_at_offset(state, offset)

		items: list[dict[str, Any]] = []
		if _looks_like_validator_context(prefix):
			items.extend(
				_completion_item(label, COMPLETION_KIND_PROPERTY, "APG field validator", label)
				for label in VALIDATORS
			)
			return _dedupe_items(items)

		if in_security_block(state.text, offset):
			return [_completion_item(
				"authentication: required;",
				COMPLETION_KIND_PROPERTY,
				"Require authentication for this security block",
				"authentication: required;",
			)]

		if re.search(r"\b(?:has_many|belongs_to)\s+$", prefix):
			return [
				_completion_item(name, COMPLETION_KIND_CLASS, "Entity relationship target", name)
				for name in state.entity_names()
			]

		if re.search(r"\bentity\s+$", prefix):
			return [
				_completion_item(name, COMPLETION_KIND_CLASS, "Entity defined in current file", name)
				for name in state.entity_names()
			]

		if re.search(r"\b[A-Za-z_][A-Za-z0-9_]*\s*:\s*$", prefix):
			return [
				_completion_item(type_name, COMPLETION_KIND_TYPE_PARAMETER, TYPE_DOCS[type_name][0], type_name)
				for type_name in TYPE_ORDER
			]

		if entity and prefix.strip() == "":
			items.extend(
				_completion_item(label, COMPLETION_KIND_KEYWORD, KEYWORD_DOCS[label], insert_text)
				for label, insert_text in RELATIONSHIP_KEYWORDS
			)
			items.extend(self._same_named_entity_field_completions(state, entity))
			return _dedupe_items(items)

		return []

	def _same_named_entity_field_completions(self, state: DocumentState, entity: EntityInfo) -> list[dict[str, Any]]:
		current_fields = {field.name for field in entity.fields}
		candidates: list[dict[str, Any]] = []
		for other in self._other_documents_for_entity_fields(state):
			other_entity = other.entity_by_name(entity.name)
			if not other_entity:
				continue
			for field_info in other_entity.fields:
				if field_info.name in current_fields:
					continue
				insert_text = f"{field_info.name}: {field_info.type_name};"
				candidates.append(_completion_item(
					field_info.name,
					COMPLETION_KIND_FIELD,
					f"Field from {entity.name} in another APG file",
					insert_text,
				))
		return candidates

	def _other_documents_for_entity_fields(self, state: DocumentState) -> list[DocumentState]:
		docs = [doc for uri, doc in self.documents.items() if uri != state.uri]
		if state.path and state.path.parent.exists():
			for path in sorted(state.path.parent.glob("*.apg"))[:50]:
				if path == state.path:
					continue
				try:
					uri = path.as_uri()
					if uri in self.documents:
						continue
					doc = DocumentState(uri=uri, text=path.read_text(encoding="utf-8"), path=path)
					doc.refresh()
					docs.append(doc)
				except OSError:
					continue
		return docs

	def _hover(self, params: dict[str, Any]) -> dict[str, Any] | None:
		uri = str(params.get("textDocument", {}).get("uri", ""))
		position = params.get("position", {})
		state = self._document_state(uri)
		word, word_range = word_at_position(state.text, position)
		if not word:
			return None
		value = self._hover_markdown(state, word, position)
		if value is None:
			return None
		return {"contents": {"kind": "markdown", "value": value}, "range": word_range}

	def _hover_markdown(self, state: DocumentState, word: str, position: dict[str, Any]) -> str | None:
		if word in TYPE_DOCS:
			description, example = TYPE_DOCS[word]
			return f"**{word}**\n\n{description}\n\nExample:\n\n```apg\n{example}\n```"
		if word in KEYWORD_DOCS:
			return f"**{word}**\n\n{KEYWORD_DOCS[word]}"
		relation = relation_target_at_position(state.text, position)
		if relation == word:
			entity = state.entity_by_name(word)
			if entity:
				return entity_summary_markdown(entity)
		entity = state.entity_by_name(word)
		if entity:
			return entity_summary_markdown(entity)
		return None

	def _definition(self, params: dict[str, Any]) -> dict[str, Any] | None:
		uri = str(params.get("textDocument", {}).get("uri", ""))
		position = params.get("position", {})
		state = self._document_state(uri)
		word, _ = word_at_position(state.text, position)
		if not word or relation_target_at_position(state.text, position) != word:
			return None
		entity = state.entity_by_name(word)
		if not entity:
			return None
		return {"uri": uri, "range": entity.range}

	def _publish_diagnostics(self, uri: str) -> None:
		state = self._document_state(uri)
		diagnostics = compile_diagnostics(state.text)
		self._write_message({
			"jsonrpc": "2.0",
			"method": "textDocument/publishDiagnostics",
			"params": {"uri": uri, "diagnostics": diagnostics},
		})

	def _document_state(self, uri: str) -> DocumentState:
		state = self.documents.get(uri)
		if state is not None:
			return state
		state = DocumentState(uri=uri, text="", path=uri_to_path(uri))
		state.refresh()
		self.documents[uri] = state
		return state

	def _log_exception(self, method: str) -> None:
		print(f"APG LSP error while handling {method}", file=sys.stderr)
		traceback.print_exc(file=sys.stderr)


class JsonRpcError(Exception):
	def __init__(self, code: int, message: str):
		super().__init__(message)
		self.code = code
		self.message = message


def create_server_capabilities() -> dict[str, Any]:
	return {
		"textDocumentSync": TEXT_DOCUMENT_SYNC_INCREMENTAL,
		"completionProvider": {"triggerCharacters": [" ", ":"]},
		"hoverProvider": True,
		"diagnosticProvider": True,
		"definitionProvider": True,
	}


def create_initialize_result() -> dict[str, Any]:
	return {
		"capabilities": create_server_capabilities(),
		"serverInfo": {"name": "apg-language-server", "version": "1.0.0"},
	}


def uri_to_path(uri: str) -> Path | None:
	parsed = urlparse(uri)
	if parsed.scheme != "file":
		return None
	path = unquote(parsed.path)
	if sys.platform.startswith("win") and re.match(r"^/[A-Za-z]:", path):
		path = path[1:]
	return Path(path)


def apply_text_change(text: str, change: dict[str, Any]) -> str:
	if "range" not in change:
		return str(change.get("text", ""))
	start = position_to_offset(text, change.get("range", {}).get("start", {}))
	end = position_to_offset(text, change.get("range", {}).get("end", {}))
	return text[:start] + str(change.get("text", "")) + text[end:]


def entity_at_offset(state: DocumentState, offset: int) -> EntityInfo | None:
	for entity in state.entities:
		if entity.body_start <= offset <= entity.body_end:
			return entity
	return None


def in_security_block(text: str, offset: int) -> bool:
	pattern = re.compile(r"\bsecurity(?:\s+[A-Za-z_][A-Za-z0-9_]*)?\s*\{")
	for match in pattern.finditer(text):
		open_brace = text.find("{", match.start())
		close_brace = _matching_brace(text, open_brace)
		if open_brace < offset <= close_brace:
			return True
	return False


def relation_target_at_position(text: str, position: dict[str, Any]) -> str | None:
	line_number = int(position.get("line", 0))
	lines = text.splitlines()
	if line_number < 0 or line_number >= len(lines):
		return None
	line = lines[line_number]
	match = re.search(r"\b(has_many|belongs_to)\s+([A-Za-z_][A-Za-z0-9_]*)\b", line)
	return match.group(2) if match else None


def entity_summary_markdown(entity: EntityInfo) -> str:
	if entity.fields:
		fields = ", ".join(f"{field.name}: {field.type_name}" for field in entity.fields)
	else:
		fields = "No fields declared."
	return f"**{entity.name}** ({entity.kind})\n\nFields: {fields}"


def compile_diagnostics(text: str) -> list[dict[str, Any]]:
	try:
		result = APGCompiler().compile_string(text, "hover")
	except Exception as error:
		return [_exception_diagnostic(error)]
	return [_diagnostic_from_error(error) for error in result.errors]


def _diagnostic_from_error(error: APGSyntaxError | SemanticError | Exception) -> dict[str, Any]:
	line = max(0, int(getattr(error, "line", 1) or 1) - 1)
	raw_column = int(getattr(error, "column", 0) or 0)
	if isinstance(error, SemanticError):
		character = max(0, raw_column - 1)
	else:
		character = max(0, raw_column)
	message = str(getattr(error, "message", "") or str(error))
	return {
		"range": {
			"start": {"line": line, "character": character},
			"end": {"line": line, "character": character + 1},
		},
		"severity": DIAGNOSTIC_SEVERITY_ERROR,
		"source": "apg",
		"message": message,
		"code": error.__class__.__name__,
	}


def _exception_diagnostic(error: Exception) -> dict[str, Any]:
	return {
		"range": {
			"start": {"line": 0, "character": 0},
			"end": {"line": 0, "character": 1},
		},
		"severity": DIAGNOSTIC_SEVERITY_ERROR,
		"source": "apg-language-server",
		"message": f"Language server diagnostic failure: {error}",
		"code": "LanguageServerError",
	}


def _looks_like_validator_context(prefix: str) -> bool:
	return prefix.endswith("@") or bool(re.search(r"@[A-Za-z_]*$", prefix))


def start_language_server(host: str = "127.0.0.1", port: int = 2087) -> None:
	"""Start the APG language server over stdio.

	``host`` and ``port`` are accepted for compatibility with the older CLI
	surface, but stdio is the only supported transport for this dependency-free
	server.
	"""
	_ = (host, port)
	APGLanguageServer().run()


def main() -> None:
	parser = argparse.ArgumentParser(description="APG Language Server over stdio")
	parser.add_argument("--stdio", action="store_true", help="Accepted for compatibility; stdio is always used")
	parser.add_argument("--host", default="127.0.0.1", help=argparse.SUPPRESS)
	parser.add_argument("--port", type=int, default=2087, help=argparse.SUPPRESS)
	parser.parse_args()
	start_language_server()


if __name__ == "__main__":
	main()
