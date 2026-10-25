"""Programs, streaming, and patch merging (spec 1.2, 1.4, 4, 7).

``Program`` is one parse of one source text. ``StreamParser`` accumulates
chunks and re-derives the program on each one, which makes the reconciliation
rule (spec 4.4) automatic: the prepared text is rebuilt from the full buffer,
so a late fence opener cannot leave a stale prefix behind.

Streaming equals parsing (spec 4) holds by construction here, since the streamed
result at end of stream is computed by the same batch path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .evaluator import Diagnostic, Evaluator, tree_summary
from .library import Library
from .parser import Literal, ParsedStatement, parse_statement
from .source import Prepared, close_implicitly, prepare_source, split_statements


@dataclass
class Program:
	"""One parse of one source text, with the statements in source order."""

	text: str
	statements: list[ParsedStatement] = field(default_factory=list)
	incomplete: bool = False
	# Statements whose text needed implicit closing. A pending one never
	# replaces a completed statement of the same name (spec 4.3).
	pending: set[str] = field(default_factory=set)

	@classmethod
	def parse(cls, source: str, *, final: bool = True) -> "Program":
		prepared: Prepared = prepare_source(source)
		pieces = split_statements(prepared.text, final=final)
		statements: list[ParsedStatement] = []
		pending: set[str] = set()
		for piece in pieces:
			text = piece.text
			if piece.needed_closing:
				text = close_implicitly(text)
			parsed = parse_statement(text, incomplete=not piece.complete, needs_closing=piece.needed_closing)
			if parsed is None:
				# A statement that does not parse is dropped (spec 8.1).
				continue
			statements.append(parsed)
			if piece.needed_closing:
				pending.add(parsed.name)
		return cls(
			text=prepared.text,
			statements=statements,
			incomplete=prepared.incomplete,
			pending=pending,
		)

	def names(self) -> list[str]:
		return [statement.name for statement in self.statements]

	def evaluate(
		self,
		library: Library,
		*,
		state: dict[str, Any] | None = None,
		functions: dict[str, Any] | None = None,
	) -> "ProgramResult":
		statements = self._without_shadowed_pending()
		evaluator = Evaluator(library, statements, state=state, functions=functions)
		result = evaluator.run()
		return ProgramResult(
			root=result.root,
			errors=result.errors,
			unresolved=result.unresolved,
			orphans=result.orphans,
			state=result.state,
			incomplete=self.incomplete,
			statements=[statement.name for statement in statements],
		)

	def _without_shadowed_pending(self) -> list[ParsedStatement]:
		"""Drop a pending statement a completed one already binds (spec 4.3)."""
		if not self.pending:
			return list(self.statements)
		completed = {
			statement.name for statement in self.statements
			if statement.name not in self.pending
		}
		kept: list[ParsedStatement] = []
		for statement in self.statements:
			if statement.name in self.pending and statement.name in completed:
				continue
			kept.append(statement)
		return kept

	def to_dict(self) -> dict[str, Any]:
		"""The canonical program shape (fixture README, Appendix B)."""
		return {
			"statements": self.names(),
			"incomplete": self.incomplete,
		}


@dataclass
class ProgramResult:
	root: Any
	errors: list[Diagnostic]
	unresolved: set[str]
	orphans: list[str]
	state: dict[str, Any]
	incomplete: bool
	statements: list[str]

	def to_dict(self, *, include_state: bool = True) -> dict[str, Any]:
		data: dict[str, Any] = {
			"root": tree_summary(self.root),
			"errors": [error.brief() for error in self.errors],
			"unresolved": sorted(self.unresolved),
			"orphans": self.orphans,
			"incomplete": self.incomplete,
		}
		if include_state:
			data["state"] = dict(self.state)
		return data

	def error_objects(self) -> list[dict[str, Any]]:
		return [error.shaped() for error in self.errors]


class StreamParser:
	"""Accumulate chunks; produce a program after every one (spec 4)."""

	def __init__(self) -> None:
		self.buffer = ""
		self._program: Program | None = None
		self._ended = False

	def feed(self, chunk: str) -> Program:
		if self._ended:
			# Chunks after the end signal are a new stream (spec 4.7).
			self.buffer = ""
			self._ended = False
		self.buffer += chunk
		self._program = Program.parse(self.buffer, final=False)
		return self._program

	def end(self) -> Program:
		self._ended = True
		self._program = Program.parse(self.buffer, final=True)
		return self._program

	@property
	def program(self) -> Program:
		if self._program is None:
			return Program.parse(self.buffer, final=False)
		return self._program

	def feed_all(self, chunks: Iterable[str]) -> Program:
		program = self.program
		for chunk in chunks:
			program = self.feed(chunk)
		return program

	def run(
		self,
		chunks: Iterable[str],
		library: Library,
		*,
		state: dict[str, Any] | None = None,
		functions: dict[str, Any] | None = None,
	) -> ProgramResult:
		self.feed_all(chunks)
		return self.end().evaluate(library, state=state, functions=functions)


# ── incremental editing (spec 7) ────────────────────────────────────────────


@dataclass
class Merged:
	text: str
	statements: list[str]


def merge_patch(current: str, patch: str, library: Library) -> Merged:
	"""Merge a patch into a program by name, then collect unreachable garbage."""
	base = _entries(current)
	incoming = _entries(patch)
	index = {entry.name: position for position, entry in enumerate(base)}
	deleted: set[str] = set()

	for entry in incoming:
		if entry.deletes:
			deleted.add(entry.name)
			if entry.name in index:
				base[index[entry.name]] = _Entry(entry.name, entry.text, True)
			continue
		if entry.name in index:
			position = index[entry.name]
			base[position] = entry
		else:
			index[entry.name] = len(base)
			base.append(entry)

	kept = [entry for entry in base if not entry.deletes and entry.name not in deleted]
	text = "\n".join(entry.text for entry in kept)
	text, statements = _collect(text, library)
	return Merged(text=text, statements=statements)


@dataclass
class _Entry:
	name: str
	text: str
	deletes: bool = False


def _entries(source: str) -> list[_Entry]:
	"""Read a text into entries, using the same statement splitter as parsing."""
	prepared = prepare_source(source)
	entries: list[_Entry] = []
	for piece in split_statements(prepared.text, final=True):
		text = close_implicitly(piece.text) if piece.needed_closing else piece.text
		parsed = parse_statement(text)
		deletes = isinstance(parsed.expression, Literal) and parsed.expression.value is None if parsed else False
		entries.append(_Entry(piece.name, text, deletes))
	return entries


def _collect(text: str, library: Library) -> tuple[str, list[str]]:
	"""Drop statements `root` cannot reach, then return the merged text.

	Spec 7: garbage collection runs after the merge, and only when the program
	has a `root` statement. State declarations are always kept, and so are
	statements reachable only through a query's or mutation's arguments.
	"""
	program = Program.parse(text)
	names = [statement.name for statement in program.statements]
	if "root" not in names:
		# Without a `root` statement nothing is collected (spec 7).
		return text, names
	reachable = _reachable_names(program, library)
	kept: list[ParsedStatement] = []
	for statement in program.statements:
		if statement.kind in ("state", "query", "mutation") or statement.name in reachable:
			kept.append(statement)
	if len(kept) == len(program.statements):
		return text, names
	collected = "\n".join(statement.text for statement in kept)
	return collected, [statement.name for statement in kept]


def _reachable_names(program: Program, library: Library) -> set[str]:
	evaluator = Evaluator(library, program.statements)
	evaluator.run()
	return set(evaluator.reachable)
