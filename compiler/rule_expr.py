"""Rule expression parser for APG `when` conditions.

Parses rule condition strings like:
    "amount > 50000"
    "amount > 50000 and stage == qualification"
    "bank_account missing"
    "agency in [FDA, EMA, HC]"
    "ae_type = serious_adverse_event AND within_24h = false"

Grammar (informal):
    expr     := or_expr
    or_expr  := and_expr ('or' and_expr)*
    and_expr := atom ('and' atom)*
    atom     := field op value
              | field 'missing'
              | field 'not' 'missing'
              | field 'in' list
              | field 'not' 'in' list
              | '(' expr ')'
    field    := IDENTIFIER ('.' IDENTIFIER)*
    op       := '==' | '!=' | '>=' | '<=' | '>' | '<' | '=' | '!=' | '<>'
    value    := STRING | NUMBER | BOOLEAN | IDENTIFIER
    list     := '[' value (',' value)* ']'

Design decisions:
  - Case-insensitive 'and' / 'or' / 'in' / 'missing' / 'not'
  - '=' is alias for '==' (common in rule DSLs)
  - '<>' is alias for '!='
  - Returns typed RuleExprNode trees for compile-time analysis
  - Field validation against an ambient field set is optional (warns, not errors)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ── AST node types ──────────────────────────────────────────────────────────

@dataclass
class CompareNode:
	"""field op value — e.g. amount > 50000"""
	field: str
	op: str
	value: Any


@dataclass
class MissingNode:
	"""field missing — e.g. bank_account missing"""
	field: str
	negated: bool = False  # 'not missing' inverts


@dataclass
class InNode:
	"""field in [...] — e.g. agency in [FDA, EMA]"""
	field: str
	values: list[Any]
	negated: bool = False  # 'not in' inverts


@dataclass
class AndNode:
	left: Any
	right: Any


@dataclass
class OrNode:
	left: Any
	right: Any


RuleExprNode = CompareNode | MissingNode | InNode | AndNode | OrNode


class RuleExprParseError(ValueError):
	pass


# ── Tokenizer ────────────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(
	r"""
	\s*
	(?:
		(?P<string>'[^']*'|"[^"]*")   # quoted string
	|	(?P<number>-?\d+(?:\.\d+)?)   # number
	|	(?P<op>==|!=|<>|>=|<=|>|<|=) # operators (multi-char first)
	|	(?P<lbracket>\[)
	|	(?P<rbracket>\])
	|	(?P<comma>,)
	|	(?P<lparen>\()
	|	(?P<rparen>\))
	|	(?P<word>[A-Za-z_][A-Za-z0-9_.]*) # identifiers and keywords
	)
	""",
	re.VERBOSE,
)

_KEYWORDS_LOWER = {"and", "or", "in", "not", "missing", "true", "false", "null"}


@dataclass
class Token:
	kind: str   # 'string', 'number', 'op', 'word', 'lbracket', 'rbracket', 'comma', 'lparen', 'rparen'
	value: Any


def _tokenize(text: str) -> list[Token]:
	tokens: list[Token] = []
	pos = 0
	while pos < len(text):
		m = _TOKEN_RE.match(text, pos)
		if not m or m.start() == m.end():
			break
		pos = m.end()
		kind = m.lastgroup
		if kind is None:
			continue
		raw = m.group(kind)  # use named group to skip leading \s*
		if kind == "string":
			tokens.append(Token("string", raw[1:-1]))
		elif kind == "number":
			v = float(raw) if "." in raw else int(raw)
			tokens.append(Token("number", v))
		elif kind == "word":
			lo = raw.lower()
			if lo in ("true", "false"):
				tokens.append(Token("bool", lo == "true"))
			elif lo in ("null", "none"):
				tokens.append(Token("null", None))
			else:
				tokens.append(Token("word", raw))
		else:
			tokens.append(Token(kind, raw))
	return tokens


# ── Parser ───────────────────────────────────────────────────────────────────

class _Parser:
	def __init__(self, tokens: list[Token]) -> None:
		self._tokens = tokens
		self._pos = 0

	def _peek(self) -> Token | None:
		if self._pos < len(self._tokens):
			return self._tokens[self._pos]
		return None

	def _consume(self) -> Token:
		tok = self._tokens[self._pos]
		self._pos += 1
		return tok

	def _match_word(self, *words: str) -> Token | None:
		tok = self._peek()
		if tok and tok.kind == "word" and tok.value.lower() in {w.lower() for w in words}:
			self._pos += 1
			return tok
		return None

	def parse(self) -> RuleExprNode:
		node = self._parse_or()
		if self._pos < len(self._tokens):
			remaining = " ".join(str(t.value) for t in self._tokens[self._pos:])
			raise RuleExprParseError(f"Unexpected tokens after expression: {remaining!r}")
		return node

	def _parse_or(self) -> RuleExprNode:
		# 'and' has higher precedence than 'or' (same as Python / SQL).
		# This is enforced structurally: _parse_and() is called inside _parse_or().
		left = self._parse_and()
		while self._match_word("or"):
			right = self._parse_and()
			left = OrNode(left=left, right=right)
		return left

	def _parse_and(self) -> RuleExprNode:
		left = self._parse_atom()
		while self._match_word("and"):
			right = self._parse_atom()
			left = AndNode(left=left, right=right)
		return left

	def _parse_atom(self) -> RuleExprNode:
		tok = self._peek()
		if tok is None:
			raise RuleExprParseError("Unexpected end of rule expression")

		# Parenthesised sub-expression
		if tok.kind == "lparen":
			self._consume()
			inner = self._parse_or()
			close = self._peek()
			if not (close and close.kind == "rparen"):
				raise RuleExprParseError("Expected ')' to close sub-expression")
			self._consume()
			return inner

		# Must start with a field identifier
		if tok.kind != "word":
			raise RuleExprParseError(f"Expected field name, got {tok.value!r}")

		field_name = tok.value
		self._consume()

		# Allow dotted field references (e.g. account.status)
		while self._peek() and self._peek().kind == "word" and self._peek().value.startswith("."):
			field_name += self._consume().value

		# Peek at what follows
		next_tok = self._peek()
		if next_tok is None:
			# Bare field reference — treat as truthy check
			return CompareNode(field=field_name, op="==", value=True)

		# field missing / field not missing
		if next_tok.kind == "word" and next_tok.value.lower() == "missing":
			self._consume()
			return MissingNode(field=field_name, negated=False)

		if next_tok.kind == "word" and next_tok.value.lower() == "not":
			self._consume()
			after_not = self._peek()
			if after_not and after_not.kind == "word" and after_not.value.lower() == "missing":
				self._consume()
				return MissingNode(field=field_name, negated=True)
			if after_not and after_not.kind == "word" and after_not.value.lower() == "in":
				self._consume()
				return InNode(field=field_name, values=self._parse_list(), negated=True)
			raise RuleExprParseError(f"Expected 'missing' or 'in' after 'not', got {after_not!r}")

		# field in [...]
		if next_tok.kind == "word" and next_tok.value.lower() == "in":
			self._consume()
			return InNode(field=field_name, values=self._parse_list(), negated=False)

		# field op value
		if next_tok.kind == "op":
			op = self._consume().value
			if op in ("=", "<>"):
				op = "==" if op == "=" else "!="
			value = self._parse_value()
			return CompareNode(field=field_name, op=op, value=value)

		# No operator follows — treat as bare truthy field reference
		return CompareNode(field=field_name, op="==", value=True)

	def _parse_list(self) -> list[Any]:
		values: list[Any] = []
		tok = self._peek()

		# Support [...] and bare comma-separated identifiers (e.g. agency in FDA, EMA, HC)
		if tok and tok.kind == "lbracket":
			self._consume()
			while True:
				next_tok = self._peek()
				if next_tok is None or next_tok.kind == "rbracket":
					break
				if next_tok.kind == "comma":
					self._consume()
					continue
				values.append(self._parse_value())
			close = self._peek()
			if not (close and close.kind == "rbracket"):
				raise RuleExprParseError("Expected ']' to close list")
			self._consume()
		else:
			# Bare list: FDA, EMA, HC (until 'and'/'or'/end)
			while True:
				next_tok = self._peek()
				if next_tok is None:
					break
				if next_tok.kind == "word" and next_tok.value.lower() in ("and", "or"):
					break
				if next_tok.kind == "comma":
					self._consume()
					continue
				values.append(self._parse_value())

		return values

	def _parse_value(self) -> Any:
		tok = self._peek()
		if tok is None:
			raise RuleExprParseError("Expected value, got end of expression")
		if tok.kind in ("string", "number", "bool", "null"):
			self._consume()
			return tok.value
		if tok.kind == "word":
			self._consume()
			# Return as string for identifier values
			return tok.value
		raise RuleExprParseError(f"Expected value, got {tok.value!r}")


# ── Public API ───────────────────────────────────────────────────────────────

def parse_rule_expr(condition: str) -> RuleExprNode | None:
	"""Parse a rule condition string into a typed AST node.

	Returns:
	    RuleExprNode  — a CompareNode, MissingNode, InNode, AndNode, or OrNode
	    None          — the condition is empty or whitespace-only (not an error)

	Raises:
	    RuleExprParseError  — the condition has unexpected tokens or invalid syntax

	Callers must handle three states: node (success), None (empty input),
	and RuleExprParseError (bad syntax). Use ``validate_rule_fields`` for a
	no-raise variant that turns parse failures into warning strings.
	"""
	condition = (condition or "").strip()
	if not condition:
		return None
	tokens = _tokenize(condition)
	if not tokens:
		return None
	return _Parser(tokens).parse()


def extract_fields(node: RuleExprNode) -> set[str]:
	"""Return all field names referenced in a rule expression.

	Dotted paths are returned as-is (e.g. ``account.status`` is returned as
	the single string ``"account.status"``, not split into ``"account"`` and
	``"status"``).  The ``validate_rule_fields`` function handles base-name
	matching so that ``{"status"}`` in ``known_fields`` also accepts
	``"account.status"`` references.
	"""
	if isinstance(node, (CompareNode, MissingNode, InNode)):
		return {node.field}
	if isinstance(node, (AndNode, OrNode)):
		return extract_fields(node.left) | extract_fields(node.right)
	return set()


def expr_to_dict(node: RuleExprNode) -> dict[str, Any]:
	"""Convert a rule expression node to a JSON-serializable dict."""
	if isinstance(node, CompareNode):
		return {"type": "compare", "field": node.field, "op": node.op, "value": node.value}
	if isinstance(node, MissingNode):
		return {"type": "missing", "field": node.field, "negated": node.negated}
	if isinstance(node, InNode):
		return {"type": "in", "field": node.field, "values": node.values, "negated": node.negated}
	if isinstance(node, AndNode):
		return {"type": "and", "left": expr_to_dict(node.left), "right": expr_to_dict(node.right)}
	if isinstance(node, OrNode):
		return {"type": "or", "left": expr_to_dict(node.left), "right": expr_to_dict(node.right)}
	return {}


def validate_rule_fields(
	condition: str,
	known_fields: "set[str] | None",
) -> list[str]:
	"""Parse condition and return warnings for unknown field references.

	known_fields semantics:
	    None    → skip field validation (syntax-only check; use in lint dry-run)
	    set()   → strict: warn on every field reference (zero known fields)
	    {name}  → warn on fields not in the set; also matches dotted paths by
	              base name (account.status validates against {'status'})

	Never raises — parse failures produce a single warning entry.
	"""
	if known_fields is None:
		# Syntax-only pass: still report parse errors as warnings
		try:
			parse_rule_expr(condition)
		except RuleExprParseError as e:
			return [f"Rule condition parse warning: {e}"]
		return []
	try:
		node = parse_rule_expr(condition)
	except RuleExprParseError as e:
		return [f"Rule condition parse warning: {e}"]
	if node is None:
		return []
	warnings: list[str] = []
	for f in extract_fields(node):
		# Strip dotted path prefix for checking (e.g. account.status → check 'status' too)
		base = f.split(".")[-1]
		if f not in known_fields and base not in known_fields:
			warnings.append(f"Rule condition references unknown field: {f!r}")
	return warnings
