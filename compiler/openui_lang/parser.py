"""OpenUI Lang expression AST and recursive-descent parser (spec 1.5, 1.6).

The parser is total: on a malformed statement it returns ``None`` and the
caller drops that statement, which is the spec's recovery rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .lexer import Token, tokenize


class ParseError(Exception):
	"""A statement did not parse. The caller drops the statement."""


# ── nodes ───────────────────────────────────────────────────────────────────


@dataclass
class Literal:
	value: Any


@dataclass
class Reference:
	name: str


@dataclass
class StateRef:
	name: str


@dataclass
class Call:
	name: str
	args: list["Node"]
	is_component: bool = False


@dataclass
class BuiltinCall:
	name: str
	args: list["Node"]


@dataclass
class ActionCall:
	name: str
	args: list["Node"]


@dataclass
class ArrayLiteral:
	items: list["Node"]


@dataclass
class ObjectLiteral:
	entries: list[tuple[str, "Node"]]


@dataclass
class Member:
	target: "Node"
	field: str


@dataclass
class Index:
	target: "Node"
	index: "Node"


@dataclass
class Unary:
	op: str
	operand: "Node"


@dataclass
class Binary:
	op: str
	left: "Node"
	right: "Node"


@dataclass
class Ternary:
	condition: "Node"
	consequent: "Node"
	alternate: "Node"


Node = (
	Literal | Reference | StateRef | Call | BuiltinCall | ActionCall
	| ArrayLiteral | ObjectLiteral | Member | Index | Unary | Binary | Ternary
)


# ── parser ──────────────────────────────────────────────────────────────────

_RESERVED_CALL_HEADS = frozenset({"Query", "Mutation", "Action"})


class ExpressionParser:
	def __init__(self, tokens: list[Token]) -> None:
		self.tokens = tokens
		self.pos = 0

	# -- token helpers --

	def peek(self) -> Token | None:
		return self.tokens[self.pos] if self.pos < len(self.tokens) else None

	def advance(self) -> Token:
		token = self.peek()
		if token is None:
			raise ParseError("unexpected end of input")
		self.pos += 1
		return token

	def at_op(self, *values: str) -> bool:
		token = self.peek()
		return token is not None and token.kind == "OP" and token.value in values

	def accept_op(self, *values: str) -> bool:
		if self.at_op(*values):
			self.pos += 1
			return True
		return False

	def expect_op(self, value: str) -> None:
		if not self.accept_op(value):
			raise ParseError(f"expected {value!r}")

	# -- grammar --

	def parse_expression(self) -> Node:
		return self.parse_ternary()

	def parse_ternary(self) -> Node:
		condition = self.parse_or()
		if self.accept_op("?"):
			consequent = self.parse_ternary()
			self.expect_op(":")
			alternate = self.parse_ternary()
			return Ternary(condition, consequent, alternate)
		return condition

	def parse_or(self) -> Node:
		node = self.parse_and()
		while self.accept_op("||"):
			node = Binary("||", node, self.parse_and())
		return node

	def parse_and(self) -> Node:
		node = self.parse_equality()
		while self.accept_op("&&"):
			node = Binary("&&", node, self.parse_equality())
		return node

	def parse_equality(self) -> Node:
		node = self.parse_comparison()
		while self.at_op("==", "!="):
			op = self.advance().value
			node = Binary(str(op), node, self.parse_comparison())
		return node

	def parse_comparison(self) -> Node:
		node = self.parse_additive()
		while self.at_op(">", "<", ">=", "<="):
			op = self.advance().value
			node = Binary(str(op), node, self.parse_additive())
		return node

	def parse_additive(self) -> Node:
		node = self.parse_multiplicative()
		while self.at_op("+", "-"):
			op = self.advance().value
			node = Binary(str(op), node, self.parse_multiplicative())
		return node

	def parse_multiplicative(self) -> Node:
		node = self.parse_unary()
		while self.at_op("*", "/", "%"):
			op = self.advance().value
			node = Binary(str(op), node, self.parse_unary())
		return node

	def parse_unary(self) -> Node:
		if self.at_op("!", "-"):
			op = str(self.advance().value)
			return Unary(op, self.parse_unary())
		return self.parse_postfix()

	def parse_postfix(self) -> Node:
		node = self.parse_primary()
		while True:
			if self.accept_op("."):
				token = self.advance()
				# A field accepts the same tokens as an object key (spec 1.5).
				node = Member(node, _field_name(token))
				continue
			if self.accept_op("["):
				index = self.parse_expression()
				self.expect_op("]")
				node = Index(node, index)
				continue
			return node

	def parse_primary(self) -> Node:
		token = self.peek()
		if token is None:
			raise ParseError("unexpected end of input")

		if token.kind == "NUMBER":
			self.advance()
			return Literal(token.value)
		if token.kind == "STRING":
			self.advance()
			return Literal(token.value)
		if token.kind == "STATE":
			self.advance()
			return StateRef(str(token.value))
		if token.kind == "AT":
			return self.parse_at_call()
		if token.kind == "NAME":
			return self.parse_name_form()
		if token.kind == "OP" and token.value == "(":
			self.advance()
			node = self.parse_expression()
			self.expect_op(")")
			return node
		if token.kind == "OP" and token.value == "[":
			return self.parse_array()
		if token.kind == "OP" and token.value == "{":
			return self.parse_object()
		raise ParseError(f"unexpected token {token.value!r}")

	def parse_at_call(self) -> Node:
		token = self.advance()
		name = str(token.value)
		if name == "Action":
			return self.parse_action_expression()
		args = self.parse_arguments() if self.at_op("(") else []
		if name in _ACTION_NAMES:
			return ActionCall(name, args)
		return BuiltinCall(name, args)

	def parse_action_expression(self) -> Node:
		"""`Action([...])` holds a list of actions (spec 6.3)."""
		if not self.at_op("("):
			return ActionCall("Action", [])
		self.expect_op("(")
		actions: list[Node] = []
		while not self.at_op(")") and self.peek() is not None:
			if self.accept_op(","):
				continue
			actions.append(self.parse_expression())
		self.expect_op(")")
		return ActionCall("Action", actions)

	def parse_name_form(self) -> Node:
		token = self.advance()
		name = str(token.value)
		if self.at_op("("):
			args = self.parse_arguments()
			is_component = bool(name) and name[0].isupper()
			if name in _RESERVED_CALL_HEADS:
				return Call(name, args, is_component=False)
			if is_component:
				return Call(name, args, is_component=True)
			# `Count(x)` parses as the reference `Count`; arguments are lost.
			return Reference(name)
		if name == "true":
			return Literal(True)
		if name == "false":
			return Literal(False)
		if name == "null":
			return Literal(None)
		return Reference(name)

	def parse_arguments(self) -> list[Node]:
		self.expect_op("(")
		args: list[Node] = []
		while not self.at_op(")") and self.peek() is not None:
			if self.accept_op(","):
				continue
			args.append(self.parse_expression())
		self.expect_op(")")
		return args

	def parse_array(self) -> Node:
		self.expect_op("[")
		items: list[Node] = []
		while not self.at_op("]") and self.peek() is not None:
			if self.accept_op(","):
				continue
			items.append(self.parse_expression())
		self.expect_op("]")
		return ArrayLiteral(items)

	def parse_object(self) -> Node:
		self.expect_op("{")
		entries: list[tuple[str, Node]] = []
		while not self.at_op("}") and self.peek() is not None:
			if self.accept_op(","):
				continue
			key = _object_key(self.advance())
			self.expect_op(":")
			entries.append((key, self.parse_expression()))
		self.expect_op("}")
		return ObjectLiteral(entries)


_ACTION_NAMES = frozenset({"Set", "Reset", "Run", "ToAssistant", "OpenUrl"})


def _field_name(token: Token) -> str:
	if token.kind == "NAME":
		return str(token.value)
	if token.kind == "NUMBER":
		value = token.value
		return str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)
	if token.kind == "STATE":
		return str(token.value)
	if token.kind == "STRING":
		return str(token.value)
	return "?"


def _object_key(token: Token) -> str:
	if token.kind == "NAME":
		return str(token.value)
	if token.kind == "STRING":
		return str(token.value)
	if token.kind == "NUMBER":
		value = token.value
		return str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)
	if token.kind == "STATE":
		return str(token.value)
	if token.kind == "AT":
		return str(token.value)
	return "?"


@dataclass
class ParsedStatement:
	name: str
	kind: str          # "value" | "state" | "query" | "mutation"
	expression: Node
	text: str = ""
	incomplete: bool = False
	needs_closing: bool = False
	extra: dict[str, Any] = field(default_factory=dict)


def parse_statement(text: str, *, incomplete: bool = False, needs_closing: bool = False) -> ParsedStatement | None:
	"""Parse one `name = expression` statement, or return None if it does not."""
	head, _, tail = text.partition("=")
	name = head.strip()
	if not name or not tail.strip():
		return None

	tokens = tokenize(tail)
	if not tokens:
		return None
	parser = ExpressionParser(tokens)
	try:
		expression = parser.parse_expression()
	except ParseError:
		return None
	if parser.pos != len(tokens):
		# Trailing tokens after a complete expression are implementation-defined
		# in 1.0 (spec 1.6); dropping the statement is the safe reading.
		return None

	kind = "value"
	if name.startswith("$"):
		kind = "state"
	if isinstance(expression, Call) and expression.name in _RESERVED_CALL_HEADS:
		if expression.name == "Query":
			kind = "query"
		elif expression.name == "Mutation":
			kind = "mutation"

	return ParsedStatement(
		name=name,
		kind=kind,
		expression=expression,
		text=text,
		incomplete=incomplete,
		needs_closing=needs_closing,
	)
