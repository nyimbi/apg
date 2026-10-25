"""OpenUI Lang lexer (spec 1.3).

Line-oriented with bracket- and string-aware continuation, so it emits one
token stream per statement. Lexing never fails: a character matching no rule is
skipped, per spec 1.3.

Token kinds: NAME, STATE ($name), AT (@name), STRING, NUMBER, OP, NEWLINE.
"""

from __future__ import annotations

from dataclasses import dataclass

# Multi-character operators, longest first so the scanner takes the longest match.
OPERATORS = (
	"==", "!=", ">=", "<=", "&&", "||",
	"=", ">", "<", "+", "-", "*", "/", "%", "!", "?", ":", ".",
	",", "(", ")", "[", "]", "{", "}",
)

_IDENT_START = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_")
_IDENT_BODY = _IDENT_START | frozenset("0123456789")
_DIGITS = frozenset("0123456789")


@dataclass(frozen=True)
class Token:
	kind: str
	value: object
	start: int
	end: int

	def __repr__(self) -> str:  # pragma: no cover - debugging aid
		return f"Token({self.kind}, {self.value!r})"


class Lexer:
	"""Tokenize one statement's text at a time.

	The caller splits the prepared source into statements (spec 1.4); this class
	only scans a single statement string, which keeps continuation logic in one
	place.
	"""

	def __init__(self, source: str) -> None:
		self.source = source
		self.pos = 0
		self.length = len(source)

	def tokenize(self) -> list[Token]:
		tokens: list[Token] = []
		while self.pos < self.length:
			token = self._next_token(tokens)
			if token is not None:
				tokens.append(token)
		return tokens

	def _next_token(self, emitted: list[Token]) -> Token | None:
		char = self.source[self.pos]

		if char in " \t\r\n":
			self.pos += 1
			return None

		# Comments run to end of line; the statement splitter has already
		# removed them, but a stray `#` or `//` inside a statement is dropped.
		if char == "#" or self.source.startswith("//", self.pos):
			while self.pos < self.length and self.source[self.pos] != "\n":
				self.pos += 1
			return None

		if char in "\"'":
			return self._string()

		if char in _IDENT_START:
			return self._name()

		if char == "$":
			return self._prefixed("STATE")

		if char == "@":
			return self._prefixed("AT")

		if char in _DIGITS or (char == "-" and self._minus_starts_number(emitted)):
			return self._number()

		return self._operator()

	def _minus_starts_number(self, emitted: list[Token]) -> bool:
		"""Spec 1.3: `-` starts a number unless the previous token is a value."""
		if self.pos + 1 >= self.length or self.source[self.pos + 1] not in _DIGITS:
			return False
		if not emitted:
			return True
		previous = emitted[-1]
		if previous.kind in ("NUMBER", "STRING"):
			return False
		if previous.kind in ("NAME", "STATE"):
			return False
		if previous.kind == "OP" and previous.value in (")", "]"):
			return False
		return True

	def _string(self) -> Token:
		start = self.pos
		quote = self.source[self.pos]
		self.pos += 1
		pieces: list[str] = []
		while self.pos < self.length:
			char = self.source[self.pos]
			if char == "\\" and self.pos + 1 < self.length:
				pieces.append(self._escape(quote))
				continue
			if char == quote:
				self.pos += 1
				return Token("STRING", "".join(pieces), start, self.pos)
			pieces.append(char)
			self.pos += 1
		# Unterminated: the caller closes it implicitly (spec 4.2) before lexing.
		return Token("STRING", "".join(pieces), start, self.pos)

	def _escape(self, quote: str) -> str:
		self.pos += 1
		char = self.source[self.pos]
		if quote == '"':
			mapping = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f",
				'"': '"', "\\": "\\", "/": "/"}
			if char == "u" and self.pos + 4 < self.length:
				digits = self.source[self.pos + 1:self.pos + 5]
				if all(d in "0123456789abcdefABCDEF" for d in digits):
					self.pos += 5
					return chr(int(digits, 16))
			self.pos += 1
			return mapping.get(char, char)
		# Single-quoted strings support only \' \\ \n \t; anything else is literal.
		mapping = {"'": "'", "\\": "\\", "n": "\n", "t": "\t"}
		self.pos += 1
		if char in mapping:
			return mapping[char]
		return "\\" + char

	def _name(self) -> Token:
		start = self.pos
		while self.pos < self.length and self.source[self.pos] in _IDENT_BODY:
			self.pos += 1
		return Token("NAME", self.source[start:self.pos], start, self.pos)

	def _prefixed(self, kind: str) -> Token:
		start = self.pos
		self.pos += 1
		body_start = self.pos
		while self.pos < self.length and self.source[self.pos] in _IDENT_BODY:
			self.pos += 1
		return Token(kind, self.source[body_start:self.pos], start, self.pos)

	def _number(self) -> Token:
		start = self.pos
		if self.source[self.pos] == "-":
			self.pos += 1
		while self.pos < self.length and self.source[self.pos] in _DIGITS:
			self.pos += 1
		# A `.` is part of the number only when a digit follows (spec 1.3).
		if (self.pos + 1 < self.length and self.source[self.pos] == "."
				and self.source[self.pos + 1] in _DIGITS):
			self.pos += 1
			while self.pos < self.length and self.source[self.pos] in _DIGITS:
				self.pos += 1
		if self.pos < self.length and self.source[self.pos] in "eE":
			lookahead = self.pos + 1
			if lookahead < self.length and self.source[lookahead] in "+-":
				lookahead += 1
			if lookahead < self.length and self.source[lookahead] in _DIGITS:
				self.pos = lookahead
				while self.pos < self.length and self.source[self.pos] in _DIGITS:
					self.pos += 1
		text = self.source[start:self.pos]
		try:
			value: object = int(text)
		except ValueError:
			try:
				value = float(text)
			except ValueError:
				value = 0
		return Token("NUMBER", value, start, self.pos)

	def _operator(self) -> Token:
		start = self.pos
		for op in OPERATORS:
			if self.source.startswith(op, self.pos):
				self.pos += len(op)
				return Token("OP", op, start, self.pos)
		# No rule matches: skip the character (spec 1.3).
		self.pos += 1
		return Token("SKIP", self.source[start:self.pos], start, self.pos)


def tokenize(source: str) -> list[Token]:
	return [token for token in Lexer(source).tokenize() if token.kind != "SKIP"]
