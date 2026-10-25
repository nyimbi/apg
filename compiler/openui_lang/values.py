"""ECMAScript value semantics for OpenUI Lang evaluation (spec section 3).

The spec defers coercion to ECMA-262 explicitly: ToNumber, IsLooselyEqual, and
Number::toString. Clients in other languages MUST reproduce them, so these are
implementations of those algorithms, not Python approximations.

The one deliberate departure is `+`: the spec overrides ECMA so that `null`
joined into a string contributes `""` rather than `"null"`.
"""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any

# ECMAScript WhiteSpace and LineTerminator, the set ToNumber strips.
_JS_WHITESPACE = (
	"\t\n\v\f\r          "
	"        　﻿"
)
_DECIMAL_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")
_HEX_RE = re.compile(r"^[+-]?0[xX][0-9a-fA-F]+$")
_OCT_RE = re.compile(r"^[+-]?0[oO][0-7]+$")
_BIN_RE = re.compile(r"^[+-]?0[bB][01]+$")


def is_number(value: Any) -> bool:
	"""True for numeric values, excluding booleans (which are their own type)."""
	return isinstance(value, (int, float)) and not isinstance(value, bool)


def to_number(value: Any) -> float | int:
	"""ECMA ToNumber. NaN never surfaces: it maps to 0 per spec section 3.1."""
	if value is True:
		return 1
	if value is False:
		return 0
	if value is None:
		return 0
	if is_number(value):
		return value
	if isinstance(value, str):
		return _string_to_number(value)
	# Arrays, objects, and component instances: ToPrimitive gives a string,
	# which for arrays is the comma join and for objects is "[object Object]".
	return _string_to_number(_to_primitive_string(value))


def _string_to_number(text: str) -> float | int:
	stripped = text.strip(_JS_WHITESPACE)
	if stripped == "":
		return 0
	sign = 1
	body = stripped
	if body[:1] in "+-":
		if body[0] == "-":
			sign = -1
		body = body[1:]
	if body in ("Infinity",):
		return sign * math.inf
	if _HEX_RE.match(stripped):
		return sign * int(body, 16)
	if _OCT_RE.match(stripped):
		return sign * int(body, 8)
	if _BIN_RE.match(stripped):
		return sign * int(body, 2)
	if _DECIMAL_RE.match(stripped):
		number = float(stripped)
		return int(number) if number.is_integer() and abs(number) < 2**53 else number
	return 0


def _to_primitive_string(value: Any) -> str:
	if isinstance(value, list):
		return ",".join("" if item is None else to_string(item) for item in value)
	if isinstance(value, dict):
		return "[object Object]"
	return to_string(value)


def to_string(value: Any) -> str:
	"""ECMA ToString."""
	if value is None:
		return "null"
	if value is True:
		return "true"
	if value is False:
		return "false"
	if is_number(value):
		return number_to_string(value)
	if isinstance(value, str):
		return value
	if isinstance(value, list):
		return _to_primitive_string(value)
	if isinstance(value, dict):
		return "[object Object]"
	return str(value)


def number_to_string(value: float | int) -> str:
	"""ECMA Number::toString for radix 10.

	Python's shortest round-trip repr agrees with ECMA on digits; the two differ
	only in exponent spelling and in the 1e21 threshold, both handled here.
	"""
	if isinstance(value, int) and not isinstance(value, bool):
		return str(value)
	if isinstance(value, float):
		if math.isnan(value):
			return "NaN"
		if math.isinf(value):
			return "Infinity" if value > 0 else "-Infinity"
		if value.is_integer() and abs(value) < 1e21:
			return str(int(value))
		text = repr(value)
	else:
		return str(value)
	if "e" not in text and "E" not in text:
		return text
	mantissa, _, exponent = text.partition("e")
	if not exponent:
		mantissa, _, exponent = text.partition("E")
	sign = "+" if not exponent.startswith("-") else "-"
	digits = exponent.lstrip("+-").lstrip("0") or "0"
	return f"{mantissa}e{sign}{digits}"


def to_boolean(value: Any) -> bool:
	"""ECMA ToBoolean. Arrays, objects, and components are always truthy."""
	if value is None or value is False:
		return False
	if value is True:
		return True
	if is_number(value):
		return value != 0 and not (isinstance(value, float) and math.isnan(value))
	if isinstance(value, str):
		return value != ""
	return True


def loose_equal(left: Any, right: Any) -> bool:
	"""ECMA IsLooselyEqual."""
	left_nullish = left is None
	right_nullish = right is None
	if left_nullish or right_nullish:
		return left_nullish and right_nullish
	if type(left) is type(right) and isinstance(left, (str, bool)):
		return left == right
	if isinstance(left, bool):
		return loose_equal(to_number(left), right)
	if isinstance(right, bool):
		return loose_equal(left, to_number(right))
	if is_number(left) and is_number(right):
		return left == right
	if is_number(left) and isinstance(right, str):
		return to_number(left) == to_number(right)
	if isinstance(left, str) and is_number(right):
		return to_number(left) == to_number(right)
	if isinstance(left, str) and isinstance(right, str):
		return left == right
	if isinstance(left, (list, dict)) and not isinstance(right, (list, dict)):
		return loose_equal(_to_primitive_string(left), right)
	if isinstance(right, (list, dict)) and not isinstance(left, (list, dict)):
		return loose_equal(left, _to_primitive_string(right))
	return left == right


def add(left: Any, right: Any) -> Any:
	"""`+` per spec 3.2: string join when either side is a string, null as ''."""
	if isinstance(left, str) or isinstance(right, str):
		return _join_string(left) + _join_string(right)
	return to_number(left) + to_number(right)


def _join_string(value: Any) -> str:
	return "" if value is None else to_string(value)


def subtract(left: Any, right: Any) -> float | int:
	return to_number(left) - to_number(right)


def multiply(left: Any, right: Any) -> float | int:
	return to_number(left) * to_number(right)


def divide(left: Any, right: Any) -> float | int:
	divisor = to_number(right)
	if divisor == 0:
		return 0
	return to_number(left) / divisor


def modulo(left: Any, right: Any) -> float | int:
	divisor = to_number(right)
	if divisor == 0:
		return 0
	return math.fmod(to_number(left), divisor)


def negate(value: Any) -> float | int:
	return -to_number(value)


def compare(left: Any, right: Any, op: str) -> bool:
	a = to_number(left)
	b = to_number(right)
	if op == ">":
		return a > b
	if op == "<":
		return a < b
	if op == ">=":
		return a >= b
	return a <= b


def collation_key(value: Any) -> tuple:
	"""Approximate ECMAScript localeCompare with no arguments.

	Primary weight is the accent-stripped casefolded text, so `é` groups with
	`e`; secondary prefers lowercase before uppercase within a base letter,
	which is what ICU does for `["b", "B"]`. Sufficient for the conformance
	fixture and stable across platforms, unlike `locale.strxfrm`.
	"""
	text = "" if value is None else to_string(value)
	stripped = "".join(
		char for char in unicodedata.normalize("NFD", text)
		if not unicodedata.combining(char)
	)
	return (stripped.casefold(), text != text.casefold(), text)


def sort_values(values: list, field: str, descending: bool = False) -> list:
	"""`@Sort` semantics: numeric when both parse as numbers, else collation."""
	def sort_key(item: Any) -> tuple:
		if field:
			target = pluck_field(item, field)
		else:
			target = item
		text = "" if target is None else to_string(target)
		numeric = _looks_numeric(text)
		return (0, to_number(target)) if numeric else (1, collation_key(target))

	ordered = sorted(values, key=sort_key, reverse=descending)
	return ordered


def _looks_numeric(text: str) -> bool:
	stripped = text.strip(_JS_WHITESPACE)
	return bool(stripped) and bool(_DECIMAL_RE.match(stripped))


def pluck_field(item: Any, field: str) -> Any:
	"""`a.b` member access, including the array-pluck form (spec 3.3).

	A numeric field on an array indexes it, which is what `m.0` means when `m`
	is a list of lists (grammar/009-member-digits-field).
	"""
	if isinstance(item, list):
		index = _index_of(field)
		if index is not None:
			return item[index] if 0 <= index < len(item) else None
		return [pluck_field(element, field) for element in item]
	if isinstance(item, dict):
		return item.get(field)
	return None


def _index_of(field: str) -> int | None:
	if not field or not field.isdigit():
		return None
	return int(field)


def is_blank(value: Any) -> bool:
	"""The empty-value test validation rules skip (spec 5.4)."""
	if value is None:
		return True
	if isinstance(value, str):
		return value == ""
	if isinstance(value, list):
		return len(value) == 0
	if isinstance(value, dict):
		return len(value) == 0
	return False
