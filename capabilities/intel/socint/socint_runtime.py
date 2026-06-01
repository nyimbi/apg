"""Small runtime helpers for APG Social Media Intelligence."""

from __future__ import annotations

from typing import Any


def normalize_code(value: Any) -> str:
	return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")


def present(value: Any) -> bool:
	return value is not None and str(value).strip() != ""


def bounded_score(value: Any) -> bool:
	try:
		number = float(value)
	except (TypeError, ValueError):
		return False
	return 0.0 <= number <= 1.0


def positive_int(value: Any) -> bool:
	try:
		return int(value) > 0
	except (TypeError, ValueError):
		return False
