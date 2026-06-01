"""Runtime helpers for APG Intelligence Analytics."""

from __future__ import annotations

from collections.abc import Sized
from typing import Any


def normalize_code(value: str) -> str:
	return value.strip().lower().replace("-", "_").replace(" ", "_")


def present(value: Any) -> bool:
	if value is None:
		return False
	if isinstance(value, str):
		return bool(value.strip())
	if isinstance(value, Sized):
		return len(value) > 0
	return True


def bounded_score(value: float) -> bool:
	try:
		number = float(value)
	except (TypeError, ValueError):
		return False
	return 0.0 <= number <= 1.0


def positive_int(value: int) -> bool:
	try:
		return int(value) > 0
	except (TypeError, ValueError):
		return False
