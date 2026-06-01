"""Runtime helpers for APG Open Source Intelligence."""

from __future__ import annotations


def normalize_code(value: str) -> str:
	return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def present(value: str) -> bool:
	return bool(str(value or "").strip())


def positive_int(value: int) -> bool:
	return isinstance(value, int) and value > 0


def bounded_score(value: float) -> bool:
	return isinstance(value, (int, float)) and 0 <= float(value) <= 1
