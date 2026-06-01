"""Runtime helpers for APG Cryptocurrency Services."""

from __future__ import annotations


def normalize_code(value: str) -> str:
	return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def normalize_symbol(value: str) -> str:
	return str(value or "").strip().upper()


def present(value: str) -> bool:
	return bool(str(value or "").strip())


def positive_int(value: int) -> bool:
	return isinstance(value, int) and value > 0


def non_negative_int(value: int) -> bool:
	return isinstance(value, int) and value >= 0
