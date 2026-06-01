"""Runtime helpers for APG Algorithmic Trading."""

from __future__ import annotations


def normalize_code(value: str) -> str:
	return value.strip().lower().replace(" ", "_").replace("-", "_")


def positive_quantity(value: float) -> bool:
	return float(value) > 0


def positive_count(value: int) -> bool:
	return int(value) > 0


def positive_value(value: float) -> bool:
	return float(value) > 0
