"""Runtime helpers for APG Portfolio Management."""

from __future__ import annotations


def normalize_code(value: str) -> str:
	return value.strip().lower().replace(" ", "_").replace("-", "_")


def normalize_currency(value: str) -> str:
	return value.strip().upper()


def allocation_totals_100(allocation: dict[str, float]) -> bool:
	return bool(allocation) and round(sum(float(value) for value in allocation.values()), 4) == 100


def positive_minor(value: int) -> bool:
	return int(value) > 0


def positive_quantity(value: float) -> bool:
	return float(value) > 0
