"""Runtime helpers for APG Wealth Management."""

from __future__ import annotations


def normalize_code(value: str) -> str:
	return value.strip().lower().replace(" ", "_").replace("-", "_")


def normalize_codes(values: list[str]) -> list[str]:
	return [normalize_code(value) for value in values if value and value.strip()]


def normalize_currency(value: str) -> str:
	return value.strip().upper()


def allocation_totals_100(allocation: dict[str, float]) -> bool:
	return bool(allocation) and round(sum(float(value) for value in allocation.values()), 4) == 100


def percent_bounded(value: float, minimum: float = 0, maximum: float = 100) -> bool:
	return minimum <= float(value) <= maximum
