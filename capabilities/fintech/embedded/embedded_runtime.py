"""Runtime helpers for APG Embedded Finance."""

from __future__ import annotations


def normalize_code(value: str) -> str:
	return value.strip().lower().replace(" ", "_").replace("-", "_")


def normalize_codes(values: list[str]) -> list[str]:
	return [normalize_code(value) for value in values if value and value.strip()]


def normalize_domain(value: str) -> str:
	return value.strip().lower().removeprefix("https://").removeprefix("http://").rstrip("/")


def public_reference(prefix: str, *parts: str) -> str:
	body = "_".join(normalize_code(part) for part in parts if part)
	return f"{prefix}_{body}"[:80]


def percent_bounded(value: float, minimum: float = 0, maximum: float = 100) -> bool:
	return minimum <= float(value) <= maximum
