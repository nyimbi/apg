"""Runtime helpers for APG InsurTech."""

from __future__ import annotations


def normalize_code(value: str) -> str:
	return value.strip().lower().replace(" ", "_").replace("-", "_")


def normalize_currency(value: str) -> str:
	return value.strip().upper()


def positive_minor(value: int) -> bool:
	return int(value) > 0


def score_present(value: float | int | None) -> bool:
	return value is not None and 0 <= float(value) <= 100
