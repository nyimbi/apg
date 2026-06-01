"""Runtime helpers for APG Crowdfunding Platform."""

from __future__ import annotations


def normalize_code(value: str) -> str:
	return value.strip().lower().replace(" ", "_").replace("-", "_")


def normalize_currency(value: str) -> str:
	return value.strip().upper()


def positive_minor(value: int) -> bool:
	return int(value) > 0
