"""Runtime helpers for APG Regulatory Technology."""

from __future__ import annotations


def normalize_code(value: str) -> str:
	return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def normalize_jurisdiction(value: str) -> str:
	return str(value or "").strip().upper()


def present(value: str) -> bool:
	return bool(str(value or "").strip())
