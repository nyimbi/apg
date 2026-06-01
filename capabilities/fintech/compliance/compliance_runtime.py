"""Runtime helpers for APG FinTech Compliance Automation."""

from __future__ import annotations


def normalize_code(value: str) -> str:
	return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def status_supported(value: str, supported: list[str]) -> bool:
	return normalize_code(value) in supported


def retention_present(value: int | str) -> bool:
	try:
		return int(value) > 0
	except (TypeError, ValueError):
		return False


def check_failed(result: str) -> bool:
	return normalize_code(result) in {"fail", "failed", "non_compliant", "exception"}
