"""Runtime helpers for APG Banking APIs."""

from __future__ import annotations

from hashlib import sha256


def normalize_code(value: str) -> str:
	return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def normalize_codes(values: list[str]) -> list[str]:
	return [normalize_scope(value) for value in values]


def normalize_scope(value: str) -> str:
	return str(value or "").strip().lower().replace(" ", ".")


def normalize_url(value: str) -> str:
	return str(value or "").strip()


def client_public_id(application_id: str, auth_flow: str) -> str:
	seed = f"{application_id}:{auth_flow}".encode("utf-8")
	return "cli_" + sha256(seed).hexdigest()[:16]


def rate_limit_allows(call_count: int, limit: int) -> bool:
	return int(call_count) <= int(limit)


def is_critical_severity(severity: str) -> bool:
	return normalize_code(severity) == "critical"
