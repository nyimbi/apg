"""Domain helpers for APG Digital Neobanking."""

from __future__ import annotations

from datetime import date


def normalize_code(value: str) -> str:
	return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")


def normalize_currency(value: str) -> str:
	return str(value or "").strip().upper()


def normalize_country(value: str) -> str:
	return str(value or "").strip().upper()


def normalize_amount(value: float | int | str | None) -> float:
	if value in (None, ""):
		return 0.0
	return round(float(value), 2)


def today_iso() -> str:
	return date.today().isoformat()


def account_number(account_id: str, country: str) -> str:
	prefix = normalize_country(country)[:2].ljust(2, "X")
	suffix = str(abs(hash(account_id)) % 10_000_000_000).zfill(10)
	return f"{prefix}{suffix}"


def transaction_direction(kind: str) -> str:
	kind = normalize_code(kind)
	if kind in {"deposit", "refund", "interest"}:
		return "credit"
	if kind in {"withdrawal", "fee", "card_purchase", "transfer_out"}:
		return "debit"
	if kind == "transfer_in":
		return "credit"
	return "unknown"
