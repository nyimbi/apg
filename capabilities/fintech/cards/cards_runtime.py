"""Domain helpers for APG Digital Cards."""

from __future__ import annotations


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


def mask_pan(card_id: str, bin_range: str) -> str:
	suffix = str(abs(hash(card_id)) % 10000).zfill(4)
	prefix = str(bin_range or "000000")[:6].ljust(6, "0")
	return f"{prefix}******{suffix}"


def authorization_decision(fraud_decision: str, aml_result: str, high_impact: bool) -> str:
	fraud = normalize_code(fraud_decision)
	aml = normalize_code(aml_result)
	if fraud == "block" or aml == "blocked":
		return "decline"
	if high_impact or fraud in {"review", "hold"} or aml == "review":
		return "review"
	return "approve"
