"""Domain helpers for APG Cross-Border Remittance."""

from __future__ import annotations


def normalize_code(value: str) -> str:
	return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")


def normalize_country(value: str) -> str:
	return str(value or "").strip().upper()


def normalize_currency(value: str) -> str:
	return str(value or "").strip().upper()


def normalize_amount(value: float | int | str | None) -> float:
	if value in (None, ""):
		return 0.0
	return round(float(value), 2)


def normalize_rate(value: float | int | str | None) -> float:
	if value in (None, ""):
		return 0.0
	return round(float(value), 8)


def corridor_key(source_country: str, destination_country: str, source_currency: str, destination_currency: str) -> str:
	return f"{normalize_country(source_country)}-{normalize_country(destination_country)}:{normalize_currency(source_currency)}-{normalize_currency(destination_currency)}"


def transfer_band(send_amount: float) -> str:
	if send_amount >= 100000:
		return "high_value"
	if send_amount >= 25000:
		return "enhanced_review"
	return "standard"


def payout_state(fraud_decision: str, aml_review: bool) -> str:
	decision = normalize_code(fraud_decision)
	if decision == "block":
		return "blocked"
	if decision in {"review", "hold"} or aml_review:
		return "review_required"
	return "ready_for_payout"
