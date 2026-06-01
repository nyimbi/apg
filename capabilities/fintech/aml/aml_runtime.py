"""Domain helpers for APG Anti Money Laundering."""

from __future__ import annotations


def normalize_code(value: str) -> str:
	return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")


def normalize_currency(value: str) -> str:
	return str(value or "").strip().upper()


def normalize_amount(value: float | int | str) -> float:
	amount = float(value)
	return round(amount, 2)


def normalize_risk_score(value: int | str) -> int:
	return int(value)


def severity_from_score(score: int) -> str:
	if score >= 90:
		return "critical"
	if score >= 75:
		return "high"
	if score >= 45:
		return "medium"
	return "low"


def typology_flags(amount: float, risk_score: int, large_threshold: float, structuring_threshold: float, sanctions_hit: bool, velocity_indicator: bool) -> list[str]:
	flags: list[str] = []
	if amount >= large_threshold:
		flags.append("large_transaction")
	if amount >= structuring_threshold and amount < large_threshold:
		flags.append("structuring")
	if sanctions_hit:
		flags.append("sanctions")
	if velocity_indicator:
		flags.append("velocity")
	if risk_score >= 75:
		flags.append("high_risk_kyc")
	return flags
