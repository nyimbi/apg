"""Runtime helpers for APG FinTech Risk Management."""

from __future__ import annotations


def normalize_code(value: str) -> str:
	return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def normalize_currency(value: str) -> str:
	return str(value or "").strip().upper()


def positive_minor(value: int | str | float) -> bool:
	try:
		return int(value) > 0
	except (TypeError, ValueError):
		return False


def score_valid(value: int | float | str) -> bool:
	try:
		score = float(value)
	except (TypeError, ValueError):
		return False
	return 0 <= score <= 100


def probability_bps_valid(value: int | str) -> bool:
	try:
		bps = int(value)
	except (TypeError, ValueError):
		return False
	return 0 <= bps <= 10000


def risk_band(score: int | float) -> str:
	if score >= 80:
		return "critical"
	if score >= 60:
		return "high"
	if score >= 35:
		return "medium"
	return "low"
