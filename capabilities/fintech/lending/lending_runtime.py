"""Domain helpers for APG Digital Lending."""

from __future__ import annotations

from datetime import date, timedelta


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


def normalize_rate(value: float | int | str | None) -> float:
	if value in (None, ""):
		return 0.0
	return round(float(value), 6)


def normalize_score(value: float | int | str | None) -> int:
	if value in (None, ""):
		return 0
	return int(round(float(value)))


def iso_due_date(days_from_today: int) -> str:
	return (date.today() + timedelta(days=days_from_today)).isoformat()


def estimate_installment(principal: float, annual_rate: float, term_days: int, frequency: str) -> float:
	frequency = normalize_code(frequency)
	periods_per_year = {"weekly": 52, "biweekly": 26, "monthly": 12, "quarterly": 4}.get(frequency, 12)
	periods = max(1, round((term_days / 365) * periods_per_year))
	period_rate = annual_rate / periods_per_year
	if period_rate <= 0:
		return round(principal / periods, 2)
	return round((principal * period_rate) / (1 - (1 + period_rate) ** -periods), 2)


def decision_category(decision: str) -> str:
	normalized = normalize_code(decision)
	if normalized in {"approve", "decline"}:
		return "final"
	if normalized in {"refer", "counteroffer"}:
		return "review"
	return "unsupported"
