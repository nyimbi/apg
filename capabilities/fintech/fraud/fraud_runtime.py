"""Domain helpers for APG Fraud Detection."""

from __future__ import annotations


MONEY_SIGNAL_TYPES = {"payment", "wallet_transfer", "card_not_present", "refund", "chargeback"}


def normalize_code(value: str) -> str:
	return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")


def normalize_currency(value: str) -> str:
	return str(value or "").strip().upper()


def normalize_amount(value: float | int | str | None) -> float:
	if value in (None, ""):
		return 0.0
	return round(float(value), 2)


def normalize_risk_score(value: int | str) -> int:
	return int(value)


def risk_band(score: int) -> str:
	if score >= 90:
		return "critical"
	if score >= 75:
		return "high"
	if score >= 45:
		return "medium"
	return "low"


def recommended_decision(score: int) -> str:
	if score >= 90:
		return "block"
	if score >= 75:
		return "hold"
	if score >= 60:
		return "step_up"
	if score >= 45:
		return "review"
	return "approve"


def collect_indicators(risk_score: int, velocity_indicator: bool, device_anomaly: bool, geo_anomaly: bool, aml_alert_present: bool, chargeback_signal: bool, account_takeover_indicator: bool) -> list[str]:
	indicators: list[str] = []
	if risk_score >= 75:
		indicators.append("high_risk_score")
	if velocity_indicator:
		indicators.append("velocity")
	if device_anomaly:
		indicators.append("device_anomaly")
	if geo_anomaly:
		indicators.append("geo_anomaly")
	if aml_alert_present:
		indicators.append("aml_alert")
	if chargeback_signal:
		indicators.append("chargeback")
	if account_takeover_indicator:
		indicators.append("account_takeover")
	return indicators
