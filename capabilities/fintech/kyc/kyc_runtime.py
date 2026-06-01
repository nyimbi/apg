"""Domain helpers for APG Know Your Customer."""

from __future__ import annotations


def normalize_code(value: str) -> str:
	return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")


def normalize_country(value: str) -> str:
	return str(value or "").strip().upper()


def normalize_confidence(value: float | int | str) -> float:
	confidence = float(value)
	if confidence < 0 or confidence > 1:
		raise ValueError("confidence must be between 0 and 1")
	return confidence


def normalize_risk_score(value: int | str) -> int:
	score = int(value)
	if score < 0 or score > 100:
		return score
	return score


def risk_band(score: int, high_threshold: int, medium_threshold: int) -> str:
	if score >= high_threshold:
		return "high"
	if score >= medium_threshold:
		return "medium"
	return "low"
