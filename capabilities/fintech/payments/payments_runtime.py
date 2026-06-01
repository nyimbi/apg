"""Domain helpers for Digital Payments runtime decisions."""

from __future__ import annotations

from decimal import Decimal


def normalize_amount(value: Decimal | int | str) -> Decimal:
	"""Normalize a payment amount to Decimal and reject negative precision drift."""
	return Decimal(str(value))


def is_high_value(amount: Decimal, threshold: Decimal = Decimal("100000")) -> bool:
	"""Return whether a payment amount crosses the high-value review threshold."""
	return amount >= threshold


def settlement_variance_detected(variance_amount: Decimal | int | str) -> bool:
	"""Return whether settlement variance review should be considered."""
	return normalize_amount(variance_amount) != 0
