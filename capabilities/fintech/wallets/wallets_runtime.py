"""Domain helpers for Digital Wallets runtime decisions."""

from __future__ import annotations

from decimal import Decimal


def normalize_amount(value: Decimal | int | str) -> Decimal:
	return Decimal(str(value))


def normalize_code(value: str) -> str:
	return str(value or "").strip().lower()


def exceeds_limit(amount: Decimal, limit: Decimal) -> bool:
	return amount > limit
