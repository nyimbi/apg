"""Runtime helpers for APG Mobile Banking."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from hashlib import sha256


HIGH_SEVERITIES = {"high", "critical"}


def normalize_code(value: str) -> str:
	return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def normalize_codes(values: list[str]) -> list[str]:
	return [normalize_code(value) for value in values]


def normalize_currency(value: str) -> str:
	return str(value or "").strip().upper()


def normalize_country(value: str) -> str:
	return str(value or "").strip().upper()


def normalize_amount(value: float | int | str | Decimal) -> float:
	try:
		amount = Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
	except (InvalidOperation, ValueError) as exc:
		raise ValueError(f"invalid amount: {value!r}") from exc
	return float(amount)


def device_fingerprint_hash(fingerprint: str) -> str:
	return sha256(str(fingerprint).encode("utf-8")).hexdigest()[:16]


def is_high_severity(severity: str) -> bool:
	return normalize_code(severity) in HIGH_SEVERITIES


def payment_direction(payment_type: str) -> str:
	payment_type = normalize_code(payment_type)
	if payment_type in {"peer_transfer", "merchant_payment", "bill_payment", "airtime", "loan_repayment", "savings_transfer", "card_payment", "wallet_cash_out"}:
		return "debit"
	return "unknown"
