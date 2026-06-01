"""Runtime helpers for APG Agency Banking."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP


CASH_OUT_SERVICES = {"cash_out", "money_transfer", "bill_payment", "airtime_topup", "loan_collection", "card_services", "insurance", "savings_products"}
CASH_IN_SERVICES = {"cash_in", "loan_disbursement", "government_payments"}


def normalize_code(value: str) -> str:
	return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


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


def normalize_codes(values: list[str]) -> list[str]:
	return [normalize_code(value) for value in values]


def service_requires_float(service: str) -> bool:
	return normalize_code(service) in CASH_OUT_SERVICES


def service_increases_float(service: str) -> bool:
	return normalize_code(service) in CASH_IN_SERVICES


def apply_float_delta(current: float, service: str, amount: float) -> float:
	if service_requires_float(service):
		return normalize_amount(current - amount)
	if service_increases_float(service):
		return normalize_amount(current + amount)
	return normalize_amount(current)


def estimate_commission(amount: float | int | str, service: str) -> float:
	value = normalize_amount(amount)
	rate = 0.005
	if normalize_code(service) in {"cash_out", "money_transfer", "bill_payment"}:
		rate = 0.01
	return normalize_amount(value * rate)
