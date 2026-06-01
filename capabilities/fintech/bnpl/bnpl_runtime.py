"""Runtime helpers for APG Buy Now Pay Later."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP


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


def normalize_score(value: float | int | str) -> int:
	try:
		score = int(Decimal(str(value)).to_integral_value(rounding=ROUND_HALF_UP))
	except (InvalidOperation, ValueError) as exc:
		raise ValueError(f"invalid score: {value!r}") from exc
	return score


def installments_for_plan(plan_type: str) -> int:
	plan_type = normalize_code(plan_type)
	if plan_type == "pay_in_3":
		return 3
	if plan_type == "pay_in_4":
		return 4
	return 1


def estimate_installment_amount(principal: float | int | str, down_payment: float | int | str, plan_type: str) -> float:
	remaining = max(normalize_amount(principal) - normalize_amount(down_payment), 0)
	count = installments_for_plan(plan_type)
	return normalize_amount(remaining / count if count else remaining)


def decision_is_final(decision: str) -> bool:
	return normalize_code(decision) in {"approve", "decline"}


def decision_is_approved(decision: str) -> bool:
	return normalize_code(decision) == "approve"
