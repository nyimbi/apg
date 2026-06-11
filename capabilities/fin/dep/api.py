"""Deposit Products Engine — REST API.

url_prefix = "/api/fin/dep"

All handlers are thin: parse → service call → serialise.
No business logic lives here.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any

try:
	from .models import (
		CompoundingFrequency, FeeConfig, FeeFrequency, InterestCalculationType,
		InterestConfig, InterestTier, MaturityInstruction, ProductTerms, ProductType,
	)
	from .service import DepositProductsService
except ImportError:  # pragma: no cover
	from models import (  # type: ignore
		CompoundingFrequency, FeeConfig, FeeFrequency, InterestCalculationType,
		InterestConfig, InterestTier, MaturityInstruction, ProductTerms, ProductType,
	)
	from service import DepositProductsService  # type: ignore


_SERVICE = DepositProductsService()


def service() -> DepositProductsService:
	"""Return the process-level singleton."""
	return _SERVICE


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _d(v: Any) -> Decimal:
	return Decimal(str(v))


def _date(v: Any) -> date:
	if isinstance(v, date):
		return v
	return date.fromisoformat(str(v))


def _interest_config(raw: dict[str, Any]) -> InterestConfig:
	tiers = [
		InterestTier(min_balance=_d(t["min_balance"]), rate=_d(t["rate"]))
		for t in raw.get("tiers", [])
	]
	return InterestConfig(
		rate=_d(raw["rate"]),
		calculation=InterestCalculationType(raw.get("calculation", "DAILY_ACCRUAL")),
		compounding=CompoundingFrequency(raw.get("compounding", "MONTHLY")),
		tiers=tiers,
		withholding_rate=_d(raw.get("withholding_rate", "0")),
	)


def _fee_config(raw: dict[str, Any]) -> FeeConfig:
	return FeeConfig(
		maintenance_fee=_d(raw.get("maintenance_fee", "0")),
		fee_frequency=FeeFrequency(raw.get("fee_frequency", "MONTHLY")),
		minimum_balance=_d(raw.get("minimum_balance", "0")),
		below_minimum_fee=_d(raw.get("below_minimum_fee", "0")),
	)


def _terms(raw: dict[str, Any]) -> ProductTerms:
	return ProductTerms(
		min_tenor_days=int(raw.get("min_tenor_days", 0)),
		max_tenor_days=int(raw.get("max_tenor_days", 0)),
		notice_period_days=int(raw.get("notice_period_days", 0)),
		auto_rollover=bool(raw.get("auto_rollover", False)),
		rollover_rate_delta=_d(raw.get("rollover_rate_delta", "0")),
		break_penalty_rate=_d(raw.get("break_penalty_rate", "0")),
		tax_exempt=bool(raw.get("tax_exempt", False)),
		allowed_currencies=raw.get("allowed_currencies", []),
		max_balance=_d(raw["max_balance"]) if raw.get("max_balance") else None,
		min_opening_amount=_d(raw.get("min_opening_amount", "0")),
	)


# ─────────────────────────────────────────────────────────────
# API handlers  (callable from Flask routes or direct testing)
# ─────────────────────────────────────────────────────────────

def health() -> dict[str, Any]:
	return _SERVICE.health_check()


def create_product(payload: dict[str, Any]) -> dict[str, Any]:
	svc = _SERVICE
	product = svc.create_product(
		tenant_id=payload["tenant_id"],
		code=payload["code"],
		name=payload["name"],
		product_type=ProductType(payload["product_type"]),
		currency=payload["currency"],
		interest_config=_interest_config(payload["interest_config"]),
		fee_config=_fee_config(payload.get("fee_config", {})),
		terms=_terms(payload.get("terms", {})),
		gl_interest_income_account=payload.get("gl_interest_income_account", ""),
		gl_interest_payable_account=payload.get("gl_interest_payable_account", ""),
		gl_wht_payable_account=payload.get("gl_wht_payable_account", ""),
		created_by=payload.get("created_by", "api"),
	)
	return product.model_dump(mode="json")


def get_product(tenant_id: str, code: str) -> dict[str, Any]:
	return _SERVICE.get_product(tenant_id, code).model_dump(mode="json")


def list_products(payload: dict[str, Any]) -> list[dict[str, Any]]:
	pt = ProductType(payload["product_type"]) if payload.get("product_type") else None
	products = _SERVICE.list_products(
		tenant_id=payload["tenant_id"],
		product_type=pt,
		active_only=payload.get("active_only", True),
	)
	return [p.model_dump(mode="json") for p in products]


def update_product(payload: dict[str, Any]) -> dict[str, Any]:
	updates: dict[str, Any] = {}
	if "name" in payload:
		updates["name"] = payload["name"]
	if "interest_config" in payload:
		updates["interest_config"] = _interest_config(payload["interest_config"])
	if "fee_config" in payload:
		updates["fee_config"] = _fee_config(payload["fee_config"])
	if "terms" in payload:
		updates["terms"] = _terms(payload["terms"])
	return _SERVICE.update_product(payload["tenant_id"], payload["code"], updates).model_dump(mode="json")


def deactivate_product(tenant_id: str, code: str) -> dict[str, Any]:
	return _SERVICE.deactivate_product(tenant_id, code).model_dump(mode="json")


def calculate_interest(payload: dict[str, Any]) -> dict[str, Any]:
	result = _SERVICE.calculate_interest(
		tenant_id=payload["tenant_id"],
		account_id=payload["account_id"],
		from_date=_date(payload["from_date"]),
		to_date=_date(payload["to_date"]),
		balance=_d(payload["balance"]),
		product_code=payload["product_code"],
	)
	return result.model_dump(mode="json")


def apply_interest(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.apply_interest(
		tenant_id=payload["tenant_id"],
		account_id=payload["account_id"],
		interest_amount=_d(payload["interest_amount"]),
		value_date=_date(payload["value_date"]),
		posting_ref=payload["posting_ref"],
	)


def apply_maintenance_fee(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.apply_maintenance_fee(
		tenant_id=payload["tenant_id"],
		account_id=payload["account_id"],
		posting_date=_date(payload["posting_date"]),
	)


def check_minimum_balance(tenant_id: str, account_id: str) -> dict[str, Any]:
	return _SERVICE.check_minimum_balance(tenant_id, account_id).model_dump(mode="json")


def process_term_deposit_maturity(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.process_term_deposit_maturity(
		tenant_id=payload["tenant_id"],
		account_id=payload["account_id"],
		instruction=MaturityInstruction(payload["instruction"]),
		partial_amount=_d(payload["partial_amount"]) if payload.get("partial_amount") else None,
		processed_by=payload.get("processed_by", "api"),
	).model_dump(mode="json")


def get_accrued_interest(tenant_id: str, account_id: str, as_of_date: str) -> dict[str, Any]:
	amount = _SERVICE.get_accrued_interest(tenant_id, account_id, _date(as_of_date))
	return {"account_id": account_id, "accrued_interest": str(amount), "as_of_date": as_of_date}


def calculate_break_penalty(tenant_id: str, account_id: str, break_date: str) -> dict[str, Any]:
	penalty = _SERVICE.calculate_break_penalty(tenant_id, account_id, _date(break_date))
	return {"account_id": account_id, "break_penalty": str(penalty), "break_date": break_date}


def get_interest_history(payload: dict[str, Any]) -> list[dict[str, Any]]:
	return _SERVICE.get_interest_history(
		tenant_id=payload["tenant_id"],
		account_id=payload["account_id"],
		from_date=_date(payload["from_date"]),
		to_date=_date(payload["to_date"]),
	)


def get_rate_schedule(tenant_id: str, product_code: str) -> list[dict[str, Any]]:
	entries = _SERVICE.get_rate_schedule(tenant_id, product_code)
	return [e.model_dump(mode="json") for e in entries]


def update_product_rate(payload: dict[str, Any]) -> dict[str, Any]:
	entry = _SERVICE.update_product_rate(
		tenant_id=payload["tenant_id"],
		product_code=payload["product_code"],
		new_rate=_d(payload["new_rate"]),
		effective_date=_date(payload["effective_date"]),
		reason=payload["reason"],
		changed_by=payload.get("changed_by", "api"),
	)
	return entry.model_dump(mode="json")


def get_products_by_balance(payload: dict[str, Any]) -> list[dict[str, Any]]:
	products = _SERVICE.get_products_by_balance(
		tenant_id=payload["tenant_id"],
		balance=_d(payload["balance"]),
		currency=payload["currency"],
	)
	return [p.model_dump(mode="json") for p in products]


def simulate_maturity(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.simulate_maturity(
		tenant_id=payload["tenant_id"],
		product_code=payload["product_code"],
		principal=_d(payload["principal"]),
		tenor_days=int(payload["tenor_days"]),
	).model_dump(mode="json")


def get_product_stats(tenant_id: str) -> dict[str, Any]:
	return _SERVICE.get_product_stats(tenant_id)


def batch_accrue_interest(payload: dict[str, Any]) -> dict[str, Any]:
	result = _SERVICE.batch_accrue_interest(
		tenant_id=payload["tenant_id"],
		accrual_date=_date(payload["accrual_date"]),
	)
	return result.model_dump(mode="json")


def get_withholding_tax_report(tenant_id: str, period_id: str) -> list[dict[str, Any]]:
	entries = _SERVICE.get_withholding_tax_report(tenant_id, period_id)
	return [e.model_dump(mode="json") for e in entries]


def register_account(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.register_account(
		tenant_id=payload["tenant_id"],
		account_id=payload["account_id"],
		product_code=payload["product_code"],
		balance=_d(payload.get("balance", "0")),
		opening_date=_date(payload["opening_date"]) if payload.get("opening_date") else None,
		maturity_date=_date(payload["maturity_date"]) if payload.get("maturity_date") else None,
		linked_account=payload.get("linked_account", ""),
	)
