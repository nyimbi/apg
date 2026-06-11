"""Unit tests for fin.dep models."""
from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from capabilities.fin.dep.models import (
	CompoundingFrequency, DepositProduct, FeeConfig, FeeFrequency,
	InterestCalculationType, InterestConfig, InterestTier, MaturityInstruction,
	ProductStatus, ProductTerms, ProductType,
)


def test_product_type_enum():
	assert ProductType.CURRENT == "CURRENT"
	assert ProductType.TERM_DEPOSIT == "TERM_DEPOSIT"
	assert len(ProductType) == 5


def test_interest_calculation_type_enum():
	assert InterestCalculationType.DAILY_ACCRUAL == "DAILY_ACCRUAL"
	assert len(InterestCalculationType) == 3


def test_compounding_frequency_enum():
	assert CompoundingFrequency.MONTHLY == "MONTHLY"
	assert len(CompoundingFrequency) == 3


def test_interest_tier_valid():
	tier = InterestTier(min_balance=Decimal("0"), rate=Decimal("3.5"))
	assert tier.min_balance == Decimal("0")
	assert tier.rate == Decimal("3.5")


def test_interest_tier_negative_balance_rejected():
	with pytest.raises(ValidationError):
		InterestTier(min_balance=Decimal("-1"), rate=Decimal("3.5"))


def test_interest_tier_rate_above_100_rejected():
	with pytest.raises(ValidationError):
		InterestTier(min_balance=Decimal("0"), rate=Decimal("101"))


def test_interest_config_defaults():
	cfg = InterestConfig(rate=Decimal("5.0"))
	assert cfg.calculation == InterestCalculationType.DAILY_ACCRUAL
	assert cfg.compounding == CompoundingFrequency.MONTHLY
	assert cfg.withholding_rate == Decimal("0")
	assert cfg.tiers == []


def test_fee_config_defaults():
	fc = FeeConfig()
	assert fc.maintenance_fee == Decimal("0")
	assert fc.fee_frequency == FeeFrequency.MONTHLY
	assert fc.minimum_balance == Decimal("0")
	assert fc.below_minimum_fee == Decimal("0")


def test_fee_config_negative_rejected():
	with pytest.raises(ValidationError):
		FeeConfig(maintenance_fee=Decimal("-100"))


def test_product_terms_defaults():
	terms = ProductTerms()
	assert terms.min_tenor_days == 0
	assert terms.auto_rollover is False
	assert terms.break_penalty_rate == Decimal("0")
	assert terms.tax_exempt is False


def test_currency_validator(savings_product):
	assert savings_product.currency == "KES"


def test_invalid_currency_rejected():
	from datetime import datetime
	with pytest.raises(ValidationError):
		DepositProduct(
			id="x",
			tenant_id="t",
			code="X",
			name="X",
			product_type=ProductType.SAVINGS,
			currency="kes",  # lowercase — invalid
			interest_config=InterestConfig(rate=Decimal("5")),
			fee_config=FeeConfig(),
			terms=ProductTerms(),
			created_at=datetime.utcnow(),
			updated_at=datetime.utcnow(),
		)


def test_maturity_instruction_enum():
	assert MaturityInstruction.ROLLOVER == "ROLLOVER"
	assert MaturityInstruction.PAYOUT == "PAYOUT"
	assert MaturityInstruction.PARTIAL == "PARTIAL"
