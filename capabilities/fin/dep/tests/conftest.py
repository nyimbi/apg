"""Shared fixtures for fin.dep tests."""
from __future__ import annotations

import sys
import os
from datetime import date
from decimal import Decimal

import pytest

# Ensure the capability is importable from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))

from capabilities.fin.dep.service import DepositProductsService
from capabilities.fin.dep.models import (
	CompoundingFrequency, FeeConfig, FeeFrequency, InterestCalculationType,
	InterestConfig, InterestTier, MaturityInstruction, ProductTerms, ProductType,
)


@pytest.fixture
def svc() -> DepositProductsService:
	return DepositProductsService()


@pytest.fixture
def tenant() -> str:
	return "bank_test"


@pytest.fixture
def savings_product(svc: DepositProductsService, tenant: str):
	return svc.create_product(
		tenant_id=tenant,
		code="SAV001",
		name="Classic Savings",
		product_type=ProductType.SAVINGS,
		currency="KES",
		interest_config=InterestConfig(
			rate=Decimal("5.5"),
			calculation=InterestCalculationType.DAILY_ACCRUAL,
			compounding=CompoundingFrequency.MONTHLY,
			withholding_rate=Decimal("15"),
		),
		fee_config=FeeConfig(
			maintenance_fee=Decimal("200"),
			fee_frequency=FeeFrequency.MONTHLY,
			minimum_balance=Decimal("1000"),
			below_minimum_fee=Decimal("50"),
		),
		terms=ProductTerms(min_opening_amount=Decimal("500")),
		gl_interest_income_account="4001-INT-INCOME",
		gl_wht_payable_account="2001-WHT-PAYABLE",
	)


@pytest.fixture
def term_product(svc: DepositProductsService, tenant: str):
	return svc.create_product(
		tenant_id=tenant,
		code="TD001",
		name="90-Day Fixed Deposit",
		product_type=ProductType.TERM_DEPOSIT,
		currency="KES",
		interest_config=InterestConfig(
			rate=Decimal("9.0"),
			calculation=InterestCalculationType.SIMPLE,
			compounding=CompoundingFrequency.ANNUALLY,
			withholding_rate=Decimal("15"),
		),
		fee_config=FeeConfig(),
		terms=ProductTerms(
			min_tenor_days=90,
			max_tenor_days=365,
			break_penalty_rate=Decimal("50"),
			min_opening_amount=Decimal("50000"),
		),
	)


@pytest.fixture
def tiered_product(svc: DepositProductsService, tenant: str):
	return svc.create_product(
		tenant_id=tenant,
		code="TIER001",
		name="Tiered Savings",
		product_type=ProductType.SAVINGS,
		currency="USD",
		interest_config=InterestConfig(
			rate=Decimal("2.0"),
			calculation=InterestCalculationType.DAILY_ACCRUAL,
			compounding=CompoundingFrequency.MONTHLY,
			withholding_rate=Decimal("0"),
			tiers=[
				InterestTier(min_balance=Decimal("0"),       rate=Decimal("2.0")),
				InterestTier(min_balance=Decimal("10000"),   rate=Decimal("3.5")),
				InterestTier(min_balance=Decimal("100000"),  rate=Decimal("5.0")),
			],
		),
		fee_config=FeeConfig(),
		terms=ProductTerms(min_opening_amount=Decimal("100")),
	)


@pytest.fixture
def savings_account(svc: DepositProductsService, tenant: str, savings_product):
	return svc.register_account(
		tenant_id=tenant,
		account_id="ACC-SAV-001",
		product_code="SAV001",
		balance=Decimal("50000"),
		opening_date=date(2025, 1, 1),
	)


@pytest.fixture
def term_account(svc: DepositProductsService, tenant: str, term_product):
	return svc.register_account(
		tenant_id=tenant,
		account_id="ACC-TD-001",
		product_code="TD001",
		balance=Decimal("100000"),
		opening_date=date(2025, 1, 1),
		maturity_date=date(2025, 4, 1),
	)
