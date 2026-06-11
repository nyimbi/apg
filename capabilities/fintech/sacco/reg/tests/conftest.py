"""Pytest fixtures for SASRA Regulatory Reporting tests."""
from __future__ import annotations

import asyncio
import pytest
from decimal import Decimal

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))

from capabilities.fintech.sacco.reg.service import SACCARegulatoryService


TENANT = "test-sacco"


def _compliant_snapshot() -> dict:
	"""A financially healthy SACCO — all SASRA ratios pass.

	Key metrics:
	  LDR: 60M loans / 100M deposits = 60% (max 70%)
	  Liquidity: 37M liquid / 115M (deposits+borrowings) = 32% (min 15%)
	  CAR: 30M institutional / ~65M RWA = ~46% (min 10%)
	  NPL: 4M / 60M = 6.7% — below 10% breach, and provisioning >= required
	"""
	return {
		# Balance sheet
		"cash_on_hand": 5_000_000,
		"bank_balances": 20_000_000,
		"government_securities": 10_000_000,
		"other_liquid_assets": 2_000_000,
		"gross_loan_portfolio": 60_000_000,   # 60% LDR against 100M deposits
		"loan_loss_provisions": 4_000_000,    # covers required provisions
		"other_investments": 5_000_000,
		"fixed_assets": 8_000_000,
		"other_assets": 3_000_000,
		"member_deposits": 100_000_000,
		"external_borrowings": 15_000_000,
		"other_liabilities": 5_000_000,
		"share_capital": 20_000_000,
		"retained_earnings": 5_000_000,
		"statutory_reserve": 3_000_000,
		"other_reserves": 2_000_000,
		# Capital
		"core_capital": 25_000_000,
		"secondary_capital": 5_000_000,
		"total_assets": 113_000_000,
		# Income
		"interest_income_loans": 10_000_000,
		"interest_income_investments": 1_500_000,
		"fee_income": 500_000,
		"other_income": 200_000,
		"interest_expense_deposits": 3_000_000,
		"interest_expense_borrowings": 500_000,
		"provision_expense": 1_000_000,
		"staff_costs": 2_000_000,
		"administrative_expenses": 1_000_000,
		"other_expenses": 300_000,
		# Loan books: NPL = 1.5M+1.5M+1M = 4M / 60M = 6.7% (below 10% breach)
		# Required provisions: 0 + 300 + 375k + 750k + 1M = ~2.13M; actual=4M > required
		"loan_books": [
			{"outstanding_balance": 54_500_000, "days_past_due": 0},    # normal 0%
			{"outstanding_balance": 2_000_000,  "days_past_due": 60},   # watch 1%
			{"outstanding_balance": 1_500_000,  "days_past_due": 120},  # substandard 25%
			{"outstanding_balance": 1_000_000,  "days_past_due": 200},  # doubtful 50%
			{"outstanding_balance": 1_000_000,  "days_past_due": 400},  # loss 100%
		],
		"sacco_name": "Test SACCO Ltd",
		"registration_number": "CS/SACCO/12345",
	}


def _breaching_snapshot() -> dict:
	"""A distressed SACCO — all SASRA ratios breach."""
	return {
		"cash_on_hand": 500_000,
		"bank_balances": 1_000_000,
		"government_securities": 0,
		"other_liquid_assets": 0,
		"gross_loan_portfolio": 95_000_000,
		"loan_loss_provisions": 500_000,
		"other_investments": 0,
		"fixed_assets": 5_000_000,
		"other_assets": 0,
		"member_deposits": 90_000_000,
		"external_borrowings": 10_000_000,
		"other_liabilities": 2_000_000,
		"share_capital": 3_000_000,
		"retained_earnings": -2_000_000,
		"statutory_reserve": 0,
		"other_reserves": 0,
		"core_capital": 1_000_000,
		"secondary_capital": 500_000,
		"total_assets": 102_000_000,
		"interest_income_loans": 5_000_000,
		"interest_income_investments": 0,
		"fee_income": 100_000,
		"other_income": 0,
		"interest_expense_deposits": 4_000_000,
		"interest_expense_borrowings": 600_000,
		"provision_expense": 2_000_000,
		"staff_costs": 3_000_000,
		"administrative_expenses": 1_500_000,
		"other_expenses": 500_000,
		"loan_books": [
			{"outstanding_balance": 30_000_000, "days_past_due": 0},
			{"outstanding_balance": 15_000_000, "days_past_due": 60},
			{"outstanding_balance": 20_000_000, "days_past_due": 150},
			{"outstanding_balance": 20_000_000, "days_past_due": 250},
			{"outstanding_balance": 10_000_000, "days_past_due": 500},
		],
		"sacco_name": "Distressed SACCO Ltd",
		"registration_number": "CS/SACCO/99999",
	}


@pytest.fixture
def svc() -> SACCARegulatoryService:
	s = SACCARegulatoryService(TENANT)
	s.seed_ledger(TENANT, "2025-03-31", _compliant_snapshot())
	s.seed_ledger(TENANT, "2025-06-30", _compliant_snapshot())
	s.seed_ledger(TENANT, "2025-09-30", _compliant_snapshot())
	s.seed_ledger(TENANT, "2025-12-31", _compliant_snapshot())
	return s


@pytest.fixture
def svc_breach() -> SACCARegulatoryService:
	s = SACCARegulatoryService(TENANT)
	s.seed_ledger(TENANT, "2025-12-31", _breaching_snapshot())
	s.seed_ledger(TENANT, "2025-03-31", _breaching_snapshot())
	return s
