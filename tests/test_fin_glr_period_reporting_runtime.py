"""Focused runtime regressions for GLR period/comparative helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from types import SimpleNamespace

import pytest

from capabilities.fin.glr.general_ledger.service import (
	AccountTypeEnum,
	FinancialReportingResult,
	GeneralLedgerService,
	TrialBalanceParams,
)
from capabilities.fin.glr.general_ledger import models


class _Session:
	def __init__(self):
		self.commits = 0

	def commit(self):
		self.commits += 1


@dataclass
class _Account:
	account_id: str
	account_code: str
	account_name: str
	type_code: AccountTypeEnum
	primary_currency: object = field(default_factory=lambda: SimpleNamespace(value="USD"))
	auto_allocation_rules: list[dict] | None = None

	@property
	def account_type(self):
		return SimpleNamespace(type_code=self.type_code)


class _GLService(GeneralLedgerService):
	def __init__(self):
		super().__init__("tenant123", "controller")
		self.session = _Session()
		self.accounts = [
			_Account("cash", "1000", "Cash", AccountTypeEnum.ASSET),
			_Account("payables", "2000", "Payables", AccountTypeEnum.LIABILITY),
			_Account("revenue", "4000", "Revenue", AccountTypeEnum.REVENUE),
			_Account(
				"expense",
				"5000",
				"Expense",
				AccountTypeEnum.EXPENSE,
				auto_allocation_rules=[{
					"name": "allocate support",
					"target_account_id": "expense_alloc",
					"percent": 25,
				}],
			),
		]

	async def _get_active_reporting_accounts(self, account_types):
		return [account for account in self.accounts if account.type_code in account_types]

	async def _get_account_balance(self, account_id, as_of_date):
		balances = {
			"cash": Decimal("125.00"),
			"payables": Decimal("25.00"),
			"revenue": Decimal("0.00"),
			"expense": Decimal("80.00"),
		}
		return balances.get(account_id, Decimal("0.00"))

	async def _get_account_period_activity(self, account_id, date_from, date_to):
		activities = {
			"revenue": Decimal("250.00"),
			"expense": Decimal("80.00"),
		}
		return activities.get(account_id, Decimal("0.00"))

	async def generate_trial_balance(self, params: TrialBalanceParams):
		return FinancialReportingResult(
			report_type="TRIAL_BALANCE",
			as_of_date=params.as_of_date,
			currency="USD",
			data={"totals": {"total_debits": 125.0, "total_credits": 125.0}},
			metadata={"balanced": True},
		)

	async def generate_balance_sheet(self, as_of_date=None, currency=None, comparative_date=None):
		return FinancialReportingResult(
			report_type="BALANCE_SHEET",
			as_of_date=as_of_date,
			currency="USD",
			data={"totals": {"total_assets": 125.0, "total_liabilities": 25.0, "total_equity": 100.0}},
			metadata={"balanced": True},
		)

	async def generate_income_statement(self, date_from, date_to, currency=None, comparative_year=None):
		return FinancialReportingResult(
			report_type="INCOME_STATEMENT",
			as_of_date=date_to,
			currency="USD",
			data={"totals": {"net_income": 170.0, "total_revenue": 250.0, "total_expenses": 80.0}},
			metadata={},
		)


def test_glr_service_imports_with_current_model_aliases():
	assert models.GLJournalEntry is models.CFGLJournalEntry
	assert models.GLJournalLine is models.CFGLJournalLine
	assert models.GLPosting is models.CFGLPosting


@pytest.mark.asyncio
async def test_comparative_balance_and_income_helpers_return_report_data():
	service = _GLService()

	balances = await service._get_comparative_balances(
		[AccountTypeEnum.ASSET, AccountTypeEnum.LIABILITY],
		date(2026, 4, 30),
	)
	income = await service._get_comparative_income_data(
		[AccountTypeEnum.REVENUE, AccountTypeEnum.EXPENSE],
		date(2026, 4, 1),
		date(2026, 4, 30),
	)

	assert balances["totals"] == {"ASSET": 125.0, "LIABILITY": 25.0}
	assert balances["sections"]["ASSET"][0]["account_code"] == "1000"
	assert income["totals"]["REVENUE"] == 250.0
	assert income["totals"]["EXPENSE"] == 80.0
	assert income["totals"]["net_income"] == 170.0


@pytest.mark.asyncio
async def test_period_allocation_and_report_helpers_record_evidence():
	service = _GLService()
	period = SimpleNamespace(
		period_id="period-2026-04",
		period_name="April 2026",
		start_date=date(2026, 4, 1),
		end_date=date(2026, 4, 30),
		closing_checklist=[],
		closing_notes=None,
	)

	await service._run_period_allocations(period)
	await service._generate_period_reports(period)

	allocation_item = period.closing_checklist[0]
	report_item = period.closing_checklist[1]
	assert allocation_item["step"] == "run_period_allocations"
	assert allocation_item["allocations"][0]["amount"] == 20.0
	assert report_item["step"] == "generate_period_reports"
	assert report_item["reports"]["trial_balance"]["balanced"] is True
	assert period.closing_notes is not None
	assert service.session.commits == 2
