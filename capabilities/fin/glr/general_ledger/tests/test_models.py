"""Unit tests for GLR Pydantic v2 models.

Tests validation, defaults, and cross-field validators.
No mocks — pure model instantiation.
"""
from __future__ import annotations

import pytest
from decimal import Decimal
from datetime import date, datetime

from capabilities.fin.glr.general_ledger.models import (
	AccountType, NormalBalance, PeriodStatus, JournalType, JournalStatus,
	ApprovalStatus, ReconciliationStatus, BudgetType, ClosingType,
	GLBase, GLAccountCreate, GLAccountUpdate, GLAccountResponse,
	GLPeriodCreate, GLPeriodUpdate, GLPeriodResponse,
	GLJournalLineCreate, GLJournalEntryCreate, GLJournalEntryUpdate,
	GLJournalEntryResponse, GLJournalLineResponse,
	GLBudgetCreate, GLBudgetUpdate, GLBudgetResponse,
	GLReconciliationCreate, GLReconciliationSubmit, GLReconciliationItem,
	GLReconciliationResponse,
	GLTrialBalanceRow, GLTrialBalanceResponse,
	GLClosingEntryResponse,
	GLCurrencyRateCreate, GLCurrencyRateResponse,
	GLReportRequest, GLBudgetVsActualRequest, GLYearEndRequest,
	GLPriorYearAdjRequest, GLConsolidationRequest, GLIntercompanyRequest,
	GLListResponse,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

def test_account_type_values():
	assert AccountType.ASSET.value == "asset"
	assert AccountType.LIABILITY.value == "liability"
	assert AccountType.EQUITY.value == "equity"
	assert AccountType.REVENUE.value == "revenue"
	assert AccountType.EXPENSE.value == "expense"
	assert AccountType.CONTRA.value == "contra"


def test_period_status_values():
	assert PeriodStatus.FUTURE.value == "future"
	assert PeriodStatus.OPEN.value == "open"
	assert PeriodStatus.CLOSED.value == "closed"
	assert PeriodStatus.LOCKED.value == "locked"


def test_journal_type_values():
	assert JournalType.STANDARD.value == "standard"
	assert JournalType.REVERSAL.value == "reversal"
	assert JournalType.INTERCOMPANY.value == "intercompany"


# ---------------------------------------------------------------------------
# GLAccountCreate
# ---------------------------------------------------------------------------

def test_account_create_basic():
	acct = GLAccountCreate(
		tenant_id="t1",
		account_code="1000",
		account_name="Cash",
		account_type=AccountType.ASSET,
	)
	assert acct.account_code == "1000"
	assert acct.normal_balance == NormalBalance.DEBIT  # derived
	assert acct.currency == "USD"
	assert acct.allow_posting is True


def test_account_create_normalises_code():
	acct = GLAccountCreate(
		tenant_id="t1",
		account_code="  1-abc  ",
		account_name="Test",
		account_type=AccountType.ASSET,
	)
	assert acct.account_code == "1-ABC"


def test_account_create_invalid_code_raises():
	with pytest.raises(Exception):
		GLAccountCreate(
			tenant_id="t1",
			account_code="!@#$",
			account_name="Bad",
			account_type=AccountType.ASSET,
		)


def test_account_create_credit_normal_for_revenue():
	acct = GLAccountCreate(
		tenant_id="t1",
		account_code="4000",
		account_name="Revenue",
		account_type=AccountType.REVENUE,
	)
	assert acct.normal_balance == NormalBalance.CREDIT


def test_account_create_credit_normal_for_liability():
	acct = GLAccountCreate(
		tenant_id="t1",
		account_code="2000",
		account_name="AP",
		account_type=AccountType.LIABILITY,
	)
	assert acct.normal_balance == NormalBalance.CREDIT


def test_account_create_expense_is_debit_normal():
	acct = GLAccountCreate(
		tenant_id="t1",
		account_code="6000",
		account_name="Rent",
		account_type=AccountType.EXPENSE,
	)
	assert acct.normal_balance == NormalBalance.DEBIT


def test_account_update_partial():
	upd = GLAccountUpdate(account_name="Cash and Equivalents")
	assert upd.account_name == "Cash and Equivalents"
	assert upd.allow_posting is None


def test_account_update_rejects_extra_fields():
	with pytest.raises(Exception):
		GLAccountUpdate(account_name="X", unknown_field="bad")


# ---------------------------------------------------------------------------
# GLPeriodCreate
# ---------------------------------------------------------------------------

def test_period_create_valid():
	p = GLPeriodCreate(
		tenant_id="t1",
		period_code="2026-01",
		fiscal_year=2026,
		period_number=1,
		start_date=date(2026, 1, 1),
		end_date=date(2026, 1, 31),
	)
	assert p.period_code == "2026-01"
	assert p.fiscal_year == 2026


def test_period_create_invalid_date_order_raises():
	with pytest.raises(Exception):
		GLPeriodCreate(
			tenant_id="t1",
			period_code="2026-01",
			fiscal_year=2026,
			period_number=1,
			start_date=date(2026, 1, 31),
			end_date=date(2026, 1, 1),
		)


def test_period_create_same_day_ok():
	p = GLPeriodCreate(
		tenant_id="t1",
		period_code="2026-ADJ",
		fiscal_year=2026,
		period_number=13,
		start_date=date(2026, 12, 31),
		end_date=date(2026, 12, 31),
		allows_adjustments=True,
	)
	assert p.allows_adjustments is True


# ---------------------------------------------------------------------------
# GLJournalLineCreate
# ---------------------------------------------------------------------------

def test_journal_line_debit_only():
	line = GLJournalLineCreate(account_id="acc-1", debit=Decimal("1000"))
	assert line.debit == Decimal("1000")
	assert line.credit == Decimal("0")


def test_journal_line_credit_only():
	line = GLJournalLineCreate(account_id="acc-1", credit=Decimal("500"))
	assert line.credit == Decimal("500")


def test_journal_line_both_sides_raises():
	with pytest.raises(Exception):
		GLJournalLineCreate(account_id="acc-1", debit=Decimal("100"), credit=Decimal("100"))


def test_journal_line_negative_raises():
	with pytest.raises(Exception):
		GLJournalLineCreate(account_id="acc-1", debit=Decimal("-50"))


# ---------------------------------------------------------------------------
# GLJournalEntryCreate
# ---------------------------------------------------------------------------

def _make_balanced_entry(**kwargs) -> GLJournalEntryCreate:
	defaults = dict(
		tenant_id="t1",
		journal_date=date(2026, 1, 15),
		description="Test entry",
		lines=[
			GLJournalLineCreate(account_id="acc-1", debit=Decimal("1000")),
			GLJournalLineCreate(account_id="acc-2", credit=Decimal("1000")),
		],
	)
	defaults.update(kwargs)
	return GLJournalEntryCreate(**defaults)


def test_journal_entry_create_balanced():
	entry = _make_balanced_entry()
	assert entry.description == "Test entry"
	assert len(entry.lines) == 2


def test_journal_entry_create_unbalanced_raises():
	with pytest.raises(Exception, match="journal_not_balanced"):
		GLJournalEntryCreate(
			tenant_id="t1",
			journal_date=date(2026, 1, 15),
			description="Bad",
			lines=[
				GLJournalLineCreate(account_id="acc-1", debit=Decimal("1000")),
				GLJournalLineCreate(account_id="acc-2", credit=Decimal("900")),
			],
		)


def test_journal_entry_create_single_line_raises():
	with pytest.raises(Exception):
		GLJournalEntryCreate(
			tenant_id="t1",
			journal_date=date(2026, 1, 15),
			description="Bad",
			lines=[GLJournalLineCreate(account_id="acc-1", debit=Decimal("100"), credit=Decimal("0"))],
		)


def test_journal_entry_create_zero_total_raises():
	with pytest.raises(Exception):
		GLJournalEntryCreate(
			tenant_id="t1",
			journal_date=date(2026, 1, 15),
			description="Zero",
			lines=[
				GLJournalLineCreate(account_id="acc-1"),
				GLJournalLineCreate(account_id="acc-2"),
			],
		)


# ---------------------------------------------------------------------------
# GLBudgetCreate
# ---------------------------------------------------------------------------

def test_budget_create():
	b = GLBudgetCreate(
		tenant_id="t1",
		budget_code="BUD-2026-01",
		fiscal_year=2026,
		account_code="6000",
		period_code="2026-01",
		amount=Decimal("50000"),
	)
	assert b.budget_type == BudgetType.ORIGINAL
	assert b.currency == "USD"


# ---------------------------------------------------------------------------
# GLReconciliationSubmit
# ---------------------------------------------------------------------------

def test_reconciliation_submit():
	sub = GLReconciliationSubmit(
		reconciled_by="controller",
		reconciling_items=[
			GLReconciliationItem(
				description="Outstanding cheque",
				amount=Decimal("1500"),
				item_type="outstanding_cheque",
			)
		],
		balance_per_statement=Decimal("98500"),
	)
	assert len(sub.reconciling_items) == 1
	assert sub.reconciling_items[0].amount == Decimal("1500")


# ---------------------------------------------------------------------------
# GLCurrencyRateCreate
# ---------------------------------------------------------------------------

def test_currency_rate_create():
	r = GLCurrencyRateCreate(
		tenant_id="t1",
		from_currency="USD",
		to_currency="KES",
		effective_date=date(2026, 1, 1),
		exchange_rate=Decimal("130.50"),
	)
	assert r.exchange_rate == Decimal("130.50")


def test_currency_rate_zero_raises():
	with pytest.raises(Exception):
		GLCurrencyRateCreate(
			tenant_id="t1",
			from_currency="USD",
			to_currency="KES",
			effective_date=date(2026, 1, 1),
			exchange_rate=Decimal("0"),
		)


# ---------------------------------------------------------------------------
# GLTrialBalanceResponse
# ---------------------------------------------------------------------------

def test_trial_balance_response():
	tb = GLTrialBalanceResponse(
		tenant_id="t1",
		period_code="2026-01",
		rows=[
			GLTrialBalanceRow(
				account_code="1000",
				account_name="Cash",
				account_type="asset",
				opening_balance=Decimal("0"),
				period_debit=Decimal("1000"),
				period_credit=Decimal("0"),
				closing_debit=Decimal("1000"),
				closing_credit=Decimal("0"),
			)
		],
		total_closing_debit=Decimal("1000"),
		total_closing_credit=Decimal("1000"),
		balanced=True,
	)
	assert tb.balanced is True
	assert len(tb.rows) == 1


# ---------------------------------------------------------------------------
# GLYearEndRequest
# ---------------------------------------------------------------------------

def test_year_end_request():
	req = GLYearEndRequest(
		tenant_id="t1",
		fiscal_year=2026,
		retained_earnings_account="3100",
	)
	assert req.fiscal_year == 2026
	assert req.executed_by == "system"


# ---------------------------------------------------------------------------
# GLListResponse
# ---------------------------------------------------------------------------

def test_list_response_defaults():
	resp = GLListResponse(items=[{"id": "x"}], total=1)
	assert resp.page == 1
	assert resp.page_size == 50
