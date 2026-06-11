"""Model validation tests for SACCO GL."""
from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from capabilities.fintech.sacco.gl.models import (
	AccountCategory,
	BalanceSheet,
	GLAccount,
	IncomeStatement,
	JournalEntry,
	JournalLine,
	NormalBalance,
	ReconciliationResult,
	SACCOAccountCode,
	STANDARD_COA,
)


def test_standard_coa_count():
	assert len(STANDARD_COA) == 30


def test_standard_coa_has_required_codes():
	codes = {d["code"] for d in STANDARD_COA}
	required = {"1001", "1010", "1100", "1110", "2100", "2110", "3200", "4100", "5100"}
	assert required.issubset(codes)


def test_account_code_enum_values():
	assert SACCOAccountCode.CASH.value == "1001"
	assert SACCOAccountCode.DEPOSITS_FOSA.value == "2100"
	assert SACCOAccountCode.RETAINED_SURPLUS.value == "3300"


def test_journal_line_non_negative():
	with pytest.raises(ValidationError):
		JournalLine(account_code="1001", debit=Decimal("-1"))


def test_journal_line_valid():
	line = JournalLine(account_code="1001", debit=Decimal("100"), narrative="test")
	assert line.debit == Decimal("100")
	assert line.credit == Decimal("0")


def test_gl_account_creation():
	acc = GLAccount(
		tenant_id="t1",
		code="1001",
		name="Cash",
		category=AccountCategory.ASSET,
		normal_balance=NormalBalance.DEBIT,
	)
	assert acc.is_active is True
	assert acc.balance == Decimal("0")
	assert acc.id  # uuid generated


def test_balance_sheet_model():
	bs = BalanceSheet(
		as_of_date="2025-12-31",
		tenant_id="t1",
		assets={"Cash": Decimal("1000")},
		liabilities={"Deposits": Decimal("800")},
		equity={"Share Capital": Decimal("200")},
		total_assets=Decimal("1000"),
		total_liabilities=Decimal("800"),
		total_equity=Decimal("200"),
		total_liabilities_equity=Decimal("1000"),
		is_balanced=True,
	)
	assert bs.is_balanced is True


def test_income_statement_surplus():
	stmt = IncomeStatement(
		from_date="2025-01-01",
		to_date="2025-12-31",
		tenant_id="t1",
		income={"Loan Interest": Decimal("50000")},
		expenses={"Staff Costs": Decimal("20000")},
		total_income=Decimal("50000"),
		total_expenses=Decimal("20000"),
		surplus_deficit=Decimal("30000"),
	)
	assert stmt.surplus_deficit == Decimal("30000")


def test_reconciliation_result():
	r = ReconciliationResult(
		tenant_id="t1",
		as_of_date="2025-06-30",
		reconciled=True,
	)
	assert r.reconciled is True
	assert r.differences == []


def test_journal_entry_forbids_extra():
	with pytest.raises(ValidationError):
		JournalEntry(
			tenant_id="t1",
			reference="REF001",
			transaction_type="deposit",
			value_date="2025-01-01",
			posted_at="2025-01-01T00:00:00Z",
			posted_by="user",
			extra_field="bad",  # should be rejected
		)
