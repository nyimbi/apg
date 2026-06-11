"""Service tests for SACCO GL — no mocks, real objects."""
from __future__ import annotations

import asyncio
from decimal import Decimal
from datetime import date

import pytest

from capabilities.fintech.sacco.gl.service import SACCOGLService


# ── helpers ────────────────────────────────────────────────────────────────────

TODAY = date.today().isoformat()


async def _init(svc: SACCOGLService, tid: str) -> None:
	await svc.initialise_sacco_coa(tid)


# ── COA initialisation ─────────────────────────────────────────────────────────

async def test_init_coa_creates_accounts():
	svc = SACCOGLService()
	result = await svc.initialise_sacco_coa("sacco1")
	assert result["total_accounts"] == 30
	assert "1001" in result["created"]
	assert "5600" in result["created"]


async def test_init_coa_idempotent():
	svc = SACCOGLService()
	await svc.initialise_sacco_coa("sacco1")
	result2 = await svc.initialise_sacco_coa("sacco1")
	assert result2["created"] == []
	assert result2["total_accounts"] == 30


async def test_init_coa_tenant_isolation():
	svc = SACCOGLService()
	await svc.initialise_sacco_coa("sacco_a")
	await svc.initialise_sacco_coa("sacco_b")
	# Each tenant has independent accounts
	result_b = await svc.initialise_sacco_coa("sacco_b")
	assert result_b["skipped"] != []


# ── Double-entry balance validation ───────────────────────────────────────────

async def test_validate_double_entry_balanced():
	svc = SACCOGLService()
	await _init(svc, "t1")
	await svc.post_member_deposit("t1", "M001", "FOSA", Decimal("5000"))
	result = await svc.validate_double_entry("t1")
	assert result["balanced"] is True
	assert Decimal(result["difference"]) == Decimal("0")


async def test_validate_double_entry_after_multiple_postings():
	svc = SACCOGLService()
	await _init(svc, "t1")
	await svc.post_member_deposit("t1", "M001", "BOSA", Decimal("10000"))
	await svc.post_share_purchase("t1", "M001", Decimal("2000"))
	await svc.post_loan_disbursement("t1", "LN001", Decimal("8000"))
	result = await svc.validate_double_entry("t1")
	assert result["balanced"] is True


# ── Member deposit ─────────────────────────────────────────────────────────────

async def test_post_deposit_fosa():
	svc = SACCOGLService()
	await _init(svc, "t1")
	result = await svc.post_member_deposit("t1", "M001", "FOSA", Decimal("3000"), "mpesa")
	assert result["transaction_type"] == "member_deposit"
	assert result["member_id"] == "M001"

	fosa_bal = await svc.get_account_balance("t1", "2100")
	# Credit-normal: balance is positive when account has a credit balance (liability)
	assert fosa_bal == Decimal("3000")

	bank_bal = await svc.get_account_balance("t1", "1010")
	assert bank_bal == Decimal("3000")


async def test_post_deposit_bosa():
	svc = SACCOGLService()
	await _init(svc, "t1")
	await svc.post_member_deposit("t1", "M002", "BOSA", Decimal("5000"), "cash")
	cash_bal = await svc.get_account_balance("t1", "1001")
	assert cash_bal == Decimal("5000")


# ── Loan disbursement ──────────────────────────────────────────────────────────

async def test_loan_disbursement_debits_loan_account():
	svc = SACCOGLService()
	await _init(svc, "t1")
	await svc.post_member_deposit("t1", "M001", "BOSA", Decimal("10000"))
	await svc.post_loan_disbursement("t1", "LN001", Decimal("8000"), "BOSA")
	loan_bal = await svc.get_account_balance("t1", "1110")
	assert loan_bal == Decimal("8000")


# ── Loan repayment ─────────────────────────────────────────────────────────────

async def test_loan_repayment_reduces_loan_balance():
	svc = SACCOGLService()
	await _init(svc, "t1")
	await svc.post_loan_disbursement("t1", "LN001", Decimal("10000"), "BOSA", "cash")
	await svc.post_loan_repayment("t1", "LN001", Decimal("1000"), Decimal("150"))
	loan_bal = await svc.get_account_balance("t1", "1110")
	assert loan_bal == Decimal("9000")
	int_bal = await svc.get_account_balance("t1", "4100")
	assert int_bal == Decimal("150")  # credit-normal income account


async def test_loan_repayment_with_penalty():
	svc = SACCOGLService()
	await _init(svc, "t1")
	await svc.post_loan_disbursement("t1", "LN001", Decimal("5000"), "BOSA", "cash")
	result = await svc.post_loan_repayment("t1", "LN001", Decimal("500"), Decimal("75"), Decimal("25"))
	assert result["transaction_type"] == "loan_repayment"
	penalty_bal = await svc.get_account_balance("t1", "4350")
	assert penalty_bal == Decimal("25")  # credit-normal income


# ── Interest earned ────────────────────────────────────────────────────────────

async def test_post_interest_earned():
	svc = SACCOGLService()
	await _init(svc, "t1")
	await svc.post_interest_earned("t1", "ACC001", Decimal("500"), "2025-01", "BOSA")
	# Expense debited
	int_exp = await svc.get_account_balance("t1", "5100")
	assert int_exp == Decimal("500")
	# Deposit credited (credit-normal liability, balance increases)
	dep_bosa = await svc.get_account_balance("t1", "2110")
	assert dep_bosa == Decimal("500")


# ── Dividends ──────────────────────────────────────────────────────────────────

async def test_post_dividend_declaration():
	svc = SACCOGLService()
	await _init(svc, "t1")
	# Seed retained surplus via income
	await svc.post_transaction(
		"t1", "income_receipt",
		[{"account_code": "1001", "debit": "10000", "credit": "0"},
		 {"account_code": "4100", "credit": "10000", "debit": "0"}],
		"REF001", TODAY, "system"
	)
	# Manually credit retained surplus to set up balance
	await svc.post_transaction(
		"t1", "year_end_transfer",
		[{"account_code": "4100", "debit": "10000", "credit": "0"},
		 {"account_code": "3300", "credit": "10000", "debit": "0"}],
		"YE001", TODAY, "system"
	)
	result = await svc.post_dividend("t1", "M001", Decimal("1000"), 2024)
	assert result["year"] == 2024
	assert result["member_id"] == "M001"


async def test_post_dividend_pay_to_deposits():
	svc = SACCOGLService()
	await _init(svc, "t1")
	result = await svc.post_dividend("t1", "M002", Decimal("500"), 2024, pay_to_deposits=True)
	assert "declaration_id" in result


# ── Share purchase ─────────────────────────────────────────────────────────────

async def test_share_purchase():
	svc = SACCOGLService()
	await _init(svc, "t1")
	await svc.post_share_purchase("t1", "M001", Decimal("2000"), "cash")
	share_cap = await svc.get_account_balance("t1", "3200")
	assert share_cap == Decimal("2000")  # credit-normal equity


# ── Withdrawal ─────────────────────────────────────────────────────────────────

async def test_withdrawal_reduces_deposit():
	svc = SACCOGLService()
	await _init(svc, "t1")
	await svc.post_member_deposit("t1", "M001", "FOSA", Decimal("5000"))
	await svc.post_withdrawal("t1", "M001", Decimal("2000"), "FOSA", "cash")
	fosa_bal = await svc.get_account_balance("t1", "2100")
	assert fosa_bal == Decimal("3000")


# ── Provision & Write-off ──────────────────────────────────────────────────────

async def test_provision_then_write_off():
	svc = SACCOGLService()
	await _init(svc, "t1")
	await svc.post_loan_disbursement("t1", "LN001", Decimal("10000"), "BOSA", "cash")
	await svc.post_provision("t1", "LN001", Decimal("10000"))
	await svc.post_write_off("t1", "LN001", Decimal("10000"), "BOSA")

	loan_bal = await svc.get_account_balance("t1", "1110")
	assert loan_bal == Decimal("0")
	prov_bal = await svc.get_account_balance("t1", "1125")
	assert prov_bal == Decimal("0")  # provision consumed by write-off


# ── Trial balance ──────────────────────────────────────────────────────────────

async def test_trial_balance_structure():
	svc = SACCOGLService()
	await _init(svc, "t1")
	await svc.post_member_deposit("t1", "M001", "FOSA", Decimal("5000"))
	rows = await svc.get_trial_balance("t1", TODAY)
	assert isinstance(rows, list)
	assert len(rows) == 30
	codes = {r["code"] for r in rows}
	assert "1001" in codes
	assert "2100" in codes


async def test_trial_balance_aggregates_correctly():
	svc = SACCOGLService()
	await _init(svc, "t1")
	await svc.post_member_deposit("t1", "M001", "FOSA", Decimal("5000"))
	await svc.post_member_deposit("t1", "M002", "FOSA", Decimal("3000"))
	rows = await svc.get_trial_balance("t1", TODAY)
	# Cash account
	cash_row = next(r for r in rows if r["code"] == "1001")
	assert cash_row["debit"] == Decimal("8000")


# ── Balance sheet ──────────────────────────────────────────────────────────────

async def test_balance_sheet_balanced():
	svc = SACCOGLService()
	await _init(svc, "t1")
	await svc.post_member_deposit("t1", "M001", "BOSA", Decimal("20000"))
	await svc.post_share_purchase("t1", "M001", Decimal("5000"))
	bs = await svc.get_balance_sheet("t1", TODAY)
	assert bs.is_balanced


async def test_balance_sheet_totals():
	svc = SACCOGLService()
	await _init(svc, "t1")
	await svc.post_member_deposit("t1", "M001", "FOSA", Decimal("10000"))
	bs = await svc.get_balance_sheet("t1", TODAY)
	assert bs.total_assets > Decimal("0")
	assert abs(bs.total_assets - bs.total_liabilities_equity) < Decimal("0.01")


# ── Income statement ───────────────────────────────────────────────────────────

async def test_income_statement():
	svc = SACCOGLService()
	await _init(svc, "t1")
	await svc.post_loan_disbursement("t1", "LN001", Decimal("50000"), "BOSA", "cash")
	await svc.post_loan_repayment("t1", "LN001", Decimal("5000"), Decimal("500"))
	stmt = await svc.get_income_statement("t1", "2000-01-01", TODAY)
	assert stmt.total_income > Decimal("0")
	assert stmt.surplus_deficit == stmt.total_income - stmt.total_expenses


# ── GL Summary ─────────────────────────────────────────────────────────────────

async def test_gl_summary():
	svc = SACCOGLService()
	await _init(svc, "t1")
	await svc.post_member_deposit("t1", "M001", "BOSA", Decimal("100000"))
	await svc.post_share_purchase("t1", "M001", Decimal("20000"))
	await svc.post_loan_disbursement("t1", "LN001", Decimal("60000"))
	from datetime import date
	period = date.today().strftime("%Y-%m")
	summary = await svc.get_gl_summary("t1", period)
	assert summary.loan_book_gross == Decimal("60000")
	assert summary.deposit_base > Decimal("0")
	assert summary.journal_entry_count == 3


# ── Period management ──────────────────────────────────────────────────────────

async def test_open_and_close_period():
	svc = SACCOGLService()
	await _init(svc, "t1")
	open_result = await svc.open_period("t1", 2025, 1)
	assert open_result["status"] in ("open", "already_open")
	close_result = await svc.close_period("t1", 2025, 1, "admin")
	assert close_result["status"] == "closed"


async def test_post_to_closed_period_raises():
	svc = SACCOGLService()
	await _init(svc, "t1")
	await svc.open_period("t1", 2024, 12)
	await svc.close_period("t1", 2024, 12, "admin")
	with pytest.raises(ValueError, match="period_closed"):
		await svc.post_member_deposit("t1", "M001", "FOSA", Decimal("1000"), value_date="2024-12-15")


# ── Reconciliation ─────────────────────────────────────────────────────────────

async def test_reconciliation_after_clean_postings():
	svc = SACCOGLService()
	await _init(svc, "t1")
	await svc.post_member_deposit("t1", "M001", "FOSA", Decimal("10000"))
	await svc.post_member_deposit("t1", "M002", "BOSA", Decimal("5000"))
	result = await svc.reconcile_subsidiary_ledgers("t1", TODAY)
	# With perfect subsidiary tracking, should reconcile
	assert result.reconciled is True


# ── Guard tests ────────────────────────────────────────────────────────────────

async def test_guard_empty_tenant_raises():
	svc = SACCOGLService()
	with pytest.raises(PermissionError, match="tenant_id_required"):
		await svc.initialise_sacco_coa("")


async def test_guard_none_tenant_raises():
	svc = SACCOGLService()
	with pytest.raises((PermissionError, TypeError)):
		await svc.initialise_sacco_coa(None)


async def test_missing_account_raises():
	svc = SACCOGLService()
	await _init(svc, "t1")
	with pytest.raises(KeyError, match="account_not_found"):
		await svc.get_account_balance("t1", "9999")


# ── Health check ───────────────────────────────────────────────────────────────

async def test_health_check():
	svc = SACCOGLService()
	result = await svc.health_check()
	assert result["status"] == "healthy"
	assert result["capability"] == "fintech_sacco_gl"


# ── Journal entries query ──────────────────────────────────────────────────────

async def test_journal_entries_filter_by_account():
	svc = SACCOGLService()
	await _init(svc, "t1")
	await svc.post_member_deposit("t1", "M001", "FOSA", Decimal("1000"))
	await svc.post_share_purchase("t1", "M001", Decimal("500"))
	entries = await svc.get_journal_entries("t1", "2000-01-01", TODAY, account_code="2100")
	assert all(any(l["account_code"] == "2100" for l in e["lines"]) for e in entries)


async def test_journal_entries_limit():
	svc = SACCOGLService()
	await _init(svc, "t1")
	for i in range(10):
		await svc.post_member_deposit("t1", f"M{i:03d}", "FOSA", Decimal("100"))
	entries = await svc.get_journal_entries("t1", "2000-01-01", TODAY, limit=5)
	assert len(entries) <= 5


# ── Tenant isolation ───────────────────────────────────────────────────────────

async def test_tenant_isolation():
	svc = SACCOGLService()
	await _init(svc, "sacco_a")
	await _init(svc, "sacco_b")
	await svc.post_member_deposit("sacco_a", "M001", "FOSA", Decimal("9999"))
	bal_b = await svc.get_account_balance("sacco_b", "1001")
	assert bal_b == Decimal("0")
