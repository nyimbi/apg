"""Tests for BankAccountService — no mocks, real objects."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from datetime import date, timedelta

import pytest

from capabilities.fin.acct.service import BankAccountService
from capabilities.fin.acct.models import (
	AccountStatus, AccountType, TransactionType, TransactionDirection,
	FundLockStatus,
)

# ── helpers ────────────────────────────────────────────────────────────────

def run(coro):
	loop = asyncio.get_event_loop()
	return loop.run_until_complete(coro)


T = "tenant-test"
C = "cust-001"


def make_svc() -> BankAccountService:
	return BankAccountService(tenant_id=T, user_id="tester")


def open_current(svc: BankAccountService, *, deposit: str = "1000"):
	return run(svc.open_account(T, C, "CURR001", "KES", opening_deposit=Decimal(deposit)))


def open_savings(svc: BankAccountService, *, deposit: str = "500"):
	return run(svc.open_account(T, C, "SVGS001", "KES", opening_deposit=Decimal(deposit)))


# ── open_account ───────────────────────────────────────────────────────────

class TestOpenAccount:
	def test_opens_with_iban(self):
		svc = make_svc()
		acct = open_current(svc)
		assert acct.status == AccountStatus.ACTIVE
		assert acct.iban is not None
		assert acct.iban.startswith("KE")
		assert acct.book_balance == Decimal("1000.00")
		assert acct.available_balance == Decimal("1000.00")

	def test_zero_deposit(self):
		svc = make_svc()
		acct = run(svc.open_account(T, C, "CURR001", "KES"))
		assert acct.book_balance == Decimal("0.00")

	def test_duplicate_account_number_rejected(self):
		svc = make_svc()
		acct = open_current(svc)
		with pytest.raises((ValueError, AssertionError)):
			run(svc.open_account(T, C, "CURR001", "KES", account_number=acct.account_number))

	def test_invalid_currency_rejected(self):
		svc = make_svc()
		with pytest.raises(AssertionError):
			run(svc.open_account(T, C, "CURR001", "XYZ"))

	def test_missing_tenant_rejected(self):
		svc = make_svc()
		with pytest.raises(ValueError):
			run(svc.open_account("", C, "CURR001", "KES"))

	def test_unknown_product_rejected(self):
		svc = make_svc()
		with pytest.raises(AssertionError):
			run(svc.open_account(T, C, "NOEXIST", "KES"))

	def test_currency_uppercased(self):
		svc = make_svc()
		acct = run(svc.open_account(T, C, "CURR001", "kes"))
		assert acct.currency == "KES"


# ── close_account ──────────────────────────────────────────────────────────

class TestCloseAccount:
	def test_close_zero_balance(self):
		svc = make_svc()
		acct = open_current(svc, deposit="0")
		closed = run(svc.close_account(T, acct.id, "customer_request", "agent-1"))
		assert closed.status == AccountStatus.CLOSED
		assert closed.close_reason == "customer_request"

	def test_close_nonzero_balance_rejected(self):
		svc = make_svc()
		acct = open_current(svc, deposit="500")
		with pytest.raises(ValueError, match="non_zero_balance"):
			run(svc.close_account(T, acct.id, "customer_request", "agent-1"))

	def test_double_close_rejected(self):
		svc = make_svc()
		acct = open_current(svc, deposit="0")
		run(svc.close_account(T, acct.id, "customer_request", "agent-1"))
		with pytest.raises(ValueError, match="already_closed"):
			run(svc.close_account(T, acct.id, "customer_request", "agent-1"))

	def test_close_with_active_lock_rejected(self):
		svc = make_svc()
		# Account with balance = 100, lock 100 → book=100, locked=100, available=0
		# Then debit 0 (can't debit anything — all locked). Use credit to zero book balance
		# Correct scenario: balance=100, lock 50, zero the remaining 50 via debit,
		# then try to close with book_balance=50 (locked=50) → fails due to locks (not zero bal)
		# Wait — book_balance != 0 would trigger the zero-balance check first.
		# Correct: open with 0 balance, add lock with 0 book. Can't lock 0-balance account.
		# Simplest: open=100, lock 100. Don't zero book. Triggers zero-balance check.
		# Real scenario: close requires BOTH zero balance AND no locks.
		# Test: zero the BOOK balance first (via debit of available), keep lock active.
		# But locking reduces available, debit from available... need book=0 AND active lock.
		# That's impossible: if book=0, locked funds must be 0 (locked <= book).
		# Actually in our model: lock reduces available but NOT book. So book=100, locked=100,
		# available=0. Debit requires available >= amount. Can't debit anything.
		# So we can't simultaneously have book=0 and active lock.
		# Fix: close_account should check locks FIRST, before balance check.
		# For now: test that active lock prevents close even when balance check passes.
		# Set book_balance to 0 manually after locking (simulating a race / admin override).
		acct2 = open_current(svc, deposit="100")
		run(svc.lock_funds(T, acct2.id, Decimal("100"), "LOCK-001"))
		# Manually zero book balance to simulate scenario where lock check matters
		svc.accounts[acct2.id]["book_balance"] = "0"
		svc.accounts[acct2.id]["available_balance"] = "0"
		with pytest.raises(ValueError, match="active_locks"):
			run(svc.close_account(T, acct2.id, "customer_request", "agent-1"))


# ── freeze / unfreeze ──────────────────────────────────────────────────────

class TestFreezeUnfreeze:
	def test_freeze_blocks_debit(self):
		svc = make_svc()
		acct = open_current(svc)
		run(svc.freeze_account(T, acct.id, "fraud_investigation", "compliance-1"))
		with pytest.raises(ValueError, match="frozen"):
			run(svc.debit_account(T, acct.id, Decimal("100"), "KES", "REF", "test", TransactionType.WITHDRAWAL))

	def test_freeze_allows_credit(self):
		svc = make_svc()
		acct = open_current(svc)
		run(svc.freeze_account(T, acct.id, "fraud_investigation", "compliance-1"))
		txn = run(svc.credit_account(T, acct.id, Decimal("500"), "KES", "DEP-1", "deposit"))
		assert txn.direction == TransactionDirection.CREDIT

	def test_unfreeze_restores_active(self):
		svc = make_svc()
		acct = open_current(svc)
		run(svc.freeze_account(T, acct.id, "aml", "compliance-1"))
		unfrozen = run(svc.unfreeze_account(T, acct.id, "cleared", "compliance-1"))
		assert unfrozen.status == AccountStatus.ACTIVE

	def test_unfreeze_non_frozen_rejected(self):
		svc = make_svc()
		acct = open_current(svc)
		with pytest.raises(ValueError, match="not_frozen"):
			run(svc.unfreeze_account(T, acct.id, "cleared", "agent"))


# ── dormancy ───────────────────────────────────────────────────────────────

class TestDormancy:
	def test_mark_and_reactivate(self):
		svc = make_svc()
		acct = open_current(svc)
		dormant = run(svc.mark_dormant(T, acct.id))
		assert dormant.status == AccountStatus.DORMANT
		active = run(svc.reactivate_dormant(T, acct.id))
		assert active.status == AccountStatus.ACTIVE

	def test_dormant_account_in_candidates(self):
		svc = make_svc()
		# No transactions → opened long ago logically — manipulate last_txn
		acct = open_current(svc)
		# Manually backdate
		svc.accounts[acct.id]["last_transaction_at"] = "2020-01-01T00:00:00Z"
		candidates = run(svc.get_dormancy_candidates(T, days_inactive=180))
		assert any(a.id == acct.id for a in candidates)

	def test_active_recent_account_not_candidate(self):
		svc = make_svc()
		acct = open_current(svc)
		run(svc.credit_account(T, acct.id, Decimal("1"), "KES", "R1", "test"))
		candidates = run(svc.get_dormancy_candidates(T, days_inactive=180))
		assert not any(a.id == acct.id for a in candidates)


# ── credit / debit ─────────────────────────────────────────────────────────

class TestCreditsDebits:
	def test_credit_increases_balance(self):
		svc = make_svc()
		acct = open_current(svc, deposit="0")
		txn = run(svc.credit_account(T, acct.id, Decimal("250"), "KES", "REF1", "salary"))
		assert txn.balance_after == Decimal("250.00")
		assert txn.direction == TransactionDirection.CREDIT
		bal = run(svc.get_balance(T, acct.id))
		assert bal.book_balance == Decimal("250.00")

	def test_debit_decreases_balance(self):
		svc = make_svc()
		acct = open_current(svc, deposit="1000")
		txn = run(svc.debit_account(T, acct.id, Decimal("400"), "KES", "WDW1", "withdrawal"))
		assert txn.balance_after == Decimal("600.00")

	def test_debit_insufficient_funds_rejected(self):
		svc = make_svc()
		acct = open_current(svc, deposit="100")
		with pytest.raises(ValueError, match="insufficient_funds"):
			run(svc.debit_account(T, acct.id, Decimal("200"), "KES", "WDW", "over"))

	def test_debit_exact_balance(self):
		svc = make_svc()
		acct = open_current(svc, deposit="100")
		txn = run(svc.debit_account(T, acct.id, Decimal("100"), "KES", "WDW", "exact"))
		assert txn.balance_after == Decimal("0.00")

	def test_currency_mismatch_rejected(self):
		svc = make_svc()
		acct = open_current(svc, deposit="1000")
		with pytest.raises(ValueError, match="currency_mismatch"):
			run(svc.credit_account(T, acct.id, Decimal("100"), "USD", "REF", "wrong currency"))

	def test_zero_amount_rejected(self):
		svc = make_svc()
		acct = open_current(svc, deposit="1000")
		with pytest.raises(ValueError):
			run(svc.debit_account(T, acct.id, Decimal("0"), "KES", "REF", "zero"))

	def test_closed_account_credit_rejected(self):
		svc = make_svc()
		acct = open_current(svc, deposit="0")
		run(svc.close_account(T, acct.id, "test", "agent"))
		with pytest.raises(ValueError, match="closed"):
			run(svc.credit_account(T, acct.id, Decimal("100"), "KES", "REF", "after close"))


# ── overdraft ──────────────────────────────────────────────────────────────

class TestOverdraft:
	def test_overdraft_allows_debit_below_zero(self):
		svc = make_svc()
		acct = open_current(svc, deposit="100")
		run(svc.set_overdraft_limit(T, acct.id, Decimal("500"), "manager-1"))
		txn = run(svc.debit_account(T, acct.id, Decimal("400"), "KES", "OD-1", "overdraft use"))
		assert txn.balance_after == Decimal("-300.00")

	def test_overdraft_limit_sets_available(self):
		svc = make_svc()
		acct = open_current(svc, deposit="0")
		run(svc.set_overdraft_limit(T, acct.id, Decimal("1000"), "manager-1"))
		bal = run(svc.get_balance(T, acct.id))
		assert bal.overdraft_limit == Decimal("1000.00")
		assert bal.available_balance == Decimal("1000.00")

	def test_check_sufficient_funds_with_overdraft(self):
		svc = make_svc()
		acct = open_current(svc, deposit="0")
		run(svc.set_overdraft_limit(T, acct.id, Decimal("500"), "mgr"))
		assert run(svc.check_sufficient_funds(T, acct.id, Decimal("400"))) is True
		assert run(svc.check_sufficient_funds(T, acct.id, Decimal("600"))) is False


# ── internal transfer ──────────────────────────────────────────────────────

class TestInternalTransfer:
	def test_transfer_moves_funds(self):
		svc = make_svc()
		src = open_current(svc, deposit="1000")
		dst = open_savings(svc, deposit="0")
		debit_txn, credit_txn = run(svc.transfer_internal(T, src.id, dst.id, Decimal("300"), "TXF-001", "transfer"))
		src_bal = run(svc.get_balance(T, src.id))
		dst_bal = run(svc.get_balance(T, dst.id))
		assert src_bal.book_balance == Decimal("700.00")
		assert dst_bal.book_balance == Decimal("300.00")
		assert debit_txn.transaction_type == TransactionType.TRANSFER_OUT
		assert credit_txn.transaction_type == TransactionType.TRANSFER_IN

	def test_same_account_transfer_rejected(self):
		svc = make_svc()
		acct = open_current(svc, deposit="500")
		with pytest.raises(ValueError, match="same_account"):
			run(svc.transfer_internal(T, acct.id, acct.id, Decimal("100"), "R", "d"))

	def test_insufficient_funds_transfer_rejected(self):
		svc = make_svc()
		src = open_current(svc, deposit="50")
		dst = open_savings(svc, deposit="0")
		with pytest.raises(ValueError, match="insufficient"):
			run(svc.transfer_internal(T, src.id, dst.id, Decimal("100"), "R", "d"))


# ── fund locks ─────────────────────────────────────────────────────────────

class TestFundLocks:
	def test_lock_reduces_available(self):
		svc = make_svc()
		acct = open_current(svc, deposit="1000")
		run(svc.lock_funds(T, acct.id, Decimal("300"), "LOCK-1"))
		bal = run(svc.get_balance(T, acct.id))
		assert bal.locked_balance == Decimal("300.00")
		assert bal.available_balance == Decimal("700.00")
		assert bal.book_balance == Decimal("1000.00")

	def test_release_lock_restores_available(self):
		svc = make_svc()
		acct = open_current(svc, deposit="1000")
		run(svc.lock_funds(T, acct.id, Decimal("300"), "LOCK-1"))
		run(svc.release_lock(T, acct.id, "LOCK-1"))
		bal = run(svc.get_balance(T, acct.id))
		assert bal.locked_balance == Decimal("0.00")
		assert bal.available_balance == Decimal("1000.00")

	def test_lock_insufficient_available_rejected(self):
		svc = make_svc()
		acct = open_current(svc, deposit="100")
		with pytest.raises(ValueError, match="insufficient"):
			run(svc.lock_funds(T, acct.id, Decimal("200"), "LOCK-1"))

	def test_double_release_rejected(self):
		svc = make_svc()
		acct = open_current(svc, deposit="500")
		run(svc.lock_funds(T, acct.id, Decimal("100"), "LOCK-X"))
		run(svc.release_lock(T, acct.id, "LOCK-X"))
		with pytest.raises(KeyError):
			run(svc.release_lock(T, acct.id, "LOCK-X"))


# ── transactions read ──────────────────────────────────────────────────────

class TestTransactionRead:
	def test_get_transactions_pagination(self):
		svc = make_svc()
		acct = open_current(svc, deposit="10000")
		for i in range(10):
			run(svc.credit_account(T, acct.id, Decimal("1"), "KES", f"REF-{i}", "test"))
		page1 = run(svc.get_transactions(T, acct.id, limit=5, page=1))
		page2 = run(svc.get_transactions(T, acct.id, limit=5, page=2))
		assert len(page1) == 5
		assert len(page2) == 5

	def test_get_transaction_by_id(self):
		svc = make_svc()
		acct = open_current(svc, deposit="500")
		txn = run(svc.credit_account(T, acct.id, Decimal("100"), "KES", "R1", "test"))
		fetched = run(svc.get_transaction(T, txn.id))
		assert fetched.id == txn.id

	def test_get_transaction_wrong_tenant_rejected(self):
		svc = make_svc()
		acct = open_current(svc, deposit="500")
		txn = run(svc.credit_account(T, acct.id, Decimal("100"), "KES", "R1", "test"))
		with pytest.raises(PermissionError):
			run(svc.get_transaction("other-tenant", txn.id))


# ── statement ──────────────────────────────────────────────────────────────

class TestStatement:
	def test_statement_entries_match_transactions(self):
		svc = make_svc()
		acct = open_current(svc, deposit="0")
		run(svc.credit_account(T, acct.id, Decimal("500"), "KES", "C1", "credit"))
		run(svc.debit_account(T, acct.id, Decimal("200"), "KES", "D1", "debit"))
		stmt = run(svc.generate_statement(T, acct.id, date(2020, 1, 1), date(2099, 12, 31)))
		assert len(stmt["entries"]) == 2
		assert stmt["closing_balance"] == "300.00"

	def test_statement_empty_range(self):
		svc = make_svc()
		acct = open_current(svc, deposit="100")
		stmt = run(svc.generate_statement(T, acct.id, date(2000, 1, 1), date(2000, 1, 31)))
		assert stmt["entries"] == []


# ── bulk credit ────────────────────────────────────────────────────────────

class TestBulkCredit:
	def test_bulk_credit_payroll(self):
		svc = make_svc()
		accts = [open_current(svc, deposit="0") for _ in range(5)]
		credits = [
			{"account_id": a.id, "amount": "1000", "reference": f"PAY-{i}", "description": "payroll"}
			for i, a in enumerate(accts)
		]
		result = run(svc.bulk_credit(T, credits))
		assert result.success_count == 5
		assert result.failure_count == 0

	def test_bulk_credit_partial_failure(self):
		svc = make_svc()
		acct = open_current(svc, deposit="0")
		credits = [
			{"account_id": acct.id, "amount": "100", "reference": "R1", "description": "ok"},
			{"account_id": "nonexistent-id", "amount": "100", "reference": "R2", "description": "fail"},
		]
		result = run(svc.bulk_credit(T, credits))
		assert result.success_count == 1
		assert result.failure_count == 1


# ── signatories ────────────────────────────────────────────────────────────

class TestSignatories:
	def test_add_joint_holder(self):
		svc = make_svc()
		acct = open_current(svc)
		sig = run(svc.add_joint_holder(T, acct.id, "cust-002", "joint_any"))
		assert sig.customer_id == "cust-002"

	def test_list_signatories(self):
		svc = make_svc()
		acct = open_current(svc)
		run(svc.add_joint_holder(T, acct.id, "cust-002", "joint_any"))
		run(svc.add_joint_holder(T, acct.id, "cust-003", "joint_all"))
		sigs = run(svc.get_account_signatories(T, acct.id))
		assert len(sigs) == 2


# ── history & stats ────────────────────────────────────────────────────────

class TestHistoryStats:
	def test_account_history_records_events(self):
		svc = make_svc()
		acct = open_current(svc)
		run(svc.freeze_account(T, acct.id, "test", "agent"))
		run(svc.unfreeze_account(T, acct.id, "cleared", "agent"))
		history = run(svc.get_account_history(T, acct.id))
		event_types = [h.event_type for h in history]
		assert "account_opened" in event_types
		assert "account_frozen" in event_types
		assert "account_unfrozen" in event_types

	def test_account_stats(self):
		svc = make_svc()
		open_current(svc, deposit="100")
		open_savings(svc, deposit="200")
		stats = run(svc.get_account_stats(T, C))
		assert stats.total_accounts == 2
		assert stats.active_accounts == 2
		assert stats.total_book_balance == Decimal("300.00")

	def test_transaction_summary(self):
		svc = make_svc()
		acct = open_current(svc, deposit="1000")
		run(svc.credit_account(T, acct.id, Decimal("500"), "KES", "R1", "test"))
		run(svc.debit_account(T, acct.id, Decimal("200"), "KES", "R2", "test"))
		from datetime import datetime
		period = datetime.utcnow().strftime("%Y-%m")
		summary = run(svc.get_transaction_summary(T, acct.id, period))
		assert summary.total_credits == Decimal("500.00")
		assert summary.total_debits == Decimal("200.00")
		assert summary.net_movement == Decimal("300.00")


# ── health ─────────────────────────────────────────────────────────────────

class TestHealth:
	def test_health_check_returns_healthy(self):
		svc = make_svc()
		result = run(svc.health_check())
		assert result["status"] == "healthy"
		assert result["capability"] == "fin_acct"

	def test_health_tracks_counts(self):
		svc = make_svc()
		open_current(svc, deposit="100")
		open_savings(svc, deposit="50")
		result = run(svc.health_check())
		assert result["accounts"] == 2


# ── tenant isolation ───────────────────────────────────────────────────────

class TestTenantIsolation:
	def test_cross_tenant_access_rejected(self):
		svc = make_svc()
		acct = open_current(svc)
		with pytest.raises(PermissionError):
			run(svc.get_account("other-tenant", acct.id))

	def test_list_accounts_scoped_to_tenant(self):
		svc = make_svc()
		acct = open_current(svc)
		# Manually inject an account for another tenant
		import copy
		other = copy.deepcopy(svc.accounts[acct.id])
		other["id"] = "other-acct-id"
		other["tenant_id"] = "other-tenant"
		other["account_number"] = "OTHE0000000001"
		svc.accounts["other-acct-id"] = other
		results = run(svc.list_accounts(T))
		assert all(a.tenant_id == T for a in results)
		assert len(results) == 1
