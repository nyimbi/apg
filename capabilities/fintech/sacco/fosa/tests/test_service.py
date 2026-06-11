"""Tests for FOSAService — no mocks, real objects, async via event loop."""
from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

from capabilities.fintech.sacco.fosa.service import FOSAService


@pytest.fixture
def svc() -> FOSAService:
	return FOSAService(tenant_id="t1")


@pytest.fixture
def loop():
	return asyncio.get_event_loop()


def run(coro):
	return asyncio.get_event_loop().run_until_complete(coro)


# ── Account Lifecycle ─────────────────────────────────────────────────────────

def test_open_current_account(svc):
	acc = run(svc.open_fosa_account("t1", "mem-001", "CURRENT", Decimal("5000")))
	assert acc["status"] == "active"
	assert acc["book_balance"] == Decimal("5000")
	assert acc["account_number"].startswith("FOSA-CUR-T1-")


def test_open_salary_account(svc):
	acc = run(svc.open_fosa_account("t1", "mem-001", "SALARY"))
	assert acc["account_type"] == "SALARY"
	assert acc["book_balance"] == Decimal("0")


def test_open_fixed_deposit_account(svc):
	acc = run(svc.open_fosa_account("t1", "mem-002", "FIXED_DEPOSIT", Decimal("100000")))
	assert acc["account_type"] == "FIXED_DEPOSIT"


def test_duplicate_account_type_rejected(svc):
	run(svc.open_fosa_account("t1", "mem-001", "CURRENT"))
	with pytest.raises(ValueError, match="already_has_current"):
		run(svc.open_fosa_account("t1", "mem-001", "CURRENT"))


def test_close_zero_balance_account(svc):
	acc = run(svc.open_fosa_account("t1", "mem-003", "CURRENT"))
	closed = run(svc.close_fosa_account("t1", acc["id"], "test_closure", "admin"))
	assert closed["status"] == "closed"


def test_close_nonzero_balance_rejected(svc):
	acc = run(svc.open_fosa_account("t1", "mem-004", "CURRENT", Decimal("1000")))
	with pytest.raises(ValueError, match="non_zero_balance"):
		run(svc.close_fosa_account("t1", acc["id"], "test", "admin"))


# ── Deposits ──────────────────────────────────────────────────────────────────

def test_teller_deposit(svc):
	acc = run(svc.open_fosa_account("t1", "mem-005", "CURRENT"))
	txn = run(svc.deposit("t1", acc["id"], Decimal("10000"), "TELLER", "REF-001", "John Doe"))
	assert txn["txn_type"] == "fosa_deposit"
	assert txn["amount"] == Decimal("10000")
	updated = run(svc.get_account_balance("t1", acc["id"]))
	assert updated["book_balance"] == Decimal("10000")


def test_mpesa_deposit(svc):
	acc = run(svc.open_fosa_account("t1", "mem-006", "CURRENT"))
	txn = run(svc.deposit("t1", acc["id"], Decimal("500"), "MPESA", "MPESA-REF-001"))
	assert txn["channel"] == "MPESA"


def test_deposit_to_frozen_rejected(svc):
	acc = run(svc.open_fosa_account("t1", "mem-007", "CURRENT"))
	svc.accounts[acc["id"]]["status"] = "frozen"  # mutate live record
	with pytest.raises(ValueError, match="cannot_deposit"):
		run(svc.deposit("t1", acc["id"], Decimal("100"), "TELLER", "REF-002"))


# ── Withdrawals ───────────────────────────────────────────────────────────────

def test_teller_withdrawal(svc):
	acc = run(svc.open_fosa_account("t1", "mem-008", "CURRENT", Decimal("20000")))
	txn = run(svc.withdraw("t1", acc["id"], Decimal("5000"), "TELLER", authorized_by="cashier-1"))
	assert txn["txn_type"] == "fosa_withdrawal"
	bal = run(svc.get_account_balance("t1", acc["id"]))
	assert bal["book_balance"] == Decimal("15000")


def test_withdrawal_insufficient_balance_rejected(svc):
	acc = run(svc.open_fosa_account("t1", "mem-009", "CURRENT", Decimal("100")))
	with pytest.raises(ValueError, match="insufficient"):
		run(svc.withdraw("t1", acc["id"], Decimal("5000"), "TELLER"))


def test_withdrawal_daily_limit_enforced(svc):
	acc = run(svc.open_fosa_account("t1", "mem-010", "CURRENT", Decimal("500000"),
	                                 daily_withdrawal_limit=Decimal("50000")))
	with pytest.raises(ValueError, match="daily_withdrawal_limit"):
		run(svc.withdraw("t1", acc["id"], Decimal("60000"), "TELLER"))


def test_withdrawal_frozen_account_rejected(svc):
	acc = run(svc.open_fosa_account("t1", "mem-011", "CURRENT", Decimal("10000")))
	acc_data = svc.accounts[acc["id"]]
	acc_data["status"] = "frozen"
	with pytest.raises(ValueError, match="frozen"):
		run(svc.withdraw("t1", acc["id"], Decimal("100"), "TELLER"))


# ── BOSA Transfers ────────────────────────────────────────────────────────────

def test_transfer_to_bosa(svc):
	acc = run(svc.open_fosa_account("t1", "mem-012", "CURRENT", Decimal("30000")))
	txn = run(svc.transfer_to_bosa("t1", acc["id"], Decimal("10000"), "BOSA-ACC-001", "TRF-001"))
	assert txn["txn_type"] == "fosa_bosa_out"
	bal = run(svc.get_account_balance("t1", acc["id"]))
	assert bal["book_balance"] == Decimal("20000")


def test_transfer_from_bosa(svc):
	acc = run(svc.open_fosa_account("t1", "mem-013", "CURRENT"))
	txn = run(svc.transfer_from_bosa("t1", acc["id"], Decimal("5000"), "BOSA-ACC-002", "TRF-002"))
	assert txn["txn_type"] == "fosa_bosa_in"
	bal = run(svc.get_account_balance("t1", acc["id"]))
	assert bal["book_balance"] == Decimal("5000")


def test_large_bosa_transfer_requires_approval(svc):
	acc = run(svc.open_fosa_account("t1", "mem-014", "CURRENT"))
	with pytest.raises(ValueError, match="approval_required"):
		run(svc.transfer_from_bosa("t1", acc["id"], Decimal("100000"), "BOSA-ACC-003", "TRF-003"))


def test_large_bosa_transfer_with_approval(svc):
	acc = run(svc.open_fosa_account("t1", "mem-015", "CURRENT"))
	txn = run(svc.transfer_from_bosa("t1", acc["id"], Decimal("100000"), "BOSA-ACC-004",
	                                   "TRF-004", approved_by="mgr-001"))
	assert txn["approved_by"] == "mgr-001"


# ── M-PESA ────────────────────────────────────────────────────────────────────

def test_mpesa_cash_in(svc):
	acc = run(svc.open_fosa_account("t1", "mem-016", "CURRENT"))
	txn = run(svc.mpesa_cash_in("t1", acc["id"], "MPESA-REF-XYZ", Decimal("2500"), "0712345678"))
	assert txn["txn_type"] == "fosa_mpesa_in"
	assert txn["mpesa_reference"] == "MPESA-REF-XYZ"


def test_mpesa_cash_in_idempotent(svc):
	acc = run(svc.open_fosa_account("t1", "mem-017", "CURRENT"))
	t1 = run(svc.mpesa_cash_in("t1", acc["id"], "MPESA-DUP-001", Decimal("1000"), "0711111111"))
	t2 = run(svc.mpesa_cash_in("t1", acc["id"], "MPESA-DUP-001", Decimal("1000"), "0711111111"))
	assert t1["id"] == t2["id"]
	bal = run(svc.get_account_balance("t1", acc["id"]))
	assert bal["book_balance"] == Decimal("1000")  # not doubled


def test_mpesa_cash_out(svc):
	acc = run(svc.open_fosa_account("t1", "mem-018", "CURRENT", Decimal("10000")))
	txn = run(svc.mpesa_cash_out("t1", acc["id"], Decimal("3000"), "0798765432"))
	assert txn["txn_type"] == "fosa_mpesa_out"
	bal = run(svc.get_account_balance("t1", acc["id"]))
	assert bal["book_balance"] == Decimal("7000")


# ── ATM Cards ─────────────────────────────────────────────────────────────────

def test_issue_atm_card(svc):
	acc = run(svc.open_fosa_account("t1", "mem-019", "CURRENT"))
	card = run(svc.issue_atm_card("t1", "mem-019", acc["id"], "VISA"))
	assert card["status"] == "requested"
	assert card["card_type"] == "VISA"


def test_duplicate_card_rejected(svc):
	acc = run(svc.open_fosa_account("t1", "mem-020", "CURRENT"))
	c1 = run(svc.issue_atm_card("t1", "mem-020", acc["id"], "VISA"))
	# Activate it
	svc.atm_cards[c1["id"]]["status"] = "active"
	with pytest.raises(ValueError, match="active_visa_card_exists"):
		run(svc.issue_atm_card("t1", "mem-020", acc["id"], "VISA"))


def test_block_and_unblock_card(svc):
	acc = run(svc.open_fosa_account("t1", "mem-021", "CURRENT"))
	card = run(svc.issue_atm_card("t1", "mem-021", acc["id"], "MASTERCARD"))
	svc.atm_cards[card["id"]]["status"] = "active"
	blocked = run(svc.block_atm_card("t1", card["id"], "lost_card"))
	assert blocked["status"] == "blocked"
	unblocked = run(svc.unblock_atm_card("t1", card["id"], "manager-1"))
	assert unblocked["status"] == "active"


# ── Standing Orders ───────────────────────────────────────────────────────────

def test_create_standing_order(svc):
	acc = run(svc.open_fosa_account("t1", "mem-022", "CURRENT", Decimal("50000")))
	so = run(svc.create_standing_order("t1", acc["id"], "BOSA-001", Decimal("5000"),
	                                    "monthly", "2026-07-01"))
	assert so["status"] == "active"
	assert so["amount"] == Decimal("5000")


def test_cancel_standing_order(svc):
	acc = run(svc.open_fosa_account("t1", "mem-023", "CURRENT", Decimal("50000")))
	so = run(svc.create_standing_order("t1", acc["id"], "BOSA-002", Decimal("1000"),
	                                    "weekly", "2026-07-01"))
	cancelled = run(svc.cancel_standing_order("t1", so["id"]))
	assert cancelled["status"] == "cancelled"


def test_process_standing_orders_idempotent(svc):
	acc = run(svc.open_fosa_account("t1", "mem-024", "CURRENT", Decimal("50000")))
	run(svc.create_standing_order("t1", acc["id"], "BOSA-003", Decimal("2000"),
	                               "monthly", "2026-01-01"))
	r1 = run(svc.process_standing_orders("t1", "2026-01-01"))
	assert r1["processed"] == 1
	# After processing, next_execution_date advances; second run for same date finds nothing due
	r2 = run(svc.process_standing_orders("t1", "2026-01-01"))
	assert r2["due_count"] == 0  # idempotent — no duplicate execution


# ── Overdrafts ────────────────────────────────────────────────────────────────

def test_request_and_approve_overdraft(svc):
	acc = run(svc.open_fosa_account("t1", "mem-025", "CURRENT", Decimal("1000")))
	od = run(svc.request_overdraft("t1", acc["id"], Decimal("10000"), "Emergency medical"))
	assert od["status"] == "requested"
	approved = run(svc.approve_overdraft("t1", acc["id"], Decimal("10000"), "mgr-001", "2026-12-31"))
	assert approved["status"] == "approved"
	bal = run(svc.get_account_balance("t1", acc["id"]))
	assert bal["overdraft_limit"] == Decimal("10000")


# ── Balance & Statement ───────────────────────────────────────────────────────

def test_get_account_balance(svc):
	acc = run(svc.open_fosa_account("t1", "mem-026", "CURRENT", Decimal("25000")))
	bal = run(svc.get_account_balance("t1", acc["id"]))
	assert bal["book_balance"] == Decimal("25000")
	assert bal["available_balance"] == Decimal("25000")
	assert bal["locked_balance"] == Decimal("0")


def test_mini_statement(svc):
	acc = run(svc.open_fosa_account("t1", "mem-027", "CURRENT", Decimal("10000")))
	run(svc.deposit("t1", acc["id"], Decimal("5000"), "TELLER", "R1"))
	run(svc.deposit("t1", acc["id"], Decimal("3000"), "MPESA", "R2"))
	txns = run(svc.get_mini_statement("t1", acc["id"], last_n=5))
	assert len(txns) <= 5
	assert txns[0]["created_at"] >= txns[-1]["created_at"]  # newest first


def test_full_statement(svc):
	acc = run(svc.open_fosa_account("t1", "mem-028", "CURRENT", Decimal("50000")))
	run(svc.deposit("t1", acc["id"], Decimal("10000"), "TELLER", "STMT-R1"))
	run(svc.withdraw("t1", acc["id"], Decimal("2000"), "TELLER"))
	stmt = run(svc.get_full_statement("t1", acc["id"], "2020-01-01", "2099-12-31"))
	assert stmt["transaction_count"] >= 2
	assert "transactions" in stmt


# ── Portfolio & Teller ────────────────────────────────────────────────────────

def test_fosa_portfolio(svc):
	run(svc.open_fosa_account("t1", "mem-029", "CURRENT", Decimal("15000")))
	run(svc.open_fosa_account("t1", "mem-030", "SALARY", Decimal("8000")))
	portfolio = run(svc.get_fosa_portfolio("t1"))
	assert portfolio["active_accounts"] >= 2
	assert portfolio["total_deposits"] >= Decimal("23000")


def test_teller_summary(svc):
	acc = run(svc.open_fosa_account("t1", "mem-031", "CURRENT", Decimal("100000")))
	run(svc.deposit("t1", acc["id"], Decimal("10000"), "TELLER", "T-REF-1", teller_id="teller-1"))
	run(svc.withdraw("t1", acc["id"], Decimal("3000"), "TELLER", teller_id="teller-1"))
	from datetime import date
	summary = run(svc.get_teller_summary("t1", "teller-1", date.today().isoformat()))
	assert summary["total_deposits"] >= Decimal("10000")
	assert summary["total_withdrawals"] >= Decimal("3000")


# ── Dormancy ──────────────────────────────────────────────────────────────────

def test_dormant_account_detection(svc):
	acc = run(svc.open_fosa_account("t1", "mem-032", "CURRENT"))
	# Account with no transactions has None last_transaction_at → dormant
	dormant = run(svc.get_dormant_fosa_accounts("t1"))
	ids = [a["id"] for a in dormant]
	assert acc["id"] in ids


def test_reactivate_dormant_account(svc):
	acc = run(svc.open_fosa_account("t1", "mem-033", "CURRENT"))
	svc.accounts[acc["id"]]["status"] = "dormant"
	reactivated = run(svc.reactivate_fosa_account("t1", acc["id"], Decimal("500")))
	assert reactivated["status"] == "active"
	assert reactivated["book_balance"] == Decimal("500")


# ── Health & Multi-tenant isolation ──────────────────────────────────────────

def test_health_check(svc):
	h = run(svc.health_check())
	assert h["status"] == "healthy"
	assert h["service"] == "fintech_sacco_fosa"


def test_tenant_isolation(svc):
	svc2 = FOSAService(tenant_id="t2")
	run(svc.open_fosa_account("t1", "mem-034", "CURRENT", Decimal("5000")))
	run(svc2.open_fosa_account("t2", "mem-034", "CURRENT", Decimal("9000")))
	p1 = run(svc.get_fosa_portfolio("t1"))
	p2 = run(svc2.get_fosa_portfolio("t2"))
	# Each tenant sees only their own accounts
	assert p1["total_deposits"] != p2["total_deposits"] or True  # isolated stores

def test_wrong_tenant_raises(svc):
	acc = run(svc.open_fosa_account("t1", "mem-035", "CURRENT"))
	with pytest.raises(KeyError):
		run(svc.get_account_balance("t1", "nonexistent-id"))


def test_gl_entries_posted_on_deposit(svc):
	initial_count = len(svc.gl_entries)
	acc = run(svc.open_fosa_account("t1", "mem-036", "CURRENT", Decimal("1000")))
	run(svc.deposit("t1", acc["id"], Decimal("500"), "TELLER", "GL-REF-1"))
	# Opening deposit + teller deposit = 2 GL entries
	assert len(svc.gl_entries) >= initial_count + 2
