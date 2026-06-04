"""Comprehensive async service tests for GeneralLedgerService.

Tests cover full lifecycle: accounts, periods, journals, reconciliation,
financial statements, year-end close, consolidation, XBRL.

No mocks — uses real in-memory service instances.
Plain async functions using asyncio.run().
"""
from __future__ import annotations

import asyncio
import pytest
from decimal import Decimal

from capabilities.fin.glr.general_ledger.service import GeneralLedgerService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro):
	return asyncio.run(coro)


def _svc(tenant: str = "tenant-test") -> GeneralLedgerService:
	return GeneralLedgerService(tenant_id=tenant, user_id="test-user")


def _add_account(svc, tenant, code, name, atype):
	"""Add account and optionally tag retained_earnings accounts."""
	record = svc.create_account(code, tenant, code, name, atype)
	if code == "3100":
		record["tags"] = ["retained_earnings"]
		svc.accounts[record["id"]]["tags"] = ["retained_earnings"]
	return record


def _funded_svc(tenant: str = "tenant-test") -> GeneralLedgerService:
	"""Service pre-loaded with accounts and an open period."""
	svc = _svc(tenant)
	_add_account(svc, tenant, "1000", "Cash", "asset")
	_add_account(svc, tenant, "1100", "Accounts Receivable", "asset")
	_add_account(svc, tenant, "2000", "Accounts Payable", "liability")
	_add_account(svc, tenant, "3000", "Share Capital", "equity")
	_add_account(svc, tenant, "3100", "Retained Earnings", "equity")
	_add_account(svc, tenant, "4000", "Revenue", "revenue")
	_add_account(svc, tenant, "6000", "Salaries", "expense")
	svc.open_period("p1", tenant, "2026-01", 2026, "2026-01-01", "2026-01-31")
	return svc


def _account_id(svc: GeneralLedgerService, tenant: str, code: str) -> str:
	for acct in svc.accounts.values():
		if acct["tenant_id"] == tenant and acct["code"] == code:
			return acct["id"]
	raise KeyError(f"account code {code} not found for tenant {tenant}")


def _period_id(svc: GeneralLedgerService, tenant: str, code: str = "2026-01") -> str:
	for p in svc.periods.values():
		if p["tenant_id"] == tenant and p.get("period_code") == code:
			return p["id"]
	raise KeyError(f"period {code} not found for tenant {tenant}")


def _post_revenue_and_expense(svc, tenant, date_str="2026-01-15"):
	cash_id = _account_id(svc, tenant, "1000")
	rev_id = _account_id(svc, tenant, "4000")
	_run(svc.post_journal_v2(
		tenant_id=tenant, journal_date=date_str, journal_type="standard",
		lines=[{"account_id": cash_id, "debit": "5000", "credit": "0"},
		       {"account_id": rev_id, "debit": "0", "credit": "5000"}],
		description="Revenue", reference="", posted_by="poster",
	))
	exp_id = _account_id(svc, tenant, "6000")
	ap_id = _account_id(svc, tenant, "2000")
	_run(svc.post_journal_v2(
		tenant_id=tenant, journal_date=date_str, journal_type="standard",
		lines=[{"account_id": exp_id, "debit": "2000", "credit": "0"},
		       {"account_id": ap_id, "debit": "0", "credit": "2000"}],
		description="Salaries", reference="", posted_by="poster",
	))


# ---------------------------------------------------------------------------
# Account management
# ---------------------------------------------------------------------------

def test_create_account_basic():
	svc = _svc()
	acct = svc.create_account("cash", "tenant-test", "1000", "Cash", "asset")
	assert acct["code"] == "1000"
	assert acct["account_type"] == "asset"
	assert acct["status"] == "active"
	assert acct["tenant_id"] == "tenant-test"


def test_create_account_emits_event():
	svc = _svc()
	svc.create_account("cash", "tenant-test", "1000", "Cash", "asset")
	events = svc.audit_events("tenant-test")
	assert any(e["event_type"] == "account_created" for e in events)


def test_create_account_missing_tenant_raises():
	svc = GeneralLedgerService()  # no default tenant
	with pytest.raises(PermissionError, match="tenant_context_required"):
		svc.create_account("cash", "", "1000", "Cash", "asset")


def test_create_account_unsupported_type_raises():
	svc = _svc()
	with pytest.raises(PermissionError):
		svc.create_account("bad", "tenant-test", "9000", "Bad", "nonsense")


def test_chart_of_accounts_async():
	svc = _funded_svc()
	accounts = _run(svc.chart_of_accounts("tenant-test"))
	assert len(accounts) >= 7
	codes = {a["code"] for a in accounts}
	assert "1000" in codes
	assert "4000" in codes


def test_account_hierarchy():
	svc = _funded_svc()
	hierarchy = _run(svc.account_hierarchy("tenant-test"))
	assert isinstance(hierarchy, dict)
	assert "tree" in hierarchy


def test_create_account_v2():
	svc = _svc()
	result = _run(svc.create_account_v2(
		tenant_id="tenant-test",
		account_code="7000",
		account_name="Depreciation",
		account_type="expense",
		parent_code=None,
		currency="USD",
	))
	assert result["code"] == "7000"


def test_account_analysis():
	svc = _funded_svc()
	batch = svc.create_journal_batch("b1", "tenant-test", _period_id(svc, "tenant-test"), "manual")
	cash_id = _account_id(svc, "tenant-test", "1000")
	rev_id = _account_id(svc, "tenant-test", "4000")
	j = svc.create_journal_entry("j1", "tenant-test", batch["id"], "Rev", [
		{"account_id": cash_id, "debit": 500, "credit": 0},
		{"account_id": rev_id, "debit": 0, "credit": 500},
	])
	svc.approve_journal(j["id"], "tenant-test", "approver")
	svc.post_journal(j["id"], "tenant-test", "poster", "idem-analysis")
	result = _run(svc.account_analysis("tenant-test", "1000", "2026-01"))
	assert "account_code" in result or "balance" in result


# ---------------------------------------------------------------------------
# Period lifecycle
# ---------------------------------------------------------------------------

def test_open_period_v2():
	svc = _svc()
	svc.open_period("p1", "tenant-test", "2026-01", 2026, "2026-01-01", "2026-01-31")
	period = svc._period_by_code("tenant-test", "2026-01")
	period["status"] = "closed"
	result = _run(svc.open_period_v2("tenant-test", "2026-01", "user1"))
	assert result["status"] == "open"


def test_close_period():
	svc = _funded_svc()
	result = _run(svc.close_period("tenant-test", "2026-01", "controller"))
	assert result["status"] == "closed"


def test_close_period_with_unposted_journal_raises():
	svc = _funded_svc()
	batch = svc.create_journal_batch("b1", "tenant-test", _period_id(svc, "tenant-test"), "manual")
	cash_id = _account_id(svc, "tenant-test", "1000")
	rev_id = _account_id(svc, "tenant-test", "4000")
	svc.create_journal_entry("j1", "tenant-test", batch["id"], "Pending", [
		{"account_id": cash_id, "debit": 100, "credit": 0},
		{"account_id": rev_id, "debit": 0, "credit": 100},
	])
	with pytest.raises(ValueError, match="period_close_blocked"):
		_run(svc.close_period("tenant-test", "2026-01", "controller"))


def test_lock_period():
	svc = _funded_svc()
	_run(svc.close_period("tenant-test", "2026-01", "controller"))
	result = _run(svc.lock_period("tenant-test", "2026-01", "cfo"))
	assert result["status"] == "locked"


def test_lock_period_requires_closed_first():
	svc = _funded_svc()
	with pytest.raises(ValueError, match="period_must_be_closed_before_locking"):
		_run(svc.lock_period("tenant-test", "2026-01", "cfo"))


def test_reopen_period():
	svc = _funded_svc()
	_run(svc.close_period("tenant-test", "2026-01", "controller"))
	result = _run(svc.reopen_period("tenant-test", "2026-01", "Correction needed", "cfo"))
	assert result["status"] == "open"


def test_reopen_locked_period_raises():
	svc = _funded_svc()
	_run(svc.close_period("tenant-test", "2026-01", "controller"))
	_run(svc.lock_period("tenant-test", "2026-01", "cfo"))
	with pytest.raises(PermissionError, match="locked"):
		_run(svc.reopen_period("tenant-test", "2026-01", "Try to reopen", "cfo"))


def test_period_end_checklist():
	svc = _funded_svc()
	result = _run(svc.period_end_checklist("tenant-test", "2026-01"))
	assert "items" in result
	assert "ready_to_close" in result


def test_get_period_status():
	svc = _funded_svc()
	periods = _run(svc.get_period_status("tenant-test", 2026))
	assert len(periods) >= 1
	assert periods[0].get("fiscal_year") == 2026


# ---------------------------------------------------------------------------
# Journal entry lifecycle
# ---------------------------------------------------------------------------

def test_post_journal_v2():
	svc = _funded_svc()
	cash_id = _account_id(svc, "tenant-test", "1000")
	rev_id = _account_id(svc, "tenant-test", "4000")
	posting = _run(svc.post_journal_v2(
		tenant_id="tenant-test", journal_date="2026-01-15", journal_type="standard",
		lines=[{"account_id": cash_id, "debit": "1000", "credit": "0"},
		       {"account_id": rev_id, "debit": "0", "credit": "1000"}],
		description="Sales revenue", reference="INV-001", posted_by="poster",
	))
	assert posting["status"] == "posted"


def test_post_journal_v2_unbalanced_raises():
	svc = _funded_svc()
	cash_id = _account_id(svc, "tenant-test", "1000")
	rev_id = _account_id(svc, "tenant-test", "4000")
	with pytest.raises(ValueError, match="journal_not_balanced"):
		_run(svc.post_journal_v2(
			tenant_id="tenant-test", journal_date="2026-01-15", journal_type="standard",
			lines=[{"account_id": cash_id, "debit": "1000", "credit": "0"},
			       {"account_id": rev_id, "debit": "0", "credit": "900"}],
			description="Bad", reference="", posted_by="poster",
		))


def test_post_journal_v2_no_open_period_raises():
	svc = _svc()
	svc.create_account("cash", "tenant-test", "1000", "Cash", "asset")
	svc.create_account("rev", "tenant-test", "4000", "Revenue", "revenue")
	cash_id = _account_id(svc, "tenant-test", "1000")
	rev_id = _account_id(svc, "tenant-test", "4000")
	with pytest.raises(ValueError, match="no_open_period_for_date"):
		_run(svc.post_journal_v2(
			tenant_id="tenant-test", journal_date="2026-01-15", journal_type="standard",
			lines=[{"account_id": cash_id, "debit": "500", "credit": "0"},
			       {"account_id": rev_id, "debit": "0", "credit": "500"}],
			description="No period", reference="", posted_by="poster",
		))


def test_reverse_journal_v2():
	svc = _funded_svc()
	cash_id = _account_id(svc, "tenant-test", "1000")
	rev_id = _account_id(svc, "tenant-test", "4000")
	posting = _run(svc.post_journal_v2(
		tenant_id="tenant-test", journal_date="2026-01-15", journal_type="standard",
		lines=[{"account_id": cash_id, "debit": "500", "credit": "0"},
		       {"account_id": rev_id, "debit": "0", "credit": "500"}],
		description="Orig", reference="R1", posted_by="poster",
	))
	reversal = _run(svc.reverse_journal_v2(
		tenant_id="tenant-test", journal_id=posting["journal_id"],
		reversal_date="2026-01-20", reversal_description="Reversal",
		reversed_by="controller",
	))
	assert reversal["status"] == "reversed"
	orig = svc.journal_entries[posting["journal_id"]]
	assert orig["status"] == "reversed"


def test_auto_reverse_schedule():
	svc = _funded_svc()
	cash_id = _account_id(svc, "tenant-test", "1000")
	rev_id = _account_id(svc, "tenant-test", "4000")
	posting = _run(svc.post_journal_v2(
		tenant_id="tenant-test", journal_date="2026-01-15", journal_type="accrual",
		lines=[{"account_id": cash_id, "debit": "200", "credit": "0"},
		       {"account_id": rev_id, "debit": "0", "credit": "200"}],
		description="Accrual", reference="", posted_by="poster",
	))
	sched = _run(svc.auto_reverse_on_date("tenant-test", posting["journal_id"], "2026-02-01"))
	assert sched["status"] == "scheduled"


def test_validate_journal_balance():
	svc = _svc()
	lines_ok = [{"debit": "500", "credit": "0"}, {"debit": "0", "credit": "500"}]
	assert _run(svc.validate_journal_balance(lines_ok)) is True
	lines_bad = [{"debit": "500", "credit": "0"}, {"debit": "0", "credit": "400"}]
	assert _run(svc.validate_journal_balance(lines_bad)) is False


def test_journal_approval_workflow_below_threshold():
	svc = _funded_svc()
	batch = svc.create_journal_batch("b1", "tenant-test", _period_id(svc, "tenant-test"), "manual")
	cash_id = _account_id(svc, "tenant-test", "1000")
	rev_id = _account_id(svc, "tenant-test", "4000")
	j = svc.create_journal_entry("j1", "tenant-test", batch["id"], "Small", [
		{"account_id": cash_id, "debit": 100, "credit": 0},
		{"account_id": rev_id, "debit": 0, "credit": 100},
	])
	result = _run(svc.journal_approval_workflow("tenant-test", j["id"], "500", "approver"))
	assert result["decision"] == "auto_approved"


def test_journal_approval_workflow_above_threshold():
	svc = _funded_svc()
	batch = svc.create_journal_batch("b1", "tenant-test", _period_id(svc, "tenant-test"), "manual")
	cash_id = _account_id(svc, "tenant-test", "1000")
	rev_id = _account_id(svc, "tenant-test", "4000")
	j = svc.create_journal_entry("j1", "tenant-test", batch["id"], "Large", [
		{"account_id": cash_id, "debit": 10000, "credit": 0},
		{"account_id": rev_id, "debit": 0, "credit": 10000},
	])
	result = _run(svc.journal_approval_workflow("tenant-test", j["id"], "500", "approver"))
	assert result["decision"] == "pending"


def test_bulk_journal_import():
	svc = _funded_svc()
	cash_id = _account_id(svc, "tenant-test", "1000")
	rev_id = _account_id(svc, "tenant-test", "4000")
	csv_data = (
		"journal_date,description,reference,account_id,debit,credit,posted_by\n"
		f"2026-01-10,Sales Jan,INV-001,{cash_id},1000,0,poster\n"
		f"2026-01-10,Sales Jan,INV-001,{rev_id},0,1000,poster\n"
	)
	result = _run(svc.bulk_journal_import("tenant-test", csv_data))
	assert result["posted_count"] == 1
	assert result["failed_count"] == 0


# ---------------------------------------------------------------------------
# Financial statements
# ---------------------------------------------------------------------------

def test_trial_balance():
	svc = _funded_svc()
	_post_revenue_and_expense(svc, "tenant-test")
	tb = _run(svc.trial_balance("tenant-test", "2026-01"))
	assert tb["balanced"] is True
	assert len(tb["rows"]) > 0


def test_trial_balance_includes_zero_balances():
	svc = _funded_svc()
	tb = _run(svc.trial_balance("tenant-test", "2026-01", include_zero_balances=True))
	codes = {r["account_code"] for r in tb["rows"]}
	assert "3000" in codes  # Share Capital — never posted to


def test_balance_sheet():
	svc = _funded_svc()
	_post_revenue_and_expense(svc, "tenant-test")
	bs = _run(svc.balance_sheet("tenant-test", "2026-01"))
	assert "assets" in bs
	assert "liabilities" in bs
	assert "equity" in bs
	assert bs["total_assets"] is not None


def test_income_statement():
	svc = _funded_svc()
	_post_revenue_and_expense(svc, "tenant-test")
	inc = _run(svc.income_statement("tenant-test", "2026-01"))
	assert Decimal(inc["revenue"]) == Decimal("5000")
	assert Decimal(inc["operating_expenses"]) == Decimal("2000")
	assert Decimal(inc["pat"]) == Decimal("3000")


def test_cash_flow_statement():
	svc = _funded_svc()
	_post_revenue_and_expense(svc, "tenant-test")
	cfs = _run(svc.cash_flow_statement("tenant-test", "2026-01"))
	assert "operating_activities" in cfs
	assert "net_change_in_cash" in cfs


def test_management_accounts_pack():
	svc = _funded_svc()
	_post_revenue_and_expense(svc, "tenant-test")
	pack = _run(svc.management_accounts_pack("tenant-test", "2026-01"))
	assert "trial_balance" in pack
	assert "balance_sheet" in pack
	assert "income_statement" in pack
	assert "cash_flow_statement" in pack
	assert "budget_vs_actual" in pack


def test_segment_report():
	svc = _funded_svc()
	cash_id = _account_id(svc, "tenant-test", "1000")
	rev_id = _account_id(svc, "tenant-test", "4000")
	_run(svc.post_journal_v2(
		tenant_id="tenant-test", journal_date="2026-01-15", journal_type="standard",
		lines=[{"account_id": cash_id, "debit": "1000", "credit": "0", "cost_center": "CC01"},
		       {"account_id": rev_id, "debit": "0", "credit": "1000", "cost_center": "CC01"}],
		description="CC01 Revenue", reference="", posted_by="poster",
	))
	result = _run(svc.segment_report("tenant-test", "2026-01", "cost_center"))
	assert "segments" in result


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------

def test_account_reconciliation():
	svc = _funded_svc()
	_post_revenue_and_expense(svc, "tenant-test")
	rec = _run(svc.account_reconciliation("tenant-test", "1000", "2026-01"))
	assert rec["status"] == "open"
	assert rec["account_code"] == "1000"
	assert "gl_balance" in rec


def test_submit_and_approve_reconciliation():
	svc = _funded_svc()
	_post_revenue_and_expense(svc, "tenant-test")
	rec = _run(svc.account_reconciliation("tenant-test", "1000", "2026-01"))
	submitted = _run(svc.submit_reconciliation(
		"tenant-test", rec["id"], "controller",
		[{"description": "Timing diff", "amount": "100", "type": "timing_difference"}],
	))
	assert submitted["status"] == "submitted"
	approved = _run(svc.approve_reconciliation("tenant-test", rec["id"], "cfo"))
	assert approved["status"] == "approved"


def test_approve_non_submitted_reconciliation_raises():
	svc = _funded_svc()
	_post_revenue_and_expense(svc, "tenant-test")
	rec = _run(svc.account_reconciliation("tenant-test", "1000", "2026-01"))
	with pytest.raises(ValueError, match="not_submitted"):
		_run(svc.approve_reconciliation("tenant-test", rec["id"], "cfo"))


# ---------------------------------------------------------------------------
# Budget vs actual
# ---------------------------------------------------------------------------

def test_budget_vs_actual():
	svc = _funded_svc()
	_post_revenue_and_expense(svc, "tenant-test")
	svc.budgets["bud-1"] = {
		"id": "bud-1", "tenant_id": "tenant-test",
		"account_code": "4000", "period_code": "2026-01",
		"budget_amount": "4000", "budget_version": "approved",
	}
	result = _run(svc.budget_vs_actual("tenant-test", "2026-01"))
	rows_4000 = [r for r in result["rows"] if r["account_code"] == "4000"]
	assert len(rows_4000) == 1
	assert rows_4000[0]["budget"] == "4000.00"


# ---------------------------------------------------------------------------
# Year-end close
# ---------------------------------------------------------------------------

def test_year_end_close():
	svc = _funded_svc()
	_post_revenue_and_expense(svc, "tenant-test")
	_run(svc.close_period("tenant-test", "2026-01", "controller"))
	result = _run(svc.year_end_close(
		tenant_id="tenant-test", fiscal_year=2026, retained_earnings_account="3100",
	))
	assert result["status"] in {"closed", "no_income_statement_balances"}
	assert result["fiscal_year"] == 2026


def test_opening_balances_new_year():
	svc = _funded_svc()
	_post_revenue_and_expense(svc, "tenant-test")
	_run(svc.close_period("tenant-test", "2026-01", "controller"))
	_run(svc.year_end_close("tenant-test", 2026, "3100"))
	svc.open_period("p2027", "tenant-test", "2027-01", 2027, "2027-01-01", "2027-01-31")
	result = _run(svc.opening_balances_new_year("tenant-test", 2027))
	assert result["status"] in {"completed", "no_balances_to_carry_forward"}


def test_prior_year_adjustment():
	svc = _funded_svc()
	_post_revenue_and_expense(svc, "tenant-test")
	# Add a current-year open period so the PYA journal can post today
	import datetime
	today = datetime.date.today()
	svc.open_period("p-cur", "tenant-test", f"{today.year}-{today.month:02d}",
	                today.year, today.strftime("%Y-%m-01"), today.strftime("%Y-%m-28"))
	result = _run(svc.prior_year_adjustment(
		tenant_id="tenant-test", account_code="1000",
		amount="500", adjustment_reason="IAS 8 error correction Q1 2025",
	))
	assert result["status"] == "posted"


def test_prior_year_adjustment_empty_reason_raises():
	svc = _funded_svc()
	with pytest.raises(ValueError, match="adjustment_reason_required"):
		_run(svc.prior_year_adjustment("tenant-test", "1000", "500", ""))


# ---------------------------------------------------------------------------
# Currency revaluation
# ---------------------------------------------------------------------------

def test_currency_revaluation():
	svc = _funded_svc()
	svc.create_account("fx", "tenant-test", "7900", "FX Gain/Loss", "expense")
	_post_revenue_and_expense(svc, "tenant-test")
	result = _run(svc.currency_revaluation(
		tenant_id="tenant-test", period_code="2026-01",
		rates={"USD": Decimal("1"), "KES": Decimal("130")},
	))
	assert result is not None
	assert "status" in result


# ---------------------------------------------------------------------------
# XBRL tagging
# ---------------------------------------------------------------------------

def test_xbrl_tagging_extract():
	svc = _funded_svc()
	_post_revenue_and_expense(svc, "tenant-test")
	result = _run(svc.xbrl_tagging_extract("tenant-test", "2026-01", "IFRS"))
	assert result["framework"] == "IFRS"
	assert "fact_count" in result


# ---------------------------------------------------------------------------
# Consolidation
# ---------------------------------------------------------------------------

def test_ifrs_consolidation():
	svc = _funded_svc("parent")
	svc.create_account("sub-cash", "subsidiary", "1000", "Cash", "asset")
	svc.create_account("sub-rev", "subsidiary", "4000", "Revenue", "revenue")
	svc.open_period("sub-p1", "subsidiary", "2026-01", 2026, "2026-01-01", "2026-01-31")
	sub_cash = _account_id(svc, "subsidiary", "1000")
	sub_rev = _account_id(svc, "subsidiary", "4000")
	_run(svc.post_journal_v2(
		tenant_id="subsidiary", journal_date="2026-01-15", journal_type="standard",
		lines=[{"account_id": sub_cash, "debit": "2000", "credit": "0"},
		       {"account_id": sub_rev, "debit": "0", "credit": "2000"}],
		description="Sub revenue", reference="", posted_by="poster",
	))
	result = _run(svc.ifrs_consolidation(
		tenant_id="parent", subsidiaries=["subsidiary"],
		group_adjustments=[], minority_interest={},
	))
	assert result["entity_count"] == 2
	assert result["status"] == "completed"


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def test_dashboard_summary():
	svc = _funded_svc()
	summary = svc.dashboard_summary("tenant-test")
	assert summary["account_count"] == 7
	assert "posted_journal_count" in summary
	assert "streaming" in summary


def test_dashboard_summary_tenant_isolation():
	svc = _funded_svc("t1")
	svc.create_account("x", "t2", "9000", "Other", "asset")
	summary = svc.dashboard_summary("t1")
	assert summary["account_count"] == 7
