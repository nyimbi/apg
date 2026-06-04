"""Unit tests for domain/rules.py — all business rules as callable functions."""
from __future__ import annotations

import pytest
from decimal import Decimal

from capabilities.fin.glr.general_ledger.domain.rules import (
	RuleViolation,
	assert_tenant_context,
	assert_write_policy,
	assert_no_cross_tenant_access,
	assert_valid_account_type,
	assert_no_parent_cycle,
	assert_posting_account,
	assert_account_code_unique,
	assert_journal_balanced,
	assert_minimum_two_lines,
	assert_line_single_sided,
	assert_line_non_negative,
	assert_journal_approved,
	assert_segregation_of_duties,
	assert_journal_not_already_posted,
	assert_journal_is_posted,
	assert_reversal_reason,
	assert_period_open_for_posting,
	assert_period_can_be_opened,
	assert_period_can_be_closed,
	assert_period_is_closed_for_locking,
	assert_period_not_locked,
	assert_reopen_authorised,
	assert_date_in_open_period,
	assert_reconciliation_submitted,
	assert_reconciliation_open,
	assert_budget_amount_positive,
	assert_exchange_rate_positive,
	assert_retained_earnings_account_exists,
	assert_fiscal_year_not_already_closed,
	assert_adjustment_reason,
	assert_intercompany_amount_positive,
	assert_intercompany_account_mapping,
	calculate_normal_balance,
	calculate_net_balance,
	calculate_variance,
)


# ---------------------------------------------------------------------------
# RuleViolation
# ---------------------------------------------------------------------------

def test_rule_violation_message():
	exc = RuleViolation("my_rule", "something went wrong", "fix_it")
	assert "my_rule" in str(exc)
	assert exc.rule_name == "my_rule"
	assert exc.required_action == "fix_it"


# ---------------------------------------------------------------------------
# Tenant / context guards
# ---------------------------------------------------------------------------

def test_assert_tenant_context_ok():
	assert_tenant_context({"tenant_id": "t1"})  # must not raise


def test_assert_tenant_context_missing_raises():
	with pytest.raises(RuleViolation, match="tenant_context_required"):
		assert_tenant_context({})


def test_assert_tenant_context_empty_string_raises():
	with pytest.raises(RuleViolation):
		assert_tenant_context({"tenant_id": ""})


def test_assert_write_policy_ok():
	assert_write_policy({"operation_type": "write", "policy_attached": True})


def test_assert_write_policy_missing_raises():
	with pytest.raises(RuleViolation, match="write_requires_policy"):
		assert_write_policy({"operation_type": "write", "policy_attached": False})


def test_assert_write_policy_read_ok():
	assert_write_policy({"operation_type": "read", "policy_attached": False})  # no raise


def test_assert_no_cross_tenant_ok():
	assert_no_cross_tenant_access("t1", "t1")


def test_assert_cross_tenant_raises():
	with pytest.raises(RuleViolation, match="cross_tenant_access_denied"):
		assert_no_cross_tenant_access("t1", "t2")


# ---------------------------------------------------------------------------
# Account rules
# ---------------------------------------------------------------------------

def test_assert_valid_account_type_all_valid():
	for t in ("asset", "liability", "equity", "revenue", "expense", "contra"):
		assert_valid_account_type(t)


def test_assert_valid_account_type_invalid_raises():
	with pytest.raises(RuleViolation, match="invalid_account_type"):
		assert_valid_account_type("bogus")


def test_assert_no_parent_cycle_ok():
	assert_no_parent_cycle("acct-1", "acct-2")
	assert_no_parent_cycle("acct-1", None)


def test_assert_parent_cycle_raises():
	with pytest.raises(RuleViolation, match="account_parent_cycle"):
		assert_no_parent_cycle("acct-1", "acct-1")


def test_assert_posting_account_active():
	assert_posting_account({"allow_posting": True, "status": "active"})


def test_assert_posting_account_disallows_posting_raises():
	with pytest.raises(RuleViolation, match="account_disallows_posting"):
		assert_posting_account({"allow_posting": False, "status": "active", "code": "1000"})


def test_assert_posting_account_inactive_raises():
	with pytest.raises(RuleViolation, match="account_inactive"):
		assert_posting_account({"allow_posting": True, "status": "inactive", "code": "1000"})


def test_assert_account_code_unique_ok():
	assert_account_code_unique({"1000", "2000"}, "3000")


def test_assert_account_code_duplicate_raises():
	with pytest.raises(RuleViolation, match="duplicate_account_code"):
		assert_account_code_unique({"1000", "2000"}, "1000")


# ---------------------------------------------------------------------------
# Journal rules
# ---------------------------------------------------------------------------

def test_assert_journal_balanced_ok():
	assert_journal_balanced(Decimal("1000"), Decimal("1000"))


def test_assert_journal_balanced_unequal_raises():
	with pytest.raises(RuleViolation, match="journal_must_balance"):
		assert_journal_balanced(Decimal("1000"), Decimal("900"))


def test_assert_journal_balanced_zero_raises():
	with pytest.raises(RuleViolation, match="journal_total_must_be_positive"):
		assert_journal_balanced(Decimal("0"), Decimal("0"))


def test_assert_minimum_two_lines_ok():
	assert_minimum_two_lines(["a", "b"])
	assert_minimum_two_lines(["a", "b", "c"])


def test_assert_minimum_two_lines_raises():
	with pytest.raises(RuleViolation, match="journal_requires_minimum_two_lines"):
		assert_minimum_two_lines(["a"])


def test_assert_line_single_sided_debit_ok():
	assert_line_single_sided(Decimal("100"), Decimal("0"))


def test_assert_line_single_sided_credit_ok():
	assert_line_single_sided(Decimal("0"), Decimal("100"))


def test_assert_line_single_sided_both_raises():
	with pytest.raises(RuleViolation, match="line_cannot_have_both_debit_and_credit"):
		assert_line_single_sided(Decimal("100"), Decimal("50"))


def test_assert_line_non_negative_ok():
	assert_line_non_negative(Decimal("0"), Decimal("0"))
	assert_line_non_negative(Decimal("500"), Decimal("0"))


def test_assert_line_non_negative_negative_debit_raises():
	with pytest.raises(RuleViolation, match="line_amount_must_be_non_negative"):
		assert_line_non_negative(Decimal("-1"), Decimal("0"))


def test_assert_journal_approved_ok():
	assert_journal_approved({"approved_by": "cfo", "id": "j1"})


def test_assert_journal_approved_missing_raises():
	with pytest.raises(RuleViolation, match="journal_approval_required"):
		assert_journal_approved({"id": "j1", "approved_by": None})


def test_assert_segregation_of_duties_ok():
	assert_segregation_of_duties("alice", "bob")
	assert_segregation_of_duties("", "bob")  # blank preparer ok
	assert_segregation_of_duties("alice", "")  # blank poster ok


def test_assert_sod_same_user_raises():
	with pytest.raises(RuleViolation, match="segregation_of_duties_required"):
		assert_segregation_of_duties("alice", "alice")


def test_assert_journal_not_already_posted_ok():
	assert_journal_not_already_posted({"status": "approved", "id": "j1"})


def test_assert_journal_already_posted_raises():
	with pytest.raises(RuleViolation, match="journal_already_posted"):
		assert_journal_not_already_posted({"status": "posted", "id": "j1"})


def test_assert_journal_is_posted_ok():
	assert_journal_is_posted({"status": "posted", "id": "j1"})


def test_assert_journal_not_posted_raises():
	with pytest.raises(RuleViolation, match="journal_not_posted"):
		assert_journal_is_posted({"status": "draft", "id": "j1"})


def test_assert_reversal_reason_ok():
	assert_reversal_reason("Correction of coding error")


def test_assert_reversal_reason_empty_raises():
	with pytest.raises(RuleViolation, match="reversal_reason_required"):
		assert_reversal_reason("")


def test_assert_reversal_reason_whitespace_raises():
	with pytest.raises(RuleViolation):
		assert_reversal_reason("   ")


# ---------------------------------------------------------------------------
# Period rules
# ---------------------------------------------------------------------------

def test_assert_period_open_for_posting_ok():
	assert_period_open_for_posting({"status": "open", "period_code": "2026-01"})


def test_assert_period_not_open_raises():
	with pytest.raises(RuleViolation, match="period_not_open"):
		assert_period_open_for_posting({"status": "closed", "period_code": "2026-01"})


def test_assert_period_can_be_opened_from_future():
	assert_period_can_be_opened({"status": "future", "period_code": "2026-01"})


def test_assert_period_can_be_opened_from_closed():
	assert_period_can_be_opened({"status": "closed", "period_code": "2026-01"})


def test_assert_period_cannot_be_opened_from_locked():
	with pytest.raises(RuleViolation, match="period_cannot_be_opened"):
		assert_period_can_be_opened({"status": "locked", "period_code": "2026-01"})


def test_assert_period_can_be_closed_ok():
	assert_period_can_be_closed({"status": "open", "period_code": "2026-01"})


def test_assert_period_cannot_be_closed_from_locked():
	with pytest.raises(RuleViolation, match="period_cannot_be_closed"):
		assert_period_can_be_closed({"status": "locked", "period_code": "2026-01"})


def test_assert_period_is_closed_for_locking_ok():
	assert_period_is_closed_for_locking({"status": "closed", "period_code": "2026-01"})


def test_assert_period_not_closed_for_locking_raises():
	with pytest.raises(RuleViolation, match="period_must_be_closed_before_locking"):
		assert_period_is_closed_for_locking({"status": "open", "period_code": "2026-01"})


def test_assert_period_not_locked_ok():
	assert_period_not_locked({"status": "open"})
	assert_period_not_locked({"status": "closed"})


def test_assert_period_locked_raises():
	with pytest.raises(RuleViolation, match="period_is_locked"):
		assert_period_not_locked({"status": "locked"})


def test_assert_reopen_authorised_ok():
	assert_reopen_authorised("Prior period error", "cfo")


def test_assert_reopen_authorised_no_reason_raises():
	with pytest.raises(RuleViolation, match="reopen_reason_required"):
		assert_reopen_authorised("", "cfo")


def test_assert_reopen_authorised_no_authoriser_raises():
	with pytest.raises(RuleViolation, match="reopen_authorisation_required"):
		assert_reopen_authorised("Reason given", "")


def test_assert_date_in_open_period_ok():
	periods = [
		{"tenant_id": "t1", "status": "open", "period_start": "2026-01-01", "period_end": "2026-01-31", "period_code": "2026-01"}
	]
	p = assert_date_in_open_period("2026-01-15", periods, "t1")
	assert p["period_code"] == "2026-01"


def test_assert_date_not_in_open_period_raises():
	periods = [
		{"tenant_id": "t1", "status": "closed", "period_start": "2026-01-01", "period_end": "2026-01-31", "period_code": "2026-01"}
	]
	with pytest.raises(RuleViolation, match="no_open_period_for_date"):
		assert_date_in_open_period("2026-01-15", periods, "t1")


def test_assert_date_in_open_period_wrong_tenant_raises():
	periods = [
		{"tenant_id": "t1", "status": "open", "period_start": "2026-01-01", "period_end": "2026-01-31", "period_code": "2026-01"}
	]
	with pytest.raises(RuleViolation, match="no_open_period_for_date"):
		assert_date_in_open_period("2026-01-15", periods, "t2")


# ---------------------------------------------------------------------------
# Reconciliation rules
# ---------------------------------------------------------------------------

def test_assert_reconciliation_submitted_ok():
	assert_reconciliation_submitted({"status": "submitted", "id": "r1"})


def test_assert_reconciliation_not_submitted_raises():
	with pytest.raises(RuleViolation, match="reconciliation_not_submitted"):
		assert_reconciliation_submitted({"status": "open", "id": "r1"})


def test_assert_reconciliation_open_ok():
	assert_reconciliation_open({"status": "open", "id": "r1"})
	assert_reconciliation_open({"status": "submitted", "id": "r1"})


def test_assert_reconciliation_approved_raises():
	with pytest.raises(RuleViolation, match="reconciliation_not_open"):
		assert_reconciliation_open({"status": "approved", "id": "r1"})


# ---------------------------------------------------------------------------
# Budget / currency rules
# ---------------------------------------------------------------------------

def test_assert_budget_amount_positive_ok():
	assert_budget_amount_positive(Decimal("0"))
	assert_budget_amount_positive(Decimal("5000"))


def test_assert_budget_amount_negative_raises():
	with pytest.raises(RuleViolation, match="budget_amount_must_be_non_negative"):
		assert_budget_amount_positive(Decimal("-1"))


def test_assert_exchange_rate_positive_ok():
	assert_exchange_rate_positive(Decimal("130"))


def test_assert_exchange_rate_zero_raises():
	with pytest.raises(RuleViolation, match="exchange_rate_must_be_positive"):
		assert_exchange_rate_positive(Decimal("0"))


def test_assert_exchange_rate_negative_raises():
	with pytest.raises(RuleViolation):
		assert_exchange_rate_positive(Decimal("-1"))


# ---------------------------------------------------------------------------
# Year-end rules
# ---------------------------------------------------------------------------

def test_assert_retained_earnings_account_exists_ok():
	accounts = [{"code": "3100", "account_type": "equity", "status": "active"}]
	acct = assert_retained_earnings_account_exists(accounts, "3100")
	assert acct["code"] == "3100"


def test_assert_retained_earnings_wrong_type_raises():
	accounts = [{"code": "3100", "account_type": "asset", "status": "active"}]
	with pytest.raises(RuleViolation, match="retained_earnings_must_be_equity"):
		assert_retained_earnings_account_exists(accounts, "3100")


def test_assert_retained_earnings_not_found_raises():
	with pytest.raises(RuleViolation, match="retained_earnings_account_not_found"):
		assert_retained_earnings_account_exists([], "3100")


def test_assert_fiscal_year_not_already_closed_ok():
	fiscal_years: dict = {}
	assert_fiscal_year_not_already_closed(fiscal_years, "t1", 2026)


def test_assert_fiscal_year_already_closed_raises():
	fiscal_years = {
		"fy1": {"tenant_id": "t1", "fiscal_year": 2026, "status": "closed"}
	}
	with pytest.raises(RuleViolation, match="fiscal_year_already_closed"):
		assert_fiscal_year_not_already_closed(fiscal_years, "t1", 2026)


# ---------------------------------------------------------------------------
# IAS 8 adjustment rules
# ---------------------------------------------------------------------------

def test_assert_adjustment_reason_ok():
	assert_adjustment_reason("Error in Q1 2025 depreciation schedule")


def test_assert_adjustment_reason_too_short_raises():
	with pytest.raises(RuleViolation, match="adjustment_reason_required"):
		assert_adjustment_reason("abc")


# ---------------------------------------------------------------------------
# Intercompany rules
# ---------------------------------------------------------------------------

def test_assert_intercompany_amount_positive_ok():
	assert_intercompany_amount_positive(Decimal("1000"))


def test_assert_intercompany_amount_zero_raises():
	with pytest.raises(RuleViolation, match="intercompany_amount_must_be_positive"):
		assert_intercompany_amount_positive(Decimal("0"))


def test_assert_intercompany_account_mapping_ok():
	assert_intercompany_account_mapping({"entity_account": "acc-1", "counterpart_account": "acc-2"})


def test_assert_intercompany_missing_entity_account_raises():
	with pytest.raises(RuleViolation, match="intercompany_entity_account_required"):
		assert_intercompany_account_mapping({"counterpart_account": "acc-2"})


def test_assert_intercompany_missing_counterpart_raises():
	with pytest.raises(RuleViolation, match="intercompany_counterpart_account_required"):
		assert_intercompany_account_mapping({"entity_account": "acc-1"})


# ---------------------------------------------------------------------------
# Calculation helpers
# ---------------------------------------------------------------------------

def test_calculate_normal_balance_asset():
	assert calculate_normal_balance("asset") == "debit"


def test_calculate_normal_balance_expense():
	assert calculate_normal_balance("expense") == "debit"


def test_calculate_normal_balance_revenue():
	assert calculate_normal_balance("revenue") == "credit"


def test_calculate_normal_balance_liability():
	assert calculate_normal_balance("liability") == "credit"


def test_calculate_normal_balance_equity():
	assert calculate_normal_balance("equity") == "credit"


def test_calculate_net_balance_debit_normal():
	result = calculate_net_balance(Decimal("100"), Decimal("500"), Decimal("200"), "debit")
	assert result == Decimal("400")


def test_calculate_net_balance_credit_normal():
	result = calculate_net_balance(Decimal("100"), Decimal("200"), Decimal("500"), "credit")
	assert result == Decimal("400")


def test_calculate_variance():
	var, pct, indicator = calculate_variance(Decimal("5000"), Decimal("4000"))
	assert var == Decimal("1000.00")
	assert pct == Decimal("25.00")
	assert indicator == "F"


def test_calculate_variance_adverse():
	var, pct, indicator = calculate_variance(Decimal("3000"), Decimal("4000"))
	assert var == Decimal("-1000.00")
	assert indicator == "A"


def test_calculate_variance_zero_budget():
	var, pct, indicator = calculate_variance(Decimal("500"), Decimal("0"))
	assert pct == Decimal("0")
