"""Unit tests for Tax Administration domain rules."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1]
if str(PKG) not in sys.path:
	sys.path.insert(0, str(PKG))

from domain.rules import (
	RuleViolation,
	assert_tenant_context,
	assert_no_cross_tenant_access,
	assert_write_policy,
	assert_tax_type_supported,
	assert_taxpayer_type_supported,
	assert_legal_name_present,
	assert_id_number_present,
	assert_pin_unique,
	assert_taxpayer_active,
	assert_taxpayer_pin_present,
	assert_period_present,
	assert_return_type_supported,
	assert_no_duplicate_return,
	assert_return_amounts_consistent,
	assert_non_negative_amounts,
	assert_assessment_type_supported,
	assert_return_exists,
	assert_assessed_amount_positive,
	assert_assessor_present,
	assert_objection_within_deadline,
	assert_objection_within_deadline_flag,
	assert_grounds_present,
	assert_amount_disputed_positive,
	assert_objection_appealable,
	assert_demand_notice_issued,
	assert_collection_method_supported,
	assert_debt_outstanding,
	assert_payment_amount_positive,
	assert_payment_reference_present,
	assert_audit_type_supported,
	assert_auditor_present,
	assert_audit_period_valid,
	assert_audit_open,
	assert_refund_amount_positive,
	assert_no_outstanding_debt_for_clearance,
	assert_eoi_urgency_valid,
	assert_treaty_partner_present,
	assert_penalty_rate_valid,
	assert_interest_rate_valid,
	assert_evidence_present,
	assert_officer_present,
	assert_agent_runtime_supported,
	assert_agent_role_supported,
	assert_event_stream_supported,
)


# ---------------------------------------------------------------------------
# RuleViolation
# ---------------------------------------------------------------------------

def test_rule_violation_has_attributes():
	exc = RuleViolation("test_rule", "test reason", "do_something")
	assert exc.rule_name == "test_rule"
	assert exc.reason == "test reason"
	assert exc.required_action == "do_something"
	assert "test_rule" in str(exc)


# ---------------------------------------------------------------------------
# Tenant isolation
# ---------------------------------------------------------------------------

def test_tenant_context_required():
	with pytest.raises(RuleViolation, match="tenant_context_required"):
		assert_tenant_context({})


def test_tenant_context_present():
	assert_tenant_context({"tenant_id": "t1"})  # no exception


def test_cross_tenant_denied():
	with pytest.raises(RuleViolation, match="cross_tenant_access_denied"):
		assert_no_cross_tenant_access("t1", "t2")


def test_same_tenant_allowed():
	assert_no_cross_tenant_access("t1", "t1")  # no exception


def test_write_policy_missing():
	with pytest.raises(RuleViolation, match="write_requires_policy"):
		assert_write_policy({"operation_type": "write", "policy_attached": False})


def test_write_policy_present():
	assert_write_policy({"operation_type": "write", "policy_attached": True})


def test_read_no_policy_needed():
	assert_write_policy({"operation_type": "read", "policy_attached": False})


# ---------------------------------------------------------------------------
# Taxpayer registration
# ---------------------------------------------------------------------------

def test_tax_type_supported():
	for t in ["income_tax", "vat", "corporate_tax", "withholding_tax", "excise_duty"]:
		assert_tax_type_supported(t)


def test_tax_type_unsupported():
	with pytest.raises(RuleViolation, match="tax_type_not_supported"):
		assert_tax_type_supported("lottery_tax")


def test_taxpayer_type_supported():
	for t in ["individual", "company", "partnership", "trust", "ngo"]:
		assert_taxpayer_type_supported(t)


def test_taxpayer_type_unsupported():
	with pytest.raises(RuleViolation, match="taxpayer_type_not_supported"):
		assert_taxpayer_type_supported("robot")


def test_legal_name_required():
	with pytest.raises(RuleViolation, match="legal_name_required"):
		assert_legal_name_present(None)
	with pytest.raises(RuleViolation, match="legal_name_required"):
		assert_legal_name_present("   ")


def test_legal_name_valid():
	assert_legal_name_present("Alice Wanjiku")


def test_id_number_required():
	with pytest.raises(RuleViolation, match="id_number_required"):
		assert_id_number_present("")


def test_id_number_valid():
	assert_id_number_present("12345678")


def test_pin_unique_duplicate():
	with pytest.raises(RuleViolation, match="duplicate_pin_denied"):
		assert_pin_unique({"A000000001X", "P000000002B"}, "A000000001X")


def test_pin_unique_new():
	assert_pin_unique({"A000000001X"}, "A000000002Y")


def test_pin_unique_case_insensitive():
	with pytest.raises(RuleViolation, match="duplicate_pin_denied"):
		assert_pin_unique({"a000000001x"}, "A000000001X")


def test_taxpayer_active_deregistered():
	with pytest.raises(RuleViolation, match="taxpayer_inactive"):
		assert_taxpayer_active("deregistered")


def test_taxpayer_active_blocked():
	with pytest.raises(RuleViolation, match="taxpayer_inactive"):
		assert_taxpayer_active("blocked")


def test_taxpayer_active_ok():
	assert_taxpayer_active("active")
	assert_taxpayer_active("pending")
	assert_taxpayer_active("suspended")


# ---------------------------------------------------------------------------
# Return filing
# ---------------------------------------------------------------------------

def test_taxpayer_pin_required():
	with pytest.raises(RuleViolation, match="taxpayer_pin_required"):
		assert_taxpayer_pin_present("")


def test_taxpayer_pin_valid():
	assert_taxpayer_pin_present("A000000001X")


def test_period_required():
	with pytest.raises(RuleViolation, match="period_required"):
		assert_period_present(None)


def test_period_valid():
	assert_period_present("2025-01")


def test_return_type_supported():
	assert_return_type_supported("monthly_vat")
	assert_return_type_supported("annual_income")
	assert_return_type_supported("corporate_annual")


def test_return_type_unsupported():
	with pytest.raises(RuleViolation, match="return_type_not_supported"):
		assert_return_type_supported("mystery_return")


def test_return_amounts_consistent():
	assert_return_amounts_consistent(
		Decimal("500000"), Decimal("50000"), Decimal("450000")
	)


def test_return_amounts_inconsistent():
	with pytest.raises(RuleViolation, match="return_amounts_inconsistent"):
		assert_return_amounts_consistent(
			Decimal("500000"), Decimal("50000"), Decimal("400000")
		)


def test_return_amounts_within_tolerance():
	# 500000 - 50000 = 450000; declared 450000.50 — within 1.00 tolerance
	assert_return_amounts_consistent(
		Decimal("500000"), Decimal("50000"), Decimal("450000.50")
	)


def test_non_negative_amounts_valid():
	assert_non_negative_amounts(Decimal("0"), Decimal("100"), field="test")


def test_non_negative_amounts_negative():
	with pytest.raises(RuleViolation, match="negative_amount_denied"):
		assert_non_negative_amounts(Decimal("-1"), field="gross_income")


# ---------------------------------------------------------------------------
# Assessment
# ---------------------------------------------------------------------------

def test_assessment_type_supported():
	for t in ["self_assessment", "audit_assessment", "best_judgement"]:
		assert_assessment_type_supported(t)


def test_assessment_type_unsupported():
	with pytest.raises(RuleViolation, match="assessment_type_not_supported"):
		assert_assessment_type_supported("magic_assessment")


def test_return_exists_none():
	with pytest.raises(RuleViolation, match="return_not_found"):
		assert_return_exists(None, "ret_001")


def test_return_exists_ok():
	assert_return_exists(object(), "ret_001")


def test_assessed_amount_positive():
	with pytest.raises(RuleViolation, match="assessed_amount_must_be_positive"):
		assert_assessed_amount_positive(Decimal("0"))


def test_assessed_amount_valid():
	assert_assessed_amount_positive(Decimal("1"))


def test_assessor_present_missing():
	with pytest.raises(RuleViolation, match="assessor_required"):
		assert_assessor_present("")


def test_assessor_present_valid():
	assert_assessor_present("officer_1")


# ---------------------------------------------------------------------------
# Objection
# ---------------------------------------------------------------------------

def test_objection_within_deadline_ok():
	assert_objection_within_deadline(date(2025, 3, 1), date(2025, 3, 15))


def test_objection_within_deadline_edge():
	assert_objection_within_deadline(date(2025, 3, 1), date(2025, 3, 31))


def test_objection_deadline_passed():
	with pytest.raises(RuleViolation, match="objection_deadline_passed"):
		assert_objection_within_deadline(date(2025, 3, 1), date(2025, 4, 15))


def test_objection_deadline_flag_false():
	with pytest.raises(RuleViolation, match="objection_deadline_passed"):
		assert_objection_within_deadline_flag(False)


def test_objection_deadline_flag_true():
	assert_objection_within_deadline_flag(True)


def test_grounds_required():
	with pytest.raises(RuleViolation, match="grounds_required"):
		assert_grounds_present(None)


def test_grounds_valid():
	assert_grounds_present("Double counting of expenses")


def test_amount_disputed_positive():
	with pytest.raises(RuleViolation, match="amount_disputed_must_be_positive"):
		assert_amount_disputed_positive(Decimal("0"))


def test_amount_disputed_valid():
	assert_amount_disputed_positive(Decimal("0.01"))


def test_objection_appealable_dismissed():
	assert_objection_appealable("dismissed")


def test_objection_appealable_partial():
	assert_objection_appealable("partially_upheld")


def test_objection_not_appealable_submitted():
	with pytest.raises(RuleViolation, match="objection_not_appealable"):
		assert_objection_appealable("submitted")


def test_objection_not_appealable_upheld():
	with pytest.raises(RuleViolation, match="objection_not_appealable"):
		assert_objection_appealable("upheld")


# ---------------------------------------------------------------------------
# Debt collection
# ---------------------------------------------------------------------------

def test_demand_notice_required():
	with pytest.raises(RuleViolation, match="demand_notice_required"):
		assert_demand_notice_issued(None)
	with pytest.raises(RuleViolation, match="demand_notice_required"):
		assert_demand_notice_issued("  ")


def test_demand_notice_present():
	assert_demand_notice_issued("DN-20250101-ABCDEF")


def test_collection_method_supported():
	assert_collection_method_supported("payment_plan")
	assert_collection_method_supported("bank_levy")
	assert_collection_method_supported("garnishment")


def test_collection_method_unsupported():
	with pytest.raises(RuleViolation, match="collection_method_not_supported"):
		assert_collection_method_supported("magic_collection")


def test_debt_outstanding_ok():
	assert_debt_outstanding("outstanding")
	assert_debt_outstanding("partially_paid")


def test_debt_not_actionable():
	with pytest.raises(RuleViolation, match="debt_not_actionable"):
		assert_debt_outstanding("paid")


def test_payment_amount_positive():
	with pytest.raises(RuleViolation, match="payment_amount_must_be_positive"):
		assert_payment_amount_positive(Decimal("0"))


def test_payment_reference_required():
	with pytest.raises(RuleViolation, match="payment_reference_required"):
		assert_payment_reference_present("")


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def test_audit_type_supported():
	for t in ["desk_audit", "field_audit", "forensic_audit", "it_audit"]:
		assert_audit_type_supported(t)


def test_audit_type_unsupported():
	with pytest.raises(RuleViolation, match="audit_type_not_supported"):
		assert_audit_type_supported("alien_audit")


def test_auditor_required():
	with pytest.raises(RuleViolation, match="auditor_required"):
		assert_auditor_present(None)


def test_audit_period_valid():
	assert_audit_period_valid(date(2024, 1, 1), date(2024, 12, 31))


def test_audit_period_invalid():
	with pytest.raises(RuleViolation, match="audit_period_invalid"):
		assert_audit_period_valid(date(2024, 12, 31), date(2024, 1, 1))


def test_audit_period_same_day():
	assert_audit_period_valid(date(2024, 6, 1), date(2024, 6, 1))


def test_audit_open_planned():
	assert_audit_open("planned")


def test_audit_open_in_progress():
	assert_audit_open("in_progress")


def test_audit_not_open_finalised():
	with pytest.raises(RuleViolation, match="audit_not_open"):
		assert_audit_open("finalised")


# ---------------------------------------------------------------------------
# Refund / clearance
# ---------------------------------------------------------------------------

def test_refund_amount_positive():
	with pytest.raises(RuleViolation, match="refund_amount_must_be_positive"):
		assert_refund_amount_positive(Decimal("0"))


def test_refund_amount_valid():
	assert_refund_amount_positive(Decimal("1"))


def test_no_outstanding_debt_for_clearance_pass():
	assert_no_outstanding_debt_for_clearance([])


def test_outstanding_debt_blocks_clearance():
	@dataclass
	class FakeDebt:
		balance: Decimal

	debts = [FakeDebt(Decimal("50000")), FakeDebt(Decimal("30000"))]
	with pytest.raises(RuleViolation, match="outstanding_debt_blocks_clearance"):
		assert_no_outstanding_debt_for_clearance(debts)


# ---------------------------------------------------------------------------
# EOI
# ---------------------------------------------------------------------------

def test_eoi_urgency_valid():
	for u in ["routine", "urgent", "spontaneous"]:
		assert_eoi_urgency_valid(u)


def test_eoi_urgency_invalid():
	with pytest.raises(RuleViolation, match="eoi_urgency_invalid"):
		assert_eoi_urgency_valid("asap")


def test_treaty_partner_required():
	with pytest.raises(RuleViolation, match="treaty_partner_required"):
		assert_treaty_partner_present("")


def test_treaty_partner_valid():
	assert_treaty_partner_present("GB")


# ---------------------------------------------------------------------------
# Penalty / interest rates
# ---------------------------------------------------------------------------

def test_penalty_rate_valid():
	assert_penalty_rate_valid(Decimal("0.05"))
	assert_penalty_rate_valid(Decimal("0"))
	assert_penalty_rate_valid(Decimal("1"))


def test_penalty_rate_invalid():
	with pytest.raises(RuleViolation, match="penalty_rate_invalid"):
		assert_penalty_rate_valid(Decimal("1.5"))


def test_interest_rate_valid():
	assert_interest_rate_valid(Decimal("0.12"))


def test_interest_rate_invalid():
	with pytest.raises(RuleViolation, match="interest_rate_invalid"):
		assert_interest_rate_valid(Decimal("-0.01"))


# ---------------------------------------------------------------------------
# Agent rules
# ---------------------------------------------------------------------------

def test_agent_runtime_supported():
	assert_agent_runtime_supported("codex")
	assert_agent_runtime_supported("bytewax")
	assert_agent_runtime_supported("langgraph")


def test_agent_runtime_unsupported():
	with pytest.raises(RuleViolation, match="agent_runtime_not_supported"):
		assert_agent_runtime_supported("fancy_new_runtime")


def test_agent_role_supported():
	assert_agent_role_supported("return_processor")
	assert_agent_role_supported("audit_analyst")


def test_agent_role_unsupported():
	with pytest.raises(RuleViolation, match="agent_role_not_supported"):
		assert_agent_role_supported("super_agent")


def test_event_stream_bytewax():
	assert_event_stream_supported("bytewax")


def test_event_stream_unsupported():
	with pytest.raises(RuleViolation, match="event_stream_not_supported"):
		assert_event_stream_supported("sqs")


# ---------------------------------------------------------------------------
# Evidence / officer
# ---------------------------------------------------------------------------

def test_evidence_required():
	with pytest.raises(RuleViolation, match="evidence_required"):
		assert_evidence_present(None)


def test_evidence_valid():
	assert_evidence_present("doc_ref_001")


def test_officer_required():
	with pytest.raises(RuleViolation, match="officer_required"):
		assert_officer_present("  ")


def test_officer_valid():
	assert_officer_present("officer_1")
