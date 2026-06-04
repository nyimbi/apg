"""Contract shape and rule evaluation tests for fintech_treasury."""
from __future__ import annotations


def test_contract_shape():
	from ..capability_contract import get_capability_contract
	contract = get_capability_contract("test")
	assert contract["capability"]
	assert contract["rule_engine"]["type"] == "deterministic"
	assert len(contract["rule_engine"]["rules"]) >= 1
	assert contract["ui"]["requires_theme"] is True


def test_rule_evaluation():
	from ..capability_contract import evaluate_capability_rules
	result = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})
	assert result["decision"] in ("allow", "deny", "require_review")


def test_rule_deny_no_tenant():
	from ..capability_contract import evaluate_capability_rules
	result = evaluate_capability_rules({"tenant_context_present": False})
	assert result["decision"] == "deny"


def test_rule_deny_sanctions_hit():
	from ..capability_contract import evaluate_capability_rules
	result = evaluate_capability_rules({
		"operation": "book_deal",
		"sanctions_hit": True,
	})
	assert result["decision"] == "deny"


def test_rule_deal_booking_allow():
	from ..capability_contract import evaluate_capability_rules
	result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "book_deal",
		"deal_type_supported": True,
		"counterparty_present": True,
		"four_eyes_recorded": True,
		"hard_limit_breached": False,
		"aml_screened": True,
		"sanctions_screened": True,
		"sanctions_hit": False,
	})
	assert result["decision"] == "allow"
