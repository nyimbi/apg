"""Contract shape and rule evaluation tests for fintech_terminal."""
from __future__ import annotations


def test_contract_shape():
	from ..capability_contract import get_capability_contract
	contract = get_capability_contract("test")
	assert contract["capability"] == "fintech_terminal"
	assert contract["rule_engine"]["type"] == "deterministic"
	assert len(contract["rule_engine"]["rules"]) >= 1
	assert contract["ui"]["requires_theme"] is True


def test_rule_evaluation_allow():
	from ..capability_contract import evaluate_capability_rules
	result = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})
	assert result["decision"] in ("allow", "deny", "require_review")


def test_rule_evaluation_deny_no_tenant():
	from ..capability_contract import evaluate_capability_rules
	result = evaluate_capability_rules({"tenant_context_present": False})
	assert result["decision"] == "deny"
	assert "tenant_context_required" in result["matched_rules"]


def test_rule_evaluation_deny_tamper():
	from ..capability_contract import evaluate_capability_rules
	result = evaluate_capability_rules({
		"operation": "process_transaction",
		"tamper_detected": True,
	})
	assert result["decision"] == "deny"


def test_rule_evaluation_allow_active_terminal():
	from ..capability_contract import evaluate_capability_rules
	result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "process_transaction",
		"terminal_status": "active",
		"tamper_detected": False,
		"terminal_key_expired": False,
	})
	assert result["decision"] == "allow"
