"""Contract shape and rule evaluation tests for grc_pol."""
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


def test_rule_deny_approver_is_requestor():
	from ..capability_contract import evaluate_capability_rules
	result = evaluate_capability_rules({
		"operation": "approve_exception",
		"approver_is_requestor": True,
	})
	assert result["decision"] == "deny"


def test_rule_allow_publish_approved():
	from ..capability_contract import evaluate_capability_rules
	result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "publish_policy",
		"approved": True,
		"review_date_overdue": False,
	})
	assert result["decision"] == "allow"
