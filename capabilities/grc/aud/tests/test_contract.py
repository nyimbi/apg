"""Contract shape and rule evaluation tests for grc_aud."""
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


def test_rule_deny_auditor_is_auditee():
	from ..capability_contract import evaluate_capability_rules
	result = evaluate_capability_rules({
		"operation": "create_audit",
		"auditor_is_auditee": True,
	})
	assert result["decision"] == "deny"


def test_rule_deny_report_approver_is_author():
	from ..capability_contract import evaluate_capability_rules
	result = evaluate_capability_rules({
		"operation": "approve_report",
		"approver_is_author": True,
	})
	assert result["decision"] == "deny"


def test_rule_allow_create_audit():
	from ..capability_contract import evaluate_capability_rules
	result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "create_audit",
		"title_present": True,
		"auditor_present": True,
		"audit_type_supported": True,
		"scope_present": True,
		"scope_type_supported": True,
		"start_date_present": True,
		"end_date_present": True,
		"auditee_present": True,
		"auditor_is_auditee": False,
	})
	assert result["decision"] == "allow"
