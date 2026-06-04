"""Contract shape and rule evaluation tests for grc_icm."""
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


def test_rule_allow_report_incident():
	from ..capability_contract import evaluate_capability_rules
	result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "report_incident",
		"title_present": True,
		"incident_type_supported": True,
		"incident_severity_supported": True,
		"reporter_present": True,
		"owner_present": True,
		"detection_time_present": True,
	})
	assert result["decision"] == "allow"
