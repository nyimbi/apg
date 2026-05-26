"""Public API tests for executable capability contracts."""

from __future__ import annotations

import capabilities


def test_capability_contract_registry_is_public_api():
	registry = capabilities.load_contract_registry()

	assert len(registry) >= 100
	assert capabilities.get_capability_contract("composition_events")["capability"] == "composition_events"
	assert capabilities.get_system_statistics()["executable_contracts"] >= 100


def test_public_rule_evaluation_api():
	result = capabilities.evaluate_capability_contract_rules(
		"composition_events",
		{"tenant_context_present": False, "operation_type": "write", "policy_attached": False},
	)

	assert result["decision"] == "deny"
	assert "tenant_context_required" in result["matched_rules"]
