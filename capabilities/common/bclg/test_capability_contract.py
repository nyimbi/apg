"""Regression coverage for the BCLG executable capability contract."""

from capabilities.common.bclg import register_capability
from capabilities.common.bclg.capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-bclg", {"transactions": {"high_value_review_threshold": 250000}})

	assert contract["capability"] == "bclg"
	assert contract["configuration"]["tenant_id"] == "tenant-bclg"
	assert contract["configuration"]["transactions"]["high_value_review_threshold"] == 250000
	assert contract["configuration_schema"]["required"] == ["tenant_id", "ledgers", "transactions", "smart_contracts", "governance", "ui", "theme"]
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "ledgers", "transactions", "contracts", "keys", "audit", "compliance", "settings"}
	assert contract["theme"]["name"] == "bclg_ledger_ops"


def test_rule_engine_enforces_bclg_guardrails():
	result = evaluate_capability_rules({"tenant_context_present": False, "operation": "create_ledger", "ledger_owner_assigned": False, "key_custody_bound": False, "transaction_value": 200000, "transaction_review_recorded": False})
	transaction_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "submit_transaction", "signature_present": False, "key_custody_bound": True})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "ledger_requires_owner", "key_custody_required", "high_value_transaction_requires_review"}
	assert transaction_result["matched_rules"] == ["transaction_requires_signature"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "bclg"
	assert "keym" in registration["dependencies"]
	assert registration["ui_components"]["contracts"] == "/bclg/contracts"
	assert "bclg:manage_contracts" in registration["permissions"]
