"""Regression coverage for the WALT executable capability contract."""

from capabilities.common.walt import register_capability
from capabilities.common.walt.capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-wallet", {"wallets": {"multi_currency_enabled": False}})

	assert contract["capability"] == "walt"
	assert contract["configuration"]["tenant_id"] == "tenant-wallet"
	assert contract["configuration"]["wallets"]["multi_currency_enabled"] is False
	assert contract["configuration_schema"]["required"] == ["tenant_id", "wallets", "payments", "settlement", "governance", "ui", "theme"]
	assert contract["theme"]["name"] == "walt_wallet_ops"
	assert contract["ui"]["api_prefix"] == "/walt/api/v1"


def test_rule_engine_enforces_wallet_guardrails():
	result = evaluate_capability_rules({"tenant_context_present": False, "operation": "create_wallet", "wallet_owner_assigned": False, "payment_instrument_present": True, "instrument_encrypted": False, "transaction_amount": 20000, "mfa_completed": False, "transaction_risk_score": 0.9, "risk_review_recorded": False})
	settle_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "settle_batch", "reconciliation_completed": False})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "wallet_requires_owner", "instrument_requires_encryption", "high_value_requires_mfa", "high_risk_transaction_requires_review"}
	assert settle_result["matched_rules"] == ["settlement_requires_reconciliation"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "walt"
	assert "comp" in registration["dependencies"]
	assert registration["ui_components"]["transactions"] == "/walt/transactions"
	assert "walt:authorize" in registration["permissions"]
