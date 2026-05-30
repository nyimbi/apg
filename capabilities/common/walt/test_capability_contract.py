"""Regression coverage for the WALT executable capability contract."""

from capabilities.common.walt import register_capability
from capabilities.common.walt.capability_contract import (
	evaluate_capability_rules,
	get_capability_contract,
	streaming_manifest,
)


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-wallet", {"wallets": {"multi_currency_enabled": False}})

	assert contract["capability"] == "walt"
	assert contract["configuration"]["tenant_id"] == "tenant-wallet"
	assert contract["configuration"]["wallets"]["multi_currency_enabled"] is False
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"wallets",
		"payments",
		"settlement",
		"walt_agents",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	]
	assert contract["requires"] == ["encr", "auth", "comp", "audl", "wflo"]
	assert contract["theme"]["name"] == "walt_wallet_ops"
	assert contract["ui"]["api_prefix"] == "/walt/api/v1"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "walt_agents" in contract["provides"]


def test_rule_engine_enforces_wallet_guardrails():
	result = evaluate_capability_rules({"tenant_context_present": False, "operation": "create_wallet", "wallet_owner_assigned": False, "ledger_ref_present": False, "compliance_policy_present": False, "payment_instrument_present": True, "instrument_encrypted": False, "transaction_amount": 20000, "mfa_completed": False, "transaction_risk_score": 0.9, "risk_review_recorded": False})
	instrument_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "register_instrument", "payment_instrument_present": True, "instrument_encrypted": False, "instrument_token_present": False, "instrument_verifier_present": False})
	settle_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "settle_batch", "reconciliation_completed": False, "settlement_approval_recorded": False, "event_stream": "local"})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "wallet_requires_owner", "wallet_requires_ledger", "wallet_requires_compliance_policy", "high_value_requires_mfa", "high_risk_transaction_requires_review"}
	assert set(instrument_result["matched_rules"]) == {"instrument_requires_encryption", "instrument_requires_token", "instrument_requires_verification"}
	assert settle_result["matched_rules"] == ["settlement_requires_reconciliation", "settlement_requires_approval", "settlement_requires_bytewax_stream"]


def test_agent_and_streaming_rules_are_exposed():
	agent_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "register_walt_agent", "agent_runtime_supported": False, "agent_role_supported": False})
	privileged_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "agent_payment_action", "privileged_scope": True, "human_approval_recorded": False})
	batch_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "batch_settlement", "event_stream": "local"})

	assert streaming_manifest()["stream"] == "apg.walt.lifecycle"
	assert agent_result["decision"] == "deny"
	assert set(agent_result["matched_rules"]) == {"walt_agent_runtime_supported", "walt_agent_role_supported"}
	assert privileged_result["matched_rules"] == ["privileged_agent_payment_action_requires_human_approval"]
	assert batch_result["matched_rules"] == ["batch_settlement_requires_bytewax"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "walt"
	assert "wflo" in registration["dependencies"]
	assert registration["ui_components"]["transactions"] == "/walt/transactions"
	assert registration["ui_components"]["agents"] == "/walt/agents"
	assert registration["streaming"]["processor"] == "bytewax"
	assert "walt:authorize" in registration["permissions"]
