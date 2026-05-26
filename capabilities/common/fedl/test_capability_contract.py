"""Regression coverage for the FEDL executable capability contract."""

from capabilities.common.fedl import register_capability
from capabilities.common.fedl.capability_contract import (
	evaluate_capability_rules,
	get_capability_contract
)


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-fed", {"privacy": {"max_privacy_epsilon": 4.0}})

	assert contract["capability"] == "fedl"
	assert contract["configuration"]["tenant_id"] == "tenant-fed"
	assert contract["configuration"]["privacy"]["max_privacy_epsilon"] == 4.0
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"federation",
		"privacy",
		"training",
		"governance",
		"ui",
		"theme"
	]
	assert len(contract["rule_engine"]["rules"]) >= 6
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"dashboard",
		"federations",
		"participants",
		"rounds",
		"privacy",
		"security",
		"models",
		"settings"
	}
	assert contract["ui"]["api_prefix"] == "/fedl/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "federation_topology" in contract["theme"]["components"]


def test_rule_engine_enforces_federated_learning_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "aggregate_updates",
		"participant_attested": False,
		"participant_count": 2,
		"secure_aggregation_enabled": False,
		"privacy_epsilon": 10.0,
		"privacy_review_recorded": False,
		"poisoning_signal_detected": True
	})
	join_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "join_federation",
		"participant_attested": False
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {
		"tenant_context_required",
		"secure_aggregation_required",
		"privacy_budget_requires_review",
		"poisoning_signal_blocks_round"
	}
	assert join_result["decision"] == "deny"
	assert join_result["matched_rules"] == ["participant_requires_attestation"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "fedl"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "fedl_privacy_mesh"
	assert registration["ui_components"]["participants"] == "/fedl/participants"
	assert "mlcm" in registration["dependencies"]
	assert "fedl:run_rounds" in registration["permissions"]
