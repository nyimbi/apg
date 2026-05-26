"""Regression coverage for the ENCR executable capability contract."""

from capabilities.common.encr import register_capability
from capabilities.common.encr.capability_contract import (
	evaluate_capability_rules,
	get_capability_contract
)


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract(
		"tenant-crypto",
		{"cryptography": {"minimum_entropy_quality": 0.98}}
	)

	assert contract["capability"] == "encr"
	assert contract["configuration"]["tenant_id"] == "tenant-crypto"
	assert contract["configuration"]["cryptography"]["minimum_entropy_quality"] == 0.98
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"cryptography",
		"key_lifecycle",
		"policy",
		"threat_adaptive",
		"compliance",
		"ui",
		"theme"
	]
	assert len(contract["rule_engine"]["rules"]) >= 6
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"dashboard",
		"operations",
		"keys",
		"policies",
		"entropy",
		"homomorphic",
		"analytics",
		"settings"
	}
	assert contract["ui"]["api_prefix"] == "/encr/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "entropy_quality_meter" in contract["theme"]["components"]


def test_rule_engine_enforces_crypto_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"data_classification": "restricted",
		"algorithm_quantum_safe": False,
		"plaintext_export_requested": True,
		"entropy_quality": 0.8,
		"operation": "generate_key",
		"algorithm_family": "legacy",
		"security_review_recorded": False,
		"active_threat_signal": True,
		"key_rotation_completed": False
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {
		"tenant_context_required",
		"restricted_data_requires_quantum_safe_algorithm",
		"plaintext_export_blocked",
		"low_entropy_blocks_key_generation",
		"legacy_algorithm_requires_review",
		"active_threat_requires_key_rotation"
	}


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "encr_quantum_guard"
	assert registration["ui_components"]["homomorphic"] == "/encr/homomorphic"
	assert "secu" in registration["dependencies"]
