"""Regression coverage for the KEYM executable capability contract."""

from capabilities.common.keym import register_capability
from capabilities.common.keym.capability_contract import (
	evaluate_capability_rules,
	get_capability_contract
)


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract(
		"tenant-vault",
		{"lifecycle": {"default_rotation_days": 60}}
	)

	assert contract["capability"] == "keym"
	assert contract["configuration"]["tenant_id"] == "tenant-vault"
	assert contract["configuration"]["lifecycle"]["default_rotation_days"] == 60
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"key_domains",
		"lifecycle",
		"access",
		"hsm",
		"compliance",
		"automation",
		"ui",
		"theme"
	]
	assert len(contract["rule_engine"]["rules"]) >= 6
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"dashboard",
		"inventory",
		"lifecycle",
		"policies",
		"hsm",
		"audit",
		"analytics",
		"settings"
	}
	assert contract["ui"]["api_prefix"] == "/keym/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "key_inventory_row" in contract["theme"]["components"]


def test_rule_engine_enforces_key_governance_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "export_key",
		"policy_attached": False,
		"key_class": "root",
		"hsm_attested": False,
		"dual_control_approved": False,
		"rotation_age_days": 120,
		"rotation_exception_recorded": False,
		"key_status": "compromised",
		"operation_is_cryptographic": True
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {
		"tenant_context_required",
		"root_key_requires_hsm_attestation",
		"export_requires_dual_control",
		"overdue_rotation_requires_review",
		"compromised_key_blocks_use"
	}


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "keym_vault_console"
	assert registration["ui_components"]["inventory"] == "/keym/keys"
	assert "secu" in registration["dependencies"]
