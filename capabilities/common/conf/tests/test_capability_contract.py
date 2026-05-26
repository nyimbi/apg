"""Regression coverage for the CONF executable capability contract."""

from .. import register_capability
from ..capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract(
		"tenant-a",
		{"automation": {"auto_remediation_enabled": True}}
	)

	assert contract["capability"] == "conf"
	assert contract["configuration"]["tenant_id"] == "tenant-a"
	assert contract["configuration"]["automation"]["auto_remediation_enabled"] is True
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"gitops",
		"security",
		"automation",
		"ui",
		"theme"
	]
	assert len(contract["rule_engine"]["rules"]) >= 4
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"dashboard",
		"resources",
		"templates",
		"policies",
		"deployments",
		"drift",
		"gitops",
		"settings"
	}
	assert contract["ui"]["api_prefix"] == "/api/v1/config"
	assert contract["theme"]["tokens"]["border.radius"] == "10px"
	assert "configuration_resource_card" in contract["theme"]["components"]


def test_rule_engine_denies_unsafe_production_rollout():
	result = evaluate_capability_rules({
		"requested_operation": "apply",
		"validation_passed": False,
		"target_environment": "production",
		"change_approved": False,
		"contains_secrets": True,
		"secrets_encrypted": False,
		"drift_detected": True,
		"remediation_plan_available": False
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {
		"validate_before_apply",
		"production_changes_require_approval",
		"encrypted_secrets_required",
		"drift_requires_remediation_plan"
	}


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "conf_control_room"
	assert registration["ui_components"]["gitops"] == "/config/gitops"
	assert "notification_engine" in registration["dependencies"]
