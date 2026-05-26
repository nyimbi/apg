"""Regression coverage for the ZTNA executable capability contract."""

from capabilities.common.ztna import register_capability
from capabilities.common.ztna.capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-zero", {"devices": {"minimum_device_trust": 0.8}})

	assert contract["capability"] == "ztna"
	assert contract["configuration"]["tenant_id"] == "tenant-zero"
	assert contract["configuration"]["devices"]["minimum_device_trust"] == 0.8
	assert contract["configuration_schema"]["required"] == ["tenant_id", "identities", "devices", "resources", "governance", "ui", "theme"]
	assert len(contract["rule_engine"]["rules"]) >= 6
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "policies", "devices", "resources", "access", "sessions", "risk", "settings"}
	assert contract["ui"]["api_prefix"] == "/ztna/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "device_posture" in contract["theme"]["components"]


def test_rule_engine_enforces_zero_trust_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"identity_verified": False,
		"device_posture_present": False,
		"resource_policy_attached": False,
		"access_level": "privileged",
		"mfa_completed": False,
		"access_risk_score": 0.95,
		"access_review_recorded": False
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {
		"tenant_context_required",
		"identity_must_be_verified",
		"device_posture_required",
		"resource_policy_required",
		"privileged_access_requires_mfa",
		"high_risk_access_requires_review"
	}


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "ztna"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "ztna_zero_trust_ops"
	assert registration["ui_components"]["resources"] == "/ztna/resources"
	assert "mfau" in registration["dependencies"]
	assert "ztna:approve_access" in registration["permissions"]
