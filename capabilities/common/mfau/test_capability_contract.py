"""Regression coverage for the MFAU executable capability contract."""

from capabilities.common.mfau import register_capability
from capabilities.common.mfau.capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-auth", {"risk": {"high_risk_threshold": 0.8}})

	assert contract["capability"] == "mfau"
	assert contract["configuration"]["tenant_id"] == "tenant-auth"
	assert contract["configuration"]["risk"]["high_risk_threshold"] == 0.8
	assert contract["configuration_schema"]["required"] == ["tenant_id", "methods", "risk", "recovery", "governance", "ui", "theme"]
	assert len(contract["rule_engine"]["rules"]) >= 6
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "methods", "enrollment", "risk", "recovery", "policies", "audit", "settings"}
	assert contract["ui"]["api_prefix"] == "/mfau/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "risk_meter" in contract["theme"]["components"]


def test_rule_engine_enforces_adaptive_mfa_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"risk_score": 0.9,
		"step_up_completed": False,
		"method_type": "biometric",
		"biometric_consent_recorded": False,
		"operation": "recover_account",
		"verified_recovery_channel": False,
		"action_risk": "admin",
		"phishing_resistant_factor_present": False,
		"device_trust_score": 0.2,
		"device_review_recorded": False
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {
		"tenant_context_required",
		"high_risk_requires_step_up",
		"biometric_method_requires_consent",
		"recovery_requires_verified_channel",
		"admin_action_requires_phishing_resistant_factor",
		"low_trust_device_requires_review"
	}


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "mfau"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "mfau_adaptive_auth_console"
	assert registration["ui_components"]["enrollment"] == "/mfau/enrollment"
	assert "auth" in registration["dependencies"]
	assert "mfau:challenge" in registration["permissions"]
