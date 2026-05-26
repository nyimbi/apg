"""Regression coverage for the BIOP executable capability contract."""

from capabilities.common.biop import register_capability
from capabilities.common.biop.capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-bio", {"modalities": {"minimum_match_confidence": 0.9}})

	assert contract["capability"] == "biop"
	assert contract["configuration"]["tenant_id"] == "tenant-bio"
	assert contract["configuration"]["modalities"]["minimum_match_confidence"] == 0.9
	assert contract["configuration_schema"]["required"] == ["tenant_id", "modalities", "templates", "liveness", "governance", "ui", "theme"]
	assert len(contract["rule_engine"]["rules"]) >= 6
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "users", "enrollments", "verification", "liveness", "compliance", "analytics", "settings"}
	assert contract["ui"]["api_prefix"] == "/biop/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "template_vault" in contract["theme"]["components"]


def test_rule_engine_enforces_biometric_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "process_biometric",
		"consent_recorded": False,
		"template_encrypted": False,
		"liveness_passed": False,
		"cross_border_processing": True,
		"privacy_review_recorded": False,
		"match_confidence": 0.5,
		"human_review_recorded": False
	})
	storage_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "store_template", "template_encrypted": False})
	auth_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "authenticate", "liveness_passed": False})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "biometric_processing_requires_consent", "cross_border_use_requires_review", "low_match_confidence_requires_review"}
	assert storage_result["matched_rules"] == ["template_storage_requires_encryption"]
	assert auth_result["matched_rules"] == ["authentication_requires_liveness"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "biop"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "biop_biometric_control"
	assert registration["ui_components"]["verification"] == "/biop/verification"
	assert "mfau" in registration["dependencies"]
	assert "biop:verify" in registration["permissions"]
