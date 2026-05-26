"""Regression coverage for the CONS executable capability contract."""

from capabilities.common.cons import register_capability
from capabilities.common.cons.capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-cons", {"consents": {"stale_review_days": 180}})

	assert contract["capability"] == "cons"
	assert contract["configuration"]["tenant_id"] == "tenant-cons"
	assert contract["configuration"]["consents"]["stale_review_days"] == 180
	assert contract["configuration_schema"]["required"] == ["tenant_id", "purposes", "consents", "privacy_requests", "governance", "ui", "theme"]
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "purposes", "notices", "consents", "requests", "preferences", "audit", "settings"}
	assert contract["theme"]["name"] == "cons_privacy_center"


def test_rule_engine_enforces_cons_guardrails():
	result = evaluate_capability_rules({"tenant_context_present": False, "operation": "create_purpose", "legal_basis_present": False, "consent_age_days": 400, "stale_consent_reviewed": False})
	privacy_request_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "process_privacy_request", "identity_verified": False})
	processing_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "process_consent_gated_data", "active_consent_present": False})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "purpose_requires_legal_basis", "stale_consent_requires_review"}
	assert privacy_request_result["matched_rules"] == ["privacy_request_requires_identity_verification"]
	assert processing_result["matched_rules"] == ["processing_requires_active_consent"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "cons"
	assert "dlpd" in registration["dependencies"]
	assert registration["ui_components"]["requests"] == "/cons/requests"
	assert "cons:process_requests" in registration["permissions"]
