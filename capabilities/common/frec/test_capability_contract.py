"""Regression coverage for the FREC executable capability contract."""

from capabilities.common.frec import register_capability
from capabilities.common.frec.capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-face", {"recognition": {"verification_threshold": 0.91}})

	assert contract["capability"] == "frec"
	assert contract["configuration"]["tenant_id"] == "tenant-face"
	assert contract["configuration"]["recognition"]["verification_threshold"] == 0.91
	assert contract["configuration_schema"]["required"] == ["tenant_id", "recognition", "liveness", "emotion", "privacy", "ui", "theme"]
	assert len(contract["rule_engine"]["rules"]) >= 6
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "enrollment", "verification", "identification", "liveness", "emotion", "watchlists", "settings"}
	assert contract["ui"]["api_prefix"] == "/frec/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "match_gallery" in contract["theme"]["components"]


def test_rule_engine_enforces_face_recognition_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "enroll_face",
		"consent_recorded": False,
		"watchlist_policy_attached": False,
		"liveness_passed": False,
		"emotion_analysis_requested": True,
		"approved_purpose_recorded": False,
		"face_quality": 0.5,
		"recapture_completed": False
	})
	identify_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "identify_face", "watchlist_policy_attached": False})
	auth_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "authenticate_face", "liveness_passed": False})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "face_enrollment_requires_consent", "emotion_analysis_requires_explicit_purpose", "low_face_quality_requires_recapture"}
	assert identify_result["matched_rules"] == ["identification_requires_watchlist_policy"]
	assert auth_result["matched_rules"] == ["authentication_requires_liveness"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "frec"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "frec_identity_vision"
	assert registration["ui_components"]["watchlists"] == "/frec/watchlists"
	assert "biop" in registration["dependencies"]
	assert "frec:identify" in registration["permissions"]
