"""Regression coverage for the AUDP executable capability contract."""

from capabilities.common.audp import register_capability
from capabilities.common.audp.capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-audio", {"transcription": {"minimum_confidence": 0.85}})

	assert contract["capability"] == "audp"
	assert contract["configuration"]["tenant_id"] == "tenant-audio"
	assert contract["configuration"]["transcription"]["minimum_confidence"] == 0.85
	assert contract["configuration_schema"]["required"] == ["tenant_id", "transcription", "synthesis", "analysis", "governance", "ui", "theme"]
	assert len(contract["rule_engine"]["rules"]) >= 6
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "transcription", "synthesis", "analysis", "sessions", "models", "quality", "settings"}
	assert contract["ui"]["api_prefix"] == "/audp/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "waveform_viewer" in contract["theme"]["components"]


def test_rule_engine_enforces_audio_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "process_recording",
		"recording_consent_recorded": False,
		"voice_owner_consent_recorded": False,
		"synthetic_audio_requested": True,
		"watermark_applied": False,
		"model_invocation": True,
		"model_policy_attached": False,
		"transcription_confidence": 0.4,
		"human_review_recorded": False
	})
	clone_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "clone_voice", "voice_owner_consent_recorded": False})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "recording_consent_required", "synthetic_audio_requires_watermark", "audio_model_requires_policy", "low_transcription_confidence_requires_review"}
	assert clone_result["matched_rules"] == ["voice_cloning_requires_consent"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "audp"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "audp_audio_intelligence"
	assert registration["ui_components"]["transcription"] == "/audp/transcription"
	assert "aicr" in registration["dependencies"]
	assert "audp:transcribe" in registration["permissions"]
