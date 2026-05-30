"""Regression coverage for the FREC executable capability contract."""

import pytest

from capabilities.common.frec import api_helpers, register_capability
from capabilities.common.frec.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.frec.face_runtime import FrecGuardrailError, FrecService
from capabilities.common.frec.view_models import (
	audit_model,
	consent_center_model,
	dashboard_model,
	emotion_model,
	identification_model,
	liveness_model,
	review_queue_model,
	settings_model,
	template_gallery_model,
	verification_model,
)


def test_contract_exposes_configuration_rules_ui_theme_and_adapters():
	contract = get_capability_contract("tenant-face", {"recognition": {"verification_threshold": 0.91}})

	assert contract["capability"] == "frec"
	assert contract["configuration"]["tenant_id"] == "tenant-face"
	assert contract["configuration"]["recognition"]["verification_threshold"] == 0.91
	assert set(contract["configuration_schema"]["required"]) >= {"tenant_id", "consent", "recognition", "enrollment", "templates", "liveness", "verification", "identification", "watchlists", "emotion", "privacy", "reviews", "security", "governance", "observability", "adapters", "ui", "theme"}
	assert len(contract["rule_engine"]["rules"]) >= 30
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "subjects", "consents", "enrollment", "templates", "verification", "identification", "liveness", "emotion", "watchlists", "reviews", "audit", "settings"}
	assert contract["ui"]["api_prefix"] == "/frec/api/v1"
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "match_gallery" in contract["theme"]["components"]
	assert "review_queue" in contract["theme"]["components"]


def test_rule_engine_enforces_face_recognition_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "enroll_face",
		"consent_recorded": False,
		"active_consent_present": False,
		"template_hash_present": False,
		"template_encrypted": False,
		"face_quality": 0.5,
		"recapture_completed": False,
	})
	identify_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "identify_face", "watchlist_policy_attached": False})
	auth_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "authenticate_face", "liveness_passed": False, "liveness_score": 0.2, "spoof_detected": True})
	emotion_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "analyze_emotion", "emotion_analysis_requested": True, "approved_purpose_recorded": False})
	batch_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "batch_face_mutation", "event_stream": "kafka"})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) >= {"tenant_context_required", "face_enrollment_requires_consent", "face_enrollment_requires_active_consent", "face_template_requires_hash", "face_template_requires_encryption", "face_quality_requires_threshold", "low_face_quality_requires_recapture"}
	assert identify_result["matched_rules"] == ["identification_requires_watchlist_policy"]
	assert set(auth_result["matched_rules"]) >= {"authentication_requires_liveness", "liveness_score_requires_threshold", "spoof_signal_blocks_face_authentication"}
	assert emotion_result["matched_rules"] == ["emotion_analysis_requires_explicit_purpose"]
	assert batch_result["matched_rules"] == ["batch_face_mutation_requires_bytewax"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "frec"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "frec_identity_vision"
	assert registration["ui_components"]["watchlists"] == "/frec/watchlists"
	assert "biop" in registration["dependencies"]
	assert registration["adapters"]["event_stream"] == "bytewax"
	assert registration["endpoints"]["audit"] == "/frec/api/v1/audit"
	assert "frec:identify" in registration["permissions"]
	assert "frec:audit" in registration["permissions"]


def test_frec_lifecycle_is_executable():
	service = FrecService()
	tenant_id = "tenant-face"

	consent = service.record_face_consent("consent-alice", tenant_id, "alice", "workforce authentication", "signed:v1")
	template = service.enroll_face("template-alice", tenant_id, "alice", consent["id"], "sha256:face-template", 0.94)
	liveness = service.record_liveness("live-alice", tenant_id, "alice", 0.93)
	verification = service.verify_face("verify-alice", tenant_id, "alice", template["id"], liveness["id"], 0.93)
	watchlist = service.create_watchlist("watchlist-access", tenant_id, "Access list", "policy-1", "security", "access governance")
	service.add_watchlist_subject(watchlist["id"], tenant_id, "alice", template["id"], "security", "authorized")
	identification = service.identify_face("identify-alice", tenant_id, watchlist["id"], "alice", 0.96, review_recorded=True)
	emotion = service.analyze_emotion("emotion-alice", tenant_id, "alice", approved_purpose_recorded=True)

	assert verification["status"] == "verified"
	assert identification["status"] == "matched"
	assert emotion["status"] == "completed"
	assert service.dashboard_summary(tenant_id)["audit_event_count"] >= 7
	assert dashboard_model(service, tenant_id)["summary"]["verification_count"] == 1
	assert consent_center_model(service, tenant_id)["consents"][0]["id"] == "consent-alice"
	assert template_gallery_model(service, tenant_id)["templates"][0]["id"] == "template-alice"
	assert verification_model(service, tenant_id)["verifications"][0]["id"] == "verify-alice"
	assert identification_model(service, tenant_id)["identifications"][0]["id"] == "identify-alice"
	assert liveness_model(service, tenant_id)["liveness"][0]["id"] == "live-alice"
	assert review_queue_model(service, tenant_id)["pending_reviews"] == []
	assert emotion_model(service, tenant_id)["emotion_events"][0]["id"] == "emotion-alice"
	assert audit_model(service, tenant_id)["audit_events"]
	assert settings_model(service, tenant_id)["configuration"]["adapters"]["event_stream"] == "bytewax"


def test_frec_runtime_rejects_guardrail_violations():
	service = FrecService()
	tenant_id = "tenant-face"
	service.record_face_consent("consent-bob", tenant_id, "bob", "login", "signed:v1")

	with pytest.raises(FrecGuardrailError) as encryption_error:
		service.enroll_face("template-plain", tenant_id, "bob", "consent-bob", "sha256:plain", 0.95, template_encrypted=False)
	assert "face_template_requires_encryption" in encryption_error.value.result["matched_rules"]

	template = service.enroll_face("template-bob", tenant_id, "bob", "consent-bob", "sha256:bob", 0.95)
	with pytest.raises(FrecGuardrailError) as liveness_error:
		service.record_liveness("live-spoof", tenant_id, "bob", 0.9, spoof_detected=True)
	assert "spoof_signal_blocks_face_authentication" in liveness_error.value.result["matched_rules"]

	liveness = service.record_liveness("live-bob", tenant_id, "bob", 0.92)
	with pytest.raises(FrecGuardrailError) as low_match_error:
		service.verify_face("verify-low", tenant_id, "bob", template["id"], liveness["id"], 0.4)
	assert low_match_error.value.result["decision"] == "require_review"

	with pytest.raises(FrecGuardrailError) as emotion_error:
		service.analyze_emotion("emotion-bob", tenant_id, "bob", approved_purpose_recorded=False)
	assert "emotion_analysis_requires_explicit_purpose" in emotion_error.value.result["matched_rules"]


def test_frec_runtime_isolates_same_record_ids_by_tenant():
	service = FrecService()

	alpha = service.record_face_consent("shared-consent", "tenant-alpha", "alice", "login", "signed:alpha")
	beta = service.record_face_consent("shared-consent", "tenant-beta", "bayo", "login", "signed:beta")

	assert alpha["tenant_id"] == "tenant-alpha"
	assert beta["tenant_id"] == "tenant-beta"
	assert service.list_consents("tenant-alpha") == [alpha]
	assert service.list_consents("tenant-beta") == [beta]

	with pytest.raises(FrecGuardrailError) as cross_tenant_error:
		service.enroll_face("template-cross", "tenant-gamma", "alice", "shared-consent", "sha256:cross", 0.96)
	assert "cross_tenant_face_access_denied" in cross_tenant_error.value.result["matched_rules"]


def test_api_helpers_wrap_runtime_operations():
	tenant_id = "tenant-api-face"
	consent = api_helpers.record_face_consent({"id": "api-consent", "tenant_id": tenant_id, "subject_id": "api-user", "purpose": "login", "evidence": "signed"})
	template = api_helpers.enroll_face({"id": "api-template", "tenant_id": tenant_id, "subject_id": "api-user", "consent_id": consent["data"]["id"], "template_hash": "sha256:api", "face_quality": 0.95})
	liveness = api_helpers.record_liveness({"id": "api-live", "tenant_id": tenant_id, "subject_id": "api-user", "liveness_score": 0.92})
	verification = api_helpers.verify_face({"id": "api-verify", "tenant_id": tenant_id, "subject_id": "api-user", "template_id": template["data"]["id"], "liveness_id": liveness["data"]["id"], "match_confidence": 0.93})

	assert consent["ok"] is True
	assert template["ok"] is True
	assert liveness["ok"] is True
	assert verification["ok"] is True
	assert api_helpers.capability_status(tenant_id)["verification_count"] == 1
