"""Regression coverage for the BIOP executable capability contract."""

from __future__ import annotations

import pytest

from capabilities.common.biop import api_helpers, register_capability
from capabilities.common.biop import view_models
from capabilities.common.biop.biometric_runtime import BiopService
from capabilities.common.biop.capability_contract import evaluate_capability_rules, get_capability_contract


def _ready_service() -> BiopService:
	service = BiopService()
	service.record_consent(
		consent_id="consent-1",
		tenant_id="tenant-bio",
		subject_id="subject-1",
		purpose="workforce authentication",
		modalities=["face", "fingerprint"],
		jurisdictions=["KE", "US"],
		granted_by="subject-1",
		evidence="signed-consent:v1",
	)
	service.enroll_template(
		template_id="template-1",
		tenant_id="tenant-bio",
		subject_id="subject-1",
		modality="face",
		template_hash="sha256:face-template",
		encrypted=True,
		quality_score=0.95,
		consent_id="consent-1",
		retention_policy="workforce-biometric-365d",
	)
	return service


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-bio", {"modalities": {"minimum_match_confidence": 0.9}})

	assert contract["capability"] == "biop"
	assert contract["configuration"]["tenant_id"] == "tenant-bio"
	assert contract["configuration"]["modalities"]["minimum_match_confidence"] == 0.9
	assert set(contract["configuration_schema"]["required"]) >= {"tenant_id", "consent", "modalities", "enrollment", "templates", "verification", "liveness", "reviews", "privacy", "retention", "security", "governance", "observability", "adapters", "ui", "theme"}
	assert len(contract["rule_engine"]["rules"]) >= 30
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"dashboard",
		"users",
		"consents",
		"enrollments",
		"templates",
		"verification",
		"liveness",
		"match_reviews",
		"privacy_reviews",
		"compliance",
		"analytics",
		"settings",
	}
	assert contract["ui"]["api_prefix"] == "/biop/api/v1"
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "template_vault" in contract["theme"]["components"]
	assert "privacy_review_queue" in contract["theme"]["components"]


def test_rule_engine_enforces_biometric_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "process_biometric",
		"consent_recorded": False,
		"active_consent_present": False,
		"active_template_present": False,
		"cross_border_processing": True,
		"privacy_review_recorded": False,
		"match_confidence": 0.5,
		"human_review_recorded": False,
	})
	storage_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "store_template", "template_encrypted": False})
	auth_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "authenticate", "liveness_passed": False})
	match_review_result = evaluate_capability_rules({"operation": "approve_match_review", "match_reviewer_same_as_requester": True})
	privacy_review_result = evaluate_capability_rules({"operation": "approve_privacy_review", "privacy_reviewer_same_as_requester": True})
	batch_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "batch_biometric_mutation", "event_stream": "kafka"})
	audit_result = evaluate_capability_rules({"tenant_context_present": True, "state_change_requested": True, "audit_event_recorded": False})
	quality_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "enroll_template", "quality_score": 0.5})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) >= {
		"tenant_context_required",
		"biometric_processing_requires_consent",
		"cross_border_use_requires_review",
		"low_match_confidence_requires_review",
		"biometric_operation_requires_active_consent",
		"verification_requires_active_template",
	}
	assert storage_result["matched_rules"] == ["template_storage_requires_encryption"]
	assert auth_result["matched_rules"] == ["authentication_requires_liveness"]
	assert match_review_result["matched_rules"] == ["match_review_requires_independent_reviewer"]
	assert privacy_review_result["matched_rules"] == ["privacy_review_requires_independent_reviewer"]
	assert batch_result["matched_rules"] == ["batch_biometric_mutation_requires_bytewax"]
	assert audit_result["matched_rules"] == ["biometric_state_change_requires_audit"]
	assert quality_result["matched_rules"] == ["template_quality_requires_threshold"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "biop"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "biop_biometric_control"
	assert registration["ui_components"]["verification"] == "/biop/verification"
	assert registration["ui_components"]["privacy_reviews"] == "/biop/reviews/privacy"
	assert "mfau" in registration["dependencies"]
	assert registration["adapters"]["event_stream"] == "bytewax"
	assert registration["endpoints"]["audit"] == "/biop/api/v1/audit"
	assert "biop:verify" in registration["permissions"]
	assert "biop:manage_consent" in registration["permissions"]
	assert "biop:review_privacy" in registration["permissions"]


def test_service_runs_consent_template_verification_review_and_audit_lifecycle():
	service = _ready_service()
	local_verified = service.request_verification(
		verification_id="verify-local",
		tenant_id="tenant-bio",
		subject_id="subject-1",
		template_id="template-1",
		modality="face",
		requested_by="access-operator",
		match_confidence=0.94,
		liveness_score=0.91,
		source_jurisdiction="KE",
		target_jurisdiction="KE",
	)
	cross_border = service.request_verification(
		verification_id="verify-cross-border",
		tenant_id="tenant-bio",
		subject_id="subject-1",
		template_id="template-1",
		modality="face",
		requested_by="access-operator",
		match_confidence=0.82,
		liveness_score=0.9,
		source_jurisdiction="KE",
		target_jurisdiction="US",
	)
	privacy_request = service.request_privacy_review(
		review_id="privacy-review-1",
		tenant_id="tenant-bio",
		verification_id="verify-cross-border",
		requested_by="access-operator",
		justification="US helpdesk recovery requires face verification.",
	)
	after_privacy = service.decide_privacy_review(
		review_id=privacy_request["id"],
		tenant_id="tenant-bio",
		reviewer="privacy-officer",
		decision="approved",
		notes="Approved under cross-border workforce support policy.",
	)
	match_request = service.request_match_review(
		review_id="match-review-1",
		tenant_id="tenant-bio",
		verification_id="verify-cross-border",
		requested_by="access-operator",
		justification="Confidence below configured threshold after privacy approval.",
	)
	after_match = service.decide_match_review(
		review_id=match_request["id"],
		tenant_id="tenant-bio",
		reviewer="identity-reviewer",
		decision="approved",
		notes="Secondary evidence confirms the subject.",
	)
	summary = service.biometric_summary("tenant-bio")
	dashboard = view_models.dashboard_model(service, "tenant-bio")

	assert local_verified["status"] == "verified"
	assert cross_border["status"] == "pending_privacy_review"
	assert after_privacy["status"] == "pending_match_review"
	assert after_match["status"] == "verified"
	assert after_match["privacy_review_id"] == "privacy-review-1"
	assert after_match["match_review_id"] == "match-review-1"
	assert summary["active_consent_count"] == 1
	assert summary["active_template_count"] == 1
	assert summary["verified_count"] == 2
	assert summary["review_count"] == 2
	assert dashboard["summary"]["audit_event_count"] >= 6


def test_service_enforces_biometric_guardrails():
	service = _ready_service()
	with pytest.raises(PermissionError, match="biometric_consent_not_active|active_biometric_consent_required"):
		service.revoke_consent("consent-1", "tenant-bio", revoked_by="privacy-admin", reason="subject request")
		service.enroll_template(
			template_id="template-revoked",
			tenant_id="tenant-bio",
			subject_id="subject-1",
			modality="face",
			template_hash="sha256:revoked",
			encrypted=True,
			quality_score=0.95,
			consent_id="consent-1",
			retention_policy="default",
		)

	service = _ready_service()
	with pytest.raises(PermissionError, match="template_encryption_required"):
		service.enroll_template(
			template_id="template-plain",
			tenant_id="tenant-bio",
			subject_id="subject-1",
			modality="face",
			template_hash="sha256:plain",
			encrypted=False,
			quality_score=0.95,
			consent_id="consent-1",
			retention_policy="default",
		)
	with pytest.raises(PermissionError, match="biometric_template_quality_too_low"):
		service.enroll_template(
			template_id="template-low-quality",
			tenant_id="tenant-bio",
			subject_id="subject-1",
			modality="face",
			template_hash="sha256:low-quality",
			encrypted=True,
			quality_score=0.2,
			consent_id="consent-1",
			retention_policy="default",
		)
	with pytest.raises(PermissionError, match="liveness_required"):
		service.request_verification(
			verification_id="verify-low-live",
			tenant_id="tenant-bio",
			subject_id="subject-1",
			template_id="template-1",
			modality="face",
			requested_by="access-operator",
			match_confidence=0.94,
			liveness_score=0.1,
			source_jurisdiction="KE",
			target_jurisdiction="KE",
		)
	cross_border = service.request_verification(
		verification_id="verify-cross-border",
		tenant_id="tenant-bio",
		subject_id="subject-1",
		template_id="template-1",
		modality="face",
		requested_by="access-operator",
		match_confidence=0.94,
		liveness_score=0.9,
		source_jurisdiction="KE",
		target_jurisdiction="US",
		privacy_review_recorded=True,
		human_review_recorded=True,
	)
	privacy_request = service.request_privacy_review(
		review_id="privacy-review-1",
		tenant_id="tenant-bio",
		verification_id=cross_border["id"],
		requested_by="access-operator",
		justification="Cross-border verification.",
	)
	with pytest.raises(ValueError, match="privacy_review_already_pending"):
		service.request_privacy_review(
			review_id="privacy-review-2",
			tenant_id="tenant-bio",
			verification_id=cross_border["id"],
			requested_by="access-operator",
			justification="Duplicate cross-border review.",
		)
	with pytest.raises(PermissionError, match="independent_privacy_reviewer_required"):
		service.decide_privacy_review(
			review_id=privacy_request["id"],
			tenant_id="tenant-bio",
			reviewer="access-operator",
			decision="approved",
			notes="Self-approved.",
		)
	with pytest.raises(ValueError, match="privacy_review_notes_required"):
		service.decide_privacy_review(
			review_id=privacy_request["id"],
			tenant_id="tenant-bio",
			reviewer="privacy-officer",
			decision="approved",
			notes="",
		)
	rejected = service.decide_privacy_review(
		review_id=privacy_request["id"],
		tenant_id="tenant-bio",
		reviewer="privacy-officer",
		decision="rejected",
		notes="No transfer impact assessment.",
	)
	with pytest.raises(ValueError, match="privacy_review_already_decided"):
		service.decide_privacy_review(
			review_id=privacy_request["id"],
			tenant_id="tenant-bio",
			reviewer="privacy-officer",
			decision="approved",
			notes="Changed later.",
		)
	low_match = service.request_verification(
		verification_id="verify-low-match",
		tenant_id="tenant-bio",
		subject_id="subject-1",
		template_id="template-1",
		modality="face",
		requested_by="access-operator",
		match_confidence=0.5,
		liveness_score=0.9,
		source_jurisdiction="KE",
		target_jurisdiction="KE",
		human_review_recorded=True,
	)
	match_request = service.request_match_review(
		review_id="match-review-1",
		tenant_id="tenant-bio",
		verification_id=low_match["id"],
		requested_by="access-operator",
		justification="Low-confidence match.",
	)
	with pytest.raises(ValueError, match="match_review_already_pending"):
		service.request_match_review(
			review_id="match-review-2",
			tenant_id="tenant-bio",
			verification_id=low_match["id"],
			requested_by="access-operator",
			justification="Duplicate match review.",
		)
	with pytest.raises(PermissionError, match="independent_match_reviewer_required"):
		service.decide_match_review(
			review_id=match_request["id"],
			tenant_id="tenant-bio",
			reviewer="access-operator",
			decision="approved",
			notes="Self-approved.",
		)
	with pytest.raises(ValueError, match="match_review_notes_required"):
		service.decide_match_review(
			review_id=match_request["id"],
			tenant_id="tenant-bio",
			reviewer="identity-reviewer",
			decision="approved",
			notes="",
		)
	match_rejected = service.decide_match_review(
		review_id=match_request["id"],
		tenant_id="tenant-bio",
		reviewer="identity-reviewer",
		decision="rejected",
		notes="Template did not match secondary evidence.",
	)
	service.retire_template("template-1", "tenant-bio", retired_by="template-admin", reason="rotation")
	with pytest.raises(PermissionError, match="biometric_template_not_active"):
		service.request_verification(
			verification_id="verify-retired",
			tenant_id="tenant-bio",
			subject_id="subject-1",
			template_id="template-1",
			modality="face",
			requested_by="access-operator",
			match_confidence=0.94,
			liveness_score=0.9,
			source_jurisdiction="KE",
			target_jurisdiction="KE",
		)

	assert cross_border["status"] == "pending_privacy_review"
	assert rejected["status"] == "rejected"
	assert rejected["decision"] == "deny"
	assert low_match["status"] == "pending_match_review"
	assert match_rejected["status"] == "rejected"


def test_service_keeps_duplicate_ids_isolated_by_tenant():
	service = BiopService()
	for tenant_id in ["tenant-a", "tenant-b"]:
		service.record_consent(
			consent_id="same-consent",
			tenant_id=tenant_id,
			subject_id="same-subject",
			purpose="authentication",
			modalities=["fingerprint"],
			jurisdictions=["KE"],
			granted_by="same-subject",
			evidence=f"evidence-{tenant_id}",
		)
		service.enroll_template(
			template_id="same-template",
			tenant_id=tenant_id,
			subject_id="same-subject",
			modality="fingerprint",
			template_hash=f"sha256:{tenant_id}",
			encrypted=True,
			quality_score=0.9,
			consent_id="same-consent",
			retention_policy="default",
		)

	assert service.list_consents("tenant-a")[0]["evidence"] == "evidence-tenant-a"
	assert service.list_consents("tenant-b")[0]["evidence"] == "evidence-tenant-b"
	assert service.list_templates("tenant-a")[0]["template_hash"] == "sha256:tenant-a"
	assert service.list_templates("tenant-b")[0]["template_hash"] == "sha256:tenant-b"
	with pytest.raises(ValueError, match="consent already exists"):
		service.record_consent(
			consent_id="same-consent",
			tenant_id="tenant-a",
			subject_id="same-subject",
			purpose="authentication",
			modalities=["fingerprint"],
			jurisdictions=["KE"],
			granted_by="same-subject",
			evidence="duplicate",
		)


def test_api_helpers_and_view_models_expose_biop_lifecycle():
	tenant_id = "tenant-api-biop"
	consent = api_helpers.record_consent({
		"id": "api-consent",
		"tenant_id": tenant_id,
		"subject_id": "api-subject",
		"purpose": "customer authentication",
		"modalities": ["face"],
		"jurisdictions": ["KE", "US"],
		"granted_by": "api-subject",
		"evidence": "api-consent-evidence",
	})
	template = api_helpers.enroll_template({
		"id": "api-template",
		"tenant_id": tenant_id,
		"subject_id": "api-subject",
		"modality": "face",
		"template_hash": "sha256:api-template",
		"encrypted": "true",
		"quality_score": 0.97,
		"consent_id": consent["id"],
		"retention_policy": "customer-biometric-365d",
	})
	verification = api_helpers.request_verification({
		"id": "api-verification",
		"tenant_id": tenant_id,
		"subject_id": "api-subject",
		"template_id": template["id"],
		"modality": "face",
		"requested_by": "api-operator",
		"match_confidence": 0.8,
		"liveness_score": 0.91,
		"source_jurisdiction": "KE",
		"target_jurisdiction": "KE",
		"human_review_recorded": "true",
	})
	match_review = api_helpers.request_match_review({
		"id": "api-match-review",
		"tenant_id": tenant_id,
		"verification_id": verification["id"],
		"requested_by": "api-operator",
		"justification": "Low-confidence API match.",
	})
	approved = api_helpers.decide_match_review({
		"id": match_review["id"],
		"tenant_id": tenant_id,
		"reviewer": "api-reviewer",
		"decision": "approved",
		"notes": "Approved from API helper path.",
	})
	dashboard = view_models.dashboard_model(tenant_id=tenant_id)
	reviews = view_models.review_queue_model(tenant_id=tenant_id)
	templates = view_models.template_vault_model(tenant_id=tenant_id)

	assert verification["status"] == "pending_match_review"
	assert approved["status"] == "verified"
	assert api_helpers.capability_status(tenant_id)["verified_count"] == 1
	assert dashboard["summary"]["review_count"] == 1
	assert reviews["decided_reviews"][0]["id"] == "api-match-review"
	assert templates["active_templates"][0]["id"] == "api-template"
