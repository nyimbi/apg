"""Regression coverage for the ESGN executable capability contract."""

import pytest

from capabilities.common.esgn import register_capability
from capabilities.common.esgn.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.esgn.service import EsgnService


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-sign", {"evidence": {"retention_policy_required": False}})

	assert contract["capability"] == "esgn"
	assert contract["configuration"]["tenant_id"] == "tenant-sign"
	assert contract["configuration"]["evidence"]["retention_policy_required"] is False
	assert contract["configuration_schema"]["required"] == ["tenant_id", "forms", "signatures", "evidence", "governance", "ui", "theme"]
	assert len(contract["rule_engine"]["rules"]) >= 6
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "forms", "builder", "submissions", "envelopes", "signing", "evidence", "settings"}
	assert contract["ui"]["api_prefix"] == "/esgn/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "signing_room" in contract["theme"]["components"]


def test_rule_engine_enforces_esign_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "create_form_template",
		"template_owner_assigned": False,
		"publication_approved": False,
		"identity_verified": False,
		"evidence_package_created": True,
		"evidence_encrypted": False,
		"regulated_form": True,
		"compliance_review_recorded": False
	})
	publish_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "publish_form", "publication_approved": False})
	sign_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "sign_envelope", "identity_verified": False})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "form_template_requires_owner", "evidence_requires_encryption", "regulated_form_requires_compliance_review"}
	assert publish_result["matched_rules"] == ["form_publication_requires_approval"]
	assert sign_result["matched_rules"] == ["signing_requires_identity_verification"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "esgn"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "esgn_forms_signing"
	assert registration["ui_components"]["signing"] == "/esgn/signing"
	assert "comp" in registration["dependencies"]
	assert "esgn:sign" in registration["permissions"]


def test_digital_form_esign_and_evidence_lifecycle_is_executable():
	service = EsgnService()

	template = service.create_template(
		template_id="tpl-nda",
		tenant_id="tenant-sign",
		name="Mutual NDA",
		owner="legal-ops",
		schema_fields=["counterparty", "effective_date"],
		compliance_framework="esign-act",
		dlp_policy="regulated-field-scan",
		retention_policy="legal-7y",
		regulated_form=True,
		compliance_review_recorded=True,
	)
	published = service.publish_template("tpl-nda", "tenant-sign", "legal-approver", True)
	submission = service.submit_form(
		submission_id="sub-001",
		tenant_id="tenant-sign",
		template_id="tpl-nda",
		submitted_by="sales-ops",
		data={"counterparty": "Acme Ltd", "effective_date": "2026-05-29"},
		evidence_ref="audit:submission-001",
	)
	envelope = service.create_envelope(
		envelope_id="env-001",
		tenant_id="tenant-sign",
		submission_id="sub-001",
		subject="Mutual NDA signature",
		sender="sales-ops",
		signature_intent="approve_nda",
		recipients=[
			{"id": "rcp-1", "name": "Ada", "email": "ada@example.com", "role": "signer", "routing_order": 1, "consent_recorded": True},
			{"id": "rcp-2", "name": "Grace", "email": "grace@example.com", "role": "signer", "routing_order": 2, "consent_recorded": True},
		],
	)
	first_ceremony = service.sign_envelope(
		ceremony_id="cer-001",
		tenant_id="tenant-sign",
		envelope_id="env-001",
		recipient_id="rcp-1",
		signature_intent="approve_nda",
		identity_verified=True,
		signed_at="2026-05-29T08:00:00+00:00",
	)
	second_ceremony = service.sign_envelope(
		ceremony_id="cer-002",
		tenant_id="tenant-sign",
		envelope_id="env-001",
		recipient_id="rcp-2",
		signature_intent="approve_nda",
		identity_verified=True,
		signed_at="2026-05-29T08:01:00+00:00",
	)
	evidence = service.create_evidence_package(
		evidence_id="evd-001",
		tenant_id="tenant-sign",
		envelope_id="env-001",
		encrypted=True,
		retention_policy="legal-7y",
		audit_trail_ref="audit:env-001",
	)
	summary = service.dashboard_summary("tenant-sign")

	assert template["schema_digest"]
	assert template["review_status"] == "approved"
	assert published["status"] == "published"
	assert submission["validation_status"] == "valid"
	assert envelope["tamper_seal"]
	assert first_ceremony["signature_hash"] != second_ceremony["signature_hash"]
	assert service.list_envelopes("tenant-sign")[0]["status"] == "completed"
	assert evidence["certificate_id"].startswith("cert:")
	assert evidence["seal_digest"]
	assert summary == {
		"template_count": 1,
		"published_template_count": 1,
		"submission_count": 1,
		"envelope_count": 1,
		"completed_envelope_count": 1,
		"ceremony_count": 2,
		"evidence_package_count": 1,
		"audit_event_count": 7,
	}


def test_esgn_service_enforces_policy_guardrails():
	service = EsgnService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.create_template("", "", "Blank", "owner", ["name"], "framework", "dlp", "retention")
	with pytest.raises(PermissionError, match="template_owner_required"):
		service.create_template("tpl", "tenant-sign", "Blank", "", ["name"], "framework", "dlp", "retention")
	with pytest.raises(PermissionError, match="schema_validation_required"):
		service.create_template("tpl", "tenant-sign", "Blank", "owner", [], "framework", "dlp", "retention")
	with pytest.raises(PermissionError, match="regulated_field_dlp_required"):
		service.create_template("tpl", "tenant-sign", "Blank", "owner", ["name"], "framework", "", "retention", regulated_form=True)

	pending = service.create_template(
		"tpl-review",
		"tenant-sign",
		"Regulated form",
		"owner",
		["name"],
		"framework",
		"dlp",
		"retention",
		regulated_form=True,
		compliance_review_recorded=False,
	)
	assert pending["status"] == "pending_review"
	assert pending["review_status"] == "required"
	with pytest.raises(PermissionError, match="compliance_review_required"):
		service.publish_template("tpl-review", "tenant-sign", "approver", True)

	service.create_template("tpl-ok", "tenant-sign", "Standard form", "owner", ["name"], "framework", "dlp", "retention")
	with pytest.raises(PermissionError, match="publication_approval_required"):
		service.publish_template("tpl-ok", "tenant-sign", "approver", False)
	service.publish_template("tpl-ok", "tenant-sign", "approver", True)
	with pytest.raises(PermissionError, match="schema_validation_required"):
		service.submit_form("sub-bad", "tenant-sign", "tpl-ok", "user", {}, "audit:bad")
	service.submit_form("sub-ok", "tenant-sign", "tpl-ok", "user", {"name": "Ada"}, "audit:ok")

	with pytest.raises(PermissionError, match="recipient_consent_required"):
		service.create_envelope(
			"env-bad",
			"tenant-sign",
			"sub-ok",
			"Needs consent",
			[{"id": "rcp", "name": "Ada", "email": "ada@example.com", "role": "signer", "routing_order": 1, "consent_recorded": False}],
		)
	with pytest.raises(PermissionError, match="delegated_signing_policy_required"):
		service.create_envelope(
			"env-delegate",
			"tenant-sign",
			"sub-ok",
			"Needs delegation policy",
			[{"id": "rcp", "name": "Ada", "email": "ada@example.com", "role": "delegate", "routing_order": 1, "consent_recorded": True}],
		)
	service.create_envelope(
		"env-ok",
		"tenant-sign",
		"sub-ok",
		"Ready to sign",
		[{"id": "rcp", "name": "Ada", "email": "ada@example.com", "role": "signer", "routing_order": 1, "consent_recorded": True}],
	)
	with pytest.raises(PermissionError, match="identity_verification_required"):
		service.sign_envelope("cer-bad", "tenant-sign", "env-ok", "rcp", "approve", False)
	service.sign_envelope("cer-ok", "tenant-sign", "env-ok", "rcp", "approve", True)
	with pytest.raises(PermissionError, match="evidence_encryption_required"):
		service.create_evidence_package("evd-bad", "tenant-sign", "env-ok", False, "retention", "audit:env-ok")
