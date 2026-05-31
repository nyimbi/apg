"""Regression coverage for the ESGN executable capability contract."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from capabilities.common.esgn import register_capability
from capabilities.common.esgn.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.esgn.service import EsgnService
from capabilities.common.esgn.views import analytics_model, lifecycle_batch_model, signing_agent_model, signing_room_model


def _future_expiry() -> str:
	return (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()


def _published_submission(service: EsgnService) -> dict[str, str]:
	service.create_template(
		"tpl-nda",
		"tenant-sign",
		"Mutual NDA",
		"legal-ops",
		["counterparty", "effective_date"],
		"esign-act",
		"regulated-field-scan",
		"legal-7y",
	)
	service.publish_template("tpl-nda", "tenant-sign", "legal-approver", True)
	service.submit_form(
		"sub-001",
		"tenant-sign",
		"tpl-nda",
		"sales-ops",
		{"counterparty": "Acme Ltd", "effective_date": "2026-05-30"},
		"audit:submission-001",
	)
	return {"template_id": "tpl-nda", "submission_id": "sub-001"}


def _recipients() -> list[dict[str, object]]:
	return [
		{"id": "rcp-1", "name": "Ada", "email": "ada@example.com", "role": "signer", "routing_order": 1, "consent_recorded": True},
		{"id": "rcp-2", "name": "Grace", "email": "grace@example.com", "role": "signer", "routing_order": 2, "consent_recorded": True},
	]


def test_contract_exposes_configuration_rules_ui_theme_and_streaming():
	contract = get_capability_contract("tenant-sign", {"evidence": {"retention_policy_required": False}})

	assert contract["capability"] == "esgn"
	assert contract["configuration"]["tenant_id"] == "tenant-sign"
	assert contract["configuration"]["evidence"]["retention_policy_required"] is False
	assert "signing_agents" in contract["configuration"]
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"forms",
		"submissions",
		"envelopes",
		"signatures",
		"evidence",
		"signing_agents",
		"agents",
		"streaming",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	]
	assert contract["agents"]["first_class"] is True
	assert contract["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert "signing_steward" in contract["agents"]["privileged_roles"]
	assert contract["streaming"]["required_processor"] == "bytewax"
	assert contract["streaming"]["lifecycle_stream"] == "esgn.lifecycle"
	assert len(contract["rule_engine"]["rules"]) >= 43
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "forms", "builder", "submissions", "envelopes", "signing", "agents", "lifecycle", "evidence", "audit", "analytics", "settings"}
	assert contract["ui"]["api_prefix"] == "/esgn/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "signing_room" in contract["theme"]["components"]
	assert contract["streaming"]["processor"] == "bytewax"


def test_rule_engine_enforces_esign_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "create_form_template",
		"template_owner_assigned": False,
		"template_name_present": False,
		"schema_fields_present": False,
		"publication_approved": False,
		"identity_verified": False,
		"evidence_package_created": True,
		"evidence_encrypted": False,
		"regulated_form": True,
		"compliance_review_recorded": False,
	})
	publish_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "publish_form", "publication_approved": False})
	sign_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "sign_envelope", "identity_verified": False, "signature_intent_present": False})
	stream_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "batch_esgn_mutation", "event_stream": "legacy_queue"})
	agent_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "register_signing_agent",
		"signing_agent_present": True,
		"agent_registered": False,
		"agent_id_present": False,
		"agent_name_present": False,
		"agent_runtime_supported": False,
		"agent_role_supported": False,
		"agent_scope_present": False,
		"agent_owner_present": False,
		"agent_purpose_present": False,
		"agent_contribution_disclosed": False,
		"privileged_role": True,
		"human_approval_required": False,
	})
	batch_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "validate_esgn_lifecycle_batch",
		"event_stream": "legacy_queue",
		"mutation_count": 0,
		"lifecycle_operation_supported": False,
	})

	assert result["decision"] == "deny"
	assert {"tenant_context_required", "form_template_requires_owner", "form_template_requires_name", "form_template_requires_schema", "evidence_requires_encryption", "regulated_form_requires_compliance_review"} <= set(result["matched_rules"])
	assert publish_result["matched_rules"] == ["form_publication_requires_approval"]
	assert sign_result["matched_rules"] == ["signing_requires_identity_verification", "signing_requires_intent"]
	assert stream_result["matched_rules"] == ["batch_esgn_mutation_requires_bytewax"]
	assert agent_result["decision"] == "deny"
	assert set(agent_result["matched_rules"]) >= {
		"signing_agent_requires_registration",
		"signing_agent_requires_id",
		"signing_agent_requires_name",
		"signing_agent_runtime_supported",
		"signing_agent_role_supported",
		"signing_agent_requires_scope",
		"signing_agent_requires_owner",
		"signing_agent_requires_purpose",
		"signing_agent_requires_disclosure",
		"signing_agent_privileged_role_requires_human_approval",
	}
	assert set(batch_result["matched_rules"]) == {
		"esgn_lifecycle_batch_requires_mutations",
		"esgn_lifecycle_operation_supported",
		"bytewax_esgn_lifecycle_stream_required",
	}


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "esgn"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "esgn_forms_signing"
	assert registration["ui_components"]["signing"] == "/esgn/signing"
	assert registration["ui_components"]["agents"] == "/esgn/agents"
	assert registration["ui_components"]["lifecycle"] == "/esgn/lifecycle"
	assert registration["agents"]["first_class"] is True
	assert registration["streaming"]["required_processor"] == "bytewax"
	assert "comp" in registration["dependencies"]
	assert "nlpc" in registration["optional_dependencies"]
	assert "esgn:audit" in registration["permissions"]


def test_digital_form_esign_and_evidence_lifecycle_is_executable():
	service = EsgnService()
	_published_submission(service)

	envelope = service.create_envelope(
		envelope_id="env-001",
		tenant_id="tenant-sign",
		submission_id="sub-001",
		subject="Mutual NDA signature",
		sender="sales-ops",
		signature_intent="approve_nda",
		recipients=_recipients(),
		document_hash="sha256:nda-v1",
		expires_at=_future_expiry(),
	)
	agent = service.register_signing_agent(
		"agent-001",
		"tenant-sign",
		"Clause reviewer",
		"codex",
		"clause_reviewer",
		"env-001",
		"legal-ops",
		True,
		purpose="Govern clause, routing, and evidence quality.",
		human_approval_required=True,
	)
	batch = service.validate_lifecycle_batch("tenant-sign", "bytewax", 2, "signing_agent_batch")
	first_ceremony = service.sign_envelope("cer-001", "tenant-sign", "env-001", "rcp-1", "approve_nda", True, signed_at="2026-05-30T08:00:00+00:00")
	second_ceremony = service.sign_envelope("cer-002", "tenant-sign", "env-001", "rcp-2", "approve_nda", True, signed_at="2026-05-30T08:01:00+00:00")
	evidence = service.create_evidence_package("evd-001", "tenant-sign", "env-001", True, "legal-7y", "audit:env-001")
	summary = service.dashboard_summary("tenant-sign")

	assert envelope["tamper_seal"]
	assert envelope["document_hash"] == "sha256:nda-v1"
	assert service.verify_tamper_seal("env-001", "tenant-sign") is True
	assert agent["runtime"] == "codex"
	assert agent["status"] == "active"
	assert batch["status"] == "accepted"
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
		"cancelled_envelope_count": 0,
		"rejected_envelope_count": 0,
		"ceremony_count": 2,
		"evidence_package_count": 1,
		"signing_agent_count": 1,
		"pending_signing_agent_review_count": 0,
		"lifecycle_batch_count": 1,
		"denied_lifecycle_batch_count": 0,
		"audit_event_count": 9,
	}
	assert signing_room_model(service, "tenant-sign")["required_controls"] == ["identity_verified", "signature_intent", "routing_order_ready", "tamper_seal_valid"]
	assert signing_agent_model(service, "tenant-sign")["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert lifecycle_batch_model(service, "tenant-sign")["accepted"][0]["operation"] == "signing_agent_batch"
	assert analytics_model(service, "tenant-sign")["summary"]["completed_envelope_count"] == 1


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
	with pytest.raises(PermissionError, match="audit_trail_required"):
		service.submit_form("sub-no-audit", "tenant-sign", "tpl-ok", "user", {"name": "Ada"}, "")
	service.submit_form("sub-ok", "tenant-sign", "tpl-ok", "user", {"name": "Ada"}, "audit:ok")

	with pytest.raises(PermissionError, match="recipient_consent_required"):
		service.create_envelope(
			"env-bad",
			"tenant-sign",
			"sub-ok",
			"Needs consent",
			[{"id": "rcp", "name": "Ada", "email": "ada@example.com", "role": "signer", "routing_order": 1, "consent_recorded": False}],
			document_hash="sha256:bad",
			expires_at=_future_expiry(),
		)
	with pytest.raises(PermissionError, match="recipient_consent_required"):
		service.create_envelope(
			"env-string-false",
			"tenant-sign",
			"sub-ok",
			"String false consent",
			[{"id": "rcp", "name": "Ada", "email": "ada@example.com", "role": "signer", "routing_order": 1, "consent_recorded": "false"}],
			document_hash="sha256:string-false",
			expires_at=_future_expiry(),
		)
	with pytest.raises(PermissionError, match="delegated_signing_policy_required"):
		service.create_envelope(
			"env-delegate",
			"tenant-sign",
			"sub-ok",
			"Needs delegation policy",
			[{"id": "rcp", "name": "Ada", "email": "ada@example.com", "role": "delegate", "routing_order": 1, "consent_recorded": True}],
			document_hash="sha256:delegate",
			expires_at=_future_expiry(),
		)
	with pytest.raises(PermissionError, match="document_hash_required"):
		service.create_envelope("env-no-doc", "tenant-sign", "sub-ok", "No hash", _recipients(), document_hash="", expires_at=_future_expiry())
	with pytest.raises(PermissionError, match="envelope_expiry_in_past"):
		service.create_envelope("env-expired", "tenant-sign", "sub-ok", "Expired", _recipients(), document_hash="sha256:expired", expires_at="2020-01-01T00:00:00+00:00")

	service.create_envelope("env-ok", "tenant-sign", "sub-ok", "Ready to sign", _recipients(), document_hash="sha256:ok", expires_at=_future_expiry())
	with pytest.raises(PermissionError, match="signer_routing_order_required"):
		service.sign_envelope("cer-order", "tenant-sign", "env-ok", "rcp-2", "approve", True)
	with pytest.raises(PermissionError, match="identity_verification_required"):
		service.sign_envelope("cer-bad", "tenant-sign", "env-ok", "rcp-1", "approve", False)
	service.sign_envelope("cer-ok-1", "tenant-sign", "env-ok", "rcp-1", "approve", True)
	with pytest.raises(PermissionError, match="recipient_already_signed"):
		service.sign_envelope("cer-duplicate", "tenant-sign", "env-ok", "rcp-1", "approve", True)
	service.sign_envelope("cer-ok-2", "tenant-sign", "env-ok", "rcp-2", "approve", True)
	with pytest.raises(PermissionError, match="evidence_encryption_required"):
		service.create_evidence_package("evd-bad", "tenant-sign", "env-ok", False, "retention", "audit:env-ok")


def test_envelope_cancellation_rejection_agent_and_batch_guardrails():
	service = EsgnService()
	_published_submission(service)
	service.create_envelope("env-cancel", "tenant-sign", "sub-001", "Cancelable", _recipients(), document_hash="sha256:cancel", expires_at=_future_expiry())

	with pytest.raises(PermissionError, match="state_change_reason_required"):
		service.cancel_envelope("env-cancel", "tenant-sign", "sender", "")
	cancelled = service.cancel_envelope("env-cancel", "tenant-sign", "sender", "Recipient requested reissue")
	assert cancelled["status"] == "cancelled"
	assert cancelled["state_reason"] == "Recipient requested reissue"
	with pytest.raises(PermissionError, match="envelope_not_signable"):
		service.sign_envelope("cer-cancelled", "tenant-sign", "env-cancel", "rcp-1", "approve", True)

	service.create_envelope("env-reject", "tenant-sign", "sub-001", "Rejectable", _recipients(), document_hash="sha256:reject", expires_at=_future_expiry())
	rejected = service.reject_envelope("env-reject", "tenant-sign", "rcp-1", "Incorrect entity")
	assert rejected["status"] == "rejected"

	with pytest.raises(PermissionError, match="signing_agent_runtime_not_supported"):
		service.register_signing_agent("agent-bad-runtime", "tenant-sign", "Assistant", "unknown", "clause_reviewer", "env-reject", "owner", True)
	with pytest.raises(PermissionError, match="signing_agent_id_required"):
		service.register_signing_agent("", "tenant-sign", "Assistant", "codex", "clause_reviewer", "env-reject", "owner", True)
	with pytest.raises(PermissionError, match="signing_agent_name_required"):
		service.register_signing_agent("agent-no-name", "tenant-sign", "", "codex", "clause_reviewer", "env-reject", "owner", True)
	with pytest.raises(PermissionError, match="signing_agent_scope_required"):
		service.register_signing_agent("agent-no-scope", "tenant-sign", "Assistant", "codex", "clause_reviewer", "", "owner", True)
	with pytest.raises(PermissionError, match="signing_agent_owner_required"):
		service.register_signing_agent("agent-no-owner", "tenant-sign", "Assistant", "codex", "clause_reviewer", "env-reject", "", True)
	with pytest.raises(PermissionError, match="signing_agent_purpose_required"):
		service.register_signing_agent("agent-no-purpose", "tenant-sign", "Assistant", "codex", "clause_reviewer", "env-reject", "owner", True, purpose=" ")
	with pytest.raises(PermissionError, match="signing_agent_disclosure_required"):
		service.register_signing_agent("agent-hidden", "tenant-sign", "Assistant", "codex", "clause_reviewer", "env-reject", "owner", False)
	with pytest.raises(PermissionError, match="signing_agent_role_not_supported"):
		service.register_signing_agent("agent-role", "tenant-sign", "Assistant", "codex", "unsupported", "env-reject", "owner", True)
	pending = service.register_signing_agent("agent-pending", "tenant-sign", "Assistant", "codex", "signing_steward", "env-reject", "owner", True, purpose="Govern privileged signing lifecycle evidence.")
	assert pending["status"] == "pending_review"
	with pytest.raises(PermissionError, match="bytewax_lifecycle_stream_required"):
		service.validate_lifecycle_batch("tenant-sign", "legacy_queue", 1, "signing_agent_batch")
	with pytest.raises(PermissionError, match="esgn_lifecycle_batch_empty"):
		service.validate_lifecycle_batch("tenant-sign", "bytewax", 0, "signing_agent_batch")
	with pytest.raises(PermissionError, match="unsupported_esgn_lifecycle_operation"):
		service.validate_lifecycle_batch("tenant-sign", "bytewax", 1, "kafka_replay")
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.validate_batch_mutation("legacy_queue")
	assert service.validate_batch_mutation("bytewax")["decision"] == "allow"
