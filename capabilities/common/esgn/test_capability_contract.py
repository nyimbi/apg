"""Regression coverage for the ESGN executable capability contract."""

from capabilities.common.esgn import register_capability
from capabilities.common.esgn.capability_contract import evaluate_capability_rules, get_capability_contract


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
