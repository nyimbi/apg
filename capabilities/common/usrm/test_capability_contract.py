"""Regression coverage for the USRM executable capability contract."""

from capabilities.common.usrm import register_capability
from capabilities.common.usrm.capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-usrm", {"lifecycle": {"bulk_action_review_threshold": 10}})

	assert contract["capability"] == "usrm"
	assert contract["configuration"]["tenant_id"] == "tenant-usrm"
	assert contract["configuration"]["lifecycle"]["bulk_action_review_threshold"] == 10
	assert contract["configuration_schema"]["required"] == ["tenant_id", "users", "lifecycle", "access", "governance", "ui", "theme"]
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "users", "profiles", "lifecycle", "access", "privacy", "deprovisioning", "settings"}
	assert contract["theme"]["name"] == "usrm_user_lifecycle"


def test_rule_engine_enforces_usrm_guardrails():
	result = evaluate_capability_rules({"tenant_context_present": False, "operation": "create_user", "unique_identity_present": False, "privileged_user": True, "mfa_enabled": False, "affected_user_count": 40, "bulk_review_recorded": False})
	invite_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "invite_user", "consent_notice_attached": False})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "user_requires_identity", "privileged_user_requires_mfa", "bulk_user_action_requires_review"}
	assert invite_result["matched_rules"] == ["invite_requires_consent_notice"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "usrm"
	assert "cons" in registration["dependencies"]
	assert registration["ui_components"]["access"] == "/usrm/access"
	assert "usrm:review_access" in registration["permissions"]
