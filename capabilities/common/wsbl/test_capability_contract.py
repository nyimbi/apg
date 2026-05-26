"""Regression coverage for the WSBL executable capability contract."""

from capabilities.common.wsbl import register_capability
from capabilities.common.wsbl.capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-wsbl", {"sites": {"multi_locale_enabled": False}})

	assert contract["capability"] == "wsbl"
	assert contract["configuration"]["tenant_id"] == "tenant-wsbl"
	assert contract["configuration"]["sites"]["multi_locale_enabled"] is False
	assert contract["configuration_schema"]["required"] == ["tenant_id", "sites", "pages", "publishing", "governance", "ui", "theme"]
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "sites", "pages", "editor", "components", "publishing", "analytics", "settings"}
	assert contract["theme"]["name"] == "wsbl_site_builder"


def test_rule_engine_enforces_wsbl_guardrails():
	result = evaluate_capability_rules({"tenant_context_present": False, "operation": "create_site", "site_owner_assigned": False, "custom_component_present": True, "component_review_recorded": False, "public_site": True, "accessibility_passed": False, "privacy_banner_required": True, "consent_policy_attached": False})
	publish_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "publish_site", "approval_recorded": False})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "site_requires_owner", "custom_component_requires_review", "public_site_requires_accessibility_pass", "privacy_banner_requires_consent_policy"}
	assert publish_result["matched_rules"] == ["publish_requires_approval"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "wsbl"
	assert "ncod" in registration["dependencies"]
	assert registration["ui_components"]["editor"] == "/wsbl/editor"
	assert "wsbl:publish" in registration["permissions"]
