"""Regression coverage for the I18N executable capability contract."""

from capabilities.common.i18n import register_capability
from capabilities.common.i18n.capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-i18n", {"translations": {"minimum_coverage_percent": 90}})

	assert contract["capability"] == "i18n"
	assert contract["configuration"]["tenant_id"] == "tenant-i18n"
	assert contract["configuration"]["translations"]["minimum_coverage_percent"] == 90
	assert contract["configuration_schema"]["required"] == ["tenant_id", "locales", "translations", "publishing", "governance", "ui", "theme"]
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "locales", "translations", "glossaries", "coverage", "publishing", "policies", "settings"}
	assert contract["theme"]["name"] == "i18n_localization_workbench"


def test_rule_engine_enforces_i18n_guardrails():
	result = evaluate_capability_rules({"tenant_context_present": False, "operation": "create_locale", "locale_owner_assigned": False, "machine_translation_used": True, "translation_review_recorded": False, "restricted_content_present": True, "rbac_filter_applied": False, "coverage_percent": 70, "coverage_review_recorded": False})
	publish_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "publish_translations", "approval_recorded": False})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "locale_requires_owner", "machine_translation_requires_review", "restricted_content_requires_filtering", "low_coverage_requires_review"}
	assert publish_result["matched_rules"] == ["publish_requires_approval"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "i18n"
	assert "nlpc" in registration["dependencies"]
	assert registration["ui_components"]["translations"] == "/i18n/translations"
	assert "i18n:translate" in registration["permissions"]
