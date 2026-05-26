"""Regression coverage for the NCOD executable capability contract."""

from capabilities.common.ncod import register_capability
from capabilities.common.ncod.capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-builder", {"apps": {"publish_approval_required": False}})

	assert contract["capability"] == "ncod"
	assert contract["configuration"]["tenant_id"] == "tenant-builder"
	assert contract["configuration"]["apps"]["publish_approval_required"] is False
	assert contract["configuration_schema"]["required"] == ["tenant_id", "apps", "builder", "extensions", "governance", "ui", "theme"]
	assert len(contract["rule_engine"]["rules"]) >= 6
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "apps", "builder", "pages", "components", "publishing", "connectors", "settings"}
	assert contract["ui"]["api_prefix"] == "/ncod/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "page_composer" in contract["theme"]["components"]


def test_rule_engine_enforces_no_code_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "create_app",
		"app_owner_assigned": False,
		"approval_recorded": False,
		"script_extension_present": True,
		"script_policy_attached": False,
		"external_connector_present": True,
		"connector_policy_attached": False,
		"production_change": True,
		"change_review_recorded": False
	})
	publish_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "publish_app", "approval_recorded": False})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "app_requires_owner", "script_extension_requires_policy", "external_connector_requires_policy", "production_change_requires_review"}
	assert publish_result["matched_rules"] == ["publish_requires_approval"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "ncod"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "ncod_app_builder"
	assert registration["ui_components"]["builder"] == "/ncod/builder"
	assert "scpt" in registration["dependencies"]
	assert "ncod:build" in registration["permissions"]
