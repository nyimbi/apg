"""Regression coverage for the PLGN executable capability contract."""

from capabilities.common.plgn import register_capability
from capabilities.common.plgn.capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-plgn", {"marketplace": {"tenant_install_policy_enabled": False}})

	assert contract["capability"] == "plgn"
	assert contract["configuration"]["tenant_id"] == "tenant-plgn"
	assert contract["configuration"]["marketplace"]["tenant_install_policy_enabled"] is False
	assert contract["configuration_schema"]["required"] == ["tenant_id", "marketplace", "plugins", "security", "governance", "ui", "theme"]
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "marketplace", "plugins", "manifests", "permissions", "sandbox", "releases", "settings"}
	assert contract["theme"]["name"] == "plgn_extension_marketplace"


def test_rule_engine_enforces_plgn_guardrails():
	result = evaluate_capability_rules({"tenant_context_present": False, "operation": "register_plugin", "plugin_owner_assigned": False, "signature_verified": False, "permissions_requested": True, "permission_review_recorded": False, "external_plugin": True, "external_review_recorded": False})
	enable_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "enable_plugin", "signature_verified": True, "sandbox_policy_attached": False})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "plugin_requires_owner", "plugin_requires_signature", "permissions_require_review", "external_plugin_requires_review"}
	assert enable_result["matched_rules"] == ["plugin_requires_sandbox"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "plgn"
	assert "secu" in registration["dependencies"]
	assert registration["ui_components"]["marketplace"] == "/plgn/marketplace"
	assert "plgn:install" in registration["permissions"]
