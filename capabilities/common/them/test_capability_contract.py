"""Regression coverage for the THEM executable capability contract."""

from capabilities.common.them import register_capability
from capabilities.common.them.capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-them", {"governance": {"large_rollout_review_threshold": 8}})

	assert contract["capability"] == "them"
	assert contract["configuration"]["tenant_id"] == "tenant-them"
	assert contract["configuration"]["governance"]["large_rollout_review_threshold"] == 8
	assert contract["configuration_schema"]["required"] == ["tenant_id", "themes", "tokens", "branding", "governance", "ui", "theme"]
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "themes", "tokens", "branding", "assets", "preview", "policies", "settings"}
	assert contract["theme"]["name"] == "them_brand_system"


def test_rule_engine_enforces_them_guardrails():
	result = evaluate_capability_rules({"tenant_context_present": False, "operation": "create_theme", "theme_owner_assigned": False, "brand_asset_present": True, "license_verified": False, "target_tenant_count": 7, "rollout_review_recorded": False})
	publish_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "publish_theme", "approval_recorded": False, "accessibility_contrast_passed": False})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "theme_requires_owner", "brand_asset_requires_license", "large_rollout_requires_review"}
	assert publish_result["matched_rules"] == ["publish_requires_approval", "accessible_contrast_required"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "them"
	assert "i18n" in registration["dependencies"]
	assert registration["ui_components"]["tokens"] == "/them/tokens"
	assert "them:publish" in registration["permissions"]
