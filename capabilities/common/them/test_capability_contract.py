"""Regression coverage for the THEM executable capability contract."""

from capabilities.common.them import register_capability
from capabilities.common.them.capability_contract import (
	evaluate_capability_rules,
	get_capability_contract,
	streaming_manifest,
)


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-them", {"governance": {"large_rollout_review_threshold": 8}})

	assert contract["capability"] == "them"
	assert contract["configuration"]["tenant_id"] == "tenant-them"
	assert contract["configuration"]["governance"]["large_rollout_review_threshold"] == 8
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"themes",
		"tokens",
		"branding",
		"them_agents",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	]
	assert contract["requires"] == ["conf", "auth", "i18n", "audl", "accs"]
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "themes", "tokens", "branding", "assets", "preview", "agents", "policies", "settings"}
	assert contract["theme"]["name"] == "them_brand_system"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "them_agents" in contract["provides"]


def test_rule_engine_enforces_them_guardrails():
	create_result = evaluate_capability_rules({"tenant_context_present": False, "operation": "create_theme", "theme_owner_assigned": False, "brand_guidelines_present": False})
	asset_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "add_brand_asset", "brand_asset_present": True, "license_verified": False, "asset_approval_recorded": False})
	publish_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "publish_theme", "approval_recorded": False, "accessibility_contrast_passed": False, "event_stream": "local"})
	batch_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "batch_theme_rollout", "event_stream": "local", "target_tenant_count": 7, "rollout_review_recorded": False})

	assert create_result["decision"] == "deny"
	assert set(create_result["matched_rules"]) == {"tenant_context_required", "theme_requires_owner", "theme_requires_guidelines"}
	assert set(asset_result["matched_rules"]) == {"brand_asset_requires_license", "brand_asset_requires_approval"}
	assert publish_result["matched_rules"] == ["publish_requires_approval", "accessible_contrast_required", "publish_requires_bytewax_stream"]
	assert set(batch_result["matched_rules"]) == {"large_rollout_requires_review", "batch_theme_rollout_requires_bytewax"}


def test_agent_and_streaming_rules_are_exposed():
	agent_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "register_them_agent", "agent_runtime_supported": False, "agent_role_supported": False})
	privileged_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "agent_theme_action", "privileged_scope": True, "human_approval_recorded": False})

	assert streaming_manifest()["stream"] == "apg.them.lifecycle"
	assert agent_result["decision"] == "deny"
	assert set(agent_result["matched_rules"]) == {"them_agent_runtime_supported", "them_agent_role_supported"}
	assert privileged_result["matched_rules"] == ["privileged_agent_theme_action_requires_human_approval"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "them"
	assert "audl" in registration["dependencies"]
	assert registration["ui_components"]["tokens"] == "/them/tokens"
	assert registration["ui_components"]["agents"] == "/them/agents"
	assert registration["streaming"]["processor"] == "bytewax"
	assert "them:publish" in registration["permissions"]
