"""Regression coverage for the WSBL executable capability contract."""

from capabilities.common.wsbl import register_capability
from capabilities.common.wsbl.capability_contract import (
	evaluate_capability_rules,
	get_capability_contract,
	streaming_manifest,
)


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-wsbl", {"sites": {"multi_locale_enabled": False}})

	assert contract["capability"] == "wsbl"
	assert contract["configuration"]["tenant_id"] == "tenant-wsbl"
	assert contract["configuration"]["sites"]["multi_locale_enabled"] is False
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"sites",
		"pages",
		"publishing",
		"wsbl_agents",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	]
	assert contract["requires"] == ["them", "auth", "ncod", "accs", "cons"]
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "sites", "pages", "editor", "components", "publishing", "analytics", "agents", "policy", "settings"}
	assert contract["theme"]["name"] == "wsbl_site_builder"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "wsbl_agents" in contract["provides"]


def test_rule_engine_enforces_wsbl_guardrails():
	result = evaluate_capability_rules({"tenant_context_present": False, "operation": "create_site", "site_owner_assigned": False})
	component_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "add_page_section", "custom_component_present": True, "component_review_recorded": False})
	publish_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "publish_site", "domain_validation_complete": False, "structured_sections_present": False, "preview_evidence_present": False, "approval_recorded": False, "event_stream": "local", "public_site": True, "accessibility_passed": False, "privacy_banner_required": True, "consent_policy_attached": False})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "site_requires_owner"}
	assert component_result["matched_rules"] == ["custom_component_requires_review"]
	assert set(publish_result["matched_rules"]) == {"domain_requires_validation_before_publish", "page_requires_structured_sections", "preview_requires_evidence", "publish_requires_approval", "publish_requires_bytewax_stream", "public_site_requires_accessibility_pass", "privacy_banner_requires_consent_policy"}


def test_agent_and_streaming_rules_are_exposed():
	agent_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "register_wsbl_agent", "agent_runtime_supported": False, "agent_role_supported": False})
	privileged_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "agent_publish_action", "privileged_scope": True, "human_approval_recorded": False})
	batch_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "batch_publish", "event_stream": "local"})

	assert streaming_manifest()["stream"] == "apg.wsbl.lifecycle"
	assert agent_result["decision"] == "deny"
	assert set(agent_result["matched_rules"]) == {"wsbl_agent_runtime_supported", "wsbl_agent_role_supported"}
	assert privileged_result["matched_rules"] == ["privileged_agent_publish_action_requires_human_approval"]
	assert batch_result["matched_rules"] == ["batch_publish_requires_bytewax"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "wsbl"
	assert "accs" in registration["dependencies"]
	assert registration["ui_components"]["editor"] == "/wsbl/editor"
	assert registration["ui_components"]["agents"] == "/wsbl/agents"
	assert registration["streaming"]["processor"] == "bytewax"
	assert "wsbl:publish" in registration["permissions"]
