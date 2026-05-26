"""Regression coverage for the MCHN executable capability contract."""

from capabilities.common.mchn import register_capability
from capabilities.common.mchn.capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-output", {"channels": {"fallback_required": False}})

	assert contract["capability"] == "mchn"
	assert contract["configuration"]["tenant_id"] == "tenant-output"
	assert contract["configuration"]["channels"]["fallback_required"] is False
	assert contract["configuration_schema"]["required"] == ["tenant_id", "channels", "rendering", "delivery", "governance", "ui", "theme"]
	assert contract["theme"]["name"] == "mchn_omnichannel_output"
	assert contract["ui"]["api_prefix"] == "/mchn/api/v1"


def test_rule_engine_enforces_multichannel_guardrails():
	result = evaluate_capability_rules({"tenant_context_present": False, "operation": "create_channel", "channel_owner_assigned": False, "sensitive_output": True, "output_encrypted": False, "channel_health": "unhealthy", "delivery_requested": True, "recipient_count": 20000, "delivery_review_recorded": False})
	template_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "publish_template", "template_approved": False})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "channel_requires_owner", "sensitive_output_requires_encryption", "unhealthy_channel_blocks_delivery", "large_delivery_requires_review"}
	assert template_result["matched_rules"] == ["template_requires_approval"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "mchn"
	assert "ntfy" in registration["dependencies"]
	assert registration["ui_components"]["render"] == "/mchn/render"
	assert "mchn:route" in registration["permissions"]
