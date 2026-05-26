"""Regression coverage for the NTFY executable capability contract."""

from capabilities.common.ntfy import register_capability
from capabilities.common.ntfy.capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-notify", {"delivery": {"max_batch_size": 1000}})

	assert contract["capability"] == "ntfy"
	assert contract["configuration"]["tenant_id"] == "tenant-notify"
	assert contract["configuration"]["delivery"]["max_batch_size"] == 1000
	assert contract["configuration_schema"]["required"] == ["tenant_id", "channels", "delivery", "preferences", "governance", "ui", "theme"]
	assert len(contract["rule_engine"]["rules"]) >= 6
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "messages", "templates", "campaigns", "preferences", "channels", "analytics", "settings"}
	assert contract["ui"]["api_prefix"] == "/ntfy/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "channel_matrix" in contract["theme"]["components"]


def test_rule_engine_enforces_notification_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"message_class": "marketing",
		"recipient_opted_in": False,
		"operation": "send_campaign",
		"template_approved": False,
		"sensitive_payload": True,
		"payload_encrypted": False,
		"provider_health": "unhealthy",
		"delivery_requested": True,
		"recipient_count": 7000,
		"batch_review_recorded": False
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {
		"tenant_context_required",
		"recipient_opt_in_required",
		"approved_template_required",
		"sensitive_payload_requires_encryption",
		"provider_health_required",
		"large_batch_requires_review"
	}


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "ntfy"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "ntfy_notification_ops"
	assert registration["ui_components"]["campaigns"] == "/ntfy/campaigns"
	assert "mqeb" in registration["dependencies"]
	assert "ntfy:send" in registration["permissions"]
