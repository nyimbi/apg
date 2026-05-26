"""Regression coverage for the MQEB executable capability contract."""

from capabilities.common.mqeb import register_capability
from capabilities.common.mqeb.capability_contract import (
	evaluate_capability_rules,
	get_capability_contract
)


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract(
		"tenant-events",
		{"broker": {"max_topics_per_tenant": 250}}
	)

	assert contract["capability"] == "mqeb"
	assert contract["configuration"]["tenant_id"] == "tenant-events"
	assert contract["configuration"]["broker"]["max_topics_per_tenant"] == 250
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"broker",
		"delivery",
		"routing",
		"security",
		"compliance",
		"scaling",
		"ui",
		"theme"
	]
	assert len(contract["rule_engine"]["rules"]) >= 6
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"dashboard",
		"topics",
		"publish",
		"subscriptions",
		"routing",
		"scaling",
		"monitoring",
		"settings"
	}
	assert contract["ui"]["api_prefix"] == "/mqeb/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "message_flow_map" in contract["theme"]["components"]


def test_rule_engine_enforces_message_governance_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "publish",
		"topic_exists": False,
		"topic_classification": "restricted",
		"message_encrypted": False,
		"cross_tenant_publish": True,
		"delivery_mode": "exactly_once",
		"dead_letter_queue_configured": False,
		"priority_messages_per_minute": 15000,
		"quota_exception_recorded": False
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {
		"tenant_context_required",
		"publish_requires_topic",
		"restricted_topic_requires_encryption",
		"cross_tenant_publish_denied",
		"guaranteed_delivery_requires_dead_letter_queue",
		"priority_quota_exhaustion_requires_review"
	}


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "mqeb_event_fabric"
	assert registration["ui_components"]["routing"] == "/mqeb/routing"
	assert "auth_rbac" in registration["dependencies"]
