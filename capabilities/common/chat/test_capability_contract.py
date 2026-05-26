"""Regression coverage for the CHAT executable capability contract."""

from capabilities.common.chat import register_capability
from capabilities.common.chat.capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-chat", {"rooms": {"max_members_per_room": 1000}})

	assert contract["capability"] == "chat"
	assert contract["configuration"]["tenant_id"] == "tenant-chat"
	assert contract["configuration"]["rooms"]["max_members_per_room"] == 1000
	assert contract["configuration_schema"]["required"] == ["tenant_id", "rooms", "messaging", "moderation", "governance", "ui", "theme"]
	assert len(contract["rule_engine"]["rules"]) >= 6
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "rooms", "messages", "presence", "moderation", "retention", "analytics", "settings"}
	assert contract["ui"]["api_prefix"] == "/chat/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "message_thread" in contract["theme"]["components"]


def test_rule_engine_enforces_chat_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "create_room",
		"room_owner_assigned": False,
		"retention_policy_attached": False,
		"external_guest_present": True,
		"guest_policy_attached": False,
		"restricted_content_detected": True,
		"moderation_completed": False,
		"member_count": 8000,
		"access_review_recorded": False
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {
		"tenant_context_required",
		"room_requires_owner",
		"retention_policy_required",
		"external_guest_requires_policy",
		"restricted_content_requires_moderation",
		"large_room_requires_review"
	}


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "chat"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "chat_team_messaging"
	assert registration["ui_components"]["rooms"] == "/chat/rooms"
	assert "ntfy" in registration["dependencies"]
	assert "chat:send" in registration["permissions"]
