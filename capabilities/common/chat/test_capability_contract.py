"""Regression coverage for the CHAT executable capability contract."""

import pytest

from capabilities.common.chat import register_capability
from capabilities.common.chat.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.chat.service import ChatService
from capabilities.common.chat.views import dashboard_model


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


def test_service_creates_rooms_messages_presence_and_dashboard_state():
	service = ChatService()
	room = service.create_room(
		room_id="ops-room",
		tenant_id="tenant-chat",
		name="Operations",
		owner="room-owner",
		members=["room-owner", "operator"],
		retention_policy="retain-90-days",
		visibility="private",
	)
	message = service.send_message(
		message_id="msg-1",
		tenant_id="tenant-chat",
		room_id="ops-room",
		sender="operator",
		body="handover complete",
		attachments=["handover.txt"],
		delivery_receipts=["room-owner"],
	)
	presence = service.update_presence(
		tenant_id="tenant-chat",
		user_id="operator",
		status="online",
		room_id="ops-room",
		typing=True,
	)
	model = dashboard_model(service, "tenant-chat")

	assert room["status"] == "active"
	assert room["retention_policy"] == "retain-90-days"
	assert message["status"] == "delivered"
	assert len(message["fingerprint"]) == 64
	assert message["delivery_receipts"] == ["room-owner"]
	assert presence["typing"] is True
	assert model["summary"]["room_count"] == 1
	assert model["summary"]["message_count"] == 1
	assert model["summary"]["presence_count"] == 1
	assert model["summary"]["audit_event_count"] >= 2


def test_service_enforces_room_message_and_moderation_guardrails():
	service = ChatService()

	with pytest.raises(PermissionError, match="room_owner_required"):
		service.create_room(
			room_id="missing-owner",
			tenant_id="tenant-chat",
			name="Missing Owner",
			owner="",
			members=["member"],
			retention_policy="retain-30-days",
		)

	with pytest.raises(PermissionError, match="retention_policy_required"):
		service.create_room(
			room_id="missing-retention",
			tenant_id="tenant-chat",
			name="Missing Retention",
			owner="room-owner",
			members=["member"],
			retention_policy="",
		)

	with pytest.raises(PermissionError, match="guest_policy_required"):
		service.create_room(
			room_id="guest-without-policy",
			tenant_id="tenant-chat",
			name="Guest Room",
			owner="room-owner",
			members=["member"],
			retention_policy="retain-30-days",
			external_guests=["guest@example.com"],
			guest_policy_attached=False,
		)

	service.create_room(
		room_id="moderated-room",
		tenant_id="tenant-chat",
		name="Moderated",
		owner="room-owner",
		members=["room-owner", "member"],
		retention_policy="retain-30-days",
	)

	with pytest.raises(PermissionError, match="moderation_required"):
		service.send_message(
			message_id="blocked-message",
			tenant_id="tenant-chat",
			room_id="moderated-room",
			sender="member",
			body="contains restricted credential",
			moderation_completed=False,
		)

	reviewed = service.review_moderation("mod:000001", reviewer="moderator", decision="rejected")
	approved_message = service.send_message(
		message_id="approved-message",
		tenant_id="tenant-chat",
		room_id="moderated-room",
		sender="member",
		body="contains restricted credential",
		moderation_completed=True,
	)

	assert reviewed["status"] == "rejected"
	assert reviewed["reason"] == "moderation_required"
	assert approved_message["moderation_status"] == "approved"
	assert service.conversation_summary("tenant-chat")["moderation_queue_count"] == 0


def test_service_routes_large_rooms_to_review_before_activation():
	service = ChatService()
	members = [f"member-{index}" for index in range(5001)]
	room = service.create_room(
		room_id="large-room",
		tenant_id="tenant-chat",
		name="Large Community",
		owner="room-owner",
		members=members,
		retention_policy="retain-30-days",
		access_review_recorded=False,
	)
	approved = service.approve_room("large-room", reviewer="access-reviewer")

	assert room["status"] == "pending_review"
	assert room["review_status"] == "required"
	assert service.list_moderation_items("tenant-chat")[0]["reason"] == "large_room_review_required"
	assert approved["status"] == "active"
