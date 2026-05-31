"""Regression coverage for the CHAT executable capability contract."""

import pytest

from capabilities.common.chat import register_capability
from capabilities.common.chat.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.chat.service import ChatService
from capabilities.common.chat import views


def test_contract_exposes_configuration_rules_ui_theme_and_adapters():
	contract = get_capability_contract("tenant-chat", {"rooms": {"max_members_per_room": 1000}})
	overridden = get_capability_contract("tenant-chat", {
		"agents": {"adapter_contract": "custom_chat_agent_adapter"},
		"streaming": {"lifecycle_stream": "chat.custom"},
	})

	assert contract["capability"] == "chat"
	assert contract["configuration"]["tenant_id"] == "tenant-chat"
	assert contract["configuration"]["rooms"]["max_members_per_room"] == 1000
	assert set(contract["configuration_schema"]["required"]) >= {
		"tenant_id",
		"rooms",
		"messaging",
		"presence",
		"moderation",
		"ai_agents",
		"security",
		"governance",
		"retention",
		"observability",
		"agents",
		"streaming",
		"adapters",
		"ui",
		"theme",
	}
	assert len(contract["rule_engine"]["rules"]) >= 42
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "rooms", "direct", "messages", "presence", "agents", "lifecycle", "moderation", "retention", "audit", "analytics", "settings"}
	assert contract["ui"]["api_prefix"] == "/chat/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "message_thread" in contract["theme"]["components"]
	assert "chat_agent_roster" in contract["theme"]["components"]
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert "codex" in contract["configuration"]["ai_agents"]["supported_runtimes"]
	assert contract["agents"]["first_class"] is True
	assert "chat_steward" in contract["agents"]["supported_roles"]
	assert contract["agents"]["adapter_contract"] == "aicr_provider_neutral_chat_agent_adapter"
	assert contract["streaming"]["required_processor"] == "bytewax"
	assert "chat_agent_batch" in contract["streaming"]["required_operations"]
	assert overridden["agents"]["adapter_contract"] == "custom_chat_agent_adapter"
	assert overridden["streaming"]["lifecycle_stream"] == "chat.custom"


def test_rule_engine_enforces_chat_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "send_message",
		"room_owner_assigned": False,
		"room_name_present": False,
		"member_present": False,
		"retention_policy_attached": False,
		"external_guest_present": True,
		"guest_policy_attached": False,
		"guest_access_expiry_present": False,
		"room_active": False,
		"sender_authenticated": False,
		"sender_is_member": False,
		"message_payload_present": False,
		"message_length_within_limit": False,
		"restricted_content_detected": True,
		"moderation_completed": False,
		"attachment_present": True,
		"attachment_scan_completed": False,
		"external_share_requested": True,
		"dlp_check_completed": False,
		"delivery_requested": True,
		"event_bus_present": False,
		"state_change_requested": True,
		"audit_event_recorded": False,
		"ai_agent_participant": True,
		"agent_registered": False,
		"agent_scope_present": False,
		"ai_response_disclosed": False,
		"duplicate_message_id": True,
		"member_count": 8000,
		"access_review_recorded": False,
	})
	stream_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "batch_chat_mutation", "event_stream": "kafka"})
	agent_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "register_chat_agent",
		"agent_runtime_supported": False,
		"agent_role_supported": False,
		"scope_present": False,
		"owner_present": False,
		"purpose_present": False,
		"contribution_disclosed": False,
		"privileged_role": True,
		"human_approval_required": False,
	})
	lifecycle_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "validate_chat_lifecycle_batch", "event_stream": "kafka", "mutation_count": 1})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) >= {
		"tenant_context_required",
		"active_room_required",
		"sender_identity_required",
		"sender_membership_required",
		"message_requires_body_or_attachment",
		"message_length_within_limit",
		"restricted_content_requires_moderation",
		"attachment_requires_scan",
		"external_share_requires_dlp",
		"delivery_requires_event_bus",
		"message_requires_audit",
		"ai_agent_requires_registration",
		"ai_agent_requires_scope",
		"ai_response_requires_disclosure",
		"duplicate_message_id_blocked",
		"large_room_requires_review",
	}
	assert stream_result["decision"] == "deny"
	assert "batch_chat_mutation_requires_bytewax" in stream_result["matched_rules"]
	assert agent_result["decision"] == "deny"
	assert set(agent_result["matched_rules"]) >= {
		"chat_agent_runtime_supported",
		"chat_agent_role_supported",
		"chat_agent_requires_scope",
		"chat_agent_requires_owner",
		"chat_agent_requires_purpose",
		"chat_agent_requires_contribution_disclosure",
		"chat_agent_privileged_role_requires_human_approval",
	}
	assert lifecycle_result["decision"] == "deny"
	assert "bytewax_chat_stream_required" in lifecycle_result["matched_rules"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "chat"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "chat_team_messaging"
	assert registration["ui_components"]["rooms"] == "/chat/rooms"
	assert registration["ui_components"]["agents"] == "/chat/agents"
	assert registration["ui_components"]["lifecycle"] == "/chat/lifecycle"
	assert "ntfy" in registration["dependencies"]
	assert registration["adapters"]["event_stream"] == "bytewax"
	assert registration["agents"]["first_class"] is True
	assert registration["streaming"]["required_processor"] == "bytewax"
	assert "chat:send" in registration["permissions"]
	assert "chat:audit" in registration["permissions"]


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
	agent = service.register_chat_agent(
		agent_id="agent-steward",
		tenant_id="tenant-chat",
		name="Chat Steward",
		runtime="codex",
		role="chat_steward",
		scope="room:ops-room",
		owner="room-owner",
		purpose="review governed chat lifecycle",
		human_approval_required=True,
	)
	batch = service.validate_chat_lifecycle_batch(
		tenant_id="tenant-chat",
		event_stream="bytewax",
		mutation_count=2,
		operation="chat_agent_batch",
		batch_id="batch-agent",
	)
	model = views.dashboard_model(service, "tenant-chat")

	assert room["status"] == "active"
	assert room["retention_policy"] == "retain-90-days"
	assert message["status"] == "delivered"
	assert len(message["fingerprint"]) == 64
	assert message["delivery_receipts"] == ["room-owner"]
	assert presence["typing"] is True
	assert agent["runtime"] == "codex"
	assert agent["status"] == "active"
	assert batch["status"] == "accepted"
	assert model["summary"]["room_count"] == 1
	assert model["summary"]["message_count"] == 1
	assert model["summary"]["presence_count"] == 1
	assert model["summary"]["chat_agent_count"] == 1
	assert model["summary"]["lifecycle_batch_count"] == 1
	assert views.chat_agent_roster_model(service, "tenant-chat")["active"][0]["id"] == "agent-steward"
	assert views.lifecycle_batch_model(service, "tenant-chat")["accepted"][0]["id"] == "batch-agent"
	assert views.lifecycle_batch_model(service, "tenant-chat")["required_processor"] == "bytewax"
	assert model["summary"]["audit_event_count"] >= 2
	assert views.analytics_model(service, "tenant-chat")["attachment_rate"] == 1.0
	assert views.agent_participant_model("tenant-chat")["enabled"] is True
	assert views.audit_model(service, "tenant-chat")["audit_events"]
	assert views.settings_model("tenant-chat")["theme"]["name"] == "chat_team_messaging"


def test_service_enforces_room_message_agent_and_moderation_guardrails():
	service = ChatService()

	with pytest.raises(PermissionError, match="room_owner_required"):
		service.create_room("missing-owner", "tenant-chat", "Missing Owner", "", ["member"], "retain-30-days")
	with pytest.raises(PermissionError, match="retention_policy_required"):
		service.create_room("missing-retention", "tenant-chat", "Missing Retention", "room-owner", ["member"], "")
	with pytest.raises(PermissionError, match="guest_policy_required"):
		service.create_room("guest-without-policy", "tenant-chat", "Guest Room", "room-owner", ["member"], "retain-30-days", external_guests=["guest@example.com"], guest_policy_attached=False)

	service.create_room("moderated-room", "tenant-chat", "Moderated", "room-owner", ["room-owner", "member"], "retain-30-days")

	with pytest.raises(PermissionError, match="moderation_required"):
		service.send_message("blocked-message", "tenant-chat", "moderated-room", "member", "contains restricted credential", moderation_completed=False)
	with pytest.raises(PermissionError, match="attachment_scan_required"):
		service.send_message("bad-attachment", "tenant-chat", "moderated-room", "member", "file attached", attachments=["payload.zip"], attachment_scan_completed=False)
	with pytest.raises(PermissionError, match="ai_agent_registration_required"):
		service.send_message("bad-agent", "tenant-chat", "moderated-room", "member", "agent says hello", ai_agent_participant=True, agent_registered=False)
	with pytest.raises(PermissionError, match="duplicate_message_id"):
		service.send_message("unique", "tenant-chat", "moderated-room", "member", "first")
		service.send_message("unique", "tenant-chat", "moderated-room", "member", "second")
	with pytest.raises(PermissionError, match="typing_room_membership_required"):
		service.update_presence("tenant-chat", "outsider", "online", room_id="moderated-room", typing=True)
	with pytest.raises(PermissionError, match="unsupported_chat_agent_runtime"):
		service.register_chat_agent("agent-unsupported", "tenant-chat", "Unsupported", "kafka_agent", "room_reviewer", "room:*", "ops", "review rooms")
	with pytest.raises(PermissionError, match="chat_agent_contribution_disclosure_required"):
		service.register_chat_agent("agent-undisclosed", "tenant-chat", "Undisclosed", "codex", "room_reviewer", "room:*", "ops", "review rooms", contribution_disclosed=False)
	pending = service.register_chat_agent("agent-pending", "tenant-chat", "Pending", "codex", "chat_steward", "room:*", "ops", "review rooms")
	with pytest.raises(ValueError, match="chat_lifecycle_batch_empty"):
		service.validate_chat_lifecycle_batch("tenant-chat", "bytewax", 0)
	with pytest.raises(ValueError, match="unsupported_chat_lifecycle_operation"):
		service.validate_chat_lifecycle_batch("tenant-chat", "bytewax", 1, "unknown_batch")
	with pytest.raises(PermissionError, match="bytewax_lifecycle_stream_required"):
		service.validate_chat_lifecycle_batch("tenant-chat", "kafka", 1, "chat_agent_batch")

	reviewed = service.review_moderation("mod:000001", reviewer="moderator", decision="rejected", tenant_id="tenant-chat")
	approved_message = service.send_message("approved-message", "tenant-chat", "moderated-room", "member", "contains restricted credential", moderation_completed=True)

	assert pending["status"] == "pending_review"
	assert reviewed["status"] == "rejected"
	assert reviewed["reason"] == "moderation_required"
	assert approved_message["moderation_status"] == "approved"


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
	approved = service.approve_room("large-room", reviewer="access-reviewer", tenant_id="tenant-chat")

	assert room["status"] == "pending_review"
	assert room["review_status"] == "required"
	assert service.list_moderation_items("tenant-chat")[0]["reason"] == "large_room_review_required"
	assert approved["status"] == "active"


def test_tenant_local_chat_records_do_not_collide():
	service = ChatService()
	service.create_room("shared-room", "tenant-alpha", "Alpha", "owner", ["owner"], "retain-30-days")
	service.create_room("shared-room", "tenant-beta", "Beta", "owner", ["owner"], "retain-30-days")
	alpha_message = service.send_message("shared-message", "tenant-alpha", "shared-room", "owner", "alpha")
	beta_message = service.send_message("shared-message", "tenant-beta", "shared-room", "owner", "beta")

	assert service.list_rooms("tenant-alpha")[0]["name"] == "Alpha"
	assert service.list_rooms("tenant-beta")[0]["name"] == "Beta"
	assert alpha_message["body"] == "alpha"
	assert beta_message["body"] == "beta"
