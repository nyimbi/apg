"""Regression coverage for the VIDC executable capability contract."""

from capabilities.common.vidc import register_capability
from capabilities.common.vidc.capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-video", {"meetings": {"max_participants": 250}})

	assert contract["capability"] == "vidc"
	assert contract["configuration"]["tenant_id"] == "tenant-video"
	assert contract["configuration"]["meetings"]["max_participants"] == 250
	assert set(contract["configuration_schema"]["required"]) >= {"tenant_id", "meetings", "media", "recordings", "meeting_agents", "governance", "observability", "adapters", "ui", "theme"}
	assert contract["configuration"]["governance"]["batch_event_stream"] == "bytewax"
	assert contract["configuration"]["meeting_agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert len(contract["rule_engine"]["rules"]) >= 25
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "meetings", "rooms", "participants", "recordings", "captions", "agents", "analytics", "audit", "settings"}
	assert contract["ui"]["api_prefix"] == "/vidc/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "recording_library" in contract["theme"]["components"]
	assert "agent_panel" in contract["theme"]["components"]


def test_rule_engine_enforces_video_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "start_meeting",
		"host_present": False,
		"external_guest_present": True,
		"guest_policy_attached": False,
		"waiting_room_enabled": False,
		"secure_transport": False,
		"screen_share_requested": True,
		"screen_share_policy_attached": False,
		"recording_requested": True,
		"recording_consent_recorded": False,
		"recording_encrypted": False,
		"recording_retention_policy_attached": False,
		"recording_access_audit_enabled": False,
		"participant_count": 800,
		"capacity_review_recorded": False,
		"meeting_agent_present": True,
		"agent_registered": False,
		"agent_runtime_supported": False,
		"agent_scope_present": False,
		"agent_contribution_disclosed": False,
		"operation": "batch_video_mutation",
		"event_stream": "legacy_bus",
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) >= {
		"tenant_context_required",
		"external_guest_requires_policy",
		"external_guest_requires_waiting_room",
		"recording_requires_consent",
		"recording_requires_encryption",
		"recording_requires_retention",
		"recording_requires_access_audit",
		"large_meeting_requires_review",
		"meeting_agent_requires_registration",
		"meeting_agent_runtime_supported",
		"meeting_agent_requires_scope",
		"meeting_agent_requires_disclosure",
		"batch_video_mutation_requires_bytewax",
	}


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "vidc"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "vidc_meeting_room"
	assert registration["ui_components"]["meetings"] == "/vidc/meetings"
	assert registration["ui_components"]["agents"] == "/vidc/agents"
	assert "colb" in registration["dependencies"]
	assert "vidc:join" in registration["permissions"]
