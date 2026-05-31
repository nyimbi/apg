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
	assert contract["agents"]["first_class"] is True
	assert contract["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert "video_meeting_steward" in contract["agents"]["privileged_roles"]
	assert contract["streaming"]["required_processor"] == "bytewax"
	assert contract["streaming"]["lifecycle_stream"] == "vidc.lifecycle"
	assert len(contract["rule_engine"]["rules"]) >= 35
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "meetings", "rooms", "participants", "recordings", "captions", "agents", "lifecycle", "analytics", "audit", "settings"}
	assert contract["ui"]["api_prefix"] == "/vidc/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "recording_library" in contract["theme"]["components"]
	assert "video_agent_roster" in contract["theme"]["components"]
	assert "bytewax_lifecycle_panel" in contract["theme"]["components"]


def test_rule_engine_enforces_video_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "register_video_agent",
		"agent_runtime_supported": False,
		"agent_role_supported": False,
		"scope_present": False,
		"owner_present": False,
		"purpose_present": False,
		"contribution_disclosed": False,
		"privileged_role": True,
		"human_approval_required": False,
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) >= {
		"video_agent_runtime_supported",
		"video_agent_role_supported",
		"video_agent_requires_scope",
		"video_agent_requires_owner",
		"video_agent_requires_purpose",
		"video_agent_requires_contribution_disclosure",
		"video_agent_privileged_role_requires_human_approval",
	}

	lifecycle = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "batch_video_mutation",
		"event_stream": "legacy_bus",
	})

	assert lifecycle["decision"] == "deny"
	assert lifecycle["matched_rules"] == ["batch_video_mutation_requires_bytewax"]

	batch = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "validate_vidc_lifecycle_batch",
		"event_stream": "legacy_bus",
		"mutation_count": 0,
		"lifecycle_operation_supported": False,
	})
	assert batch["decision"] == "deny"
	assert set(batch["matched_rules"]) == {
		"vidc_lifecycle_batch_requires_mutations",
		"vidc_lifecycle_operation_supported",
		"bytewax_vidc_stream_required",
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
	assert registration["ui_components"]["lifecycle"] == "/vidc/lifecycle"
	assert registration["agents"]["first_class"] is True
	assert registration["streaming"]["required_processor"] == "bytewax"
	assert "colb" in registration["dependencies"]
	assert "vidc:join" in registration["permissions"]
