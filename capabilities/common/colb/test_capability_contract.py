"""Regression coverage for the COLB executable capability contract."""

from capabilities.common.colb import register_capability
from capabilities.common.colb.capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-collab", {"workspaces": {"max_participants_per_workspace": 250}})

	assert contract["capability"] == "colb"
	assert contract["configuration"]["tenant_id"] == "tenant-collab"
	assert contract["configuration"]["workspaces"]["max_participants_per_workspace"] == 250
	assert contract["configuration_schema"]["required"] == ["tenant_id", "workspaces", "sessions", "protocols", "governance", "ui", "theme"]
	assert len(contract["rule_engine"]["rules"]) >= 6
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "workspaces", "sessions", "presence", "annotations", "protocols", "analytics", "settings"}
	assert contract["ui"]["api_prefix"] == "/colb/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "session_canvas" in contract["theme"]["components"]


def test_rule_engine_enforces_collaboration_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "create_workspace",
		"workspace_owner_assigned": False,
		"external_participant_present": True,
		"external_policy_attached": False,
		"realtime_session": True,
		"secure_transport": False,
		"shared_artifact_present": True,
		"artifact_policy_attached": False,
		"participant_count": 1500,
		"membership_review_recorded": False
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {
		"tenant_context_required",
		"workspace_requires_owner",
		"external_collaboration_requires_policy",
		"secure_transport_required",
		"artifact_policy_required",
		"large_workspace_requires_review"
	}


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "colb"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "colb_collaboration_workspace"
	assert registration["ui_components"]["workspaces"] == "/colb/workspaces"
	assert "chat" in registration["dependencies"]
	assert "colb:collaborate" in registration["permissions"]
