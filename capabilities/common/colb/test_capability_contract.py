"""Regression coverage for the COLB executable capability contract."""

import pytest

from capabilities.common.colb import register_capability
from capabilities.common.colb.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.colb.collaboration_runtime import CollaborationRuntime
from capabilities.common.colb import view_models


def test_contract_exposes_configuration_rules_ui_theme_and_adapters():
	contract = get_capability_contract("tenant-collab", {"workspaces": {"max_participants_per_workspace": 250}})

	assert contract["capability"] == "colb"
	assert contract["configuration"]["tenant_id"] == "tenant-collab"
	assert contract["configuration"]["workspaces"]["max_participants_per_workspace"] == 250
	assert set(contract["configuration_schema"]["required"]) >= {
		"tenant_id",
		"workspaces",
		"sessions",
		"artifacts",
		"annotations",
		"presence",
		"protocols",
		"ai_agents",
		"security",
		"governance",
		"retention",
		"observability",
		"adapters",
		"ui",
		"theme",
	}
	assert len(contract["rule_engine"]["rules"]) >= 30
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "workspaces", "sessions", "presence", "artifacts", "annotations", "decisions", "agents", "protocols", "analytics", "audit", "settings"}
	assert contract["ui"]["api_prefix"] == "/colb/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "session_canvas" in contract["theme"]["components"]
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert contract["configuration"]["adapters"]["production_app"] == "production_app.py"
	assert "codex" in contract["configuration"]["ai_agents"]["supported_runtimes"]


def test_rule_engine_enforces_collaboration_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "share_artifact",
		"workspace_owner_assigned": False,
		"workspace_name_present": False,
		"participant_present": False,
		"retention_policy_attached": False,
		"external_participant_present": True,
		"external_policy_attached": False,
		"external_access_expiry_present": False,
		"participant_count": 1500,
		"membership_review_recorded": False,
		"realtime_session": True,
		"secure_transport": False,
		"protocol_health": "unhealthy",
		"event_bus_present": False,
		"artifact_policy_attached": False,
		"version_history_enabled": False,
		"external_share_requested": True,
		"dlp_check_completed": False,
		"state_change_requested": True,
		"audit_event_recorded": False,
		"ai_agent_participant": True,
		"agent_registered": False,
		"agent_scope_present": False,
		"ai_contribution_disclosed": False,
		"duplicate_artifact_id": True,
	})
	stream_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "batch_collaboration_mutation", "event_stream": "kafka"})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) >= {
		"tenant_context_required",
		"external_collaboration_requires_policy",
		"session_requires_secure_transport",
		"session_requires_protocol_health",
		"session_requires_event_bus",
		"artifact_policy_required",
		"artifact_requires_version_history",
		"external_artifact_requires_dlp",
		"collaboration_state_change_requires_audit",
		"ai_agent_requires_registration",
		"ai_agent_requires_scope",
		"ai_contribution_requires_disclosure",
		"duplicate_artifact_id_blocked",
		"large_workspace_requires_review",
	}
	assert stream_result["decision"] == "deny"
	assert "batch_collaboration_mutation_requires_bytewax" in stream_result["matched_rules"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "colb"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "colb_collaboration_workspace"
	assert registration["ui_components"]["workspaces"] == "/colb/workspaces"
	assert registration["ui_components"]["agents"] == "/colb/agents"
	assert "chat" in registration["dependencies"]
	assert registration["adapters"]["event_stream"] == "bytewax"
	assert "colb:collaborate" in registration["permissions"]
	assert "colb:audit" in registration["permissions"]


def _build_runtime() -> CollaborationRuntime:
	runtime = CollaborationRuntime()
	runtime.create_workspace("tenant-collab", "workspace-1", "Finance Close", "owner", ["owner", "analyst"], "retain-180-days")
	return runtime


def test_runtime_executes_workspace_session_artifact_decision_and_views():
	runtime = _build_runtime()
	session = runtime.start_session("tenant-collab", "session-1", "workspace-1", "owner")
	runtime.join_session("tenant-collab", "session-1", "analyst")
	presence = runtime.update_presence("tenant-collab", "session-1", "analyst", "online", {"row": 7})
	artifact = runtime.share_artifact("tenant-collab", "artifact-1", "workspace-1", "Close Checklist", "owner", "document")
	annotation = runtime.add_annotation("tenant-collab", "annotation-1", "artifact-1", "analyst", "Need controller approval")
	decision = runtime.record_decision("tenant-collab", "decision-1", "annotation-1", "owner", "Approved", ["approval-link"])
	dashboard = view_models.dashboard_model(runtime, "tenant-collab")

	assert session["status"] == "active"
	assert presence["cursor"] == {"row": 7}
	assert artifact["version"] == "v1"
	assert annotation["status"] == "open"
	assert decision["evidence"] == ["approval-link"]
	assert dashboard["summary"]["workspace_count"] == 1
	assert view_models.artifact_model(runtime, "tenant-collab")["decisions"][0]["id"] == "decision-1"
	assert view_models.agent_model("tenant-collab")["enabled"] is True
	assert view_models.analytics_model(runtime, "tenant-collab")["artifact_density"] == 1.0
	assert view_models.audit_model(runtime, "tenant-collab")["audit_events"]


def test_runtime_enforces_workspace_session_artifact_decision_and_agent_guardrails():
	runtime = _build_runtime()

	with pytest.raises(PermissionError, match="external_policy_required"):
		runtime.create_workspace("tenant-collab", "guest", "Guest", "owner", ["owner"], "retain-180-days", external_participants=["guest@example.com"], external_policy_attached=False)
	with pytest.raises(PermissionError, match="secure_transport_required"):
		runtime.start_session("tenant-collab", "insecure", "workspace-1", "owner", secure_transport=False)
	with pytest.raises(PermissionError, match="protocol_unhealthy"):
		runtime.start_session("tenant-collab", "unhealthy", "workspace-1", "owner", protocol_healthy=False)
	with pytest.raises(PermissionError, match="session_owner_not_workspace_member"):
		runtime.start_session("tenant-collab", "bad-owner", "workspace-1", "outsider")
	with pytest.raises(PermissionError, match="participant_not_workspace_member"):
		runtime.start_session("tenant-collab", "session-guard", "workspace-1", "owner")
		runtime.join_session("tenant-collab", "session-guard", "outsider")
	with pytest.raises(PermissionError, match="artifact_policy_required"):
		runtime.share_artifact("tenant-collab", "bad-artifact", "workspace-1", "Bad", "owner", "document", artifact_policy_attached=False)
	with pytest.raises(PermissionError, match="dlp_check_required"):
		runtime.share_artifact("tenant-collab", "external-artifact", "workspace-1", "External", "owner", "document", external_share=True, dlp_check_completed=False)
	with pytest.raises(PermissionError, match="duplicate_artifact_id"):
		runtime.share_artifact("tenant-collab", "dup-artifact", "workspace-1", "First", "owner", "document")
		runtime.share_artifact("tenant-collab", "dup-artifact", "workspace-1", "Second", "owner", "document")
	with pytest.raises(PermissionError, match="decision_evidence_required"):
		runtime.share_artifact("tenant-collab", "decision-artifact", "workspace-1", "Decision", "owner", "document")
		runtime.add_annotation("tenant-collab", "decision-annotation", "decision-artifact", "analyst", "Decide")
		runtime.record_decision("tenant-collab", "bad-decision", "decision-annotation", "owner", "Approved", [])

	agent_result = runtime.evaluate({"tenant_context_present": True, "ai_agent_participant": True, "agent_registered": False, "agent_scope_present": False, "ai_contribution_disclosed": False})
	assert agent_result["decision"] == "deny"
	assert "ai_agent_requires_registration" in agent_result["matched_rules"]


def test_runtime_routes_large_workspaces_to_review_before_activation():
	runtime = CollaborationRuntime()
	participants = [f"user-{index}" for index in range(1001)]
	workspace = runtime.create_workspace("tenant-collab", "large", "Large", "owner", participants, "retain-180-days", membership_review_recorded=False)
	approved = runtime.approve_workspace("tenant-collab", "large", "reviewer")

	assert workspace["status"] == "pending_review"
	assert workspace["review_status"] == "required"
	assert approved["status"] == "active"


def test_tenant_local_records_do_not_collide():
	runtime = CollaborationRuntime()
	runtime.create_workspace("tenant-alpha", "shared", "Alpha", "owner", ["owner"], "retain-180-days")
	runtime.create_workspace("tenant-beta", "shared", "Beta", "owner", ["owner"], "retain-180-days")
	alpha = runtime.share_artifact("tenant-alpha", "artifact", "shared", "Alpha Artifact", "owner", "document")
	beta = runtime.share_artifact("tenant-beta", "artifact", "shared", "Beta Artifact", "owner", "document")

	assert runtime.list_workspaces("tenant-alpha")[0]["name"] == "Alpha"
	assert runtime.list_workspaces("tenant-beta")[0]["name"] == "Beta"
	assert alpha["name"] == "Alpha Artifact"
	assert beta["name"] == "Beta Artifact"
