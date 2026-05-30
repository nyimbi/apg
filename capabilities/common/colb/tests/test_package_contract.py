"""Collaboration Tools package runtime tests."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

from capabilities.capability_contract_registry import validate_contract_shape
from capabilities.common.colb import package_api, view_models
from capabilities.common.colb.collaboration_runtime import CollaborationRuntime


PACKAGE_DIR = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules[name] = module
	spec.loader.exec_module(module)
	return module


def test_package_contract_shape_and_entrypoint_are_publishable():
	contract_module = _load_module("colb_contract_runtime", PACKAGE_DIR / "capability_contract.py")
	app_module = _load_module("colb_app_runtime", PACKAGE_DIR / "app.py")
	contract = contract_module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	self_test = app_module.self_test()
	manifest = app_module.component_manifest()
	model = app_module.semantic_model()

	assert contract["capability"] == "colb"
	assert len(contract["ui"]["routes"]) >= 12
	assert len(contract["rule_engine"]["rules"]) >= 30
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["colb"]["streaming"]["engine"] == "bytewax"
	assert model["capabilities"]["colb"]["runtime"]["service"] == "collaboration_runtime.CollaborationRuntime"


def test_package_api_executes_collaboration_lifecycle():
	package_api.RUNTIME = CollaborationRuntime()
	workspace = package_api.create_workspace({"tenant_id": "tenant-api", "workspace_id": "api-workspace", "name": "API Workspace", "owner": "owner", "participants": ["owner", "member"], "retention_policy": "retain-180-days"})
	session = package_api.start_session({"tenant_id": "tenant-api", "session_id": "api-session", "workspace_id": workspace["id"], "owner": "owner"})
	package_api.join_session({"tenant_id": "tenant-api", "session_id": session["id"], "participant_id": "member"})
	presence = package_api.update_presence({"tenant_id": "tenant-api", "session_id": session["id"], "participant_id": "member", "status": "online"})
	artifact = package_api.share_artifact({"tenant_id": "tenant-api", "artifact_id": "api-artifact", "workspace_id": workspace["id"], "name": "API Artifact", "owner": "owner", "artifact_type": "document"})
	annotation = package_api.add_annotation({"tenant_id": "tenant-api", "annotation_id": "api-annotation", "artifact_id": artifact["id"], "author": "member", "body": "Looks good"})
	decision = package_api.record_decision({"tenant_id": "tenant-api", "decision_id": "api-decision", "annotation_id": annotation["id"], "owner": "owner", "decision": "Approved", "evidence": ["ticket"]})
	state = package_api.collaboration_state("tenant-api")

	assert workspace["status"] == "active"
	assert session["status"] == "active"
	assert presence["status"] == "online"
	assert decision["decision"] == "Approved"
	assert state["summary"]["workspace_count"] == 1
	assert state["audit_events"]


def test_view_models_match_runtime_state():
	runtime = CollaborationRuntime()
	runtime.create_workspace("tenant-view", "view-workspace", "View Workspace", "owner", ["owner", "member"], "retain-180-days")
	runtime.start_session("tenant-view", "view-session", "view-workspace", "owner")
	runtime.join_session("tenant-view", "view-session", "member")
	runtime.update_presence("tenant-view", "view-session", "member", "online")
	runtime.share_artifact("tenant-view", "view-artifact", "view-workspace", "Artifact", "owner", "document")
	runtime.add_annotation("tenant-view", "view-annotation", "view-artifact", "member", "Comment")
	runtime.record_decision("tenant-view", "view-decision", "view-annotation", "owner", "Approved", ["ticket"])

	dashboard = view_models.dashboard_model(runtime, "tenant-view")
	workspaces = view_models.workspace_model(runtime, "tenant-view")
	sessions = view_models.session_model(runtime, "tenant-view")
	artifacts = view_models.artifact_model(runtime, "tenant-view")
	agents = view_models.agent_model("tenant-view")
	analytics = view_models.analytics_model(runtime, "tenant-view")
	audit = view_models.audit_model(runtime, "tenant-view")
	settings = view_models.settings_model("tenant-view")

	assert dashboard["summary"]["workspace_count"] == 1
	assert workspaces["active"][0]["id"] == "view-workspace"
	assert sessions["presence"][0]["participant_id"] == "member"
	assert artifacts["decisions"][0]["id"] == "view-decision"
	assert agents["enabled"] is True
	assert analytics["artifact_density"] == 1.0
	assert audit["audit_events"]
	assert settings["theme"]["name"] == "colb_collaboration_workspace"
