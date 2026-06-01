"""Executable Intelligence Fusion capability package tests."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

import pytest

from capabilities.capability_contract_registry import validate_contract_shape


PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
	sys.path.insert(0, str(PACKAGE_DIR))


def _load_module(name: str, path: Path):
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules[name] = module
	spec.loader.exec_module(module)
	return module


def test_contract_shape_streaming_routes_and_agents_are_valid():
	module = _load_module("contract_intel_fusion", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "intel_fusion"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "fusion_agent_workflow" in contract["provides"]
	assert "/intel-fusion/agents" in [route["path"] for route in contract["ui"]["routes"]]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert contract["configuration"]["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]


def test_rule_engine_blocks_missing_context_non_bytewax_and_prohibited_agent_actions():
	module = _load_module("rules_intel_fusion", PACKAGE_DIR / "capability_contract.py")

	assert module.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "fusion_batch", "event_stream": "queue"})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "fusion_agent_action", "privileged_scope": True, "human_approval_recorded": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "fusion_agent_action", "evidence_fabrication_scope": True})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "fusion_agent_action", "source_tampering_scope": True})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "fusion_agent_action", "privacy_bypass_scope": True})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "fusion_agent_action", "unsupported_identity_resolution_scope": True})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "fusion_agent_action", "autonomous_dissemination_scope": True})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "fusion_agent_action", "unapproved_attribution_scope": True})["decision"] == "deny"


def test_service_executes_fusion_lifecycle():
	service_module = _load_module("service_intel_fusion", PACKAGE_DIR / "service.py")
	service = service_module.IntelligenceFusionService()

	authority = service.record_authority("auth-1", "tenant-test", "mission_order", "scope-ref", "confidential", "approver-1", "2026-12-31", "authority-evidence")
	workspace = service.record_workspace("workspace-1", "tenant-test", "case_fusion", "Case Fusion", "confidential", authority["id"], "workspace-evidence")
	source = service.register_source("source-1", "tenant-test", "osint", "source-ref", "custodian-1", authority["id"], "lineage-ref", "source-evidence")
	artifact = service.record_artifact("artifact-1", "tenant-test", workspace["id"], source["id"], "report", "artifact-ref", "sha256:abc", 0.82, "artifact-evidence")
	correlation = service.record_correlation("correlation-1", "tenant-test", artifact["id"], "cross_source_confirmation", 0.86, "analyst-1", "correlation-evidence")
	hypothesis = service.record_hypothesis("hypothesis-1", "tenant-test", correlation["id"], "risk", "claim-ref", 0.77, "analyst-1", "hypothesis-evidence")
	assessment = service.record_assessment("assessment-1", "tenant-test", hypothesis["id"], "threat", "high", 0.81, "analyst-1", "assessment-evidence")
	referral = service.record_referral("referral-1", "tenant-test", assessment["id"], "incident_response", "response-team", "approval-ref", "referral-evidence")
	dissemination = service.record_dissemination("dissemination-1", "tenant-test", assessment["id"], "command-team", "CONFIDENTIAL", "approval-ref", "dissemination-evidence")
	review = service.record_review("review-1", "tenant-test", dissemination["id"], "reviewer-1", "approved", "review-evidence")
	agent = service.register_fusion_agent("agent-1", "tenant-test", "Fusion Agent", "codex", "correlation_analyst", "correlation support")
	bytewax_batch = service.validate_batch("tenant-test", 3)
	summary = service.dashboard_summary("tenant-test")

	assert authority["authority_type"] == "mission_order"
	assert workspace["workspace_type"] == "case_fusion"
	assert source["source_type"] == "osint"
	assert artifact["artifact_type"] == "report"
	assert correlation["correlation_type"] == "cross_source_confirmation"
	assert hypothesis["hypothesis_type"] == "risk"
	assert assessment["risk_level"] == "high"
	assert referral["referral_type"] == "incident_response"
	assert dissemination["release_marking"] == "CONFIDENTIAL"
	assert review["status"] == "approved"
	assert agent["runtime"] == "codex"
	assert bytewax_batch["processor"] == "bytewax"
	assert summary["audit_event_count"] == 11


def test_service_keys_state_by_tenant_and_record_id():
	service_module = _load_module("tenant_service_intel_fusion", PACKAGE_DIR / "service.py")
	service = service_module.IntelligenceFusionService()

	tenant_a = service.record_authority("shared-auth", "tenant-a", "mission_order", "scope-a", "confidential", "approver-a", "2026-12-31", "evidence-a")
	tenant_b = service.record_authority("shared-auth", "tenant-b", "consent", "scope-b", "unclassified", "approver-b", "2026-12-31", "evidence-b")
	service.record_workspace("shared-workspace", "tenant-a", "case_fusion", "Workspace A", "confidential", tenant_a["id"], "evidence-a")
	service.record_workspace("shared-workspace", "tenant-b", "public_safety", "Workspace B", "unclassified", tenant_b["id"], "evidence-b")

	dashboard_a = service.dashboard_summary("tenant-a")
	dashboard_b = service.dashboard_summary("tenant-b")

	assert dashboard_a["authority_count"] == 1
	assert dashboard_b["authority_count"] == 1
	assert dashboard_a["workspace_count"] == 1
	assert dashboard_b["workspace_count"] == 1
	assert service._tenant_workspace_or_none("shared-workspace", "tenant-a").name == "Workspace A"
	assert service._tenant_workspace_or_none("shared-workspace", "tenant-b").name == "Workspace B"


def test_service_guardrails_reject_invalid_fusion_actions():
	service_module = _load_module("guardrail_service_intel_fusion", PACKAGE_DIR / "service.py")
	service = service_module.IntelligenceFusionService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.record_authority("auth", "", "mission_order", "scope", "confidential", "approver", "2026-12-31", "evidence")
	with pytest.raises(PermissionError, match="authority_type_not_supported"):
		service.record_authority("auth", "tenant-test", "unknown", "scope", "confidential", "approver", "2026-12-31", "evidence")
	authority = service.record_authority("auth-ok", "tenant-test", "mission_order", "scope", "confidential", "approver", "2026-12-31", "evidence")
	with pytest.raises(PermissionError, match="lawful_authority_required"):
		service.record_workspace("workspace", "tenant-test", "case_fusion", "workspace", "confidential", "missing-auth", "evidence")
	workspace = service.record_workspace("workspace-ok", "tenant-test", "case_fusion", "workspace", "confidential", authority["id"], "evidence")
	with pytest.raises(PermissionError, match="source_lineage_required"):
		service.register_source("source", "tenant-test", "osint", "ref", "custodian", authority["id"], "", "evidence")
	source = service.register_source("source-ok", "tenant-test", "osint", "ref", "custodian", authority["id"], "lineage", "evidence")
	other_authority = service.record_authority("auth-other", "tenant-test", "consent", "scope", "confidential", "approver", "2026-12-31", "evidence")
	other_workspace = service.record_workspace("workspace-other", "tenant-test", "case_fusion", "other", "confidential", other_authority["id"], "evidence")
	with pytest.raises(PermissionError, match="authority_mismatch"):
		service.record_artifact("artifact", "tenant-test", other_workspace["id"], source["id"], "report", "ref", "hash", 0.8, "evidence")
	with pytest.raises(PermissionError, match="confidence_score_invalid"):
		service.record_artifact("artifact", "tenant-test", workspace["id"], source["id"], "report", "ref", "hash", 1.8, "evidence")
	artifact = service.record_artifact("artifact-ok", "tenant-test", workspace["id"], source["id"], "report", "ref", "hash", 0.8, "evidence")
	with pytest.raises(PermissionError, match="correlation_type_not_supported"):
		service.record_correlation("correlation", "tenant-test", artifact["id"], "unknown", 0.8, "analyst", "evidence")
	correlation = service.record_correlation("correlation-ok", "tenant-test", artifact["id"], "entity_match", 0.8, "analyst", "evidence")
	with pytest.raises(PermissionError, match="hypothesis_type_not_supported"):
		service.record_hypothesis("hypothesis", "tenant-test", correlation["id"], "unknown", "claim", 0.8, "analyst", "evidence")
	hypothesis = service.record_hypothesis("hypothesis-ok", "tenant-test", correlation["id"], "risk", "claim", 0.8, "analyst", "evidence")
	with pytest.raises(PermissionError, match="risk_level_not_supported"):
		service.record_assessment("assessment", "tenant-test", hypothesis["id"], "threat", "unknown", 0.8, "analyst", "evidence")
	assessment = service.record_assessment("assessment-ok", "tenant-test", hypothesis["id"], "threat", "medium", 0.8, "analyst", "evidence")
	with pytest.raises(PermissionError, match="referral_approval_required"):
		service.record_referral("referral", "tenant-test", assessment["id"], "incident_response", "team", "", "evidence")
	with pytest.raises(PermissionError, match="dissemination_approval_required"):
		service.record_dissemination("dissemination", "tenant-test", assessment["id"], "team", "CONFIDENTIAL", "", "evidence")
	with pytest.raises(PermissionError, match="reviewer_required"):
		service.record_review("review", "tenant-test", assessment["id"], "", "approved", "evidence")
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.validate_batch("tenant-test", 1, event_stream="queue")
	with pytest.raises(PermissionError, match="fusion_agent_runtime_not_supported"):
		service.register_fusion_agent("agent", "tenant-test", "Bad Agent", "unsupported", "correlation_analyst", "scope")
	with pytest.raises(PermissionError, match="human_approval_required"):
		service.validate_agent_action("tenant-test", privileged_scope=True, human_approval_recorded=False)
	with pytest.raises(PermissionError, match="evidence_fabrication_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, evidence_fabrication_scope=True)
	with pytest.raises(PermissionError, match="source_tampering_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, source_tampering_scope=True)
	with pytest.raises(PermissionError, match="privacy_bypass_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, privacy_bypass_scope=True)
	with pytest.raises(PermissionError, match="unsupported_identity_resolution_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, unsupported_identity_resolution_scope=True)
	with pytest.raises(PermissionError, match="autonomous_dissemination_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, autonomous_dissemination_scope=True)
	with pytest.raises(PermissionError, match="unapproved_attribution_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, unapproved_attribution_scope=True)


def test_api_views_and_app_are_executable():
	api = _load_module("api_intel_fusion", PACKAGE_DIR / "api.py")
	views = _load_module("views_intel_fusion", PACKAGE_DIR / "views.py")
	app = _load_module("app_intel_fusion", PACKAGE_DIR / "app.py")

	authority = api.record_authority({"tenant_id": "tenant-api", "authority_id": "api-auth", "authority_type": "consent", "scope_reference": "scope", "classification": "unclassified", "approver_id": "approver", "expires_at": "2026-12-31", "evidence_reference": "evidence"})
	workspace = api.record_workspace({"tenant_id": "tenant-api", "workspace_id": "api-workspace", "workspace_type": "public_safety", "name": "Public Safety", "classification": "unclassified", "authority_id": authority["id"], "evidence_reference": "evidence"})
	source = api.register_source({"tenant_id": "tenant-api", "source_id": "api-source", "source_type": "partner_report", "source_reference": "source-ref", "custodian_id": "custodian", "authority_id": authority["id"], "lineage_reference": "lineage", "evidence_reference": "evidence"})
	api.record_artifact({"tenant_id": "tenant-api", "artifact_id": "api-artifact", "workspace_id": workspace["id"], "source_id": source["id"], "artifact_type": "report", "artifact_reference": "artifact-ref", "content_fingerprint": "sha256:abc", "confidence_score": 0.72, "evidence_reference": "evidence"})
	agent = api.register_fusion_agent({"tenant_id": "tenant-api", "agent_id": "api-agent", "name": "Fusion Agent", "runtime": "claude_code", "role": "correlation_analyst"})
	batch = api.validate_batch({"tenant_id": "tenant-api", "item_count": 2})
	dashboard = views.dashboard_model(api.service(), "tenant-api")
	console = views.fusion_console_model(api.service(), "tenant-api")
	self_test = app.self_test()
	semantic = app.semantic_model()

	assert agent["role"] == "correlation_analyst"
	assert batch["processor"] == "bytewax"
	assert dashboard["summary"]["authority_count"] == 1
	assert console["sources"][0]["id"] == source["id"]
	assert self_test["passed"] is True
	assert semantic["capabilities"]["intel_fusion"]["screens"]["agents"]["route"] == "/intel-fusion/agents"


def test_app_entrypoint_is_publishable():
	module = _load_module("publishable_app_intel_fusion", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["intel_fusion"]["streaming"]["processor"] == "bytewax"
