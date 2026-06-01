"""Executable Data Correlation capability package tests."""

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
	module = _load_module("contract_intel_correlation", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "intel_correlation"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "correlation_agent_workflow" in contract["provides"]
	assert "/intel-correlation/agents" in [route["path"] for route in contract["ui"]["routes"]]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert contract["configuration"]["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]


def test_rule_engine_blocks_missing_context_non_bytewax_and_prohibited_agent_actions():
	module = _load_module("rules_intel_correlation", PACKAGE_DIR / "capability_contract.py")

	assert module.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "correlation_batch", "event_stream": "queue"})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "correlation_agent_action", "privileged_scope": True, "human_approval_recorded": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "correlation_agent_action", "unapproved_identity_merge_scope": True})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "correlation_agent_action", "source_tampering_scope": True})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "correlation_agent_action", "privacy_bypass_scope": True})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "correlation_agent_action", "evidence_fabrication_scope": True})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "correlation_agent_action", "autonomous_referral_scope": True})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "correlation_agent_action", "unreviewed_high_impact_match_scope": True})["decision"] == "deny"


def test_service_executes_correlation_lifecycle():
	service_module = _load_module("service_intel_correlation", PACKAGE_DIR / "service.py")
	service = service_module.DataCorrelationService()

	authority = service.record_authority("auth-1", "tenant-test", "mission_order", "scope-ref", "confidential", "approver-1", "2026-12-31", "authority-evidence")
	workspace = service.record_workspace("workspace-1", "tenant-test", "entity_resolution", "Entity Resolution", "confidential", authority["id"], "workspace-evidence")
	source = service.register_source("source-1", "tenant-test", workspace["id"], "fusion_extract", "source-ref", "custodian-1", "lineage-ref", "source-evidence")
	entity = service.record_entity("entity-1", "tenant-test", source["id"], "person", "entity-ref", 0.82, "entity-evidence")
	observation = service.record_observation("observation-1", "tenant-test", entity["id"], "attribute", "observation-ref", "2026-06-01T00:00:00Z", 0.86, "observation-evidence")
	rule = service.record_rule("rule-1", "tenant-test", workspace["id"], "fuzzy_match", "rule-ref", 0.76, "analyst-1", "rule-evidence")
	run = service.record_run("run-1", "tenant-test", rule["id"], "batch", "result-ref", 0.81, "analyst-1", "run-evidence")
	cluster = service.record_cluster("cluster-1", "tenant-test", run["id"], "entity_cluster", "cluster-ref", 0.79, "analyst-1", "cluster-evidence")
	decision = service.record_decision("decision-1", "tenant-test", cluster["id"], "possible_match", "rationale-ref", "approval-ref", "decision-evidence")
	referral = service.record_referral("referral-1", "tenant-test", decision["id"], "analyst_review", "review-team", "approval-ref", "referral-evidence")
	review = service.record_review("review-1", "tenant-test", referral["id"], "reviewer-1", "approved", "review-evidence")
	agent = service.register_correlation_agent("agent-1", "tenant-test", "Correlation Agent", "codex", "cluster_analyst", "cluster support")
	bytewax_batch = service.validate_batch("tenant-test", 3)
	summary = service.dashboard_summary("tenant-test")

	assert authority["authority_type"] == "mission_order"
	assert workspace["workspace_type"] == "entity_resolution"
	assert source["source_type"] == "fusion_extract"
	assert entity["entity_type"] == "person"
	assert observation["observation_type"] == "attribute"
	assert rule["rule_type"] == "fuzzy_match"
	assert run["run_type"] == "batch"
	assert cluster["cluster_type"] == "entity_cluster"
	assert decision["decision_type"] == "possible_match"
	assert referral["referral_type"] == "analyst_review"
	assert review["status"] == "approved"
	assert agent["runtime"] == "codex"
	assert bytewax_batch["processor"] == "bytewax"
	assert summary["audit_event_count"] == 12


def test_service_keys_state_by_tenant_and_record_id():
	service_module = _load_module("tenant_service_intel_correlation", PACKAGE_DIR / "service.py")
	service = service_module.DataCorrelationService()

	tenant_a = service.record_authority("shared-auth", "tenant-a", "mission_order", "scope-a", "confidential", "approver-a", "2026-12-31", "evidence-a")
	tenant_b = service.record_authority("shared-auth", "tenant-b", "consent", "scope-b", "unclassified", "approver-b", "2026-12-31", "evidence-b")
	service.record_workspace("shared-workspace", "tenant-a", "entity_resolution", "Workspace A", "confidential", tenant_a["id"], "evidence-a")
	service.record_workspace("shared-workspace", "tenant-b", "link_analysis", "Workspace B", "unclassified", tenant_b["id"], "evidence-b")

	dashboard_a = service.dashboard_summary("tenant-a")
	dashboard_b = service.dashboard_summary("tenant-b")

	assert dashboard_a["authority_count"] == 1
	assert dashboard_b["authority_count"] == 1
	assert dashboard_a["workspace_count"] == 1
	assert dashboard_b["workspace_count"] == 1
	assert service._tenant_workspace_or_none("shared-workspace", "tenant-a").name == "Workspace A"
	assert service._tenant_workspace_or_none("shared-workspace", "tenant-b").name == "Workspace B"


def test_service_guardrails_reject_invalid_correlation_actions():
	service_module = _load_module("guardrail_service_intel_correlation", PACKAGE_DIR / "service.py")
	service = service_module.DataCorrelationService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.record_authority("auth", "", "mission_order", "scope", "confidential", "approver", "2026-12-31", "evidence")
	with pytest.raises(PermissionError, match="authority_type_not_supported"):
		service.record_authority("auth", "tenant-test", "unknown", "scope", "confidential", "approver", "2026-12-31", "evidence")
	authority = service.record_authority("auth-ok", "tenant-test", "mission_order", "scope", "confidential", "approver", "2026-12-31", "evidence")
	with pytest.raises(PermissionError, match="lawful_authority_required"):
		service.record_workspace("workspace", "tenant-test", "entity_resolution", "workspace", "confidential", "missing-auth", "evidence")
	workspace = service.record_workspace("workspace-ok", "tenant-test", "entity_resolution", "workspace", "confidential", authority["id"], "evidence")
	with pytest.raises(PermissionError, match="source_lineage_required"):
		service.register_source("source", "tenant-test", workspace["id"], "fusion_extract", "ref", "custodian", "", "evidence")
	source = service.register_source("source-ok", "tenant-test", workspace["id"], "fusion_extract", "ref", "custodian", "lineage", "evidence")
	with pytest.raises(PermissionError, match="entity_type_not_supported"):
		service.record_entity("entity", "tenant-test", source["id"], "unknown", "entity", 0.8, "evidence")
	with pytest.raises(PermissionError, match="confidence_score_invalid"):
		service.record_entity("entity", "tenant-test", source["id"], "person", "entity", 1.8, "evidence")
	entity = service.record_entity("entity-ok", "tenant-test", source["id"], "person", "entity", 0.8, "evidence")
	with pytest.raises(PermissionError, match="observed_at_required"):
		service.record_observation("observation", "tenant-test", entity["id"], "attribute", "obs", "", 0.8, "evidence")
	with pytest.raises(PermissionError, match="rule_type_not_supported"):
		service.record_rule("rule", "tenant-test", workspace["id"], "unknown", "rule", 0.8, "analyst", "evidence")
	with pytest.raises(PermissionError, match="threshold_score_invalid"):
		service.record_rule("rule", "tenant-test", workspace["id"], "fuzzy_match", "rule", 1.8, "analyst", "evidence")
	rule = service.record_rule("rule-ok", "tenant-test", workspace["id"], "fuzzy_match", "rule", 0.8, "analyst", "evidence")
	with pytest.raises(PermissionError, match="run_type_not_supported"):
		service.record_run("run", "tenant-test", rule["id"], "unknown", "result", 0.8, "analyst", "evidence")
	run = service.record_run("run-ok", "tenant-test", rule["id"], "batch", "result", 0.8, "analyst", "evidence")
	with pytest.raises(PermissionError, match="cluster_type_not_supported"):
		service.record_cluster("cluster", "tenant-test", run["id"], "unknown", "cluster", 0.8, "analyst", "evidence")
	cluster = service.record_cluster("cluster-ok", "tenant-test", run["id"], "entity_cluster", "cluster", 0.8, "analyst", "evidence")
	with pytest.raises(PermissionError, match="decision_approval_required"):
		service.record_decision("decision", "tenant-test", cluster["id"], "possible_match", "rationale", "", "evidence")
	decision = service.record_decision("decision-ok", "tenant-test", cluster["id"], "possible_match", "rationale", "approval", "evidence")
	with pytest.raises(PermissionError, match="referral_approval_required"):
		service.record_referral("referral", "tenant-test", decision["id"], "analyst_review", "team", "", "evidence")
	with pytest.raises(PermissionError, match="reviewer_required"):
		service.record_review("review", "tenant-test", decision["id"], "", "approved", "evidence")
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.validate_batch("tenant-test", 1, event_stream="queue")
	with pytest.raises(PermissionError, match="correlation_agent_runtime_not_supported"):
		service.register_correlation_agent("agent", "tenant-test", "Bad Agent", "unsupported", "cluster_analyst", "scope")
	with pytest.raises(PermissionError, match="human_approval_required"):
		service.validate_agent_action("tenant-test", privileged_scope=True, human_approval_recorded=False)
	with pytest.raises(PermissionError, match="unapproved_identity_merge_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, unapproved_identity_merge_scope=True)
	with pytest.raises(PermissionError, match="source_tampering_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, source_tampering_scope=True)
	with pytest.raises(PermissionError, match="privacy_bypass_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, privacy_bypass_scope=True)
	with pytest.raises(PermissionError, match="evidence_fabrication_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, evidence_fabrication_scope=True)
	with pytest.raises(PermissionError, match="autonomous_referral_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, autonomous_referral_scope=True)
	with pytest.raises(PermissionError, match="unreviewed_high_impact_match_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, unreviewed_high_impact_match_scope=True)


def test_api_views_and_app_are_executable():
	api = _load_module("api_intel_correlation", PACKAGE_DIR / "api.py")
	views = _load_module("views_intel_correlation", PACKAGE_DIR / "views.py")
	app = _load_module("app_intel_correlation", PACKAGE_DIR / "app.py")

	authority = api.record_authority({"tenant_id": "tenant-api", "authority_id": "api-auth", "authority_type": "consent", "scope_reference": "scope", "classification": "unclassified", "approver_id": "approver", "expires_at": "2026-12-31", "evidence_reference": "evidence"})
	workspace = api.record_workspace({"tenant_id": "tenant-api", "workspace_id": "api-workspace", "workspace_type": "link_analysis", "name": "Link Analysis", "classification": "unclassified", "authority_id": authority["id"], "evidence_reference": "evidence"})
	source = api.register_source({"tenant_id": "tenant-api", "source_id": "api-source", "workspace_id": workspace["id"], "source_type": "partner_dataset", "source_reference": "source-ref", "custodian_id": "custodian", "lineage_reference": "lineage", "evidence_reference": "evidence"})
	api.record_entity({"tenant_id": "tenant-api", "entity_id": "api-entity", "source_id": source["id"], "entity_type": "person", "entity_reference": "entity-ref", "confidence_score": 0.72, "evidence_reference": "evidence"})
	agent = api.register_correlation_agent({"tenant_id": "tenant-api", "agent_id": "api-agent", "name": "Correlation Agent", "runtime": "claude_code", "role": "cluster_analyst"})
	batch = api.validate_batch({"tenant_id": "tenant-api", "item_count": 2})
	dashboard = views.dashboard_model(api.service(), "tenant-api")
	console = views.correlation_console_model(api.service(), "tenant-api")
	self_test = app.self_test()
	semantic = app.semantic_model()

	assert agent["role"] == "cluster_analyst"
	assert batch["processor"] == "bytewax"
	assert dashboard["summary"]["authority_count"] == 1
	assert console["sources"][0]["id"] == source["id"]
	assert self_test["passed"] is True
	assert semantic["capabilities"]["intel_correlation"]["screens"]["agents"]["route"] == "/intel-correlation/agents"


def test_app_entrypoint_is_publishable():
	module = _load_module("publishable_app_intel_correlation", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["intel_correlation"]["streaming"]["processor"] == "bytewax"
