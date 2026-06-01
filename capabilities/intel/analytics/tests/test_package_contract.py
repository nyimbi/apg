"""Executable Intelligence Analytics capability package tests."""

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
	module = _load_module("contract_intel_analytics", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "intel_analytics"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "analytics_agent_workflow" in contract["provides"]
	assert "/intel-analytics/agents" in [route["path"] for route in contract["ui"]["routes"]]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert contract["configuration"]["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]


def test_rule_engine_blocks_missing_context_non_bytewax_and_prohibited_agent_actions():
	module = _load_module("rules_intel_analytics", PACKAGE_DIR / "capability_contract.py")

	assert module.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "analytics_batch", "event_stream": "queue"})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "analytics_agent_action", "privileged_scope": True, "human_approval_recorded": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "analytics_agent_action", "hallucinated_insight_scope": True})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "analytics_agent_action", "training_data_leakage_scope": True})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "analytics_agent_action", "privacy_bypass_scope": True})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "analytics_agent_action", "unsupported_automated_decision_scope": True})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "analytics_agent_action", "unapproved_model_deployment_scope": True})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "analytics_agent_action", "autonomous_dissemination_scope": True})["decision"] == "deny"


def test_service_executes_analytics_lifecycle():
	service_module = _load_module("service_intel_analytics", PACKAGE_DIR / "service.py")
	service = service_module.IntelligenceAnalyticsService()

	authority = service.record_authority("auth-1", "tenant-test", "mission_order", "scope-ref", "confidential", "approver-1", "2026-12-31", "authority-evidence")
	workspace = service.record_workspace("workspace-1", "tenant-test", "threat_analytics", "Threat Analytics", "confidential", authority["id"], "workspace-evidence")
	dataset = service.register_dataset("dataset-1", "tenant-test", workspace["id"], "fusion_extract", "dataset-ref", "owner-1", "lineage-ref", "standard", "dataset-evidence")
	feature_set = service.record_feature_set("features-1", "tenant-test", dataset["id"], "indicator_features", "feature-ref", 0.82, "analyst-1", "feature-evidence")
	model = service.record_model("model-1", "tenant-test", feature_set["id"], "graph_analytics", "Detect threat communities", "validation-ref", "medium", "model-evidence")
	run = service.record_run("run-1", "tenant-test", model["id"], "batch", "result-ref", 0.86, "analyst-1", "run-evidence")
	insight = service.record_insight("insight-1", "tenant-test", run["id"], "risk_signal", "claim-ref", 0.77, "analyst-1", "insight-evidence")
	dashboard = service.record_dashboard("dashboard-1", "tenant-test", insight["id"], "Threat View", "command-team", "CONFIDENTIAL", "approval-ref", "dashboard-evidence")
	narrative = service.record_narrative("narrative-1", "tenant-test", insight["id"], "briefing", "summary-ref", "approval-ref", "narrative-evidence")
	recommendation = service.record_recommendation("recommendation-1", "tenant-test", insight["id"], "investigate", "action-ref", "approval-ref", "recommendation-evidence")
	review = service.record_review("review-1", "tenant-test", recommendation["id"], "reviewer-1", "approved", "review-evidence")
	agent = service.register_analytics_agent("agent-1", "tenant-test", "Analytics Agent", "codex", "insight_analyst", "insight support")
	bytewax_batch = service.validate_batch("tenant-test", 3)
	summary = service.dashboard_summary("tenant-test")

	assert authority["authority_type"] == "mission_order"
	assert workspace["workspace_type"] == "threat_analytics"
	assert dataset["dataset_type"] == "fusion_extract"
	assert feature_set["feature_type"] == "indicator_features"
	assert model["model_type"] == "graph_analytics"
	assert run["run_type"] == "batch"
	assert insight["insight_type"] == "risk_signal"
	assert dashboard["release_marking"] == "CONFIDENTIAL"
	assert narrative["narrative_type"] == "briefing"
	assert recommendation["recommendation_type"] == "investigate"
	assert review["status"] == "approved"
	assert agent["runtime"] == "codex"
	assert bytewax_batch["processor"] == "bytewax"
	assert summary["audit_event_count"] == 12


def test_service_keys_state_by_tenant_and_record_id():
	service_module = _load_module("tenant_service_intel_analytics", PACKAGE_DIR / "service.py")
	service = service_module.IntelligenceAnalyticsService()

	tenant_a = service.record_authority("shared-auth", "tenant-a", "mission_order", "scope-a", "confidential", "approver-a", "2026-12-31", "evidence-a")
	tenant_b = service.record_authority("shared-auth", "tenant-b", "consent", "scope-b", "unclassified", "approver-b", "2026-12-31", "evidence-b")
	service.record_workspace("shared-workspace", "tenant-a", "threat_analytics", "Workspace A", "confidential", tenant_a["id"], "evidence-a")
	service.record_workspace("shared-workspace", "tenant-b", "public_safety_analytics", "Workspace B", "unclassified", tenant_b["id"], "evidence-b")

	dashboard_a = service.dashboard_summary("tenant-a")
	dashboard_b = service.dashboard_summary("tenant-b")

	assert dashboard_a["authority_count"] == 1
	assert dashboard_b["authority_count"] == 1
	assert dashboard_a["workspace_count"] == 1
	assert dashboard_b["workspace_count"] == 1
	assert service._tenant_workspace_or_none("shared-workspace", "tenant-a").name == "Workspace A"
	assert service._tenant_workspace_or_none("shared-workspace", "tenant-b").name == "Workspace B"


def test_service_guardrails_reject_invalid_analytics_actions():
	service_module = _load_module("guardrail_service_intel_analytics", PACKAGE_DIR / "service.py")
	service = service_module.IntelligenceAnalyticsService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.record_authority("auth", "", "mission_order", "scope", "confidential", "approver", "2026-12-31", "evidence")
	with pytest.raises(PermissionError, match="authority_type_not_supported"):
		service.record_authority("auth", "tenant-test", "unknown", "scope", "confidential", "approver", "2026-12-31", "evidence")
	authority = service.record_authority("auth-ok", "tenant-test", "mission_order", "scope", "confidential", "approver", "2026-12-31", "evidence")
	with pytest.raises(PermissionError, match="lawful_authority_required"):
		service.record_workspace("workspace", "tenant-test", "threat_analytics", "workspace", "confidential", "missing-auth", "evidence")
	workspace = service.record_workspace("workspace-ok", "tenant-test", "threat_analytics", "workspace", "confidential", authority["id"], "evidence")
	with pytest.raises(PermissionError, match="dataset_lineage_required"):
		service.register_dataset("dataset", "tenant-test", workspace["id"], "fusion_extract", "ref", "owner", "", "standard", "evidence")
	with pytest.raises(PermissionError, match="retention_class_not_supported"):
		service.register_dataset("dataset", "tenant-test", workspace["id"], "fusion_extract", "ref", "owner", "lineage", "forever", "evidence")
	dataset = service.register_dataset("dataset-ok", "tenant-test", workspace["id"], "fusion_extract", "ref", "owner", "lineage", "standard", "evidence")
	with pytest.raises(PermissionError, match="confidence_score_invalid"):
		service.record_feature_set("features", "tenant-test", dataset["id"], "indicator_features", "feature-ref", 1.8, "analyst", "evidence")
	feature_set = service.record_feature_set("features-ok", "tenant-test", dataset["id"], "indicator_features", "feature-ref", 0.8, "analyst", "evidence")
	with pytest.raises(PermissionError, match="model_validation_required"):
		service.record_model("model", "tenant-test", feature_set["id"], "ruleset", "objective", "", "medium", "evidence")
	model = service.record_model("model-ok", "tenant-test", feature_set["id"], "ruleset", "objective", "validation", "medium", "evidence")
	with pytest.raises(PermissionError, match="run_type_not_supported"):
		service.record_run("run", "tenant-test", model["id"], "unknown", "result", 0.8, "analyst", "evidence")
	run = service.record_run("run-ok", "tenant-test", model["id"], "batch", "result", 0.8, "analyst", "evidence")
	with pytest.raises(PermissionError, match="insight_type_not_supported"):
		service.record_insight("insight", "tenant-test", run["id"], "unknown", "claim", 0.8, "analyst", "evidence")
	insight = service.record_insight("insight-ok", "tenant-test", run["id"], "risk_signal", "claim", 0.8, "analyst", "evidence")
	with pytest.raises(PermissionError, match="dashboard_approval_required"):
		service.record_dashboard("dashboard", "tenant-test", insight["id"], "Dash", "team", "CONFIDENTIAL", "", "evidence")
	with pytest.raises(PermissionError, match="narrative_type_not_supported"):
		service.record_narrative("narrative", "tenant-test", insight["id"], "unknown", "summary", "approval", "evidence")
	with pytest.raises(PermissionError, match="recommendation_approval_required"):
		service.record_recommendation("recommendation", "tenant-test", insight["id"], "investigate", "action", "", "evidence")
	with pytest.raises(PermissionError, match="reviewer_required"):
		service.record_review("review", "tenant-test", insight["id"], "", "approved", "evidence")
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.validate_batch("tenant-test", 1, event_stream="queue")
	with pytest.raises(PermissionError, match="analytics_agent_runtime_not_supported"):
		service.register_analytics_agent("agent", "tenant-test", "Bad Agent", "unsupported", "insight_analyst", "scope")
	with pytest.raises(PermissionError, match="human_approval_required"):
		service.validate_agent_action("tenant-test", privileged_scope=True, human_approval_recorded=False)
	with pytest.raises(PermissionError, match="hallucinated_insight_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, hallucinated_insight_scope=True)
	with pytest.raises(PermissionError, match="training_data_leakage_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, training_data_leakage_scope=True)
	with pytest.raises(PermissionError, match="privacy_bypass_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, privacy_bypass_scope=True)
	with pytest.raises(PermissionError, match="unsupported_automated_decision_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, unsupported_automated_decision_scope=True)
	with pytest.raises(PermissionError, match="unapproved_model_deployment_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, unapproved_model_deployment_scope=True)
	with pytest.raises(PermissionError, match="autonomous_dissemination_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, autonomous_dissemination_scope=True)


def test_api_views_and_app_are_executable():
	api = _load_module("api_intel_analytics", PACKAGE_DIR / "api.py")
	views = _load_module("views_intel_analytics", PACKAGE_DIR / "views.py")
	app = _load_module("app_intel_analytics", PACKAGE_DIR / "app.py")

	authority = api.record_authority({"tenant_id": "tenant-api", "authority_id": "api-auth", "authority_type": "consent", "scope_reference": "scope", "classification": "unclassified", "approver_id": "approver", "expires_at": "2026-12-31", "evidence_reference": "evidence"})
	workspace = api.record_workspace({"tenant_id": "tenant-api", "workspace_id": "api-workspace", "workspace_type": "public_safety_analytics", "name": "Public Safety", "classification": "unclassified", "authority_id": authority["id"], "evidence_reference": "evidence"})
	dataset = api.register_dataset({"tenant_id": "tenant-api", "dataset_id": "api-dataset", "workspace_id": workspace["id"], "dataset_type": "partner_dataset", "source_reference": "source-ref", "owner_id": "owner", "lineage_reference": "lineage", "retention_class": "standard", "evidence_reference": "evidence"})
	api.record_feature_set({"tenant_id": "tenant-api", "feature_set_id": "api-features", "dataset_id": dataset["id"], "feature_type": "entity_features", "feature_reference": "feature-ref", "confidence_score": 0.72, "analyst_id": "analyst", "evidence_reference": "evidence"})
	agent = api.register_analytics_agent({"tenant_id": "tenant-api", "agent_id": "api-agent", "name": "Analytics Agent", "runtime": "claude_code", "role": "insight_analyst"})
	batch = api.validate_batch({"tenant_id": "tenant-api", "item_count": 2})
	dashboard = views.dashboard_model(api.service(), "tenant-api")
	console = views.analytics_console_model(api.service(), "tenant-api")
	self_test = app.self_test()
	semantic = app.semantic_model()

	assert agent["role"] == "insight_analyst"
	assert batch["processor"] == "bytewax"
	assert dashboard["summary"]["authority_count"] == 1
	assert console["datasets"][0]["id"] == dataset["id"]
	assert self_test["passed"] is True
	assert semantic["capabilities"]["intel_analytics"]["screens"]["agents"]["route"] == "/intel-analytics/agents"


def test_app_entrypoint_is_publishable():
	module = _load_module("publishable_app_intel_analytics", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["intel_analytics"]["streaming"]["processor"] == "bytewax"
