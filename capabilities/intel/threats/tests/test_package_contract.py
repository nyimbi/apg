"""Executable Threat Intelligence capability package tests."""

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
	module = _load_module("contract_intel_threats", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "intel_threats"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "threat_agent_workflow" in contract["provides"]
	assert "/intel-threats/agents" in [route["path"] for route in contract["ui"]["routes"]]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert contract["configuration"]["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]


def test_rule_engine_blocks_missing_context_non_bytewax_and_prohibited_agent_actions():
	module = _load_module("rules_intel_threats", PACKAGE_DIR / "capability_contract.py")

	assert module.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "threat_batch", "event_stream": "queue"})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "threat_agent_action", "privileged_scope": True, "human_approval_recorded": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "threat_agent_action", "unsupported_attribution_scope": True})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "threat_agent_action", "fabricated_indicator_scope": True})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "threat_agent_action", "source_tampering_scope": True})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "threat_agent_action", "privacy_bypass_scope": True})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "threat_agent_action", "autonomous_mitigation_scope": True})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "threat_agent_action", "unapproved_publication_scope": True})["decision"] == "deny"


def test_service_executes_threat_lifecycle():
	service_module = _load_module("service_intel_threats", PACKAGE_DIR / "service.py")
	service = service_module.ThreatIntelligenceService()

	authority = service.record_authority("auth-1", "tenant-test", "mission_order", "scope-ref", "confidential", "approver-1", "2026-12-31", "authority-evidence")
	workspace = service.record_workspace("workspace-1", "tenant-test", "cyber_threat", "Cyber Threats", "confidential", authority["id"], "workspace-evidence")
	source = service.register_source("source-1", "tenant-test", workspace["id"], "osint", "source-ref", "custodian-1", "lineage-ref", "source-evidence")
	indicator = service.record_indicator("indicator-1", "tenant-test", source["id"], "ioc", "indicator-ref", 0.82, "indicator-evidence")
	actor = service.record_actor("actor-1", "tenant-test", workspace["id"], "criminal_group", "actor-ref", 0.76, "actor-evidence")
	campaign = service.record_campaign("campaign-1", "tenant-test", actor["id"], "intrusion_campaign", "campaign-ref", "high", "campaign-evidence")
	assessment = service.record_assessment("assessment-1", "tenant-test", campaign["id"], "risk_assessment", "high", 0.79, "analyst-1", "assessment-evidence")
	report = service.record_report("report-1", "tenant-test", assessment["id"], "advisory", "report-ref", "approval-ref", "report-evidence")
	mitigation = service.record_mitigation("mitigation-1", "tenant-test", assessment["id"], "harden", "action-ref", "approval-ref", "mitigation-evidence")
	review = service.record_review("review-1", "tenant-test", mitigation["id"], "reviewer-1", "approved", "review-evidence")
	agent = service.register_threat_agent("agent-1", "tenant-test", "Threat Agent", "codex", "actor_analyst", "actor support")
	bytewax_batch = service.validate_batch("tenant-test", 3)
	summary = service.dashboard_summary("tenant-test")

	assert authority["authority_type"] == "mission_order"
	assert workspace["workspace_type"] == "cyber_threat"
	assert source["source_type"] == "osint"
	assert indicator["indicator_type"] == "ioc"
	assert actor["actor_type"] == "criminal_group"
	assert campaign["campaign_type"] == "intrusion_campaign"
	assert assessment["assessment_type"] == "risk_assessment"
	assert report["report_type"] == "advisory"
	assert mitigation["mitigation_type"] == "harden"
	assert review["status"] == "approved"
	assert agent["runtime"] == "codex"
	assert bytewax_batch["processor"] == "bytewax"
	assert summary["audit_event_count"] == 11


def test_service_keys_state_by_tenant_and_record_id():
	service_module = _load_module("tenant_service_intel_threats", PACKAGE_DIR / "service.py")
	service = service_module.ThreatIntelligenceService()

	tenant_a = service.record_authority("shared-auth", "tenant-a", "mission_order", "scope-a", "confidential", "approver-a", "2026-12-31", "evidence-a")
	tenant_b = service.record_authority("shared-auth", "tenant-b", "consent", "scope-b", "unclassified", "approver-b", "2026-12-31", "evidence-b")
	service.record_workspace("shared-workspace", "tenant-a", "cyber_threat", "Workspace A", "confidential", tenant_a["id"], "evidence-a")
	service.record_workspace("shared-workspace", "tenant-b", "fraud_threat", "Workspace B", "unclassified", tenant_b["id"], "evidence-b")

	dashboard_a = service.dashboard_summary("tenant-a")
	dashboard_b = service.dashboard_summary("tenant-b")

	assert dashboard_a["authority_count"] == 1
	assert dashboard_b["authority_count"] == 1
	assert dashboard_a["workspace_count"] == 1
	assert dashboard_b["workspace_count"] == 1
	assert service._tenant_workspace_or_none("shared-workspace", "tenant-a").name == "Workspace A"
	assert service._tenant_workspace_or_none("shared-workspace", "tenant-b").name == "Workspace B"


def test_service_guardrails_reject_invalid_threat_actions():
	service_module = _load_module("guardrail_service_intel_threats", PACKAGE_DIR / "service.py")
	service = service_module.ThreatIntelligenceService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.record_authority("auth", "", "mission_order", "scope", "confidential", "approver", "2026-12-31", "evidence")
	with pytest.raises(PermissionError, match="authority_type_not_supported"):
		service.record_authority("auth", "tenant-test", "unknown", "scope", "confidential", "approver", "2026-12-31", "evidence")
	authority = service.record_authority("auth-ok", "tenant-test", "mission_order", "scope", "confidential", "approver", "2026-12-31", "evidence")
	with pytest.raises(PermissionError, match="lawful_authority_required"):
		service.record_workspace("workspace", "tenant-test", "cyber_threat", "workspace", "confidential", "missing-auth", "evidence")
	workspace = service.record_workspace("workspace-ok", "tenant-test", "cyber_threat", "workspace", "confidential", authority["id"], "evidence")
	with pytest.raises(PermissionError, match="source_lineage_required"):
		service.register_source("source", "tenant-test", workspace["id"], "osint", "source", "custodian", "", "evidence")
	source = service.register_source("source-ok", "tenant-test", workspace["id"], "osint", "source", "custodian", "lineage", "evidence")
	with pytest.raises(PermissionError, match="confidence_score_invalid"):
		service.record_indicator("indicator", "tenant-test", source["id"], "ioc", "indicator", 1.8, "evidence")
	with pytest.raises(PermissionError, match="actor_type_not_supported"):
		service.record_actor("actor", "tenant-test", workspace["id"], "ghost_network", "actor", 0.8, "evidence")
	actor = service.record_actor("actor-ok", "tenant-test", workspace["id"], "criminal_group", "actor", 0.8, "evidence")
	with pytest.raises(PermissionError, match="risk_level_not_supported"):
		service.record_campaign("campaign", "tenant-test", actor["id"], "intrusion_campaign", "campaign", "severe", "evidence")
	campaign = service.record_campaign("campaign-ok", "tenant-test", actor["id"], "intrusion_campaign", "campaign", "high", "evidence")
	with pytest.raises(PermissionError, match="analyst_required"):
		service.record_assessment("assessment", "tenant-test", campaign["id"], "risk_assessment", "high", 0.8, "", "evidence")
	assessment = service.record_assessment("assessment-ok", "tenant-test", campaign["id"], "risk_assessment", "high", 0.8, "analyst", "evidence")
	with pytest.raises(PermissionError, match="report_approval_required"):
		service.record_report("report", "tenant-test", assessment["id"], "advisory", "report", "", "evidence")
	with pytest.raises(PermissionError, match="mitigation_approval_required"):
		service.record_mitigation("mitigation", "tenant-test", assessment["id"], "harden", "action", "", "evidence")
	with pytest.raises(PermissionError, match="reviewer_required"):
		service.record_review("review", "tenant-test", assessment["id"], "", "approved", "evidence")
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.validate_batch("tenant-test", 1, event_stream="queue")
	with pytest.raises(PermissionError, match="threat_agent_runtime_not_supported"):
		service.register_threat_agent("agent", "tenant-test", "Bad Agent", "unsupported", "actor_analyst", "scope")
	with pytest.raises(PermissionError, match="threat_agent_scope_required"):
		service.register_threat_agent("agent", "tenant-test", "Threat Agent", "codex", "actor_analyst", "")
	with pytest.raises(PermissionError, match="human_approval_required"):
		service.validate_agent_action("tenant-test", privileged_scope=True, human_approval_recorded=False)
	with pytest.raises(PermissionError, match="unsupported_attribution_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, unsupported_attribution_scope=True)
	with pytest.raises(PermissionError, match="fabricated_indicator_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, fabricated_indicator_scope=True)
	with pytest.raises(PermissionError, match="source_tampering_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, source_tampering_scope=True)
	with pytest.raises(PermissionError, match="privacy_bypass_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, privacy_bypass_scope=True)
	with pytest.raises(PermissionError, match="autonomous_mitigation_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, autonomous_mitigation_scope=True)
	with pytest.raises(PermissionError, match="unapproved_publication_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, unapproved_publication_scope=True)


def test_api_views_and_app_are_executable():
	api = _load_module("api_intel_threats", PACKAGE_DIR / "api.py")
	views = _load_module("views_intel_threats", PACKAGE_DIR / "views.py")
	app = _load_module("app_intel_threats", PACKAGE_DIR / "app.py")

	authority = api.record_authority({"tenant_id": "tenant-api", "authority_id": "api-auth", "authority_type": "consent", "scope_reference": "scope", "classification": "unclassified", "approver_id": "approver", "expires_at": "2026-12-31", "evidence_reference": "evidence"})
	workspace = api.record_workspace({"tenant_id": "tenant-api", "workspace_id": "api-workspace", "workspace_type": "fraud_threat", "name": "Fraud Threats", "classification": "unclassified", "authority_id": authority["id"], "evidence_reference": "evidence"})
	source = api.register_source({"tenant_id": "tenant-api", "source_id": "api-source", "workspace_id": workspace["id"], "source_type": "partner_report", "source_reference": "source-ref", "custodian_id": "custodian", "lineage_reference": "lineage", "evidence_reference": "evidence"})
	api.record_indicator({"tenant_id": "tenant-api", "indicator_id": "api-indicator", "source_id": source["id"], "indicator_type": "financial_signal", "indicator_reference": "indicator-ref", "confidence_score": 0.72, "evidence_reference": "evidence"})
	agent = api.register_threat_agent({"tenant_id": "tenant-api", "agent_id": "api-agent", "name": "Threat Agent", "runtime": "claude_code", "role": "indicator_curator"})
	batch = api.validate_batch({"tenant_id": "tenant-api", "item_count": 2})
	dashboard = views.dashboard_model(api.service(), "tenant-api")
	console = views.threat_console_model(api.service(), "tenant-api")
	workbench = views.agent_workbench_model(api.service(), "tenant-api")
	self_test = app.self_test()
	semantic = app.semantic_model()

	assert agent["role"] == "indicator_curator"
	assert batch["processor"] == "bytewax"
	assert dashboard["summary"]["authority_count"] == 1
	assert console["sources"][0]["id"] == source["id"]
	assert workbench["agents"][0]["id"] == agent["id"]
	assert self_test["passed"] is True
	assert semantic["capabilities"]["intel_threats"]["screens"]["agents"]["route"] == "/intel-threats/agents"


def test_app_entrypoint_is_publishable():
	module = _load_module("publishable_app_intel_threats", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["intel_threats"]["streaming"]["processor"] == "bytewax"
