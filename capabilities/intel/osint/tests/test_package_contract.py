"""Executable Open Source Intelligence capability package tests."""

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
	module = _load_module("contract_intel_osint", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "intel_osint"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "osint_agent_workflow" in contract["provides"]
	assert "/intel-osint/agents" in [route["path"] for route in contract["ui"]["routes"]]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert contract["configuration"]["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]


def test_rule_engine_blocks_missing_context_non_bytewax_and_privileged_agent_action():
	module = _load_module("rules_intel_osint", PACKAGE_DIR / "capability_contract.py")

	assert module.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "osint_batch", "event_stream": "queue"})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "osint_agent_action", "privileged_scope": True, "human_approval_recorded": False})["decision"] == "deny"


def test_service_executes_osint_lifecycle():
	service_module = _load_module("service_intel_osint", PACKAGE_DIR / "service.py")
	service = service_module.OpenSourceIntelligenceService()

	requirement = service.register_requirement("req-1", "tenant-test", "Critical infrastructure monitoring", "high", "requester-1", "confidential", "req-evidence")
	source = service.register_source("source-1", "tenant-test", "news", "https://example.com/feed", "owner-1", "terms-review", "medium", "source-evidence")
	plan = service.record_collection_plan("plan-1", "tenant-test", requirement["id"], source["id"], "rss_feed", "hourly", "", "plan-evidence")
	evidence = service.record_evidence("evidence-1", "tenant-test", plan["id"], "object://content", "sha256:abc", 0.91, "evidence-evidence")
	triage = service.record_triage("triage-1", "tenant-test", evidence["id"], "relevant", "analyst-1", "triage-evidence")
	assessment = service.record_assessment("assessment-1", "tenant-test", requirement["id"], "threat", 0.84, "analyst-1", "assessment-evidence")
	package = service.record_dissemination("package-1", "tenant-test", assessment["id"], "security-leads", "confidential", "approval-ref", "package-evidence")
	review = service.record_review("review-1", "tenant-test", package["id"], "reviewer-1", "approved", "review-evidence")
	agent = service.register_osint_agent("agent-1", "tenant-test", "OSINT Agent", "codex", "source_scout", "scout sources")
	batch = service.validate_batch("tenant-test", 3)
	summary = service.dashboard_summary("tenant-test")

	assert requirement["priority"] == "high"
	assert source["source_type"] == "news"
	assert plan["method"] == "rss_feed"
	assert evidence["confidence_score"] == 0.91
	assert triage["decision"] == "relevant"
	assert assessment["assessment_type"] == "threat"
	assert package["audience"] == "security-leads"
	assert review["status"] == "approved"
	assert agent["runtime"] == "codex"
	assert batch["processor"] == "bytewax"
	assert summary["audit_event_count"] == 9


def test_service_guardrails_reject_invalid_osint_actions():
	service_module = _load_module("guardrail_service_intel_osint", PACKAGE_DIR / "service.py")
	service = service_module.OpenSourceIntelligenceService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.register_requirement("req", "", "Topic", "high", "requester", "confidential", "evidence")
	with pytest.raises(PermissionError, match="priority_not_supported"):
		service.register_requirement("req", "tenant-test", "Topic", "unknown", "requester", "confidential", "evidence")
	requirement = service.register_requirement("req-ok", "tenant-test", "Topic", "high", "requester", "confidential", "evidence")
	with pytest.raises(PermissionError, match="source_type_not_supported"):
		service.register_source("source", "tenant-test", "unknown", "ref", "owner", "terms", "low", "evidence")
	with pytest.raises(PermissionError, match="terms_review_required"):
		service.register_source("source", "tenant-test", "news", "ref", "owner", "", "low", "evidence")
	source = service.register_source("source-ok", "tenant-test", "news", "ref", "owner", "terms", "high", "evidence")
	with pytest.raises(PermissionError, match="collection_approval_required"):
		service.record_collection_plan("plan", "tenant-test", requirement["id"], source["id"], "rss_feed", "hourly", "", "evidence")
	plan = service.record_collection_plan("plan-ok", "tenant-test", requirement["id"], source["id"], "rss_feed", "hourly", "approval", "evidence")
	with pytest.raises(PermissionError, match="confidence_score_invalid"):
		service.record_evidence("evidence", "tenant-test", plan["id"], "content", "fingerprint", 1.2, "evidence")
	evidence = service.record_evidence("evidence-ok", "tenant-test", plan["id"], "content", "fingerprint", 0.9, "evidence")
	with pytest.raises(PermissionError, match="triage_decision_not_supported"):
		service.record_triage("triage", "tenant-test", evidence["id"], "maybe", "analyst", "evidence")
	with pytest.raises(PermissionError, match="assessment_type_not_supported"):
		service.record_assessment("assessment", "tenant-test", requirement["id"], "unknown", 0.8, "analyst", "evidence")
	assessment = service.record_assessment("assessment-ok", "tenant-test", requirement["id"], "threat", 0.8, "analyst", "evidence")
	with pytest.raises(PermissionError, match="dissemination_approval_required"):
		service.record_dissemination("package", "tenant-test", assessment["id"], "audience", "confidential", "", "evidence")
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.validate_batch("tenant-test", 1, event_stream="queue")
	with pytest.raises(PermissionError, match="osint_agent_runtime_not_supported"):
		service.register_osint_agent("agent", "tenant-test", "Bad Agent", "unsupported", "source_scout", "scope")
	with pytest.raises(PermissionError, match="human_approval_required"):
		service.validate_agent_action("tenant-test", privileged_scope=True, human_approval_recorded=False)


def test_api_views_and_app_are_executable():
	api = _load_module("api_intel_osint", PACKAGE_DIR / "api.py")
	views = _load_module("views_intel_osint", PACKAGE_DIR / "views.py")
	app = _load_module("app_intel_osint", PACKAGE_DIR / "app.py")

	requirement = api.register_requirement({"tenant_id": "tenant-api", "requirement_id": "api-req", "topic": "Infrastructure", "priority": "medium", "requester_id": "requester", "classification": "unclassified", "evidence_reference": "evidence"})
	source = api.register_source({"tenant_id": "tenant-api", "source_id": "api-source", "source_type": "web", "source_reference": "https://example.com", "owner_id": "owner", "terms_review_reference": "terms", "risk_tier": "low", "evidence_reference": "evidence"})
	api.record_collection_plan({"tenant_id": "tenant-api", "plan_id": "api-plan", "requirement_id": requirement["id"], "source_id": source["id"], "method": "manual_review", "cadence": "daily", "evidence_reference": "evidence"})
	agent = api.register_osint_agent({"tenant_id": "tenant-api", "agent_id": "api-agent", "name": "OSINT Agent", "runtime": "claude_code", "role": "collection_planner"})
	dashboard = views.dashboard_model(api.service(), "tenant-api")
	console = views.osint_console_model(api.service(), "tenant-api")
	self_test = app.self_test()
	semantic = app.semantic_model()

	assert agent["role"] == "collection_planner"
	assert dashboard["summary"]["requirement_count"] == 1
	assert console["collection_plans"][0]["id"] == "api-plan"
	assert self_test["passed"] is True
	assert semantic["capabilities"]["intel_osint"]["screens"]["agents"]["route"] == "/intel-osint/agents"


def test_app_entrypoint_is_publishable():
	module = _load_module("publishable_app_intel_osint", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["intel_osint"]["streaming"]["processor"] == "bytewax"
