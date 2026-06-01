"""Executable Human Intelligence capability package tests."""

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
	module = _load_module("contract_intel_humint", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "intel_humint"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "humint_agent_workflow" in contract["provides"]
	assert "/intel-humint/agents" in [route["path"] for route in contract["ui"]["routes"]]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert contract["configuration"]["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]


def test_rule_engine_blocks_missing_context_non_bytewax_and_prohibited_agent_actions():
	module = _load_module("rules_intel_humint", PACKAGE_DIR / "capability_contract.py")

	assert module.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "humint_batch", "event_stream": "queue"})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "humint_agent_action", "privileged_scope": True, "human_approval_recorded": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "humint_agent_action", "coercive_scope": True})["decision"] == "deny"


def test_service_executes_humint_lifecycle():
	service_module = _load_module("service_intel_humint", PACKAGE_DIR / "service.py")
	service = service_module.HumanIntelligenceService()

	authority = service.record_authority("auth-1", "tenant-test", "mission_order", "scope-ref", "secret", "approver-1", "2026-12-31", "authority-evidence")
	source = service.register_source("source-1", "tenant-test", "voluntary_source", "active", "medium", "owner-1", authority["id"], "protection-ref", "source-evidence")
	plan = service.record_contact_plan("plan-1", "tenant-test", authority["id"], source["id"], "secure_call", "objective-ref", "safety-ref", "approval-ref", "plan-evidence")
	report = service.record_contact_report("report-1", "tenant-test", plan["id"], "report-ref", "handler-1", 0.92, "report-evidence")
	debriefing = service.record_debriefing("debriefing-1", "tenant-test", report["id"], "network activity", "secret", 0.86, "analyst-1", "debriefing-evidence")
	reliability = service.record_reliability("reliability-1", "tenant-test", source["id"], "b", 0.81, "analyst-1", "reliability-evidence")
	lead = service.record_lead("lead-1", "tenant-test", debriefing["id"], "network", "high", "analyst-1", "lead-evidence")
	dissemination = service.record_dissemination("dissemination-1", "tenant-test", lead["id"], "watch-center", "REL TO PARTNER", "release-approval", "dissemination-evidence")
	review = service.record_review("review-1", "tenant-test", dissemination["id"], "reviewer-1", "approved", "review-evidence")
	agent = service.register_humint_agent("agent-1", "tenant-test", "HUMINT Agent", "codex", "source_manager", "source management")
	bytewax_batch = service.validate_batch("tenant-test", 3)
	summary = service.dashboard_summary("tenant-test")

	assert authority["authority_type"] == "mission_order"
	assert source["handling_status"] == "active"
	assert plan["contact_method"] == "secure_call"
	assert report["source_welfare_score"] == 0.92
	assert debriefing["classification"] == "secret"
	assert reliability["reliability_grade"] == "b"
	assert lead["priority"] == "high"
	assert dissemination["release_marking"] == "REL TO PARTNER"
	assert review["status"] == "approved"
	assert agent["runtime"] == "codex"
	assert bytewax_batch["processor"] == "bytewax"
	assert summary["audit_event_count"] == 10


def test_service_keys_state_by_tenant_and_record_id():
	service_module = _load_module("tenant_service_intel_humint", PACKAGE_DIR / "service.py")
	service = service_module.HumanIntelligenceService()

	tenant_a = service.record_authority("shared-auth", "tenant-a", "mission_order", "scope-a", "secret", "approver-a", "2026-12-31", "evidence-a")
	tenant_b = service.record_authority("shared-auth", "tenant-b", "consent", "scope-b", "confidential", "approver-b", "2026-12-31", "evidence-b")
	service.register_source("shared-source", "tenant-a", "voluntary_source", "active", "medium", "owner-a", tenant_a["id"], "protect-a", "evidence-a")
	service.register_source("shared-source", "tenant-b", "partner_liaison", "paused", "low", "owner-b", tenant_b["id"], "protect-b", "evidence-b")

	dashboard_a = service.dashboard_summary("tenant-a")
	dashboard_b = service.dashboard_summary("tenant-b")

	assert dashboard_a["authority_count"] == 1
	assert dashboard_b["authority_count"] == 1
	assert dashboard_a["source_count"] == 1
	assert dashboard_b["source_count"] == 1
	assert service._tenant_authority_or_none("shared-auth", "tenant-a").authority_type == "mission_order"
	assert service._tenant_authority_or_none("shared-auth", "tenant-b").authority_type == "consent"


def test_service_guardrails_reject_invalid_humint_actions():
	service_module = _load_module("guardrail_service_intel_humint", PACKAGE_DIR / "service.py")
	service = service_module.HumanIntelligenceService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.record_authority("auth", "", "mission_order", "scope", "secret", "approver", "2026-12-31", "evidence")
	with pytest.raises(PermissionError, match="authority_type_not_supported"):
		service.record_authority("auth", "tenant-test", "unknown", "scope", "secret", "approver", "2026-12-31", "evidence")
	authority = service.record_authority("auth-ok", "tenant-test", "mission_order", "scope", "secret", "approver", "2026-12-31", "evidence")
	with pytest.raises(PermissionError, match="lawful_authority_required"):
		service.register_source("source", "tenant-test", "voluntary_source", "active", "medium", "owner", "missing-auth", "protect", "evidence")
	with pytest.raises(PermissionError, match="risk_level_not_supported"):
		service.register_source("source", "tenant-test", "voluntary_source", "active", "unknown", "owner", authority["id"], "protect", "evidence")
	source = service.register_source("source-ok", "tenant-test", "voluntary_source", "active", "medium", "owner", authority["id"], "protect", "evidence")
	other_authority = service.record_authority("auth-other", "tenant-test", "consent", "scope", "secret", "approver", "2026-12-31", "evidence")
	with pytest.raises(PermissionError, match="source_authority_mismatch"):
		service.record_contact_plan("plan", "tenant-test", other_authority["id"], source["id"], "secure_call", "objective", "safety", "approval", "evidence")
	with pytest.raises(PermissionError, match="safety_plan_required"):
		service.record_contact_plan("plan", "tenant-test", authority["id"], source["id"], "secure_call", "objective", "", "approval", "evidence")
	plan = service.record_contact_plan("plan-ok", "tenant-test", authority["id"], source["id"], "secure_call", "objective", "safety", "approval", "evidence")
	with pytest.raises(PermissionError, match="source_welfare_score_invalid"):
		service.record_contact_report("report", "tenant-test", plan["id"], "report", "handler", 1.2, "evidence")
	report = service.record_contact_report("report-ok", "tenant-test", plan["id"], "report", "handler", 0.9, "evidence")
	with pytest.raises(PermissionError, match="classification_not_supported"):
		service.record_debriefing("debriefing", "tenant-test", report["id"], "topic", "unknown", 0.8, "analyst", "evidence")
	debriefing = service.record_debriefing("debriefing-ok", "tenant-test", report["id"], "topic", "secret", 0.8, "analyst", "evidence")
	with pytest.raises(PermissionError, match="reliability_grade_not_supported"):
		service.record_reliability("reliability", "tenant-test", source["id"], "z", 0.8, "analyst", "evidence")
	with pytest.raises(PermissionError, match="priority_not_supported"):
		service.record_lead("lead", "tenant-test", debriefing["id"], "network", "unknown", "analyst", "evidence")
	lead = service.record_lead("lead-ok", "tenant-test", debriefing["id"], "network", "high", "analyst", "evidence")
	with pytest.raises(PermissionError, match="dissemination_approval_required"):
		service.record_dissemination("dissemination", "tenant-test", lead["id"], "audience", "REL", "", "evidence")
	with pytest.raises(PermissionError, match="reviewer_required"):
		service.record_review("review", "tenant-test", lead["id"], "", "approved", "evidence")
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.validate_batch("tenant-test", 1, event_stream="queue")
	with pytest.raises(PermissionError, match="humint_agent_runtime_not_supported"):
		service.register_humint_agent("agent", "tenant-test", "Bad Agent", "unsupported", "source_manager", "scope")
	with pytest.raises(PermissionError, match="human_approval_required"):
		service.validate_agent_action("tenant-test", privileged_scope=True, human_approval_recorded=False)
	with pytest.raises(PermissionError, match="coercive_humint_action_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, coercive_scope=True)


def test_api_views_and_app_are_executable():
	api = _load_module("api_intel_humint", PACKAGE_DIR / "api.py")
	views = _load_module("views_intel_humint", PACKAGE_DIR / "views.py")
	app = _load_module("app_intel_humint", PACKAGE_DIR / "app.py")

	authority = api.record_authority({"tenant_id": "tenant-api", "authority_id": "api-auth", "authority_type": "consent", "scope_reference": "scope", "classification": "confidential", "approver_id": "approver", "expires_at": "2026-12-31", "evidence_reference": "evidence"})
	source = api.register_source({"tenant_id": "tenant-api", "source_id": "api-source", "source_type": "partner_liaison", "handling_status": "active", "risk_level": "low", "owner_id": "owner", "authority_id": authority["id"], "protection_reference": "protect", "evidence_reference": "evidence"})
	api.record_contact_plan({"tenant_id": "tenant-api", "plan_id": "api-plan", "authority_id": authority["id"], "source_id": source["id"], "contact_method": "partner_channel", "objective_reference": "objective", "safety_plan_reference": "safety", "approval_reference": "approval", "evidence_reference": "evidence"})
	agent = api.register_humint_agent({"tenant_id": "tenant-api", "agent_id": "api-agent", "name": "HUMINT Agent", "runtime": "claude_code", "role": "contact_planner"})
	dashboard = views.dashboard_model(api.service(), "tenant-api")
	console = views.humint_console_model(api.service(), "tenant-api")
	self_test = app.self_test()
	semantic = app.semantic_model()

	assert agent["role"] == "contact_planner"
	assert dashboard["summary"]["authority_count"] == 1
	assert console["contact_plans"][0]["id"] == "api-plan"
	assert self_test["passed"] is True
	assert semantic["capabilities"]["intel_humint"]["screens"]["agents"]["route"] == "/intel-humint/agents"


def test_app_entrypoint_is_publishable():
	module = _load_module("publishable_app_intel_humint", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["intel_humint"]["streaming"]["processor"] == "bytewax"
