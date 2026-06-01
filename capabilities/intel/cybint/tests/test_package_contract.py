"""Executable Cyber Intelligence capability package tests."""

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
	module = _load_module("contract_intel_cybint", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "intel_cybint"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "cybint_agent_workflow" in contract["provides"]
	assert "/intel-cybint/agents" in [route["path"] for route in contract["ui"]["routes"]]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert contract["configuration"]["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]


def test_rule_engine_blocks_missing_context_non_bytewax_and_prohibited_agent_actions():
	module = _load_module("rules_intel_cybint", PACKAGE_DIR / "capability_contract.py")

	assert module.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "cybint_batch", "event_stream": "queue"})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "cybint_agent_action", "privileged_scope": True, "human_approval_recorded": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "cybint_agent_action", "offensive_or_exploit_scope": True})["decision"] == "deny"


def test_service_executes_cybint_lifecycle():
	service_module = _load_module("service_intel_cybint", PACKAGE_DIR / "service.py")
	service = service_module.CyberIntelligenceService()

	authority = service.record_authority("auth-1", "tenant-test", "defensive_operations_authority", "scope-ref", "confidential", "approver-1", "2026-12-31", "authority-evidence")
	indicator = service.record_indicator("indicator-1", "tenant-test", "domain", "example.invalid", "amber", 0.82, authority["id"], "indicator-evidence")
	sighting = service.record_sighting("sighting-1", "tenant-test", indicator["id"], "siem:event", "2026-06-01T00:00:00Z", "high", "sighting-evidence")
	enrichment = service.record_enrichment("enrichment-1", "tenant-test", indicator["id"], "reputation", "provider-ref", 0.77, "analyst-1", "enrichment-evidence")
	profile = service.record_profile("profile-1", "tenant-test", "campaign", "Campaign A", "confidential", 0.74, "analyst-1", "profile-evidence")
	risk = service.record_risk("risk-1", "tenant-test", indicator["id"], profile["id"], "high", 0.86, "analyst-1", "risk-evidence")
	incident = service.record_incident_link("incident-1", "tenant-test", risk["id"], "case-123", "triage", "owner-1", "incident-evidence")
	dissemination = service.record_dissemination("dissemination-1", "tenant-test", risk["id"], "soc-team", "TLP:AMBER", "approval-ref", "dissemination-evidence")
	review = service.record_review("review-1", "tenant-test", dissemination["id"], "reviewer-1", "approved", "review-evidence")
	agent = service.register_cybint_agent("agent-1", "tenant-test", "CYBINT Agent", "codex", "indicator_triage", "indicator triage")
	bytewax_batch = service.validate_batch("tenant-test", 3)
	summary = service.dashboard_summary("tenant-test")

	assert authority["authority_type"] == "defensive_operations_authority"
	assert indicator["tlp"] == "amber"
	assert sighting["severity"] == "high"
	assert enrichment["enrichment_type"] == "reputation"
	assert profile["profile_type"] == "campaign"
	assert risk["risk_level"] == "high"
	assert incident["response_priority"] == "triage"
	assert dissemination["release_marking"] == "TLP:AMBER"
	assert review["status"] == "approved"
	assert agent["runtime"] == "codex"
	assert bytewax_batch["processor"] == "bytewax"
	assert summary["audit_event_count"] == 10


def test_service_keys_state_by_tenant_and_record_id():
	service_module = _load_module("tenant_service_intel_cybint", PACKAGE_DIR / "service.py")
	service = service_module.CyberIntelligenceService()

	tenant_a = service.record_authority("shared-auth", "tenant-a", "defensive_operations_authority", "scope-a", "confidential", "approver-a", "2026-12-31", "evidence-a")
	tenant_b = service.record_authority("shared-auth", "tenant-b", "consent", "scope-b", "unclassified", "approver-b", "2026-12-31", "evidence-b")
	service.record_indicator("shared-indicator", "tenant-a", "domain", "a.example.invalid", "amber", 0.8, tenant_a["id"], "evidence-a")
	service.record_indicator("shared-indicator", "tenant-b", "domain", "b.example.invalid", "green", 0.7, tenant_b["id"], "evidence-b")

	dashboard_a = service.dashboard_summary("tenant-a")
	dashboard_b = service.dashboard_summary("tenant-b")

	assert dashboard_a["authority_count"] == 1
	assert dashboard_b["authority_count"] == 1
	assert dashboard_a["indicator_count"] == 1
	assert dashboard_b["indicator_count"] == 1
	assert service._tenant_indicator_or_none("shared-indicator", "tenant-a").indicator_value == "a.example.invalid"
	assert service._tenant_indicator_or_none("shared-indicator", "tenant-b").indicator_value == "b.example.invalid"


def test_service_guardrails_reject_invalid_cybint_actions():
	service_module = _load_module("guardrail_service_intel_cybint", PACKAGE_DIR / "service.py")
	service = service_module.CyberIntelligenceService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.record_authority("auth", "", "defensive_operations_authority", "scope", "confidential", "approver", "2026-12-31", "evidence")
	with pytest.raises(PermissionError, match="authority_type_not_supported"):
		service.record_authority("auth", "tenant-test", "unknown", "scope", "confidential", "approver", "2026-12-31", "evidence")
	authority = service.record_authority("auth-ok", "tenant-test", "defensive_operations_authority", "scope", "confidential", "approver", "2026-12-31", "evidence")
	with pytest.raises(PermissionError, match="lawful_authority_required"):
		service.record_indicator("indicator", "tenant-test", "domain", "example.invalid", "amber", 0.8, "missing-auth", "evidence")
	with pytest.raises(PermissionError, match="tlp_not_supported"):
		service.record_indicator("indicator", "tenant-test", "domain", "example.invalid", "unknown", 0.8, authority["id"], "evidence")
	indicator = service.record_indicator("indicator-ok", "tenant-test", "domain", "example.invalid", "amber", 0.8, authority["id"], "evidence")
	with pytest.raises(PermissionError, match="severity_not_supported"):
		service.record_sighting("sighting", "tenant-test", indicator["id"], "source", "2026-06-01", "unknown", "evidence")
	with pytest.raises(PermissionError, match="confidence_score_invalid"):
		service.record_enrichment("enrichment", "tenant-test", indicator["id"], "reputation", "provider", 1.2, "analyst", "evidence")
	with pytest.raises(PermissionError, match="profile_type_not_supported"):
		service.record_profile("profile", "tenant-test", "unknown", "Name", "confidential", 0.7, "analyst", "evidence")
	profile = service.record_profile("profile-ok", "tenant-test", "campaign", "Campaign", "confidential", 0.7, "analyst", "evidence")
	with pytest.raises(PermissionError, match="risk_level_not_supported"):
		service.record_risk("risk", "tenant-test", indicator["id"], profile["id"], "unknown", 0.8, "analyst", "evidence")
	risk = service.record_risk("risk-ok", "tenant-test", indicator["id"], profile["id"], "high", 0.8, "analyst", "evidence")
	with pytest.raises(PermissionError, match="response_priority_not_supported"):
		service.record_incident_link("incident", "tenant-test", risk["id"], "case", "unknown", "owner", "evidence")
	with pytest.raises(PermissionError, match="dissemination_approval_required"):
		service.record_dissemination("dissemination", "tenant-test", risk["id"], "audience", "TLP:AMBER", "", "evidence")
	with pytest.raises(PermissionError, match="reviewer_required"):
		service.record_review("review", "tenant-test", risk["id"], "", "approved", "evidence")
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.validate_batch("tenant-test", 1, event_stream="queue")
	with pytest.raises(PermissionError, match="cybint_agent_runtime_not_supported"):
		service.register_cybint_agent("agent", "tenant-test", "Bad Agent", "unsupported", "indicator_triage", "scope")
	with pytest.raises(PermissionError, match="human_approval_required"):
		service.validate_agent_action("tenant-test", privileged_scope=True, human_approval_recorded=False)
	with pytest.raises(PermissionError, match="offensive_or_exploit_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, offensive_or_exploit_scope=True)


def test_api_views_and_app_are_executable():
	api = _load_module("api_intel_cybint", PACKAGE_DIR / "api.py")
	views = _load_module("views_intel_cybint", PACKAGE_DIR / "views.py")
	app = _load_module("app_intel_cybint", PACKAGE_DIR / "app.py")

	authority = api.record_authority({"tenant_id": "tenant-api", "authority_id": "api-auth", "authority_type": "consent", "scope_reference": "scope", "classification": "unclassified", "approver_id": "approver", "expires_at": "2026-12-31", "evidence_reference": "evidence"})
	api.record_indicator({"tenant_id": "tenant-api", "indicator_id": "api-indicator", "indicator_type": "domain", "indicator_value": "example.invalid", "tlp": "green", "confidence_score": 0.7, "authority_id": authority["id"], "evidence_reference": "evidence"})
	agent = api.register_cybint_agent({"tenant_id": "tenant-api", "agent_id": "api-agent", "name": "CYBINT Agent", "runtime": "claude_code", "role": "indicator_triage"})
	dashboard = views.dashboard_model(api.service(), "tenant-api")
	console = views.cybint_console_model(api.service(), "tenant-api")
	self_test = app.self_test()
	semantic = app.semantic_model()

	assert agent["role"] == "indicator_triage"
	assert dashboard["summary"]["authority_count"] == 1
	assert console["indicators"][0]["id"] == "api-indicator"
	assert self_test["passed"] is True
	assert semantic["capabilities"]["intel_cybint"]["screens"]["agents"]["route"] == "/intel-cybint/agents"


def test_app_entrypoint_is_publishable():
	module = _load_module("publishable_app_intel_cybint", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["intel_cybint"]["streaming"]["processor"] == "bytewax"
