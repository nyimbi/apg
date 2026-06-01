"""Executable Dark Web Monitoring capability package tests."""

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
	module = _load_module("contract_intel_darkweb", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "intel_darkweb"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "darkweb_agent_workflow" in contract["provides"]
	assert "/intel-darkweb/agents" in [route["path"] for route in contract["ui"]["routes"]]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert contract["configuration"]["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]


def test_rule_engine_blocks_missing_context_non_bytewax_and_prohibited_agent_actions():
	module = _load_module("rules_intel_darkweb", PACKAGE_DIR / "capability_contract.py")

	assert module.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "darkweb_batch", "event_stream": "queue"})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "darkweb_agent_action", "privileged_scope": True, "human_approval_recorded": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "darkweb_agent_action", "credential_use_scope": True})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "darkweb_agent_action", "contraband_transaction_scope": True})["decision"] == "deny"


def test_service_executes_darkweb_lifecycle():
	service_module = _load_module("service_intel_darkweb", PACKAGE_DIR / "service.py")
	service = service_module.DarkWebMonitoringService()

	authority = service.record_authority("auth-1", "tenant-test", "security_monitoring_authority", "scope-ref", "confidential", "approver-1", "2026-12-31", "authority-evidence")
	program = service.record_program("program-1", "tenant-test", "credential_exposure", "Credential leak watch", "high", authority["id"], "program-evidence")
	source = service.register_source("source-1", "tenant-test", "paste", "paste_site", "paste-watch", "custodian-1", authority["id"], "access-review", "source-evidence")
	observation = service.record_observation("observation-1", "tenant-test", program["id"], source["id"], "paste", "paste-ref", "sha256:abc", "2026-06-01T00:00:00Z", 0.82, "observation-evidence")
	indicator = service.record_indicator("indicator-1", "tenant-test", observation["id"], "credential_exposure", "high", 0.86, "analyst-1", "indicator-evidence")
	marketplace_risk = service.record_marketplace_risk("risk-1", "tenant-test", indicator["id"], "exposure_risk", "high", 0.84, "analyst-1", "risk-evidence")
	threat_actor = service.record_threat_actor("actor-1", "tenant-test", indicator["id"], "actor-ref", "medium", 0.79, "analyst-1", "actor-evidence")
	referral = service.record_referral("referral-1", "tenant-test", marketplace_risk["id"], "incident_response", "ir-team", "approval-ref", "referral-evidence")
	dissemination = service.record_dissemination("dissemination-1", "tenant-test", threat_actor["id"], "security-team", "CONFIDENTIAL", "approval-ref", "dissemination-evidence")
	review = service.record_review("review-1", "tenant-test", dissemination["id"], "reviewer-1", "approved", "review-evidence")
	agent = service.register_darkweb_agent("agent-1", "tenant-test", "Dark Web Agent", "codex", "exposure_analyst", "exposure analysis")
	bytewax_batch = service.validate_batch("tenant-test", 3)
	summary = service.dashboard_summary("tenant-test")

	assert authority["authority_type"] == "security_monitoring_authority"
	assert program["program_type"] == "credential_exposure"
	assert source["network_type"] == "paste_site"
	assert observation["observation_type"] == "paste"
	assert indicator["indicator_type"] == "credential_exposure"
	assert marketplace_risk["assessment_type"] == "exposure_risk"
	assert threat_actor["actor_reference"] == "actor-ref"
	assert referral["referral_type"] == "incident_response"
	assert dissemination["release_marking"] == "CONFIDENTIAL"
	assert review["status"] == "approved"
	assert agent["runtime"] == "codex"
	assert bytewax_batch["processor"] == "bytewax"
	assert summary["audit_event_count"] == 11


def test_service_keys_state_by_tenant_and_record_id():
	service_module = _load_module("tenant_service_intel_darkweb", PACKAGE_DIR / "service.py")
	service = service_module.DarkWebMonitoringService()

	tenant_a = service.record_authority("shared-auth", "tenant-a", "security_monitoring_authority", "scope-a", "confidential", "approver-a", "2026-12-31", "evidence-a")
	tenant_b = service.record_authority("shared-auth", "tenant-b", "consent", "scope-b", "unclassified", "approver-b", "2026-12-31", "evidence-b")
	service.record_program("shared-program", "tenant-a", "brand_protection", "Brand A", "medium", tenant_a["id"], "evidence-a")
	service.record_program("shared-program", "tenant-b", "watchlist", "Watch B", "low", tenant_b["id"], "evidence-b")

	dashboard_a = service.dashboard_summary("tenant-a")
	dashboard_b = service.dashboard_summary("tenant-b")

	assert dashboard_a["authority_count"] == 1
	assert dashboard_b["authority_count"] == 1
	assert dashboard_a["program_count"] == 1
	assert dashboard_b["program_count"] == 1
	assert service._tenant_program_or_none("shared-program", "tenant-a").name == "Brand A"
	assert service._tenant_program_or_none("shared-program", "tenant-b").name == "Watch B"


def test_service_guardrails_reject_invalid_darkweb_actions():
	service_module = _load_module("guardrail_service_intel_darkweb", PACKAGE_DIR / "service.py")
	service = service_module.DarkWebMonitoringService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.record_authority("auth", "", "security_monitoring_authority", "scope", "confidential", "approver", "2026-12-31", "evidence")
	with pytest.raises(PermissionError, match="authority_type_not_supported"):
		service.record_authority("auth", "tenant-test", "unknown", "scope", "confidential", "approver", "2026-12-31", "evidence")
	authority = service.record_authority("auth-ok", "tenant-test", "security_monitoring_authority", "scope", "confidential", "approver", "2026-12-31", "evidence")
	with pytest.raises(PermissionError, match="lawful_authority_required"):
		service.record_program("program", "tenant-test", "brand_protection", "brand", "medium", "missing-auth", "evidence")
	program = service.record_program("program-ok", "tenant-test", "brand_protection", "brand", "medium", authority["id"], "evidence")
	with pytest.raises(PermissionError, match="source_access_review_required"):
		service.register_source("source", "tenant-test", "forum", "tor", "source-ref", "custodian", authority["id"], "", "evidence")
	source = service.register_source("source-ok", "tenant-test", "forum", "tor", "source-ref", "custodian", authority["id"], "access-review", "evidence")
	other_authority = service.record_authority("auth-other", "tenant-test", "consent", "scope", "confidential", "approver", "2026-12-31", "evidence")
	other_program = service.record_program("program-other", "tenant-test", "watchlist", "watch", "low", other_authority["id"], "evidence")
	with pytest.raises(PermissionError, match="authority_mismatch"):
		service.record_observation("observation", "tenant-test", other_program["id"], source["id"], "post", "ref", "hash", "2026-06-01", 0.8, "evidence")
	with pytest.raises(PermissionError, match="confidence_score_invalid"):
		service.record_observation("observation", "tenant-test", program["id"], source["id"], "post", "ref", "hash", "2026-06-01", 1.8, "evidence")
	observation = service.record_observation("observation-ok", "tenant-test", program["id"], source["id"], "post", "ref", "hash", "2026-06-01", 0.8, "evidence")
	with pytest.raises(PermissionError, match="indicator_type_not_supported"):
		service.record_indicator("indicator", "tenant-test", observation["id"], "unknown", "high", 0.8, "analyst", "evidence")
	indicator = service.record_indicator("indicator-ok", "tenant-test", observation["id"], "brand_abuse", "medium", 0.8, "analyst", "evidence")
	with pytest.raises(PermissionError, match="assessment_type_not_supported"):
		service.record_marketplace_risk("risk", "tenant-test", indicator["id"], "unknown", "medium", 0.8, "analyst", "evidence")
	risk = service.record_marketplace_risk("risk-ok", "tenant-test", indicator["id"], "marketplace_risk", "medium", 0.8, "analyst", "evidence")
	with pytest.raises(PermissionError, match="actor_reference_required"):
		service.record_threat_actor("actor", "tenant-test", indicator["id"], "", "medium", 0.8, "analyst", "evidence")
	with pytest.raises(PermissionError, match="referral_approval_required"):
		service.record_referral("referral", "tenant-test", risk["id"], "incident_response", "team", "", "evidence")
	with pytest.raises(PermissionError, match="dissemination_approval_required"):
		service.record_dissemination("dissemination", "tenant-test", risk["id"], "team", "CONFIDENTIAL", "", "evidence")
	with pytest.raises(PermissionError, match="reviewer_required"):
		service.record_review("review", "tenant-test", risk["id"], "", "approved", "evidence")
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.validate_batch("tenant-test", 1, event_stream="queue")
	with pytest.raises(PermissionError, match="darkweb_agent_runtime_not_supported"):
		service.register_darkweb_agent("agent", "tenant-test", "Bad Agent", "unsupported", "exposure_analyst", "scope")
	with pytest.raises(PermissionError, match="human_approval_required"):
		service.validate_agent_action("tenant-test", privileged_scope=True, human_approval_recorded=False)
	with pytest.raises(PermissionError, match="credential_use_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, credential_use_scope=True)
	with pytest.raises(PermissionError, match="contraband_transaction_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, contraband_transaction_scope=True)


def test_api_views_and_app_are_executable():
	api = _load_module("api_intel_darkweb", PACKAGE_DIR / "api.py")
	views = _load_module("views_intel_darkweb", PACKAGE_DIR / "views.py")
	app = _load_module("app_intel_darkweb", PACKAGE_DIR / "app.py")

	authority = api.record_authority({"tenant_id": "tenant-api", "authority_id": "api-auth", "authority_type": "consent", "scope_reference": "scope", "classification": "unclassified", "approver_id": "approver", "expires_at": "2026-12-31", "evidence_reference": "evidence"})
	program = api.record_program({"tenant_id": "tenant-api", "program_id": "api-program", "program_type": "brand_protection", "name": "Brand", "priority": "medium", "authority_id": authority["id"], "evidence_reference": "evidence"})
	source = api.register_source({"tenant_id": "tenant-api", "source_id": "api-source", "source_type": "paste", "network_type": "paste_site", "source_reference": "source-ref", "custodian_id": "custodian", "authority_id": authority["id"], "access_review_reference": "access-review", "evidence_reference": "evidence"})
	api.record_observation({"tenant_id": "tenant-api", "observation_id": "api-observation", "program_id": program["id"], "source_id": source["id"], "observation_type": "paste", "observation_reference": "obs-ref", "content_fingerprint": "hash", "observed_at": "2026-06-01", "confidence_score": 0.8, "evidence_reference": "evidence"})
	agent = api.register_darkweb_agent({"tenant_id": "tenant-api", "agent_id": "api-agent", "name": "Dark Web Agent", "runtime": "claude_code", "role": "exposure_analyst"})
	dashboard = views.dashboard_model(api.service(), "tenant-api")
	console = views.darkweb_console_model(api.service(), "tenant-api")
	self_test = app.self_test()
	semantic = app.semantic_model()

	assert agent["role"] == "exposure_analyst"
	assert dashboard["summary"]["authority_count"] == 1
	assert console["sources"][0]["id"] == source["id"]
	assert self_test["passed"] is True
	assert semantic["capabilities"]["intel_darkweb"]["screens"]["agents"]["route"] == "/intel-darkweb/agents"


def test_app_entrypoint_is_publishable():
	module = _load_module("publishable_app_intel_darkweb", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["intel_darkweb"]["streaming"]["processor"] == "bytewax"
