"""Executable Real-Time Monitoring capability package tests."""

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
	module = _load_module("contract_intel_monitoring", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "intel_monitoring"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "monitoring_agent_workflow" in contract["provides"]
	assert "/intel-monitoring/agents" in [route["path"] for route in contract["ui"]["routes"]]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert contract["configuration"]["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]


def test_rule_engine_blocks_missing_context_non_bytewax_and_prohibited_agent_actions():
	module = _load_module("rules_intel_monitoring", PACKAGE_DIR / "capability_contract.py")

	assert module.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "monitoring_batch", "event_stream": "queue"})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "monitoring_agent_action", "privileged_scope": True, "human_approval_recorded": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "monitoring_agent_action", "destructive_action_scope": True})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "monitoring_agent_action", "data_exfiltration_scope": True})["decision"] == "deny"


def test_service_executes_monitoring_lifecycle():
	service_module = _load_module("service_intel_monitoring", PACKAGE_DIR / "service.py")
	service = service_module.RealTimeMonitoringService()

	authority = service.record_authority("auth-1", "tenant-test", "security_monitoring_authority", "scope-ref", "confidential", "approver-1", "2026-12-31", "authority-evidence")
	policy = service.record_policy("policy-1", "tenant-test", "security", "Security monitoring", "high", authority["id"], "policy-evidence")
	source = service.register_source("source-1", "tenant-test", "event_stream", "events-prod", "owner-1", authority["id"], "access-review", "source-evidence")
	watch = service.record_watch("watch-1", "tenant-test", policy["id"], source["id"], "pattern", "event.type == login_failure", "standard", "watch-evidence")
	event = service.record_event("event-1", "tenant-test", watch["id"], "alert", "event-ref", "sha256:abc", "2026-06-01T00:00:00Z", 0.82, "event-evidence")
	signal = service.record_signal("signal-1", "tenant-test", event["id"], "threat_signal", "high", 0.86, "analyst-1", "signal-evidence")
	incident = service.record_incident("incident-1", "tenant-test", signal["id"], "security_incident", "high", 0.84, "analyst-1", "incident-evidence")
	referral = service.record_referral("referral-1", "tenant-test", incident["id"], "incident_response", "soc-team", "approval-ref", "referral-evidence")
	dissemination = service.record_dissemination("dissemination-1", "tenant-test", incident["id"], "security-team", "CONFIDENTIAL", "approval-ref", "dissemination-evidence")
	review = service.record_review("review-1", "tenant-test", dissemination["id"], "reviewer-1", "approved", "review-evidence")
	agent = service.register_monitoring_agent("agent-1", "tenant-test", "Monitoring Agent", "codex", "signal_analyst", "signal analysis")
	bytewax_batch = service.validate_batch("tenant-test", 3)
	summary = service.dashboard_summary("tenant-test")

	assert authority["authority_type"] == "security_monitoring_authority"
	assert policy["policy_type"] == "security"
	assert source["source_type"] == "event_stream"
	assert watch["watch_type"] == "pattern"
	assert event["event_type"] == "alert"
	assert signal["signal_type"] == "threat_signal"
	assert incident["incident_type"] == "security_incident"
	assert referral["referral_type"] == "incident_response"
	assert dissemination["release_marking"] == "CONFIDENTIAL"
	assert review["status"] == "approved"
	assert agent["runtime"] == "codex"
	assert bytewax_batch["processor"] == "bytewax"
	assert summary["audit_event_count"] == 11


def test_service_keys_state_by_tenant_and_record_id():
	service_module = _load_module("tenant_service_intel_monitoring", PACKAGE_DIR / "service.py")
	service = service_module.RealTimeMonitoringService()

	tenant_a = service.record_authority("shared-auth", "tenant-a", "security_monitoring_authority", "scope-a", "confidential", "approver-a", "2026-12-31", "evidence-a")
	tenant_b = service.record_authority("shared-auth", "tenant-b", "consent", "scope-b", "unclassified", "approver-b", "2026-12-31", "evidence-b")
	service.record_policy("shared-policy", "tenant-a", "security", "Policy A", "medium", tenant_a["id"], "evidence-a")
	service.record_policy("shared-policy", "tenant-b", "operations", "Policy B", "low", tenant_b["id"], "evidence-b")

	dashboard_a = service.dashboard_summary("tenant-a")
	dashboard_b = service.dashboard_summary("tenant-b")

	assert dashboard_a["authority_count"] == 1
	assert dashboard_b["authority_count"] == 1
	assert dashboard_a["policy_count"] == 1
	assert dashboard_b["policy_count"] == 1
	assert service._tenant_policy_or_none("shared-policy", "tenant-a").name == "Policy A"
	assert service._tenant_policy_or_none("shared-policy", "tenant-b").name == "Policy B"


def test_service_guardrails_reject_invalid_monitoring_actions():
	service_module = _load_module("guardrail_service_intel_monitoring", PACKAGE_DIR / "service.py")
	service = service_module.RealTimeMonitoringService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.record_authority("auth", "", "security_monitoring_authority", "scope", "confidential", "approver", "2026-12-31", "evidence")
	with pytest.raises(PermissionError, match="authority_type_not_supported"):
		service.record_authority("auth", "tenant-test", "unknown", "scope", "confidential", "approver", "2026-12-31", "evidence")
	authority = service.record_authority("auth-ok", "tenant-test", "security_monitoring_authority", "scope", "confidential", "approver", "2026-12-31", "evidence")
	with pytest.raises(PermissionError, match="lawful_authority_required"):
		service.record_policy("policy", "tenant-test", "security", "policy", "medium", "missing-auth", "evidence")
	policy = service.record_policy("policy-ok", "tenant-test", "security", "policy", "medium", authority["id"], "evidence")
	with pytest.raises(PermissionError, match="source_access_review_required"):
		service.register_source("source", "tenant-test", "event_stream", "ref", "owner", authority["id"], "", "evidence")
	source = service.register_source("source-ok", "tenant-test", "event_stream", "ref", "owner", authority["id"], "access-review", "evidence")
	other_authority = service.record_authority("auth-other", "tenant-test", "consent", "scope", "confidential", "approver", "2026-12-31", "evidence")
	other_policy = service.record_policy("policy-other", "tenant-test", "operations", "ops", "low", other_authority["id"], "evidence")
	with pytest.raises(PermissionError, match="authority_mismatch"):
		service.record_watch("watch", "tenant-test", other_policy["id"], source["id"], "pattern", "expr", "standard", "evidence")
	watch = service.record_watch("watch-ok", "tenant-test", policy["id"], source["id"], "pattern", "expr", "standard", "evidence")
	with pytest.raises(PermissionError, match="confidence_score_invalid"):
		service.record_event("event", "tenant-test", watch["id"], "alert", "ref", "hash", "2026-06-01", 1.8, "evidence")
	event = service.record_event("event-ok", "tenant-test", watch["id"], "alert", "ref", "hash", "2026-06-01", 0.8, "evidence")
	with pytest.raises(PermissionError, match="signal_type_not_supported"):
		service.record_signal("signal", "tenant-test", event["id"], "unknown", "high", 0.8, "analyst", "evidence")
	signal = service.record_signal("signal-ok", "tenant-test", event["id"], "threat_signal", "medium", 0.8, "analyst", "evidence")
	with pytest.raises(PermissionError, match="incident_type_not_supported"):
		service.record_incident("incident", "tenant-test", signal["id"], "unknown", "medium", 0.8, "analyst", "evidence")
	incident = service.record_incident("incident-ok", "tenant-test", signal["id"], "security_incident", "medium", 0.8, "analyst", "evidence")
	with pytest.raises(PermissionError, match="referral_approval_required"):
		service.record_referral("referral", "tenant-test", incident["id"], "incident_response", "team", "", "evidence")
	with pytest.raises(PermissionError, match="dissemination_approval_required"):
		service.record_dissemination("dissemination", "tenant-test", incident["id"], "team", "CONFIDENTIAL", "", "evidence")
	with pytest.raises(PermissionError, match="reviewer_required"):
		service.record_review("review", "tenant-test", incident["id"], "", "approved", "evidence")
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.validate_batch("tenant-test", 1, event_stream="queue")
	with pytest.raises(PermissionError, match="monitoring_agent_runtime_not_supported"):
		service.register_monitoring_agent("agent", "tenant-test", "Bad Agent", "unsupported", "signal_analyst", "scope")
	with pytest.raises(PermissionError, match="human_approval_required"):
		service.validate_agent_action("tenant-test", privileged_scope=True, human_approval_recorded=False)
	with pytest.raises(PermissionError, match="destructive_action_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, destructive_action_scope=True)
	with pytest.raises(PermissionError, match="data_exfiltration_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, data_exfiltration_scope=True)


def test_api_views_and_app_are_executable():
	api = _load_module("api_intel_monitoring", PACKAGE_DIR / "api.py")
	views = _load_module("views_intel_monitoring", PACKAGE_DIR / "views.py")
	app = _load_module("app_intel_monitoring", PACKAGE_DIR / "app.py")

	authority = api.record_authority({"tenant_id": "tenant-api", "authority_id": "api-auth", "authority_type": "consent", "scope_reference": "scope", "classification": "unclassified", "approver_id": "approver", "expires_at": "2026-12-31", "evidence_reference": "evidence"})
	policy = api.record_policy({"tenant_id": "tenant-api", "policy_id": "api-policy", "policy_type": "security", "name": "Security", "severity_floor": "medium", "authority_id": authority["id"], "evidence_reference": "evidence"})
	source = api.register_source({"tenant_id": "tenant-api", "source_id": "api-source", "source_type": "event_stream", "source_reference": "source-ref", "owner_id": "owner", "authority_id": authority["id"], "access_review_reference": "access-review", "evidence_reference": "evidence"})
	api.record_watch({"tenant_id": "tenant-api", "watch_id": "api-watch", "policy_id": policy["id"], "source_id": source["id"], "watch_type": "pattern", "watch_expression": "expr", "retention_class": "standard", "evidence_reference": "evidence"})
	agent = api.register_monitoring_agent({"tenant_id": "tenant-api", "agent_id": "api-agent", "name": "Monitoring Agent", "runtime": "claude_code", "role": "signal_analyst"})
	dashboard = views.dashboard_model(api.service(), "tenant-api")
	console = views.monitoring_console_model(api.service(), "tenant-api")
	self_test = app.self_test()
	semantic = app.semantic_model()

	assert agent["role"] == "signal_analyst"
	assert dashboard["summary"]["authority_count"] == 1
	assert console["sources"][0]["id"] == source["id"]
	assert self_test["passed"] is True
	assert semantic["capabilities"]["intel_monitoring"]["screens"]["agents"]["route"] == "/intel-monitoring/agents"


def test_app_entrypoint_is_publishable():
	module = _load_module("publishable_app_intel_monitoring", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["intel_monitoring"]["streaming"]["processor"] == "bytewax"
