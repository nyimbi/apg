"""Executable Alert Management capability package tests."""

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
	module = _load_module("contract_intel_alerts", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "intel_alerts"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "alert_agent_workflow" in contract["provides"]
	assert "/intel-alerts/agents" in [route["path"] for route in contract["ui"]["routes"]]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert contract["configuration"]["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]


def test_rule_engine_blocks_missing_context_non_bytewax_and_prohibited_agent_actions():
	module = _load_module("rules_intel_alerts", PACKAGE_DIR / "capability_contract.py")

	assert module.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "alert_batch", "event_stream": "queue"})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "alert_agent_action", "privileged_scope": True, "human_approval_recorded": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "alert_agent_action", "unapproved_escalation_scope": True})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "alert_agent_action", "unapproved_notification_scope": True})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "alert_agent_action", "alert_suppression_scope": True})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "alert_agent_action", "evidence_fabrication_scope": True})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "alert_agent_action", "privacy_bypass_scope": True})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "alert_agent_action", "autonomous_closure_scope": True})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "alert_agent_action", "severity_downgrade_scope": True})["decision"] == "deny"


def test_service_executes_alert_lifecycle():
	service_module = _load_module("service_intel_alerts", PACKAGE_DIR / "service.py")
	service = service_module.AlertManagementService()

	authority = service.record_authority("auth-1", "tenant-test", "mission_order", "scope-ref", "confidential", "approver-1", "2026-12-31", "authority-evidence")
	workspace = service.record_workspace("workspace-1", "tenant-test", "watch_center", "Watch Center", "confidential", authority["id"], "workspace-evidence")
	rule = service.record_rule("rule-1", "tenant-test", workspace["id"], "threshold", "rule-ref", "high", "owner-1", "rule-evidence")
	signal = service.record_signal("signal-1", "tenant-test", rule["id"], "metric", "signal-ref", 0.82, "signal-evidence")
	alert = service.record_alert("alert-1", "tenant-test", signal["id"], "critical_alert", "critical", "alert-ref", "alert-evidence")
	escalation = service.record_escalation("escalation-1", "tenant-test", alert["id"], "supervisor", "target-ref", "approval-ref", "escalation-evidence")
	notification = service.record_notification("notification-1", "tenant-test", alert["id"], "in_app", "recipient-ref", "approval-ref", "notification-evidence")
	assignment = service.record_assignment("assignment-1", "tenant-test", alert["id"], "analyst", "assignee-1", "assignment-evidence")
	resolution = service.record_resolution("resolution-1", "tenant-test", alert["id"], "confirmed", "resolution-ref", "approval-ref", "resolution-evidence")
	review = service.record_review("review-1", "tenant-test", resolution["id"], "reviewer-1", "approved", "review-evidence")
	agent = service.register_alert_agent("agent-1", "tenant-test", "Alert Agent", "codex", "signal_triage", "triage support")
	bytewax_batch = service.validate_batch("tenant-test", 3)
	summary = service.dashboard_summary("tenant-test")

	assert authority["authority_type"] == "mission_order"
	assert workspace["workspace_type"] == "watch_center"
	assert rule["rule_type"] == "threshold"
	assert signal["signal_type"] == "metric"
	assert alert["alert_type"] == "critical_alert"
	assert escalation["escalation_type"] == "supervisor"
	assert notification["notification_type"] == "in_app"
	assert assignment["assignment_type"] == "analyst"
	assert resolution["resolution_type"] == "confirmed"
	assert review["status"] == "approved"
	assert agent["runtime"] == "codex"
	assert bytewax_batch["processor"] == "bytewax"
	assert summary["audit_event_count"] == 11


def test_service_keys_state_by_tenant_and_record_id():
	service_module = _load_module("tenant_service_intel_alerts", PACKAGE_DIR / "service.py")
	service = service_module.AlertManagementService()

	tenant_a = service.record_authority("shared-auth", "tenant-a", "mission_order", "scope-a", "confidential", "approver-a", "2026-12-31", "evidence-a")
	tenant_b = service.record_authority("shared-auth", "tenant-b", "consent", "scope-b", "unclassified", "approver-b", "2026-12-31", "evidence-b")
	service.record_workspace("shared-workspace", "tenant-a", "watch_center", "Workspace A", "confidential", tenant_a["id"], "evidence-a")
	service.record_workspace("shared-workspace", "tenant-b", "partner_alerting", "Workspace B", "unclassified", tenant_b["id"], "evidence-b")

	dashboard_a = service.dashboard_summary("tenant-a")
	dashboard_b = service.dashboard_summary("tenant-b")

	assert dashboard_a["authority_count"] == 1
	assert dashboard_b["authority_count"] == 1
	assert dashboard_a["workspace_count"] == 1
	assert dashboard_b["workspace_count"] == 1
	assert service._tenant_workspace_or_none("shared-workspace", "tenant-a").name == "Workspace A"
	assert service._tenant_workspace_or_none("shared-workspace", "tenant-b").name == "Workspace B"


def test_service_guardrails_reject_invalid_alert_actions():
	service_module = _load_module("guardrail_service_intel_alerts", PACKAGE_DIR / "service.py")
	service = service_module.AlertManagementService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.record_authority("auth", "", "mission_order", "scope", "confidential", "approver", "2026-12-31", "evidence")
	with pytest.raises(PermissionError, match="authority_type_not_supported"):
		service.record_authority("auth", "tenant-test", "unknown", "scope", "confidential", "approver", "2026-12-31", "evidence")
	authority = service.record_authority("auth-ok", "tenant-test", "mission_order", "scope", "confidential", "approver", "2026-12-31", "evidence")
	with pytest.raises(PermissionError, match="lawful_authority_required"):
		service.record_workspace("workspace", "tenant-test", "watch_center", "workspace", "confidential", "missing-auth", "evidence")
	workspace = service.record_workspace("workspace-ok", "tenant-test", "watch_center", "workspace", "confidential", authority["id"], "evidence")
	with pytest.raises(PermissionError, match="rule_owner_required"):
		service.record_rule("rule", "tenant-test", workspace["id"], "threshold", "rule", "high", "", "evidence")
	rule = service.record_rule("rule-ok", "tenant-test", workspace["id"], "threshold", "rule", "high", "owner", "evidence")
	with pytest.raises(PermissionError, match="confidence_score_invalid"):
		service.record_signal("signal", "tenant-test", rule["id"], "metric", "signal", 1.8, "evidence")
	signal = service.record_signal("signal-ok", "tenant-test", rule["id"], "metric", "signal", 0.8, "evidence")
	with pytest.raises(PermissionError, match="alert_type_not_supported"):
		service.record_alert("alert", "tenant-test", signal["id"], "unknown", "critical", "alert", "evidence")
	alert = service.record_alert("alert-ok", "tenant-test", signal["id"], "critical_alert", "critical", "alert", "evidence")
	with pytest.raises(PermissionError, match="escalation_approval_required"):
		service.record_escalation("escalation", "tenant-test", alert["id"], "supervisor", "target", "", "evidence")
	with pytest.raises(PermissionError, match="notification_approval_required"):
		service.record_notification("notification", "tenant-test", alert["id"], "in_app", "recipient", "", "evidence")
	with pytest.raises(PermissionError, match="assignee_required"):
		service.record_assignment("assignment", "tenant-test", alert["id"], "analyst", "", "evidence")
	with pytest.raises(PermissionError, match="resolution_approval_required"):
		service.record_resolution("resolution", "tenant-test", alert["id"], "confirmed", "resolution", "", "evidence")
	with pytest.raises(PermissionError, match="reviewer_required"):
		service.record_review("review", "tenant-test", alert["id"], "", "approved", "evidence")
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.validate_batch("tenant-test", 1, event_stream="queue")
	with pytest.raises(PermissionError, match="alert_agent_runtime_not_supported"):
		service.register_alert_agent("agent", "tenant-test", "Bad Agent", "unsupported", "signal_triage", "scope")
	with pytest.raises(PermissionError, match="alert_agent_scope_required"):
		service.register_alert_agent("agent", "tenant-test", "Alert Agent", "codex", "signal_triage", "")
	with pytest.raises(PermissionError, match="human_approval_required"):
		service.validate_agent_action("tenant-test", privileged_scope=True, human_approval_recorded=False)
	with pytest.raises(PermissionError, match="unapproved_escalation_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, unapproved_escalation_scope=True)
	with pytest.raises(PermissionError, match="unapproved_notification_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, unapproved_notification_scope=True)
	with pytest.raises(PermissionError, match="alert_suppression_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, alert_suppression_scope=True)
	with pytest.raises(PermissionError, match="evidence_fabrication_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, evidence_fabrication_scope=True)
	with pytest.raises(PermissionError, match="privacy_bypass_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, privacy_bypass_scope=True)
	with pytest.raises(PermissionError, match="autonomous_closure_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, autonomous_closure_scope=True)
	with pytest.raises(PermissionError, match="severity_downgrade_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, severity_downgrade_scope=True)


def test_api_views_and_app_are_executable():
	api = _load_module("api_intel_alerts", PACKAGE_DIR / "api.py")
	views = _load_module("views_intel_alerts", PACKAGE_DIR / "views.py")
	app = _load_module("app_intel_alerts", PACKAGE_DIR / "app.py")

	authority = api.record_authority({"tenant_id": "tenant-api", "authority_id": "api-auth", "authority_type": "consent", "scope_reference": "scope", "classification": "unclassified", "approver_id": "approver", "expires_at": "2026-12-31", "evidence_reference": "evidence"})
	workspace = api.record_workspace({"tenant_id": "tenant-api", "workspace_id": "api-workspace", "workspace_type": "partner_alerting", "name": "Partner Alerts", "classification": "unclassified", "authority_id": authority["id"], "evidence_reference": "evidence"})
	rule = api.record_rule({"tenant_id": "tenant-api", "rule_id": "api-rule", "workspace_id": workspace["id"], "rule_type": "watchlist", "rule_reference": "rule-ref", "severity": "medium", "owner_id": "owner", "evidence_reference": "evidence"})
	signal = api.record_signal({"tenant_id": "tenant-api", "signal_id": "api-signal", "rule_id": rule["id"], "signal_type": "partner_notice", "signal_reference": "signal-ref", "confidence_score": 0.72, "evidence_reference": "evidence"})
	alert = api.record_alert({"tenant_id": "tenant-api", "alert_id": "api-alert", "signal_id": signal["id"], "alert_type": "watchlist_hit", "severity": "medium", "alert_reference": "alert-ref", "evidence_reference": "evidence"})
	agent = api.register_alert_agent({"tenant_id": "tenant-api", "agent_id": "api-agent", "name": "Alert Agent", "runtime": "claude_code", "role": "signal_triage"})
	batch = api.validate_batch({"tenant_id": "tenant-api", "item_count": 2})
	dashboard = views.dashboard_model(api.service(), "tenant-api")
	console = views.alert_console_model(api.service(), "tenant-api")
	workbench = views.agent_workbench_model(api.service(), "tenant-api")
	self_test = app.self_test()
	semantic = app.semantic_model()

	assert agent["role"] == "signal_triage"
	assert batch["processor"] == "bytewax"
	assert dashboard["summary"]["authority_count"] == 1
	assert console["alerts"][0]["id"] == alert["id"]
	assert workbench["agents"][0]["id"] == agent["id"]
	assert self_test["passed"] is True
	assert semantic["capabilities"]["intel_alerts"]["screens"]["agents"]["route"] == "/intel-alerts/agents"


def test_app_entrypoint_is_publishable():
	module = _load_module("publishable_app_intel_alerts", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["intel_alerts"]["streaming"]["processor"] == "bytewax"

