"""Executable Intelligence Dashboard capability package tests."""

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
	module = _load_module("contract_intel_dashboard", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-test")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "intel_dashboard"
	assert contract["streaming"]["processor"] == "bytewax"
	assert "dashboard_agent_workflow" in contract["provides"]
	assert "/intel-dashboard/agents" in [route["path"] for route in contract["ui"]["routes"]]
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert contract["configuration"]["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]


def test_rule_engine_blocks_missing_context_non_bytewax_and_prohibited_agent_actions():
	module = _load_module("rules_intel_dashboard", PACKAGE_DIR / "capability_contract.py")

	assert module.evaluate_capability_rules({"tenant_context_present": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "dashboard_batch", "event_stream": "queue"})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "dashboard_agent_action", "privileged_scope": True, "human_approval_recorded": False})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "dashboard_agent_action", "uncited_metric_scope": True})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "dashboard_agent_action", "classification_leak_scope": True})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "dashboard_agent_action", "source_tampering_scope": True})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "dashboard_agent_action", "privacy_bypass_scope": True})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "dashboard_agent_action", "autonomous_share_scope": True})["decision"] == "deny"
	assert module.evaluate_capability_rules({"tenant_id": "tenant-test", "tenant_context_present": True, "operation": "dashboard_agent_action", "unapproved_public_view_scope": True})["decision"] == "deny"


def test_service_executes_dashboard_lifecycle():
	service_module = _load_module("service_intel_dashboard", PACKAGE_DIR / "service.py")
	service = service_module.IntelligenceDashboardService()

	authority = service.record_authority("auth-1", "tenant-test", "mission_order", "scope-ref", "confidential", "approver-1", "2026-12-31", "authority-evidence")
	workspace = service.record_workspace("workspace-1", "tenant-test", "operations_center", "Operations Center", "confidential", authority["id"], "workspace-evidence")
	dashboard = service.record_dashboard("dashboard-1", "tenant-test", workspace["id"], "operational", "Operations", "owner-1", "confidential", "dashboard-evidence")
	source = service.record_source("source-1", "tenant-test", dashboard["id"], "capability_summary", "source-ref", "custodian-1", "source-evidence")
	metric = service.record_metric("metric-1", "tenant-test", source["id"], "risk_score", "metric-ref", 0.82, "metric-evidence")
	widget = service.record_widget("widget-1", "tenant-test", dashboard["id"], "kpi_tile", "widget-ref", metric["id"], "widget-evidence")
	filter_item = service.record_filter("filter-1", "tenant-test", dashboard["id"], "time_range", "filter-ref", "filter-evidence")
	view = service.record_view("view-1", "tenant-test", dashboard["id"], "analyst", "view-ref", "analyst", "view-evidence")
	share = service.record_share("share-1", "tenant-test", dashboard["id"], "internal", "recipient-ref", "approval-ref", "share-evidence")
	review = service.record_review("review-1", "tenant-test", share["id"], "reviewer-1", "approved", "review-evidence")
	agent = service.register_dashboard_agent("agent-1", "tenant-test", "Dashboard Agent", "codex", "layout_designer", "layout support")
	bytewax_batch = service.validate_batch("tenant-test", 3)
	summary = service.dashboard_summary("tenant-test")

	assert authority["authority_type"] == "mission_order"
	assert workspace["workspace_type"] == "operations_center"
	assert dashboard["dashboard_type"] == "operational"
	assert source["source_type"] == "capability_summary"
	assert metric["metric_type"] == "risk_score"
	assert widget["widget_type"] == "kpi_tile"
	assert filter_item["filter_type"] == "time_range"
	assert view["view_type"] == "analyst"
	assert share["share_type"] == "internal"
	assert review["status"] == "approved"
	assert agent["runtime"] == "codex"
	assert bytewax_batch["processor"] == "bytewax"
	assert summary["audit_event_count"] == 11


def test_service_keys_state_by_tenant_and_record_id():
	service_module = _load_module("tenant_service_intel_dashboard", PACKAGE_DIR / "service.py")
	service = service_module.IntelligenceDashboardService()

	tenant_a = service.record_authority("shared-auth", "tenant-a", "mission_order", "scope-a", "confidential", "approver-a", "2026-12-31", "evidence-a")
	tenant_b = service.record_authority("shared-auth", "tenant-b", "consent", "scope-b", "unclassified", "approver-b", "2026-12-31", "evidence-b")
	service.record_workspace("shared-workspace", "tenant-a", "operations_center", "Workspace A", "confidential", tenant_a["id"], "evidence-a")
	service.record_workspace("shared-workspace", "tenant-b", "partner_view", "Workspace B", "unclassified", tenant_b["id"], "evidence-b")

	dashboard_a = service.dashboard_summary("tenant-a")
	dashboard_b = service.dashboard_summary("tenant-b")

	assert dashboard_a["authority_count"] == 1
	assert dashboard_b["authority_count"] == 1
	assert dashboard_a["workspace_count"] == 1
	assert dashboard_b["workspace_count"] == 1
	assert service._tenant_workspace_or_none("shared-workspace", "tenant-a").name == "Workspace A"
	assert service._tenant_workspace_or_none("shared-workspace", "tenant-b").name == "Workspace B"


def test_service_guardrails_reject_invalid_dashboard_actions():
	service_module = _load_module("guardrail_service_intel_dashboard", PACKAGE_DIR / "service.py")
	service = service_module.IntelligenceDashboardService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.record_authority("auth", "", "mission_order", "scope", "confidential", "approver", "2026-12-31", "evidence")
	with pytest.raises(PermissionError, match="authority_type_not_supported"):
		service.record_authority("auth", "tenant-test", "unknown", "scope", "confidential", "approver", "2026-12-31", "evidence")
	authority = service.record_authority("auth-ok", "tenant-test", "mission_order", "scope", "confidential", "approver", "2026-12-31", "evidence")
	with pytest.raises(PermissionError, match="lawful_authority_required"):
		service.record_workspace("workspace", "tenant-test", "operations_center", "workspace", "confidential", "missing-auth", "evidence")
	workspace = service.record_workspace("workspace-ok", "tenant-test", "operations_center", "workspace", "confidential", authority["id"], "evidence")
	with pytest.raises(PermissionError, match="dashboard_owner_required"):
		service.record_dashboard("dashboard", "tenant-test", workspace["id"], "operational", "title", "", "confidential", "evidence")
	dashboard = service.record_dashboard("dashboard-ok", "tenant-test", workspace["id"], "operational", "title", "owner", "confidential", "evidence")
	with pytest.raises(PermissionError, match="source_custodian_required"):
		service.record_source("source", "tenant-test", dashboard["id"], "capability_summary", "source", "", "evidence")
	source = service.record_source("source-ok", "tenant-test", dashboard["id"], "capability_summary", "source", "custodian", "evidence")
	with pytest.raises(PermissionError, match="confidence_score_invalid"):
		service.record_metric("metric", "tenant-test", source["id"], "risk_score", "metric", 1.8, "evidence")
	metric = service.record_metric("metric-ok", "tenant-test", source["id"], "risk_score", "metric", 0.8, "evidence")
	with pytest.raises(PermissionError, match="widget_type_not_supported"):
		service.record_widget("widget", "tenant-test", dashboard["id"], "unknown", "widget", metric["id"], "evidence")
	with pytest.raises(PermissionError, match="filter_type_not_supported"):
		service.record_filter("filter", "tenant-test", dashboard["id"], "unknown", "filter", "evidence")
	with pytest.raises(PermissionError, match="viewer_role_required"):
		service.record_view("view", "tenant-test", dashboard["id"], "analyst", "view", "", "evidence")
	with pytest.raises(PermissionError, match="share_approval_required"):
		service.record_share("share", "tenant-test", dashboard["id"], "internal", "recipient", "", "evidence")
	with pytest.raises(PermissionError, match="reviewer_required"):
		service.record_review("review", "tenant-test", dashboard["id"], "", "approved", "evidence")
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.validate_batch("tenant-test", 1, event_stream="queue")
	with pytest.raises(PermissionError, match="dashboard_agent_runtime_not_supported"):
		service.register_dashboard_agent("agent", "tenant-test", "Bad Agent", "unsupported", "layout_designer", "scope")
	with pytest.raises(PermissionError, match="dashboard_agent_scope_required"):
		service.register_dashboard_agent("agent", "tenant-test", "Dashboard Agent", "codex", "layout_designer", "")
	with pytest.raises(PermissionError, match="human_approval_required"):
		service.validate_agent_action("tenant-test", privileged_scope=True, human_approval_recorded=False)
	with pytest.raises(PermissionError, match="uncited_metric_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, uncited_metric_scope=True)
	with pytest.raises(PermissionError, match="classification_leak_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, classification_leak_scope=True)
	with pytest.raises(PermissionError, match="source_tampering_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, source_tampering_scope=True)
	with pytest.raises(PermissionError, match="privacy_bypass_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, privacy_bypass_scope=True)
	with pytest.raises(PermissionError, match="autonomous_share_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, autonomous_share_scope=True)
	with pytest.raises(PermissionError, match="unapproved_public_view_scope_denied"):
		service.validate_agent_action("tenant-test", privileged_scope=False, human_approval_recorded=False, unapproved_public_view_scope=True)


def test_api_views_and_app_are_executable():
	api = _load_module("api_intel_dashboard", PACKAGE_DIR / "api.py")
	views = _load_module("views_intel_dashboard", PACKAGE_DIR / "views.py")
	app = _load_module("app_intel_dashboard", PACKAGE_DIR / "app.py")

	authority = api.record_authority({"tenant_id": "tenant-api", "authority_id": "api-auth", "authority_type": "consent", "scope_reference": "scope", "classification": "unclassified", "approver_id": "approver", "expires_at": "2026-12-31", "evidence_reference": "evidence"})
	workspace = api.record_workspace({"tenant_id": "tenant-api", "workspace_id": "api-workspace", "workspace_type": "partner_view", "name": "Partner Dashboard", "classification": "unclassified", "authority_id": authority["id"], "evidence_reference": "evidence"})
	dashboard = api.record_dashboard({"tenant_id": "tenant-api", "dashboard_id": "api-dashboard", "workspace_id": workspace["id"], "dashboard_type": "partner", "title": "Partner View", "owner_id": "owner", "classification": "unclassified", "evidence_reference": "evidence"})
	source = api.record_source({"tenant_id": "tenant-api", "source_id": "api-source", "dashboard_id": dashboard["id"], "source_type": "reporting_product", "source_reference": "source-ref", "custodian_id": "custodian", "evidence_reference": "evidence"})
	api.record_metric({"tenant_id": "tenant-api", "metric_id": "api-metric", "source_id": source["id"], "metric_type": "count", "metric_reference": "metric-ref", "confidence_score": 0.72, "evidence_reference": "evidence"})
	agent = api.register_dashboard_agent({"tenant_id": "tenant-api", "agent_id": "api-agent", "name": "Dashboard Agent", "runtime": "claude_code", "role": "layout_designer"})
	batch = api.validate_batch({"tenant_id": "tenant-api", "item_count": 2})
	dashboard_model = views.dashboard_model(api.service(), "tenant-api")
	console = views.dashboard_console_model(api.service(), "tenant-api")
	workbench = views.agent_workbench_model(api.service(), "tenant-api")
	self_test = app.self_test()
	semantic = app.semantic_model()

	assert agent["role"] == "layout_designer"
	assert batch["processor"] == "bytewax"
	assert dashboard_model["summary"]["authority_count"] == 1
	assert console["dashboards"][0]["id"] == dashboard["id"]
	assert workbench["agents"][0]["id"] == agent["id"]
	assert self_test["passed"] is True
	assert semantic["capabilities"]["intel_dashboard"]["screens"]["agents"]["route"] == "/intel-dashboard/agents"


def test_app_entrypoint_is_publishable():
	module = _load_module("publishable_app_intel_dashboard", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert model["capabilities"]["intel_dashboard"]["streaming"]["processor"] == "bytewax"

