"""Regression coverage for the PLFD executable capability contract."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from capabilities.common.plfd import register_capability
from capabilities.common.plfd.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.plfd.service import PlfdService
from capabilities.common.plfd import views


PACKAGE_DIR = Path(__file__).resolve().parent


def _load_module(name: str, path: Path):
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules[name] = module
	spec.loader.exec_module(module)
	return module


def test_contract_exposes_configuration_rules_ui_theme_and_streaming():
	contract = get_capability_contract("tenant-plfd", {"operations": {"change_window_required": False}})

	assert contract["capability"] == "plfd"
	assert contract["configuration"]["tenant_id"] == "tenant-plfd"
	assert contract["configuration"]["operations"]["change_window_required"] is False
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"foundation",
		"baselines",
		"operations",
		"plfd_agents",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	]
	assert contract["provides"] == [
		"foundation_registry",
		"dependency_posture",
		"configuration_baselines",
		"readiness_gates",
		"platform_governance",
		"plfd_agents",
	]
	assert contract["requires"] == ["conf", "mten", "auth", "audl"]
	assert contract["streaming"]["processor"] == "bytewax"
	assert contract["configuration"]["plfd_agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "services", "dependencies", "baselines", "readiness", "changes", "agents", "governance", "audit", "settings"}
	assert contract["theme"]["name"] == "plfd_platform_foundation"


def test_rule_engine_enforces_plfd_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "register_foundation_service",
		"service_owner_assigned": False,
		"tier_classified": False,
		"readiness_score_present": False,
		"configuration_baseline_present": False,
		"affected_capability_count": 12,
		"broad_review_recorded": False,
	})
	change_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "approve_platform_change",
		"dependencies_healthy": False,
		"approval_recorded": False,
		"configuration_baseline_present": True,
		"security_review_recorded": False,
		"change_window_present": False,
		"rollback_plan_present": False,
		"event_stream": "other-stream",
	})
	baseline_result = evaluate_capability_rules({"operation": "attach_baseline", "baseline_evidence_present": False, "baseline_approver_present": False})
	agent_result = evaluate_capability_rules({"plfd_agent_present": True, "agent_runtime_supported": False})
	batch_result = evaluate_capability_rules({"requested_operation": "batch_foundation_mutation", "event_stream": "other-stream"})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {
		"tenant_context_required",
		"foundation_service_requires_owner",
		"foundation_service_requires_tier",
		"foundation_service_requires_readiness_score",
		"configuration_baseline_required",
		"broad_platform_change_requires_review",
	}
	assert set(change_result["matched_rules"]) == {
		"dependency_health_required",
		"platform_change_requires_approval",
		"platform_change_requires_security_review",
		"platform_change_requires_window",
		"platform_change_requires_rollback",
		"platform_change_requires_bytewax_stream",
	}
	assert set(baseline_result["matched_rules"]) == {"baseline_requires_evidence", "baseline_requires_approver"}
	assert agent_result["matched_rules"] == ["plfd_agent_runtime_supported"]
	assert batch_result["matched_rules"] == ["batch_foundation_mutation_requires_bytewax"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "plfd"
	assert "mten" in registration["dependencies"]
	assert registration["ui_components"]["baselines"] == "/plfd/baselines"
	assert registration["ui_components"]["agents"] == "/plfd/agents"
	assert registration["streaming"]["processor"] == "bytewax"
	assert "plfd:manage_baselines" in registration["permissions"]


def test_service_runs_foundation_readiness_and_change_lifecycle():
	service = PlfdService()

	service.register_foundation_service(
		"conf",
		"tenant-plfd",
		"Configuration service",
		"platform-owner",
		"core",
		readiness_score=96,
		configuration_baseline_present=True,
		monitoring_enabled=True,
		rollback_plan_ref="rollback-conf",
		change_window_ref="cw-001",
	)
	foundation = service.register_foundation_service(
		"plfd-core",
		"tenant-plfd",
		"Platform foundation",
		"foundation-owner",
		"core",
		dependencies=["conf"],
		readiness_score=94,
		configuration_baseline_present=True,
		monitoring_enabled=True,
		rollback_plan_ref="rollback-plfd",
		change_window_ref="cw-001",
	)
	dependency = service.record_dependency("dep-plfd-conf", "tenant-plfd", "plfd-core", "conf", "healthy", evidence_ref="health-conf")
	for baseline_type in ("configuration", "tenant", "auth", "audit"):
		service.attach_baseline(
			f"base-{baseline_type}",
			"tenant-plfd",
			"plfd-core",
			baseline_type,
			evidence_ref=f"evidence-{baseline_type}",
			approved_by="platform-reviewer",
		)
	readiness = service.assess_readiness("ready-plfd", "tenant-plfd", "plfd-core")
	change = service.propose_platform_change(
		"change-plfd",
		"tenant-plfd",
		"plfd-core",
		"Update platform baseline",
		"foundation-owner",
		affected_capability_count=8,
		approval_recorded=True,
		security_review_recorded=True,
	)
	approved = service.approve_platform_change("change-plfd", "tenant-plfd", "platform-approver")
	agent = service.register_plfd_agent(
		tenant_id="tenant-plfd",
		name="Readiness reviewer",
		runtime="codex",
		role="readiness_reviewer",
		scope="review baseline, dependency, monitoring, rollback, and change-window gates",
	)
	summary = service.dashboard_summary("tenant-plfd")
	dashboard = views.dashboard_model(service, "tenant-plfd")
	dependency_map = views.dependency_map_model(service, "tenant-plfd")
	change_queue = views.change_queue_model(service, "tenant-plfd")

	assert foundation["tier"] == "core"
	assert dependency["health_status"] == "healthy"
	assert readiness["status"] == "ready"
	assert readiness["issues"] == []
	assert change["status"] == "pending_approval"
	assert approved["status"] == "approved"
	assert approved["approval_recorded"] is True
	assert agent["runtime"] == "codex"
	assert agent["role"] == "readiness_reviewer"
	assert summary["ready_service_count"] == 1
	assert summary["approved_change_count"] == 1
	assert summary["plfd_agent_count"] == 1
	assert service.validate_batch_foundation_mutation("bytewax")["decision"] == "allow"
	assert service.validate_batch_foundation_mutation("other-stream")["decision"] == "deny"
	assert dashboard["summary"]["service_count"] == 2
	assert dashboard["streaming"]["processor"] == "bytewax"
	assert len(dependency_map["edges"]) == 1
	assert change_queue["changes"][0]["id"] == "change-plfd"
	assert views.plfd_agent_model(service, "tenant-plfd")["plfd_agents"][0]["role"] == "readiness_reviewer"
	assert views.audit_trail_model(service, "tenant-plfd")["audit_events"]
	assert views.foundation_policy_model(service, "tenant-plfd")["streaming"]["processor"] == "bytewax"
	assert len(service.list_audit_events("tenant-plfd")) >= 10


def test_service_enforces_platform_foundation_guardrails():
	service = PlfdService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.register_foundation_service("svc-no-tenant", "", "No tenant", "owner", "core")
	with pytest.raises(PermissionError, match="service_owner_required"):
		service.register_foundation_service("svc-no-owner", "tenant-plfd", "No owner", "", "core")
	with pytest.raises(PermissionError, match="tier_classification_required"):
		service.register_foundation_service("svc-no-tier", "tenant-plfd", "No tier", "owner", "")
	with pytest.raises(PermissionError, match="configuration_baseline_required"):
		service.register_foundation_service("svc-no-baseline", "tenant-plfd", "No baseline", "owner", "core", configuration_baseline_present=False)

	service.register_foundation_service("auth", "tenant-plfd", "Auth service", "auth-owner", "core", readiness_score=95, monitoring_enabled=True, rollback_plan_ref="rollback-auth", change_window_ref="cw-auth")
	service.register_foundation_service("plfd-core", "tenant-plfd", "Platform foundation", "owner", "core", readiness_score=95, monitoring_enabled=True, rollback_plan_ref="rollback-plfd", change_window_ref="cw-plfd")

	with pytest.raises(PermissionError, match="dependency_evidence_required"):
		service.record_dependency("dep-no-evidence", "tenant-plfd", "plfd-core", "auth", "healthy")
	with pytest.raises(PermissionError, match="baseline_evidence_required"):
		service.attach_baseline("base-bad", "tenant-plfd", "plfd-core", "configuration", "", "reviewer")
	with pytest.raises(PermissionError, match="baseline_approver_required"):
		service.attach_baseline("base-no-approver", "tenant-plfd", "plfd-core", "configuration", "evidence", "")

	service.record_dependency("dep-bad", "tenant-plfd", "plfd-core", "auth", "unhealthy", evidence_ref="health-auth")
	service.propose_platform_change("change-unhealthy", "tenant-plfd", "plfd-core", "Unhealthy dependency change", "owner", affected_capability_count=1, approval_recorded=True, security_review_recorded=True)
	with pytest.raises(PermissionError, match="dependency_health_required"):
		service.approve_platform_change("change-unhealthy", "tenant-plfd", "approver")

	service.register_foundation_service("regy", "tenant-plfd", "Registry", "registry-owner", "shared", readiness_score=90, monitoring_enabled=True, rollback_plan_ref="rollback-regy", change_window_ref="cw-regy")
	for baseline_type in ("configuration", "tenant", "auth", "audit"):
		service.attach_baseline(f"regy-{baseline_type}", "tenant-plfd", "regy", baseline_type, f"ev-{baseline_type}", "reviewer")

	with pytest.raises(PermissionError, match="change_owner_required"):
		service.propose_platform_change("change-no-owner", "tenant-plfd", "regy", "No owner", "", 1)
	with pytest.raises(PermissionError, match="affected_capability_required"):
		service.propose_platform_change("change-no-scope", "tenant-plfd", "regy", "No scope", "owner", 0)
	service.propose_platform_change("change-no-approval", "tenant-plfd", "regy", "No approval", "owner", 1)
	with pytest.raises(PermissionError, match="platform_change_approval_required"):
		service.approve_platform_change("change-no-approval", "tenant-plfd", "approver", approval_recorded=False)
	service.propose_platform_change("change-broad", "tenant-plfd", "regy", "Broad platform change", "owner", 12, approval_recorded=True, security_review_recorded=True)
	with pytest.raises(PermissionError, match="broad_platform_review_required"):
		service.approve_platform_change("change-broad", "tenant-plfd", "approver")
	service.propose_platform_change("change-no-security", "tenant-plfd", "regy", "No security review", "owner", 1, approval_recorded=True, security_review_recorded=False)
	with pytest.raises(PermissionError, match="security_review_required"):
		service.approve_platform_change("change-no-security", "tenant-plfd", "approver")
	service.propose_platform_change("change-other-stream", "tenant-plfd", "regy", "Other stream", "owner", 1, approval_recorded=True, security_review_recorded=True)
	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.approve_platform_change("change-other-stream", "tenant-plfd", "approver", event_stream="other-stream")
	with pytest.raises(PermissionError, match="plfd_agent_runtime_not_supported"):
		service.register_plfd_agent("tenant-plfd", "Unsupported", "unsupported", "readiness_reviewer", "review")
	with pytest.raises(PermissionError, match="plfd_agent_scope_required"):
		service.register_plfd_agent("tenant-plfd", "No scope", "codex", "readiness_reviewer", "")


def test_lifecycle_ids_are_tenant_scoped():
	service = PlfdService()

	for tenant_id, owner, score in (
		("tenant-a", "owner-a", 91),
		("tenant-b", "owner-b", 83),
	):
		service.register_foundation_service("shared-service", tenant_id, "Shared", owner, "core", readiness_score=score, monitoring_enabled=True, rollback_plan_ref="rollback", change_window_ref="cw")
		service.attach_baseline("shared-baseline", tenant_id, "shared-service", "configuration", "evidence", owner)
		service.assess_readiness("shared-assessment", tenant_id, "shared-service")
		service.register_plfd_agent(tenant_id, "Reviewer", "codex", "readiness_reviewer", "review tenant foundation", agent_id="shared-agent")

	assert service.list_services("tenant-a")[0]["owner"] == "owner-a"
	assert service.list_services("tenant-b")[0]["owner"] == "owner-b"
	assert service.list_readiness_assessments("tenant-a")[0]["score"] == 91
	assert service.list_readiness_assessments("tenant-b")[0]["score"] == 83
	assert service.list_plfd_agents("tenant-a")[0]["id"] == "shared-agent"
	assert service.list_plfd_agents("tenant-b")[0]["id"] == "shared-agent"


def test_readiness_blocks_incomplete_foundation_service():
	service = PlfdService()
	service.register_foundation_service(
		"plfd-core",
		"tenant-plfd",
		"Platform foundation",
		"owner",
		"core",
		readiness_score=55,
		monitoring_enabled=False,
		rollback_plan_ref="",
		change_window_ref="",
	)
	readiness = service.assess_readiness("ready-blocked", "tenant-plfd", "plfd-core")

	assert readiness["status"] == "blocked"
	assert set(readiness["issues"]) >= {"readiness_score_below_threshold", "baselines_incomplete", "monitoring_required", "rollback_plan_required", "change_window_required"}
	assert service.dashboard_summary("tenant-plfd")["blocked_service_count"] == 1


def test_generated_evidence_and_docs_are_current():
	app = _load_module("plfd_app_under_test", PACKAGE_DIR / "app.py")
	model = app.semantic_model()
	committed_model = json.loads((PACKAGE_DIR / "semantic_model.json").read_text(encoding="utf-8"))

	assert app.self_test()["passed"] is True
	assert model == committed_model
	assert model["capabilities"]["plfd"]["streaming"]["processor"] == "bytewax"
	assert model["capabilities"]["plfd"]["screens"]["agents"]["route"] == "/plfd/agents"
	for name in ("README.md", "SPECIFICATION.md", "PLAN.md"):
		assert (PACKAGE_DIR / name).exists()
