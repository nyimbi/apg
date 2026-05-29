"""Regression coverage for the PLFD executable capability contract."""

import pytest

from capabilities.common.plfd import register_capability
from capabilities.common.plfd.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.plfd.service import PlfdService
from capabilities.common.plfd import views


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-plfd", {"operations": {"change_window_required": False}})

	assert contract["capability"] == "plfd"
	assert contract["configuration"]["tenant_id"] == "tenant-plfd"
	assert contract["configuration"]["operations"]["change_window_required"] is False
	assert contract["configuration_schema"]["required"] == ["tenant_id", "foundation", "baselines", "operations", "governance", "ui", "theme"]
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "services", "dependencies", "baselines", "readiness", "changes", "governance", "settings"}
	assert contract["theme"]["name"] == "plfd_platform_foundation"


def test_rule_engine_enforces_plfd_guardrails():
	result = evaluate_capability_rules({"tenant_context_present": False, "operation": "register_foundation_service", "service_owner_assigned": False, "configuration_baseline_present": False, "affected_capability_count": 12, "broad_review_recorded": False})
	change_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "approve_platform_change", "dependencies_healthy": False, "approval_recorded": False, "configuration_baseline_present": True})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "foundation_service_requires_owner", "configuration_baseline_required", "broad_platform_change_requires_review"}
	assert change_result["matched_rules"] == ["dependency_health_required", "platform_change_requires_approval"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "plfd"
	assert "mten" in registration["dependencies"]
	assert registration["ui_components"]["baselines"] == "/plfd/baselines"
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
	assert summary["ready_service_count"] == 1
	assert summary["approved_change_count"] == 1
	assert dashboard["summary"]["service_count"] == 2
	assert len(dependency_map["edges"]) == 1
	assert change_queue["changes"][0]["id"] == "change-plfd"
	assert len(service.list_audit_events("tenant-plfd")) >= 9


def test_service_enforces_platform_foundation_guardrails():
	service = PlfdService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.register_foundation_service("svc-no-tenant", "", "No tenant", "owner", "core")

	with pytest.raises(PermissionError, match="service_owner_required"):
		service.register_foundation_service("svc-no-owner", "tenant-plfd", "No owner", "", "core")

	with pytest.raises(PermissionError, match="configuration_baseline_required"):
		service.register_foundation_service(
			"svc-no-baseline",
			"tenant-plfd",
			"No baseline",
			"owner",
			"core",
			configuration_baseline_present=False,
		)

	service.register_foundation_service(
		"auth",
		"tenant-plfd",
		"Auth service",
		"auth-owner",
		"core",
		readiness_score=95,
		monitoring_enabled=True,
		rollback_plan_ref="rollback-auth",
		change_window_ref="cw-auth",
	)
	service.register_foundation_service(
		"plfd-core",
		"tenant-plfd",
		"Platform foundation",
		"owner",
		"core",
		readiness_score=95,
		monitoring_enabled=True,
		rollback_plan_ref="rollback-plfd",
		change_window_ref="cw-plfd",
	)

	with pytest.raises(PermissionError, match="baseline_evidence_required"):
		service.attach_baseline("base-bad", "tenant-plfd", "plfd-core", "configuration", "", "reviewer")

	service.record_dependency("dep-bad", "tenant-plfd", "plfd-core", "auth", "unhealthy", evidence_ref="health-auth")
	service.propose_platform_change(
		"change-unhealthy",
		"tenant-plfd",
		"plfd-core",
		"Unhealthy dependency change",
		"owner",
		affected_capability_count=1,
		approval_recorded=True,
		security_review_recorded=True,
	)
	with pytest.raises(PermissionError, match="dependency_health_required"):
		service.approve_platform_change("change-unhealthy", "tenant-plfd", "approver")

	service.register_foundation_service(
		"regy",
		"tenant-plfd",
		"Registry",
		"registry-owner",
		"shared",
		readiness_score=90,
		monitoring_enabled=True,
		rollback_plan_ref="rollback-regy",
		change_window_ref="cw-regy",
	)
	for baseline_type in ("configuration", "tenant", "auth", "audit"):
		service.attach_baseline(f"regy-{baseline_type}", "tenant-plfd", "regy", baseline_type, f"ev-{baseline_type}", "reviewer")

	service.propose_platform_change("change-no-approval", "tenant-plfd", "regy", "No approval", "owner", 1)
	with pytest.raises(PermissionError, match="platform_change_approval_required"):
		service.approve_platform_change("change-no-approval", "tenant-plfd", "approver", approval_recorded=False)

	service.propose_platform_change(
		"change-broad",
		"tenant-plfd",
		"regy",
		"Broad platform change",
		"owner",
		affected_capability_count=12,
		approval_recorded=True,
		security_review_recorded=True,
	)
	with pytest.raises(PermissionError, match="broad_platform_review_required"):
		service.approve_platform_change("change-broad", "tenant-plfd", "approver")

	service.propose_platform_change(
		"change-no-security",
		"tenant-plfd",
		"regy",
		"No security review",
		"owner",
		affected_capability_count=1,
		approval_recorded=True,
		security_review_recorded=False,
	)
	with pytest.raises(PermissionError, match="security_review_required"):
		service.approve_platform_change("change-no-security", "tenant-plfd", "approver")


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
