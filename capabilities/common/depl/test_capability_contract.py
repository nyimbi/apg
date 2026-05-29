"""Regression coverage for the DEPL executable capability contract."""

import pytest

from capabilities.common.depl import register_capability
from capabilities.common.depl.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.depl.service import DeplService


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-deploy", {"rollouts": {"max_canary_percent": 10}})

	assert contract["capability"] == "depl"
	assert contract["configuration"]["tenant_id"] == "tenant-deploy"
	assert contract["configuration"]["rollouts"]["max_canary_percent"] == 10
	assert contract["configuration_schema"]["required"] == ["tenant_id", "releases", "rollouts", "evidence", "governance", "ui", "theme"]
	assert contract["theme"]["name"] == "depl_release_ops"


def test_rule_engine_enforces_deployment_guardrails():
	result = evaluate_capability_rules({"tenant_context_present": False, "operation": "create_release", "release_owner_assigned": False, "target_environment": "production", "approval_recorded": False, "canary_percent": 50, "canary_review_recorded": False})
	deploy_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "deploy", "health_gate_passed": False, "rollback_plan_attached": False})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "release_requires_owner", "production_requires_approval", "large_canary_requires_review"}
	assert set(deploy_result["matched_rules"]) == {"deployment_requires_health_gate", "rollback_requires_plan"}


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "depl"
	assert "logt" in registration["dependencies"]
	assert registration["ui_components"]["rollback"] == "/depl/rollback"
	assert "depl:deploy" in registration["permissions"]


def test_service_runs_release_deployment_and_rollback_lifecycle():
	service = DeplService()

	environment = service.register_environment(
		"env-prod",
		"tenant-depl",
		"Production",
		"production",
		"platform-owner",
		"prod-change-policy",
		["ops-approver"],
	)
	release = service.create_release(
		"rel-2026-05",
		"tenant-depl",
		"2026.05",
		"release-owner",
		{"service": "erp-core", "version": "2026.05"},
		"sha256:artifact",
		"sigstore:signature",
		"CHG-1001",
		"release-owner",
	)
	rollback = service.attach_rollback_plan(
		"rbp-2026-05",
		"tenant-depl",
		"rel-2026-05",
		"release-owner",
		["switch traffic to previous slot", "restore previous artifact"],
		tested=True,
	)
	health = service.record_health_gate(
		"hlg-2026-05",
		"tenant-depl",
		"rel-2026-05",
		{"smoke": True, "latency": True, "error_budget": True},
		"health-report:1001",
		"trace:deploy-1001",
		"sre",
	)
	plan = service.create_deployment_plan(
		"plan-2026-05",
		"tenant-depl",
		"rel-2026-05",
		"env-prod",
		"canary",
		"release-owner",
		approval_recorded=True,
		rollback_plan_id="rbp-2026-05",
		health_gate_id="hlg-2026-05",
		change_ticket="CHG-1001",
		canary_percent=10,
	)
	run = service.execute_deployment(
		"run-2026-05",
		"tenant-depl",
		"plan-2026-05",
		"release-owner",
		"trace:deploy-1001",
		"health-report:1001",
	)
	rollback_event = service.execute_rollback(
		"rollback-2026-05",
		"tenant-depl",
		"run-2026-05",
		"sre",
		"synthetic rollback drill",
	)
	summary = service.dashboard_summary("tenant-depl")

	assert environment["tier"] == "production"
	assert release["artifact_signature"] == "sigstore:signature"
	assert rollback["tested"] is True
	assert health["status"] == "passed"
	assert plan["status"] == "approved"
	assert run["status"] == "deployed"
	assert run["fingerprint"].startswith("depl-")
	assert rollback_event["status"] == "rolled_back"
	assert summary["environment_count"] == 1
	assert summary["release_count"] == 1
	assert summary["rollback_count"] == 1
	assert summary["governance_posture"] == "ready"
	assert len(service.list_audit_events("tenant-depl")) >= 7


def test_service_enforces_deployment_guardrails():
	service = DeplService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.register_environment("env-missing-tenant", "", "Prod", "production", "owner", "policy", ["approver"])

	with pytest.raises(PermissionError, match="environment_policy_required"):
		service.register_environment("env-no-policy", "tenant-depl", "Prod", "production", "owner", "", ["approver"])

	service.register_environment("env-prod", "tenant-depl", "Production", "production", "platform-owner", "policy", ["approver"])

	with pytest.raises(PermissionError, match="release_owner_required"):
		service.create_release(
			"rel-no-owner",
			"tenant-depl",
			"2026.05",
			"",
			{"service": "erp-core"},
			"sha256:artifact",
			"sigstore:signature",
			"CHG-1001",
			"creator",
		)

	with pytest.raises(PermissionError, match="manifest_required"):
		service.create_release(
			"rel-no-manifest",
			"tenant-depl",
			"2026.05",
			"owner",
			{},
			"sha256:artifact",
			"sigstore:signature",
			"CHG-1001",
			"creator",
		)

	service.create_release(
		"rel-2026-05",
		"tenant-depl",
		"2026.05",
		"owner",
		{"service": "erp-core"},
		"sha256:artifact",
		"sigstore:signature",
		"CHG-1001",
		"creator",
	)

	with pytest.raises(PermissionError, match="rollback_plan_test_required"):
		service.attach_rollback_plan("rbp-untested", "tenant-depl", "rel-2026-05", "owner", ["restore"], tested=False)

	service.attach_rollback_plan("rbp-2026-05", "tenant-depl", "rel-2026-05", "owner", ["restore"], tested=True)
	failed_gate = service.record_health_gate(
		"hlg-failed",
		"tenant-depl",
		"rel-2026-05",
		{"smoke": True, "error_budget": False},
		"health-report:failed",
		"trace:failed",
		"sre",
	)
	assert failed_gate["status"] == "failed"

	with pytest.raises(PermissionError, match="health_gate_required"):
		service.create_deployment_plan(
			"plan-failed-health",
			"tenant-depl",
			"rel-2026-05",
			"env-prod",
			"rolling",
			"owner",
			approval_recorded=True,
			rollback_plan_id="rbp-2026-05",
			health_gate_id="hlg-failed",
			change_ticket="CHG-1001",
		)

	service.record_health_gate(
		"hlg-passed",
		"tenant-depl",
		"rel-2026-05",
		{"smoke": True, "error_budget": True},
		"health-report:passed",
		"trace:passed",
		"sre",
	)

	with pytest.raises(PermissionError, match="production_approval_required"):
		service.create_deployment_plan(
			"plan-no-approval",
			"tenant-depl",
			"rel-2026-05",
			"env-prod",
			"rolling",
			"owner",
			approval_recorded=False,
			rollback_plan_id="rbp-2026-05",
			health_gate_id="hlg-passed",
			change_ticket="CHG-1001",
		)


def test_large_canary_requires_review_before_execution():
	service = DeplService()
	service.register_environment("env-stage", "tenant-depl", "Stage", "staging", "platform-owner", "stage-policy", [])
	service.create_release(
		"rel-2026-05",
		"tenant-depl",
		"2026.05",
		"owner",
		{"service": "erp-core"},
		"sha256:artifact",
		"sigstore:signature",
		"CHG-1001",
		"creator",
	)
	service.attach_rollback_plan("rbp-2026-05", "tenant-depl", "rel-2026-05", "owner", ["restore"], tested=True)
	service.record_health_gate(
		"hlg-passed",
		"tenant-depl",
		"rel-2026-05",
		{"smoke": True, "error_budget": True},
		"health-report:passed",
		"trace:passed",
		"sre",
	)

	plan = service.create_deployment_plan(
		"plan-large-canary",
		"tenant-depl",
		"rel-2026-05",
		"env-stage",
		"canary",
		"owner",
		approval_recorded=True,
		rollback_plan_id="rbp-2026-05",
		health_gate_id="hlg-passed",
		change_ticket="CHG-1001",
		canary_percent=50,
		canary_review_recorded=False,
	)

	assert plan["status"] == "pending_review"
	assert plan["review_status"] == "required"
	with pytest.raises(PermissionError, match="deployment_plan_not_approved"):
		service.execute_deployment(
			"run-large-canary",
			"tenant-depl",
			"plan-large-canary",
			"owner",
			"trace:deploy",
			"health-report:passed",
		)
	approved = service.approve_deployment_plan("plan-large-canary", "tenant-depl", "release-manager")
	run = service.execute_deployment(
		"run-large-canary",
		"tenant-depl",
		"plan-large-canary",
		"owner",
		"trace:deploy",
		"health-report:passed",
	)

	assert approved["status"] == "approved"
	assert run["status"] == "deployed"
