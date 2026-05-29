"""Regression coverage for the BKUP executable capability contract."""

import pytest

from capabilities.common.bkup import register_capability
from capabilities.common.bkup.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.bkup.service import BkupService
from capabilities.common.bkup.views import dashboard_model


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-backup", {"plans": {"rpo_minutes": 15}})

	assert contract["capability"] == "bkup"
	assert contract["configuration"]["tenant_id"] == "tenant-backup"
	assert contract["configuration"]["plans"]["rpo_minutes"] == 15
	assert contract["configuration_schema"]["required"] == ["tenant_id", "plans", "snapshots", "restore", "governance", "ui", "theme"]
	assert contract["theme"]["name"] == "bkup_continuity_ops"


def test_rule_engine_enforces_backup_guardrails():
	result = evaluate_capability_rules({"tenant_context_present": False, "operation": "create_backup_plan", "plan_owner_assigned": False, "target_environment": "production", "approval_recorded": False, "days_since_restore_test": 120, "restore_test_review_recorded": False})
	snapshot_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "create_snapshot", "snapshot_encrypted": False})
	restore_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "restore", "integrity_check_passed": False})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "backup_plan_requires_owner", "production_restore_requires_approval", "stale_restore_test_requires_review"}
	assert snapshot_result["matched_rules"] == ["snapshot_requires_encryption"]
	assert restore_result["matched_rules"] == ["restore_requires_integrity_check"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "bkup"
	assert "encr" in registration["dependencies"]
	assert registration["ui_components"]["restore"] == "/bkup/restore"
	assert "bkup:restore" in registration["permissions"]


def test_service_creates_plans_snapshots_restores_and_reports():
	service = BkupService()
	plan = service.create_backup_plan(
		plan_id="orders-db",
		tenant_id="tenant-backup",
		name="Orders Database",
		owner="platform-owner",
		schedule="0 */6 * * *",
		sources=["orders-primary", "orders-replica"],
		retention_days=35,
		rpo_minutes=30,
	)
	snapshot = service.create_snapshot(
		snapshot_id="snap-1",
		tenant_id="tenant-backup",
		plan_id="orders-db",
		source_id="orders-primary",
		size_bytes=4096,
		encrypted=True,
		integrity_check_passed=True,
		data_fingerprint="orders-v1",
	)
	restore = service.restore_snapshot(
		restore_id="restore-1",
		tenant_id="tenant-backup",
		snapshot_id="snap-1",
		target_environment="staging",
		requested_by="recovery-owner",
		integrity_check_passed=True,
		rto_minutes=45,
	)
	report = service.record_restore_test(
		report_id="report-1",
		tenant_id="tenant-backup",
		plan_id="orders-db",
		rto_minutes=45,
		days_since_restore_test=10,
	)
	summary = service.continuity_summary("tenant-backup")
	model = dashboard_model(service, "tenant-backup")

	assert plan["sources"] == ["orders-primary", "orders-replica"]
	assert snapshot["encrypted"] is True
	assert snapshot["integrity_status"] == "passed"
	assert len(snapshot["snapshot_hash"]) == 64
	assert restore["status"] == "completed"
	assert report["restore_test_status"] == "passed"
	assert summary["plan_count"] == 1
	assert summary["snapshot_count"] == 1
	assert summary["completed_restore_count"] == 1
	assert summary["continuity_report_count"] == 1
	assert model["summary"]["audit_event_count"] >= 4


def test_service_enforces_backup_snapshot_and_restore_guardrails():
	service = BkupService()

	with pytest.raises(PermissionError, match="plan_owner_required"):
		service.create_backup_plan(
			plan_id="missing-owner",
			tenant_id="tenant-backup",
			name="Missing Owner",
			owner="",
			schedule="0 * * * *",
			sources=["db"],
		)

	service.create_backup_plan(
		plan_id="core-db",
		tenant_id="tenant-backup",
		name="Core Database",
		owner="platform-owner",
		schedule="0 * * * *",
		sources=["core-primary"],
	)

	with pytest.raises(PermissionError, match="snapshot_encryption_required"):
		service.create_snapshot(
			snapshot_id="unencrypted",
			tenant_id="tenant-backup",
			plan_id="core-db",
			source_id="core-primary",
			size_bytes=1024,
			encrypted=False,
		)

	service.create_snapshot(
		snapshot_id="snap-core",
		tenant_id="tenant-backup",
		plan_id="core-db",
		source_id="core-primary",
		size_bytes=2048,
		encrypted=True,
		integrity_check_passed=True,
	)

	with pytest.raises(PermissionError, match="integrity_check_required"):
		service.restore_snapshot(
			restore_id="bad-restore",
			tenant_id="tenant-backup",
			snapshot_id="snap-core",
			target_environment="staging",
			requested_by="operator",
			integrity_check_passed=False,
		)

	with pytest.raises(PermissionError, match="production_restore_approval_required"):
		service.restore_snapshot(
			restore_id="prod-without-approval",
			tenant_id="tenant-backup",
			snapshot_id="snap-core",
			target_environment="production",
			requested_by="operator",
			integrity_check_passed=True,
			approval_recorded=False,
		)

	stale_restore = service.restore_snapshot(
		restore_id="stale-test-restore",
		tenant_id="tenant-backup",
		snapshot_id="snap-core",
		target_environment="staging",
		requested_by="operator",
		integrity_check_passed=True,
		days_since_restore_test=120,
		restore_test_review_recorded=False,
	)
	approved = service.approve_restore("stale-test-restore", reviewer="continuity-reviewer")
	stale_report = service.record_restore_test(
		report_id="stale-report",
		tenant_id="tenant-backup",
		plan_id="core-db",
		rto_minutes=300,
		days_since_restore_test=120,
		restore_test_review_recorded=False,
	)

	assert stale_restore["status"] == "pending_review"
	assert stale_restore["review_status"] == "required"
	assert approved["status"] == "completed"
	assert stale_report["review_status"] == "required"
	assert "restore test older than 90 days" in stale_report["findings"]
