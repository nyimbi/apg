"""Regression coverage for the BKUP executable capability contract."""

from __future__ import annotations

import pytest

from capabilities.common.bkup import api, register_capability
from capabilities.common.bkup import views
from capabilities.common.bkup.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.bkup.service import BkupService


def _ready_service(legal_hold: bool = False) -> BkupService:
	service = BkupService()
	service.create_backup_plan(
		plan_id="core-db",
		tenant_id="tenant-backup",
		name="Core Database",
		owner="platform-owner",
		schedule="0 * * * *",
		sources=["core-primary", "core-replica"],
		retention_days=35,
		rpo_minutes=30,
		legal_hold=legal_hold,
	)
	service.create_snapshot(
		snapshot_id="snap-core",
		tenant_id="tenant-backup",
		plan_id="core-db",
		source_id="core-primary",
		size_bytes=2048,
		encrypted=True,
		integrity_check_passed=True,
		data_fingerprint="core-v1",
	)
	return service


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-backup", {"plans": {"rpo_minutes": 15}})

	assert contract["capability"] == "bkup"
	assert contract["configuration"]["tenant_id"] == "tenant-backup"
	assert contract["configuration"]["plans"]["rpo_minutes"] == 15
	assert contract["configuration_schema"]["required"] == ["tenant_id", "plans", "snapshots", "restore", "governance", "ui", "theme"]
	assert len(contract["rule_engine"]["rules"]) >= 10
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"dashboard",
		"plans",
		"snapshots",
		"backup",
		"restore",
		"restore_approvals",
		"retention",
		"retention_dispositions",
		"reports",
		"audit",
		"settings",
	}
	assert contract["theme"]["name"] == "bkup_continuity_ops"
	assert "restore_approval_queue" in contract["theme"]["components"]
	assert "retention_disposition_queue" in contract["theme"]["components"]


def test_rule_engine_enforces_backup_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "create_backup_plan",
		"plan_owner_assigned": False,
		"target_environment": "production",
		"approval_recorded": False,
		"days_since_restore_test": 120,
		"restore_test_review_recorded": False,
	})
	snapshot_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "create_snapshot",
		"snapshot_encrypted": False,
		"snapshot_integrity_passed": False,
	})
	restore_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "restore", "integrity_check_passed": False})
	restore_review_result = evaluate_capability_rules({"operation": "approve_restore", "restore_reviewer_same_as_requester": True})
	retention_result = evaluate_capability_rules({"operation": "retention_disposition", "legal_hold_active": True})
	retention_review_result = evaluate_capability_rules({"operation": "approve_retention_disposition", "retention_reviewer_same_as_requester": True})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {
		"tenant_context_required",
		"backup_plan_requires_owner",
		"production_restore_requires_approval",
		"stale_restore_test_requires_review",
	}
	assert set(snapshot_result["matched_rules"]) == {"snapshot_requires_encryption", "snapshot_requires_integrity"}
	assert restore_result["matched_rules"] == ["restore_requires_integrity_check"]
	assert restore_review_result["matched_rules"] == ["restore_review_requires_independent_reviewer"]
	assert retention_result["matched_rules"] == ["retention_disposition_blocks_legal_hold"]
	assert retention_review_result["matched_rules"] == ["retention_review_requires_independent_reviewer"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "bkup"
	assert "encr" in registration["dependencies"]
	assert registration["ui_components"]["restore"] == "/bkup/restore"
	assert registration["ui_components"]["restore_approvals"] == "/bkup/restore/approvals"
	assert registration["ui_components"]["retention_dispositions"] == "/bkup/retention/dispositions"
	assert "bkup:restore" in registration["permissions"]
	assert "bkup:approve_restore" in registration["permissions"]
	assert "bkup:approve_retention" in registration["permissions"]


def test_service_creates_plans_snapshots_restores_reports_retention_and_audit():
	service = _ready_service()
	restore_approval = service.request_restore_approval(
		approval_id="restore-approval-1",
		tenant_id="tenant-backup",
		snapshot_id="snap-core",
		target_environment="production",
		requested_by="recovery-owner",
		justification="Production recovery drill approved by change window.",
		point_in_time="2026-05-30T00:00:00Z",
	)
	approved_restore = service.decide_restore_approval(
		approval_id=restore_approval["id"],
		tenant_id="tenant-backup",
		reviewer="continuity-reviewer",
		decision="approved",
		notes="Change window, integrity, and rollback checklist verified.",
	)
	restore = service.restore_snapshot(
		restore_id="restore-1",
		tenant_id="tenant-backup",
		snapshot_id="snap-core",
		target_environment="production",
		requested_by="recovery-owner",
		integrity_check_passed=True,
		point_in_time="2026-05-30T00:00:00Z",
		rto_minutes=45,
		approval_id=approved_restore["id"],
	)
	report = service.record_restore_test(
		report_id="report-1",
		tenant_id="tenant-backup",
		plan_id="core-db",
		rto_minutes=45,
		days_since_restore_test=10,
	)
	disposition = service.request_retention_disposition(
		disposition_id="dispose-1",
		tenant_id="tenant-backup",
		snapshot_id="snap-core",
		action="archive",
		requested_by="retention-owner",
		reason="Move tested snapshot to archive tier.",
	)
	approved_disposition = service.decide_retention_disposition(
		disposition_id=disposition["id"],
		tenant_id="tenant-backup",
		reviewer="records-reviewer",
		decision="approved",
		notes="Retention policy permits archive transition.",
	)
	summary = service.continuity_summary("tenant-backup")
	model = views.dashboard_model(service, "tenant-backup")

	assert restore["status"] == "completed"
	assert restore["approval_id"] == "restore-approval-1"
	assert report["restore_test_status"] == "passed"
	assert approved_disposition["status"] == "approved"
	assert service.list_snapshots("tenant-backup")[0]["status"] == "archived"
	assert summary["plan_count"] == 1
	assert summary["snapshot_count"] == 1
	assert summary["completed_restore_count"] == 1
	assert summary["restore_approval_count"] == 1
	assert summary["retention_disposition_count"] == 1
	assert model["summary"]["audit_event_count"] >= 8


def test_service_enforces_backup_snapshot_restore_and_retention_guardrails():
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

	service = _ready_service()
	with pytest.raises(PermissionError, match="snapshot_encryption_required"):
		service.create_snapshot(
			snapshot_id="unencrypted",
			tenant_id="tenant-backup",
			plan_id="core-db",
			source_id="core-primary",
			size_bytes=1024,
			encrypted=False,
		)
	with pytest.raises(PermissionError, match="snapshot_integrity_check_required"):
		service.create_snapshot(
			snapshot_id="bad-integrity",
			tenant_id="tenant-backup",
			plan_id="core-db",
			source_id="core-primary",
			size_bytes=1024,
			encrypted=True,
			integrity_check_passed=False,
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
			approval_recorded=True,
		)
	approval_request = service.request_restore_approval(
		approval_id="restore-approval-rejected",
		tenant_id="tenant-backup",
		snapshot_id="snap-core",
		target_environment="production",
		requested_by="operator",
		justification="Emergency restore.",
	)
	with pytest.raises(PermissionError, match="independent_restore_reviewer_required"):
		service.decide_restore_approval(
			approval_id=approval_request["id"],
			tenant_id="tenant-backup",
			reviewer="operator",
			decision="approved",
			notes="Self-approved.",
		)
	with pytest.raises(ValueError, match="restore_approval_notes_required"):
		service.decide_restore_approval(
			approval_id=approval_request["id"],
			tenant_id="tenant-backup",
			reviewer="continuity-reviewer",
			decision="approved",
			notes="",
		)
	rejected_approval = service.decide_restore_approval(
		approval_id=approval_request["id"],
		tenant_id="tenant-backup",
		reviewer="continuity-reviewer",
		decision="rejected",
		notes="Rollback evidence incomplete.",
	)
	with pytest.raises(PermissionError, match="restore_approval_not_approved"):
		service.restore_snapshot(
			restore_id="restore-with-rejected-approval",
			tenant_id="tenant-backup",
			snapshot_id="snap-core",
			target_environment="production",
			requested_by="operator",
			approval_id=rejected_approval["id"],
		)
	stale_restore = service.restore_snapshot(
		restore_id="stale-test-restore",
		tenant_id="tenant-backup",
		snapshot_id="snap-core",
		target_environment="staging",
		requested_by="operator",
		integrity_check_passed=True,
		days_since_restore_test=120,
		restore_test_review_recorded=True,
	)
	with pytest.raises(PermissionError, match="independent_restore_reviewer_required"):
		service.approve_restore("stale-test-restore", reviewer="operator", tenant_id="tenant-backup")
	approved = service.approve_restore("stale-test-restore", reviewer="continuity-reviewer", tenant_id="tenant-backup", notes="Restore test reviewed.")
	stale_report = service.record_restore_test(
		report_id="stale-report",
		tenant_id="tenant-backup",
		plan_id="core-db",
		rto_minutes=300,
		days_since_restore_test=120,
		restore_test_review_recorded=True,
	)

	hold_service = _ready_service(legal_hold=True)
	with pytest.raises(PermissionError, match="legal_hold_blocks_disposition"):
		hold_service.request_retention_disposition(
			disposition_id="dispose-held",
			tenant_id="tenant-backup",
			snapshot_id="snap-core",
			action="delete",
			requested_by="records-owner",
			reason="Retention expired.",
		)

	disposition = service.request_retention_disposition(
		disposition_id="dispose-core",
		tenant_id="tenant-backup",
		snapshot_id="snap-core",
		action="delete",
		requested_by="records-owner",
		reason="Retention expired.",
	)
	with pytest.raises(PermissionError, match="independent_retention_reviewer_required"):
		service.decide_retention_disposition(
			disposition_id=disposition["id"],
			tenant_id="tenant-backup",
			reviewer="records-owner",
			decision="approved",
			notes="Self-approved.",
		)
	with pytest.raises(ValueError, match="retention_disposition_notes_required"):
		service.decide_retention_disposition(
			disposition_id=disposition["id"],
			tenant_id="tenant-backup",
			reviewer="records-reviewer",
			decision="approved",
			notes="",
		)

	assert stale_restore["status"] == "pending_review"
	assert stale_restore["review_status"] == "required"
	assert approved["status"] == "completed"
	assert stale_report["review_status"] == "required"
	assert "restore test older than 90 days" in stale_report["findings"]


def test_service_keeps_duplicate_ids_isolated_by_tenant():
	service = BkupService()
	for tenant_id in ["tenant-a", "tenant-b"]:
		service.create_backup_plan(
			plan_id="same-plan",
			tenant_id=tenant_id,
			name=f"Plan {tenant_id}",
			owner="owner",
			schedule="0 * * * *",
			sources=["same-source"],
		)
		service.create_snapshot(
			snapshot_id="same-snapshot",
			tenant_id=tenant_id,
			plan_id="same-plan",
			source_id="same-source",
			size_bytes=1024,
			data_fingerprint=tenant_id,
		)

	assert service.list_plans("tenant-a")[0]["tenant_id"] == "tenant-a"
	assert service.list_plans("tenant-b")[0]["tenant_id"] == "tenant-b"
	assert service.list_snapshots("tenant-a")[0]["tenant_id"] == "tenant-a"
	assert service.list_snapshots("tenant-b")[0]["tenant_id"] == "tenant-b"
	with pytest.raises(ValueError, match="backup plan already exists"):
		service.create_backup_plan(
			plan_id="same-plan",
			tenant_id="tenant-a",
			name="Duplicate",
			owner="owner",
			schedule="0 * * * *",
			sources=["same-source"],
		)


def test_api_helpers_and_view_models_expose_bkup_lifecycle():
	tenant_id = "tenant-api-bkup"
	api.create_backup_plan({
		"id": "api-plan",
		"tenant_id": tenant_id,
		"name": "API Plan",
		"owner": "api-owner",
		"schedule": "0 * * * *",
		"sources": ["api-source"],
		"legal_hold": "false",
	})
	snapshot = api.create_snapshot({
		"id": "api-snapshot",
		"tenant_id": tenant_id,
		"plan_id": "api-plan",
		"source_id": "api-source",
		"size_bytes": 4096,
		"encrypted": "true",
		"integrity_check_passed": "true",
		"data_fingerprint": "api-v1",
	})
	approval_request = api.request_restore_approval({
		"id": "api-restore-approval",
		"tenant_id": tenant_id,
		"snapshot_id": snapshot["id"],
		"target_environment": "production",
		"requested_by": "api-operator",
		"justification": "API production restore.",
	})
	approval = api.decide_restore_approval({
		"id": approval_request["id"],
		"tenant_id": tenant_id,
		"reviewer": "api-reviewer",
		"decision": "approved",
		"notes": "Approved API restore.",
	})
	restore = api.restore_snapshot({
		"id": "api-restore",
		"tenant_id": tenant_id,
		"snapshot_id": snapshot["id"],
		"target_environment": "production",
		"requested_by": "api-operator",
		"approval_id": approval["id"],
	})
	disposition = api.request_retention_disposition({
		"id": "api-disposition",
		"tenant_id": tenant_id,
		"snapshot_id": snapshot["id"],
		"action": "archive",
		"requested_by": "api-retention-owner",
		"reason": "Move to archive.",
	})
	api.decide_retention_disposition({
		"id": disposition["id"],
		"tenant_id": tenant_id,
		"reviewer": "api-records-reviewer",
		"decision": "approved",
		"notes": "Archive approved.",
	})
	dashboard = views.dashboard_model(tenant_id=tenant_id)
	restore_approvals = views.restore_approval_model(tenant_id=tenant_id)
	retention = views.retention_disposition_model(tenant_id=tenant_id)

	assert restore["status"] == "completed"
	assert api.capability_status(tenant_id)["restore_approval_count"] == 1
	assert dashboard["summary"]["retention_disposition_count"] == 1
	assert restore_approvals["decided_approvals"][0]["id"] == "api-restore-approval"
	assert retention["decided_dispositions"][0]["id"] == "api-disposition"
