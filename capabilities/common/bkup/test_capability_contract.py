"""Regression coverage for the BKUP executable capability contract."""

from capabilities.common.bkup import register_capability
from capabilities.common.bkup.capability_contract import evaluate_capability_rules, get_capability_contract


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
