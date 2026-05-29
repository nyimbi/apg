"""Executable capability contract for APG Backup and Restore."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"plans": {"plan_owner_required": True, "schedule_required": True, "source_inventory_required": True, "rpo_minutes": 60},
	"snapshots": {"encryption_required": True, "integrity_check_required": True, "cross_region_copy_supported": True, "lineage_required": True},
	"restore": {"restore_test_required": True, "approval_required_for_production": True, "point_in_time_supported": True, "rto_minutes": 240},
	"governance": {"require_tenant_context": True, "audit_backup_events": True, "retention_policy_required": True, "legal_hold_supported": True},
	"ui": {"enable_backup_dashboard": True, "enable_plan_manager": True, "enable_restore_console": True, "enable_restore_approval_queue": True, "enable_retention_disposition_queue": True, "enable_continuity_reports": True},
	"theme": {"default_theme": "bkup_continuity_ops", "allow_tenant_overrides": True}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {"type": "object", "required": ["tenant_id", "plans", "snapshots", "restore", "governance", "ui", "theme"], "properties": {key: {"type": "object"} for key in ["plans", "snapshots", "restore", "governance", "ui", "theme"]} | {"tenant_id": {"type": "string", "minLength": 1}}}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All backup operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "backup_plan_requires_owner", "description": "Backup plans require an accountable owner.", "condition": {"operation": "create_backup_plan", "plan_owner_assigned": False}, "effect": {"decision": "deny", "reason": "plan_owner_required", "required_action": "assign_plan_owner"}},
	{"name": "snapshot_requires_encryption", "description": "Snapshots must be encrypted.", "condition": {"operation": "create_snapshot", "snapshot_encrypted": False}, "effect": {"decision": "deny", "reason": "snapshot_encryption_required", "required_action": "encrypt_snapshot"}},
	{"name": "snapshot_requires_integrity", "description": "Snapshots require integrity evidence.", "condition": {"operation": "create_snapshot", "snapshot_integrity_passed": False}, "effect": {"decision": "deny", "reason": "snapshot_integrity_check_required", "required_action": "pass_snapshot_integrity_check"}},
	{"name": "restore_requires_integrity_check", "description": "Restore operations require integrity checks.", "condition": {"operation": "restore", "integrity_check_passed": False}, "effect": {"decision": "deny", "reason": "integrity_check_required", "required_action": "pass_integrity_check"}},
	{"name": "production_restore_requires_approval", "description": "Production restores require approval.", "condition": {"target_environment": "production", "approval_recorded": False}, "effect": {"decision": "deny", "reason": "production_restore_approval_required", "required_action": "record_restore_approval"}},
	{"name": "stale_restore_test_requires_review", "description": "Stale restore tests require review.", "condition": {"days_since_restore_test_gt": 90, "restore_test_review_recorded": False}, "effect": {"decision": "require_review", "reason": "restore_test_review_required", "required_action": "review_restore_test"}},
	{"name": "restore_review_requires_independent_reviewer", "description": "Restore approvals and reviews require an independent reviewer.", "condition": {"operation": "approve_restore", "restore_reviewer_same_as_requester": True}, "effect": {"decision": "deny", "reason": "independent_restore_reviewer_required", "required_action": "route_to_independent_restore_reviewer"}},
	{"name": "retention_disposition_blocks_legal_hold", "description": "Snapshots under legal hold cannot be disposed.", "condition": {"operation": "retention_disposition", "legal_hold_active": True}, "effect": {"decision": "deny", "reason": "legal_hold_blocks_disposition", "required_action": "release_legal_hold_before_disposition"}},
	{"name": "retention_review_requires_independent_reviewer", "description": "Retention disposition approvals require an independent reviewer.", "condition": {"operation": "approve_retention_disposition", "retention_reviewer_same_as_requester": True}, "effect": {"decision": "deny", "reason": "independent_retention_reviewer_required", "required_action": "route_to_independent_retention_reviewer"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/bkup/dashboard", "component": "BKUPDashboard", "permission": "bkup:view", "nav_group": "Overview"},
	{"name": "plans", "path": "/bkup/plans", "component": "BackupPlans", "permission": "bkup:manage_plans", "nav_group": "Plans"},
	{"name": "snapshots", "path": "/bkup/snapshots", "component": "SnapshotVault", "permission": "bkup:view", "nav_group": "Backups"},
	{"name": "backup", "path": "/bkup/backup", "component": "BackupRuns", "permission": "bkup:run_backup", "nav_group": "Backups"},
	{"name": "restore", "path": "/bkup/restore", "component": "RestoreConsole", "permission": "bkup:restore", "nav_group": "Recovery"},
	{"name": "restore_approvals", "path": "/bkup/restore/approvals", "component": "RestoreApprovalQueue", "permission": "bkup:approve_restore", "nav_group": "Recovery"},
	{"name": "retention", "path": "/bkup/retention", "component": "RetentionPolicy", "permission": "bkup:admin", "nav_group": "Governance"},
	{"name": "retention_dispositions", "path": "/bkup/retention/dispositions", "component": "RetentionDispositionQueue", "permission": "bkup:approve_retention", "nav_group": "Governance"},
	{"name": "reports", "path": "/bkup/reports", "component": "ContinuityReports", "permission": "bkup:view", "nav_group": "Governance"},
	{"name": "audit", "path": "/bkup/audit", "component": "BackupAudit", "permission": "bkup:view", "nav_group": "Governance"},
	{"name": "settings", "path": "/bkup/settings", "component": "BKUPSettings", "permission": "bkup:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {"name": "bkup_continuity_ops", "tokens": {"color.primary": "#214E34", "color.accent": "#2B6CB0", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F7F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"}, "components": {"backup_plan": {"icon": "database-backup", "status_indicator": "rpo-pill", "risk_style": "retention-band"}, "snapshot_vault": {"visual": "snapshot-list", "highlight": "encryption-chip"}, "restore_console": {"visual": "restore-timeline", "status_style": "integrity-chip"}, "restore_approval_queue": {"visual": "approval-lane", "status_style": "restore-review-chip"}, "retention_disposition_queue": {"visual": "legal-hold-lane", "status_style": "retention-chip"}, "continuity_report": {"visual": "rto-rpo-card", "status_style": "test-chip"}, "backup_audit": {"visual": "event-ledger", "status_style": "decision-chip"}}}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {"capability": "bkup", "display_name": "Backup and Restore", "configuration": config, "configuration_schema": CONFIGURATION_SCHEMA, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "view_module": "views.py", "api_prefix": "/bkup/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True}, "theme": deepcopy(THEME)}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	matched: list[str] = []
	actions: list[dict[str, Any]] = []
	decision = "allow"
	for rule in RULES:
		if _matches(rule["condition"], context):
			matched.append(rule["name"])
			effect = rule["effect"]
			actions.append(effect)
			if effect["decision"] == "deny":
				decision = "deny"
			elif effect["decision"] == "require_review" and decision != "deny":
				decision = "require_review"
	return {"decision": decision, "matched_rules": matched, "actions": actions, "context": context}


def _matches(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if key.endswith("_lt"):
			if not context.get(key[:-3], 0) < expected:
				return False
		elif key.endswith("_gt"):
			if not context.get(key[:-3], 0) > expected:
				return False
		elif context.get(key) != expected:
			return False
	return True


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
	for key, value in source.items():
		if isinstance(value, dict) and isinstance(target.get(key), dict):
			_deep_merge(target[key], value)
		else:
			target[key] = value
