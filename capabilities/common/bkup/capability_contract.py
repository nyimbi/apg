"""Executable capability contract for APG Backup and Restore."""

from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Any

SUPPORTED_BACKUP_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_BACKUP_AGENT_ROLES = ["plan_reviewer", "snapshot_reviewer", "restore_reviewer", "retention_reviewer", "continuity_reviewer"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"plans": {"plan_owner_required": True, "schedule_required": True, "source_inventory_required": True, "rpo_minutes": 60},
	"snapshots": {"encryption_required": True, "integrity_check_required": True, "cross_region_copy_supported": True, "lineage_required": True},
	"restore": {"restore_test_required": True, "approval_required_for_production": True, "point_in_time_supported": True, "rto_minutes": 240},
	"backup_agents": {"agent_assist_enabled": True, "agent_registration_required": True, "agent_runtime_required": True, "agent_scope_required": True, "agent_contribution_disclosure_required": True, "supported_runtimes": SUPPORTED_BACKUP_AGENT_RUNTIMES, "allowed_roles": SUPPORTED_BACKUP_AGENT_ROLES},
	"governance": {"require_tenant_context": True, "audit_backup_events": True, "retention_policy_required": True, "legal_hold_supported": True, "batch_event_stream": "bytewax"},
	"observability": {"audit_required": True, "trace_required": True, "continuity_metrics_required": True, "agent_activity_required": True, "event_stream": "bytewax"},
	"adapters": {"generated_app_runtime": "service.BkupService", "api_helpers": "api.py", "view_models": "views.py", "event_stream": "bytewax", "audit_sink": "audl", "encryption": "encr", "configuration": "conf", "scheduler": "schd", "monitoring": "moni", "compliance": "comp"},
	"ui": {"enable_backup_dashboard": True, "enable_plan_manager": True, "enable_restore_console": True, "enable_restore_approval_queue": True, "enable_retention_disposition_queue": True, "enable_continuity_reports": True, "enable_agent_panel": True, "enable_audit": True, "enable_analytics": True},
	"theme": {"default_theme": "bkup_continuity_ops", "allow_tenant_overrides": True}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {"type": "object", "required": ["tenant_id", "plans", "snapshots", "restore", "backup_agents", "governance", "observability", "adapters", "ui", "theme"], "properties": {key: {"type": "object"} for key in ["plans", "snapshots", "restore", "backup_agents", "governance", "observability", "adapters", "ui", "theme"]} | {"tenant_id": {"type": "string", "minLength": 1}}}

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
	{"name": "retention_review_requires_independent_reviewer", "description": "Retention disposition approvals require an independent reviewer.", "condition": {"operation": "approve_retention_disposition", "retention_reviewer_same_as_requester": True}, "effect": {"decision": "deny", "reason": "independent_retention_reviewer_required", "required_action": "route_to_independent_retention_reviewer"}},
	{"name": "backup_agent_requires_registration", "description": "AI backup agents must be registered.", "condition": {"backup_agent_present": True, "agent_registered": False}, "effect": {"decision": "deny", "reason": "backup_agent_registration_required", "required_action": "register_backup_agent"}},
	{"name": "backup_agent_runtime_supported", "description": "AI backup agents must use a supported runtime.", "condition": {"backup_agent_present": True, "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "backup_agent_runtime_not_supported", "required_action": "choose_supported_backup_agent_runtime"}},
	{"name": "backup_agent_role_supported", "description": "AI backup agents must use a supported role.", "condition": {"backup_agent_present": True, "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "backup_agent_role_not_supported", "required_action": "choose_supported_backup_agent_role"}},
	{"name": "backup_agent_requires_scope", "description": "AI backup agents require explicit scope.", "condition": {"backup_agent_present": True, "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "backup_agent_scope_required", "required_action": "set_backup_agent_scope"}},
	{"name": "backup_agent_requires_disclosure", "description": "AI backup-agent contributions require disclosure.", "condition": {"backup_agent_present": True, "agent_contribution_disclosed": False}, "effect": {"decision": "deny", "reason": "backup_agent_disclosure_required", "required_action": "disclose_backup_agent"}},
	{"name": "backup_state_change_requires_audit", "description": "BKUP lifecycle state changes require audit evidence.", "condition": {"state_change_requested": True, "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "backup_audit_event_required", "required_action": "record_backup_audit_event"}},
	{"name": "batch_backup_mutation_requires_bytewax", "description": "Batch BKUP mutations must use Bytewax event streams.", "condition": {"requested_operation": "batch_backup_mutation", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}}
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
	{"name": "backup_agents", "path": "/bkup/agents", "component": "BackupAgentPanel", "permission": "bkup:approve_restore", "nav_group": "Governance"},
	{"name": "reports", "path": "/bkup/reports", "component": "ContinuityReports", "permission": "bkup:view", "nav_group": "Governance"},
	{"name": "audit", "path": "/bkup/audit", "component": "BackupAudit", "permission": "bkup:view", "nav_group": "Governance"},
	{"name": "analytics", "path": "/bkup/analytics", "component": "BackupAnalytics", "permission": "bkup:view", "nav_group": "Operations"},
	{"name": "settings", "path": "/bkup/settings", "component": "BKUPSettings", "permission": "bkup:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {"name": "bkup_continuity_ops", "tokens": {"color.primary": "#214E34", "color.accent": "#2B6CB0", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F7F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"}, "components": {"backup_plan": {"icon": "database-backup", "status_indicator": "rpo-pill", "risk_style": "retention-band"}, "snapshot_vault": {"visual": "snapshot-list", "highlight": "encryption-chip"}, "restore_console": {"visual": "restore-timeline", "status_style": "integrity-chip"}, "restore_approval_queue": {"visual": "approval-lane", "status_style": "restore-review-chip"}, "retention_disposition_queue": {"visual": "legal-hold-lane", "status_style": "retention-chip"}, "continuity_report": {"visual": "rto-rpo-card", "status_style": "test-chip"}, "backup_audit": {"visual": "event-ledger", "status_style": "decision-chip"}, "backup_agent_panel": {"icon": "bot", "status_style": "scope-chip"}, "stream_health": {"visual": "event-lane", "status_style": "stream-chip"}}}


def streaming_manifest() -> dict[str, Any]:
	return {
		"processor": "bytewax",
		"topic": "apg.bkup.lifecycle",
		"state": ["plans", "snapshots", "restores", "restore_approvals", "retention_dispositions", "continuity_reports", "backup_agents", "audit_events"],
		"events": ["backup_plan_created", "snapshot_created", "restore_approval_requested", "restore_approval_decided", "restore_requested", "restore_review_approved", "restore_test_recorded", "retention_disposition_requested", "retention_disposition_decided", "backup_agent_registered"],
		"batch_mutation_guardrail": "batch_backup_mutation_requires_bytewax"
	}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {"capability": "bkup", "display_name": "Backup and Restore", "provides": ["backup_plan_governance", "snapshot_vault", "restore_governance", "retention_governance", "continuity_reporting", "backup_agents"], "requires": ["encr", "conf", "audl"], "configuration": config, "configuration_schema": CONFIGURATION_SCHEMA, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "view_module": "views.py", "api_prefix": "/bkup/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True}, "theme": deepcopy(THEME), "streaming": streaming_manifest()}


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
		if key.endswith("_lte"):
			actual = context.get(key[:-4])
			if not isinstance(actual, Number) or not actual <= expected:
				return False
		elif key.endswith("_lt"):
			actual = context.get(key[:-3])
			if not isinstance(actual, Number) or not actual < expected:
				return False
		elif key.endswith("_gt"):
			actual = context.get(key[:-3])
			if not isinstance(actual, Number) or not actual > expected:
				return False
		elif key.endswith("_gte"):
			actual = context.get(key[:-4])
			if not isinstance(actual, Number) or not actual >= expected:
				return False
		elif key.endswith("_ne"):
			if context.get(key[:-3]) == expected:
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
