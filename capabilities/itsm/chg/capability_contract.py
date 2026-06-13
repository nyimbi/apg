"""Executable capability contract for APG ITSM Change Management (CHG)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "itsm_chg"
CAPABILITY_NAME = "Change Management"
CAPABILITY_VERSION = "1.0.0"
CHG_EVENT_STREAM = "apg.itsm.chg.lifecycle"

SUPPORTED_CHANGE_TYPES = ["standard", "normal", "emergency"]
SUPPORTED_CHANGE_STATUSES = [
	"draft", "submitted", "cab_pending", "cab_approved", "cab_rejected",
	"scheduled", "in_progress", "implemented", "failed", "rolled_back", "closed", "cancelled",
]
SUPPORTED_RISK_LEVELS = ["low", "medium", "high", "critical"]
SUPPORTED_IMPACT_LEVELS = ["low", "medium", "high", "critical"]
SUPPORTED_CAB_VOTE_OUTCOMES = ["approve", "reject", "defer", "abstain"]
SUPPORTED_REVIEW_OUTCOMES = ["success", "partial_success", "failure", "inconclusive"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"changes": {
		"supported_types": SUPPORTED_CHANGE_TYPES,
		"supported_statuses": SUPPORTED_CHANGE_STATUSES,
		"title_required": True,
		"implementer_required": True,
		"standard_changes_skip_cab": True,
		"emergency_changes_expedited": True,
	},
	"cab": {
		"quorum_pct": 0.5,
		"approval_threshold_pct": 0.6,
		"vote_outcomes": SUPPORTED_CAB_VOTE_OUTCOMES,
		"human_gate_via_temporal": True,
	},
	"schedule": {
		"conflict_detection": True,
		"freeze_windows_supported": True,
	},
	"pir": {
		"required_for_failed_changes": True,
		"required_for_emergency_changes": True,
	},
	"governance": {
		"require_tenant_context": True,
		"audit_events": True,
		"rollback_plan_required": True,
	},
	"observability": {
		"event_stream": CHG_EVENT_STREAM,
		"stream_processor": "bytewax",
	},
	"adapters": {"auth": "auth", "audit": "audl", "cmdb": "itsm_cmdb", "incidents": "itsm_inc", "temporal": "tmprl"},
	"nats": {
		"subscribes": ["itsm_cmdb.ci.change_requested"],
	},
}

PROVIDES = [
	"change_lifecycle_workflow",
	"cab_approval_workflow",
	"change_schedule_management",
	"change_conflict_detection",
	"post_implementation_review",
	"emergency_change_workflow",
]
REQUIRES = ["auth", "audl", "itsm_cmdb", "tmprl"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/itsm-chg/dashboard", "component": "ChangeDashboard", "permission": "itsm_chg:view", "nav_group": "Overview"},
	{"name": "changes", "path": "/itsm-chg/changes", "component": "ChangeQueue", "permission": "itsm_chg:changes", "nav_group": "Operations"},
	{"name": "cab_calendar", "path": "/itsm-chg/cab", "component": "CabCalendar", "permission": "itsm_chg:cab", "nav_group": "Governance"},
	{"name": "schedule", "path": "/itsm-chg/schedule", "component": "ChangeSchedule", "permission": "itsm_chg:view", "nav_group": "Planning"},
	{"name": "pir", "path": "/itsm-chg/pir", "component": "PostImplementationReview", "permission": "itsm_chg:pir", "nav_group": "Improvement"},
	{"name": "settings", "path": "/itsm-chg/settings", "component": "ChangeSettings", "permission": "itsm_chg:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "itsm_chg_control",
	"tokens": {
		"color.primary": "#0F766E",
		"color.accent": "#1D4ED8",
		"color.success": "#166534",
		"color.warning": "#A16207",
		"surface.canvas": "#F0FDF4",
		"surface.panel": "#FFFFFF",
		"border.radius": "8px",
		"density": "comfortable",
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": CHG_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"change_submitted", "change_cab_approved", "change_cab_rejected",
		"change_scheduled", "change_implemented", "change_failed",
		"change_rolled_back", "change_pir_completed",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "change_title_required", "condition": {"operation": "create_change", "title_present": False}, "effect": {"decision": "deny", "reason": "change_title_required", "required_action": "provide_change_title"}},
	{"name": "change_type_supported", "condition": {"operation": "create_change", "change_type_supported": False}, "effect": {"decision": "deny", "reason": "change_type_not_supported", "required_action": "select_supported_change_type"}},
	{"name": "rollback_plan_required", "condition": {"operation": "submit_change", "rollback_plan_present": False}, "effect": {"decision": "deny", "reason": "rollback_plan_required", "required_action": "provide_rollback_plan"}},
	{"name": "cab_vote_outcome_supported", "condition": {"operation": "record_cab_vote", "vote_outcome_supported": False}, "effect": {"decision": "deny", "reason": "vote_outcome_not_supported", "required_action": "select_supported_vote_outcome"}},
	{"name": "pir_required_for_failed", "condition": {"operation": "close_change", "is_failed": True, "pir_completed": False}, "effect": {"decision": "deny", "reason": "pir_required_before_close", "required_action": "complete_pir_first"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	return {
		"capability": CAPABILITY_ID,
		"name": CAPABILITY_NAME,
		"version": CAPABILITY_VERSION,
		"provides": list(PROVIDES),
		"requires": list(REQUIRES),
		"configuration": configuration,
		"rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)},
		"ui": {"shell": "apg_python", "api_prefix": "/itsm-chg/api/v1", "requires_theme": True, "routes": deepcopy(UI_ROUTES)},
		"theme": deepcopy(THEME),
		"streaming": deepcopy(STREAMING),
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	actions: list[dict[str, Any]] = []
	for rule in RULES:
		if _matches(rule["condition"], context):
			actions.append(rule["effect"] | {"rule": rule["name"]})
	if not actions:
		return {"decision": "allow", "actions": [], "context": dict(context)}
	return {"decision": "deny", "actions": actions, "context": dict(context)}


def _matches(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if context.get(key) != expected:
			return False
	return True
