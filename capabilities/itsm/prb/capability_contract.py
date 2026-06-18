"""Executable capability contract for APG ITSM Problem Management (PRB)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "itsm_prb"
CAPABILITY_NAME = "Problem Management"
CAPABILITY_VERSION = "1.0.0"
PRB_EVENT_STREAM = "apg.itsm.prb.lifecycle"

SUPPORTED_PROBLEM_STATUSES = ["new", "under_investigation", "root_cause_identified", "known_error", "resolved", "closed"]
SUPPORTED_RCA_METHODS = ["five_whys", "fishbone", "fault_tree", "timeline_analysis", "kepner_tregoe"]
SUPPORTED_WORKAROUND_TYPES = ["manual", "automated", "configuration_change", "bypass", "restart"]
SUPPORTED_FIX_TYPES = ["permanent", "temporary", "vendor_patch", "configuration", "architecture_change"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"problems": {
		"supported_statuses": SUPPORTED_PROBLEM_STATUSES,
		"title_required": True,
		"incident_linkage_required": False,
	},
	"rca": {
		"supported_methods": SUPPORTED_RCA_METHODS,
		"method_required": True,
	},
	"kedb": {
		"enabled": True,
		"workaround_required": True,
	},
	"governance": {
		"require_tenant_context": True,
		"audit_events": True,
	},
	"observability": {
		"event_stream": PRB_EVENT_STREAM,
		"stream_processor": "bytewax",
	},
	"adapters": {"auth": "auth", "audit": "audl", "incidents": "itsm_inc", "changes": "itsm_chg"},
}

PROVIDES = ["problem_lifecycle_workflow", "rca_workflow", "known_error_database", "workaround_management"]
REQUIRES = ["auth", "audl", "itsm_inc"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/itsm-prb/dashboard", "component": "ProblemDashboard", "permission": "itsm_prb:view", "nav_group": "Overview"},
	{"name": "problems", "path": "/itsm-prb/problems", "component": "ProblemQueue", "permission": "itsm_prb:problems", "nav_group": "Operations"},
	{"name": "rca", "path": "/itsm-prb/rca", "component": "RcaWorkbench", "permission": "itsm_prb:rca", "nav_group": "Analysis"},
	{"name": "kedb", "path": "/itsm-prb/kedb", "component": "KnownErrorDatabase", "permission": "itsm_prb:view", "nav_group": "Knowledge"},
	{"name": "settings", "path": "/itsm-prb/settings", "component": "ProblemSettings", "permission": "itsm_prb:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "itsm_prb_control",
	"tokens": {
		"color.primary": "#7C3AED",
		"color.accent": "#0F766E",
		"color.success": "#166534",
		"color.warning": "#A16207",
		"surface.canvas": "#FAF5FF",
		"surface.panel": "#FFFFFF",
		"border.radius": "8px",
		"density": "comfortable",
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": PRB_EVENT_STREAM,
	"key": "tenant_id",
	"events": ["problem_created", "rca_completed", "known_error_registered", "problem_resolved"],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "problem_title_required", "condition": {"operation": "create_problem", "title_present": False}, "effect": {"decision": "deny", "reason": "problem_title_required", "required_action": "provide_problem_title"}},
	{"name": "rca_method_supported", "condition": {"operation": "record_rca", "method_supported": False}, "effect": {"decision": "deny", "reason": "rca_method_not_supported", "required_action": "select_supported_rca_method"}},
	{"name": "kedb_workaround_required", "condition": {"operation": "register_known_error", "workaround_present": False}, "effect": {"decision": "deny", "reason": "workaround_required_for_kedb", "required_action": "provide_workaround"}},
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
		"ui": {"shell": "apg_python", "api_prefix": "/itsm-prb/api/v1", "requires_theme": True, "routes": deepcopy(UI_ROUTES)},
		"theme": deepcopy(THEME),
		"streaming": deepcopy(STREAMING),
	
		"configuration_schema": {
			"type": "object",
			"required": ['tenant_id'],
			"properties": {
				"tenant_id": {"type": "string"},
				"problems": {"type": "object"},
				"rca": {"type": "object"},
				"kedb": {"type": "object"},
				"governance": {"type": "object"},
				"observability": {"type": "object"},
				"adapters": {"type": "object"},
			},
		},
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
