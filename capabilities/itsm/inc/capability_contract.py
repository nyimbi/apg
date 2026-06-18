"""Executable capability contract for APG ITSM Incident Management (INC)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "itsm_inc"
CAPABILITY_NAME = "Incident Management"
CAPABILITY_VERSION = "1.0.0"
INC_EVENT_STREAM = "apg.itsm.inc.lifecycle"

# ITIL v4 lifecycle
SUPPORTED_STATUSES = ["new", "acknowledged", "in_progress", "resolved", "closed", "cancelled"]
SUPPORTED_PRIORITIES = ["P1", "P2", "P3", "P4"]
SUPPORTED_CATEGORIES = [
	"hardware", "software", "network", "security", "application",
	"database", "cloud", "access", "performance", "data_loss", "other",
]
SUPPORTED_RESOLUTION_CODES = [
	"fixed", "workaround_applied", "user_error", "no_fault_found",
	"configuration_change", "vendor_fix", "duplicate", "known_error",
]
SUPPORTED_ESCALATION_LEVELS = ["L1", "L2", "L3", "vendor", "major_incident"]
SUPPORTED_IMPACT_LEVELS = ["low", "medium", "high", "critical"]
SUPPORTED_URGENCY_LEVELS = ["low", "medium", "high", "critical"]
SUPPORTED_UPDATE_TYPES = ["note", "workaround", "status_change", "assignment_change", "escalation", "resolution"]

# SLA resolve-by thresholds in minutes — ITIL recommended
SLA_MINUTES: dict[str, int] = {
	"P1": 60,		# 1 hour
	"P2": 240,		# 4 hours
	"P3": 480,		# 8 hours
	"P4": 1440,		# 24 hours
}

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"incidents": {
		"supported_statuses": SUPPORTED_STATUSES,
		"supported_priorities": SUPPORTED_PRIORITIES,
		"supported_categories": SUPPORTED_CATEGORIES,
		"supported_impact_levels": SUPPORTED_IMPACT_LEVELS,
		"supported_urgency_levels": SUPPORTED_URGENCY_LEVELS,
		"sla_minutes": SLA_MINUTES,
		"title_required": True,
		"category_required": True,
	},
	"updates": {
		"supported_update_types": SUPPORTED_UPDATE_TYPES,
		"incident_required": True,
		"author_required": True,
	},
	"sla": {
		"enabled": True,
		"breach_alert": True,
		"sla_minutes": SLA_MINUTES,
	},
	"escalation": {
		"supported_levels": SUPPORTED_ESCALATION_LEVELS,
		"auto_escalate_on_sla_breach": True,
	},
	"major_incident": {
		"declaration_threshold": "P1",
		"requires_incident_commander": True,
		"post_incident_review_required": True,
	},
	"resolutions": {
		"supported_codes": SUPPORTED_RESOLUTION_CODES,
		"workaround_required_before_close": False,
	},
	"governance": {
		"require_tenant_context": True,
		"audit_events": True,
		"itil_v4_lifecycle": True,
	},
	"observability": {
		"event_stream": INC_EVENT_STREAM,
		"stream_processor": "bytewax",
	},
	"adapters": {
		"auth": "auth",
		"audit": "audl",
		"cmdb": "itsm_cmdb",
		"notifications": "ntfy",
		"problem": "itsm_prb",
		"change": "itsm_chg",
	},
	"ui": {
		"enable_incident_queue": True,
		"enable_sla_tracker": True,
		"enable_major_incident_room": True,
		"enable_analytics": True,
		"enable_pir": True,
	},
	"theme": {
		"default_theme": "itsm_inc_control",
		"allow_tenant_overrides": True,
	},
	"nats": {
		"publishes": ["incident.created", "incident.resolved", "incident.escalated", "incident.major_declared"],
		"subscribes": ["itsm_cmdb.ci_failure", "intel_alerts.alert_created"],
	},
}

PROVIDES = [
	"incident_lifecycle_workflow",
	"incident_sla_tracking",
	"incident_escalation_workflow",
	"major_incident_workflow",
	"post_incident_review",
	"incident_analytics",
]
REQUIRES = ["auth", "audl", "ntfy", "itsm_cmdb"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/itsm-inc/dashboard", "component": "IncidentDashboard", "permission": "itsm_inc:view", "nav_group": "Overview"},
	{"name": "incident_queue", "path": "/itsm-inc/incidents", "component": "IncidentQueue", "permission": "itsm_inc:incidents", "nav_group": "Operations"},
	{"name": "major_incidents", "path": "/itsm-inc/major", "component": "MajorIncidentRoom", "permission": "itsm_inc:major", "nav_group": "Operations"},
	{"name": "sla_tracker", "path": "/itsm-inc/sla", "component": "SlaTracker", "permission": "itsm_inc:view", "nav_group": "Compliance"},
	{"name": "post_incident_review", "path": "/itsm-inc/pir", "component": "PostIncidentReview", "permission": "itsm_inc:pir", "nav_group": "Improvement"},
	{"name": "analytics", "path": "/itsm-inc/analytics", "component": "IncidentAnalytics", "permission": "itsm_inc:view", "nav_group": "Reporting"},
	{"name": "settings", "path": "/itsm-inc/settings", "component": "IncidentSettings", "permission": "itsm_inc:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "itsm_inc_control",
	"tokens": {
		"color.primary": "#DC2626",
		"color.accent": "#EA580C",
		"color.success": "#166534",
		"color.warning": "#A16207",
		"color.danger": "#991B1B",
		"surface.canvas": "#FFF7F7",
		"surface.panel": "#FFFFFF",
		"text.primary": "#111827",
		"text.secondary": "#4B5563",
		"border.radius": "6px",
		"density": "compact",
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": INC_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"incident_created", "incident_acknowledged", "incident_in_progress",
		"incident_resolved", "incident_closed", "incident_escalated",
		"incident_major_declared", "incident_pir_completed",
		"sla_breach_detected",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "incident_title_required", "condition": {"operation": "create_incident", "title_present": False}, "effect": {"decision": "deny", "reason": "incident_title_required", "required_action": "provide_incident_title"}},
	{"name": "incident_priority_supported", "condition": {"operation": "create_incident", "priority_supported": False}, "effect": {"decision": "deny", "reason": "priority_not_supported", "required_action": "select_supported_priority"}},
	{"name": "incident_category_supported", "condition": {"operation": "create_incident", "category_supported": False}, "effect": {"decision": "deny", "reason": "category_not_supported", "required_action": "select_supported_category"}},
	{"name": "resolution_requires_code", "condition": {"operation": "resolve_incident", "resolution_code_present": False}, "effect": {"decision": "deny", "reason": "resolution_code_required", "required_action": "select_resolution_code"}},
	{"name": "close_requires_resolution", "condition": {"operation": "close_incident", "is_resolved": False}, "effect": {"decision": "deny", "reason": "incident_must_be_resolved_before_close", "required_action": "resolve_incident_first"}},
	{"name": "major_incident_requires_commander", "condition": {"operation": "declare_major_incident", "commander_present": False}, "effect": {"decision": "deny", "reason": "incident_commander_required", "required_action": "assign_incident_commander"}},
	{"name": "escalation_level_supported", "condition": {"operation": "escalate_incident", "escalation_level_supported": False}, "effect": {"decision": "deny", "reason": "escalation_level_not_supported", "required_action": "select_supported_escalation_level"}},
	{"name": "update_author_required", "condition": {"operation": "add_update", "author_present": False}, "effect": {"decision": "deny", "reason": "update_author_required", "required_action": "provide_update_author"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	return {
		"capability": CAPABILITY_ID,
		"name": CAPABILITY_NAME,
		"display_name": CAPABILITY_NAME,
		"version": CAPABILITY_VERSION,
		"provides": list(PROVIDES),
		"requires": list(REQUIRES),
		"configuration": configuration,
		"rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)},
		"ui": {"shell": "apg_python", "api_prefix": "/itsm-inc/api/v1", "requires_theme": True, "routes": deepcopy(UI_ROUTES)},
		"theme": deepcopy(THEME),
		"streaming": deepcopy(STREAMING),
	
		"configuration_schema": {
			"type": "object",
			"required": ['tenant_id'],
			"properties": {
				"tenant_id": {"type": "string"},
				"incidents": {"type": "object"},
				"updates": {"type": "object"},
				"sla": {"type": "object"},
				"escalation": {"type": "object"},
				"major_incident": {"type": "object"},
				"resolutions": {"type": "object"},
				"governance": {"type": "object"},
				"observability": {"type": "object"},
				"adapters": {"type": "object"},
				"ui": {"type": "object"},
				"theme": {"type": "object"},
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
		if key.endswith("_ne"):
			if context.get(key[:-3]) == expected:
				return False
			continue
		if context.get(key) != expected:
			return False
	return True
