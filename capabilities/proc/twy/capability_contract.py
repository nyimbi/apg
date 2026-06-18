"""Executable capability contract for APG Three-Way Match Engine (proc_twy)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "proc_twy"
CAPABILITY_NAME = "Three-Way Match Engine"
CAPABILITY_VERSION = "1.0.0"
TWY_EVENT_STREAM = "apg.proc.twy.lifecycle"

# Supported match outcome codes
MATCH_OUTCOMES = ["matched", "partial_match", "exception"]

# Exception status lifecycle
EXCEPTION_STATUSES = ["open", "pending_review", "escalated", "resolved", "cancelled"]

# Exception resolution types
EXCEPTION_RESOLUTION_TYPES = [
	"approved_with_variance",
	"rejected",
	"duplicate_invoice",
	"cancelled_po",
	"goods_not_received",
	"price_correction",
	"quantity_correction",
	"date_extension",
	"manual_override",
]

# Tolerance scopes — determines how a rule is applied
TOLERANCE_SCOPES = ["global", "vendor", "category", "line_item"]

# Document types participating in the match
DOCUMENT_TYPES = ["purchase_order", "goods_receipt", "vendor_invoice"]

# Variance types surfaced during matching
VARIANCE_TYPES = ["price", "quantity", "date", "line_missing", "document_missing"]

# Escalation targets
ESCALATION_TARGETS = ["ap_manager", "procurement_manager", "cfo", "vendor_manager", "compliance"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"matching": {
		"default_price_tolerance_pct": 2.0,
		"default_quantity_tolerance_pct": 5.0,
		"default_date_tolerance_days": 30,
		"require_all_three_documents": True,
		"allow_two_way_match_fallback": False,
		"auto_approve_within_tolerance": True,
		"line_level_matching": True,
		"header_level_matching": True,
	},
	"exceptions": {
		"supported_statuses": EXCEPTION_STATUSES,
		"supported_resolution_types": EXCEPTION_RESOLUTION_TYPES,
		"auto_escalation_age_days": 7,
		"max_resolution_days": 30,
		"require_resolution_note": True,
	},
	"tolerance_rules": {
		"supported_scopes": TOLERANCE_SCOPES,
		"allow_zero_tolerance": True,
		"allow_per_vendor_overrides": True,
		"allow_per_category_overrides": True,
	},
	"governance": {
		"require_tenant_context": True,
		"audit_events": True,
		"cross_tenant_match_denied": True,
		"auto_approve_over_threshold_denied": False,
		"policy_attached_for_writes": True,
	},
	"observability": {
		"event_stream": TWY_EVENT_STREAM,
		"stream_processor": "bytewax",
	},
	"adapters": {
		"auth": "auth",
		"audit": "audl",
		"notifications": "ntfy",
		"fin_arc": "fin_arc",
		"scm_prc": "scm_prc",
		"scm_wms": "scm_wms",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_match_queue": True,
		"enable_exceptions": True,
		"enable_tolerance_rules": True,
		"enable_analytics": True,
		"enable_settings": True,
	},
	"theme": {
		"default_theme": "proc_twy_control",
		"allow_tenant_overrides": True,
	},
}

PROVIDES = [
	"three_way_match",
	"exception_management",
	"tolerance_rules",
	"match_analytics",
]

REQUIRES = ["auth", "audl", "ntfy", "fin_arc", "scm_prc", "scm_wms"]

NATS_PUBLISHES = ["match.completed", "exception.raised", "exception.resolved", "auto_approved"]

NATS_SUBSCRIBES = [
	{
		"source_capability": "fin_arc",
		"event_type": "invoice.received",
		"handler": "on_invoice_received",
	},
]

UI_ROUTES = [
	{"name": "dashboard", "path": "/proc-twy/dashboard", "component": "TwyDashboard", "permission": "proc_twy:view", "nav_group": "Overview"},
	{"name": "match_queue", "path": "/proc-twy/queue", "component": "TwyMatchQueue", "permission": "proc_twy:match", "nav_group": "Operations"},
	{"name": "exceptions", "path": "/proc-twy/exceptions", "component": "TwyExceptionConsole", "permission": "proc_twy:exceptions", "nav_group": "Operations"},
	{"name": "tolerance_rules", "path": "/proc-twy/tolerance-rules", "component": "TwyToleranceRuleEditor", "permission": "proc_twy:admin", "nav_group": "Configuration"},
	{"name": "analytics", "path": "/proc-twy/analytics", "component": "TwyAnalytics", "permission": "proc_twy:view", "nav_group": "Insights"},
	{"name": "settings", "path": "/proc-twy/settings", "component": "TwySettings", "permission": "proc_twy:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "proc_twy_control",
	"tokens": {
		"color.primary": "#1D4ED8",
		"color.accent": "#0369A1",
		"color.success": "#166534",
		"color.warning": "#A16207",
		"color.danger": "#991B1B",
		"surface.canvas": "#F0F9FF",
		"surface.panel": "#FFFFFF",
		"text.primary": "#0F172A",
		"text.secondary": "#475569",
		"border.radius": "6px",
		"density": "comfortable",
	},
	"components": {
		"match_queue": {"icon": "git-merge", "status_indicator": "match-outcome-chip"},
		"exceptions": {"icon": "alert-triangle", "status_indicator": "exception-status-chip"},
		"tolerance_rules": {"icon": "sliders", "status_indicator": "tolerance-scope-chip"},
		"analytics": {"icon": "bar-chart-2", "status_indicator": "kpi-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": TWY_EVENT_STREAM,
	"key": "tenant_id",
	"publishes": NATS_PUBLISHES,
	"subscribes": NATS_SUBSCRIBES,
	"guardrails": [
		"cross_tenant_match_denied",
		"auto_approve_over_threshold_denied",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "match_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "match_policy_required", "required_action": "attach_match_policy"}},
	{"name": "cross_tenant_match_denied", "condition": {"operation": "match_documents", "cross_tenant": True}, "effect": {"decision": "deny", "reason": "cross_tenant_match_not_allowed", "required_action": "use_single_tenant_documents"}},
	{"name": "document_type_supported", "condition": {"operation": "create_document", "document_type_supported": False}, "effect": {"decision": "deny", "reason": "document_type_not_supported", "required_action": "select_supported_document_type"}},
	{"name": "tolerance_scope_supported", "condition": {"operation": "create_tolerance_rule", "tolerance_scope_supported": False}, "effect": {"decision": "deny", "reason": "tolerance_scope_not_supported", "required_action": "select_supported_tolerance_scope"}},
	{"name": "exception_status_supported", "condition": {"operation": "update_exception", "exception_status_supported": False}, "effect": {"decision": "deny", "reason": "exception_status_not_supported", "required_action": "select_supported_exception_status"}},
	{"name": "exception_resolution_type_supported", "condition": {"operation": "resolve_exception", "resolution_type_supported": False}, "effect": {"decision": "deny", "reason": "resolution_type_not_supported", "required_action": "select_supported_resolution_type"}},
	{"name": "exception_resolution_note_required", "condition": {"operation": "resolve_exception", "resolution_note_present": False}, "effect": {"decision": "deny", "reason": "resolution_note_required", "required_action": "add_resolution_note"}},
	{"name": "escalation_target_supported", "condition": {"operation": "escalate_exception", "escalation_target_supported": False}, "effect": {"decision": "deny", "reason": "escalation_target_not_supported", "required_action": "select_supported_escalation_target"}},
	{"name": "auto_approve_requires_tolerance_check", "condition": {"operation": "auto_approve", "within_tolerance": False}, "effect": {"decision": "deny", "reason": "auto_approve_only_within_tolerance", "required_action": "resolve_exception_manually"}},
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
		"configuration_schema": {
			"type": "object",
			"required": list(configuration),
			"properties": {key: {"type": "object"} for key in configuration if key != "tenant_id"} | {"tenant_id": {"type": "string", "minLength": 1}},
		},
		"rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"api_prefix": "/proc-twy/api/v1",
			"requires_theme": True,
			"view_module": "views.py",
			"template_roots": ["templates/", "static/"],
			"routes": deepcopy(UI_ROUTES),
		},
		"theme": deepcopy(THEME),
		"streaming": deepcopy(STREAMING),
		"nats": {
			"publishes": list(NATS_PUBLISHES),
			"subscribes": deepcopy(NATS_SUBSCRIBES),
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
