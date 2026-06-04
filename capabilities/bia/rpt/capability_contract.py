"""Executable capability contract for APG Report Builder (bia_rpt)."""

from __future__ import annotations
from copy import deepcopy
from typing import Any

CAPABILITY_ID = "bia_rpt"
CAPABILITY_NAME = "Report Builder"
CAPABILITY_VERSION = "1.0.0"
RPT_EVENT_STREAM = "apg.bia.rpt.lifecycle"

SUPPORTED_REPORT_TYPES = ["tabular", "summary", "cross_tab", "chart", "dashboard_export", "letter", "invoice", "custom"]
SUPPORTED_OUTPUT_FORMATS = ["pdf", "xlsx", "csv", "html", "docx", "json", "xml"]
SUPPORTED_SCHEDULE_FREQUENCIES = ["once", "daily", "weekly", "monthly", "quarterly", "annual", "cron"]
SUPPORTED_DISTRIBUTION_CHANNELS = ["email", "sftp", "s3", "webhook", "in_app", "sharepoint", "api"]
SUPPORTED_PARAMETER_TYPES = ["date", "date_range", "string", "number", "boolean", "dropdown", "multi_select"]
SUPPORTED_REPORT_STATES = ["draft", "published", "scheduled", "archived", "deprecated"]
SUPPORTED_AUDIT_ACTIONS = ["created", "updated", "published", "run", "distributed", "archived", "deleted"]
SUPPORTED_SECTION_TYPES = ["header", "body", "footer", "summary", "chart", "table", "text", "image", "page_break"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["report_author", "template_designer", "schedule_manager", "distribution_reviewer", "audit_reviewer"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"reports": {"supported_types": SUPPORTED_REPORT_TYPES, "supported_states": SUPPORTED_REPORT_STATES, "require_owner": True, "max_parameters": 30, "versioning_enabled": True},
	"output": {"supported_formats": SUPPORTED_OUTPUT_FORMATS, "default_format": "pdf", "max_pages": 500, "watermark_enabled": True},
	"scheduling": {"supported_frequencies": SUPPORTED_SCHEDULE_FREQUENCIES, "max_schedules_per_report": 5, "require_owner": True},
	"distribution": {"supported_channels": SUPPORTED_DISTRIBUTION_CHANNELS, "require_recipient": True, "require_approval_for_external": True},
	"parameters": {"supported_types": SUPPORTED_PARAMETER_TYPES, "require_default_value": False},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "cross_tenant_report_denied": True, "distribution_approval_required": True},
	"observability": {"event_stream": RPT_EVENT_STREAM, "stream_processor": "bytewax"},
	"theme": {"default_theme": "bia_rpt_reports", "allow_tenant_overrides": True},
}

PROVIDES = ["parameterised_report_authoring", "report_scheduling", "report_distribution", "multi_format_export", "report_audit_trail", "report_template_library", "report_versioning", "report_bursting"]

REQUIRES = ["auth", "audl", "mten", "conf", "schd", "mqeb", "ntfy", "bia_anl"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/bia/rpt/dashboard", "component": "ReportDashboard", "permission": "bia_rpt:view", "nav_group": "Overview"},
	{"name": "report_library", "path": "/bia/rpt/reports", "component": "ReportLibrary", "permission": "bia_rpt:view", "nav_group": "Reports"},
	{"name": "report_detail", "path": "/bia/rpt/reports/<id>", "component": "ReportDetail", "permission": "bia_rpt:view", "nav_group": "Reports"},
	{"name": "report_builder", "path": "/bia/rpt/reports/<id>/build", "component": "ReportBuilder", "permission": "bia_rpt:edit", "nav_group": "Reports"},
	{"name": "report_new", "path": "/bia/rpt/reports/new", "component": "ReportCreate", "permission": "bia_rpt:create", "nav_group": "Reports"},
	{"name": "report_run", "path": "/bia/rpt/reports/<id>/run", "component": "ReportRunner", "permission": "bia_rpt:run", "nav_group": "Reports"},
	{"name": "schedules", "path": "/bia/rpt/schedules", "component": "ReportScheduleManager", "permission": "bia_rpt:schedule", "nav_group": "Scheduling"},
	{"name": "schedule_detail", "path": "/bia/rpt/schedules/<id>", "component": "ScheduleDetail", "permission": "bia_rpt:schedule", "nav_group": "Scheduling"},
	{"name": "distribution", "path": "/bia/rpt/distribution", "component": "DistributionManager", "permission": "bia_rpt:distribute", "nav_group": "Distribution"},
	{"name": "templates", "path": "/bia/rpt/templates", "component": "TemplateLibrary", "permission": "bia_rpt:view", "nav_group": "Templates"},
	{"name": "run_history", "path": "/bia/rpt/history", "component": "RunHistory", "permission": "bia_rpt:view", "nav_group": "History"},
	{"name": "audit_log", "path": "/bia/rpt/audit", "component": "ReportAuditLog", "permission": "bia_rpt:admin", "nav_group": "Governance"},
	{"name": "settings", "path": "/bia/rpt/settings", "component": "ReportSettings", "permission": "bia_rpt:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "bia_rpt_reports",
	"tokens": {"color.primary": "#1B5E20", "color.accent": "#0277BD", "color.success": "#2E7D32", "color.warning": "#E65100", "color.danger": "#B71C1C", "surface.canvas": "#F1F8E9", "surface.panel": "#FFFFFF", "text.primary": "#1B2A1B", "text.secondary": "#546E57", "border.radius": "4px", "density": "compact"},
	"components": {
		"report": {"icon": "file-text", "status_indicator": "report-state-chip"},
		"schedule": {"icon": "calendar-clock", "status_indicator": "schedule-freq-chip"},
		"distribution": {"icon": "send", "status_indicator": "channel-chip"},
		"template": {"icon": "layout-template", "status_indicator": "template-type-chip"},
		"run": {"icon": "play-circle", "status_indicator": "run-status-chip"},
	},
}

STREAMING = {
	"processor": "bytewax", "stream": RPT_EVENT_STREAM, "key": "tenant_id",
	"events": ["report_created", "report_published", "report_run_started", "report_run_completed", "report_distributed", "report_scheduled", "report_archived", "report_template_created", "distribution_approved", "distribution_rejected"],
	"guardrails": ["cross_tenant_report_denied", "distribution_approval_required_for_external", "max_pages_enforced", "audit_all_runs"],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "write_policy_required", "required_action": "attach_policy"}},
	{"name": "cross_tenant_report_denied", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_report_not_permitted", "required_action": "restrict_to_tenant"}},
	{"name": "report_type_supported", "condition": {"operation": "create_report", "report_type_supported": False}, "effect": {"decision": "deny", "reason": "report_type_not_supported", "required_action": "select_supported_report_type"}},
	{"name": "report_owner_required", "condition": {"operation": "create_report", "owner_present": False}, "effect": {"decision": "deny", "reason": "report_owner_required", "required_action": "attach_owner"}},
	{"name": "output_format_supported", "condition": {"operation": "run_report", "format_supported": False}, "effect": {"decision": "deny", "reason": "output_format_not_supported", "required_action": "select_supported_output_format"}},
	{"name": "run_requires_published", "condition": {"operation": "run_report", "report_state": "draft"}, "effect": {"decision": "deny", "reason": "draft_report_cannot_be_run", "required_action": "publish_report_first"}},
	{"name": "schedule_frequency_supported", "condition": {"operation": "create_schedule", "frequency_supported": False}, "effect": {"decision": "deny", "reason": "schedule_frequency_not_supported", "required_action": "select_supported_frequency"}},
	{"name": "schedule_limit_enforced", "condition": {"operation": "create_schedule", "schedule_limit_exceeded": True}, "effect": {"decision": "deny", "reason": "max_schedules_per_report_exceeded", "required_action": "delete_existing_schedule_first"}},
	{"name": "distribution_channel_supported", "condition": {"operation": "distribute_report", "channel_supported": False}, "effect": {"decision": "deny", "reason": "distribution_channel_not_supported", "required_action": "select_supported_channel"}},
	{"name": "external_distribution_requires_approval", "condition": {"operation": "distribute_report", "is_external_channel": True, "distribution_approved": False}, "effect": {"decision": "deny", "reason": "external_distribution_requires_approval", "required_action": "submit_distribution_for_approval"}},
	{"name": "distribution_recipient_required", "condition": {"operation": "distribute_report", "recipient_present": False}, "effect": {"decision": "deny", "reason": "distribution_recipient_required", "required_action": "specify_recipient"}},
	{"name": "parameter_type_supported", "condition": {"operation": "add_parameter", "parameter_type_supported": False}, "effect": {"decision": "deny", "reason": "parameter_type_not_supported", "required_action": "select_supported_parameter_type"}},
	{"name": "parameter_limit_enforced", "condition": {"operation": "add_parameter", "parameter_limit_exceeded": True}, "effect": {"decision": "deny", "reason": "parameter_limit_exceeded", "required_action": "reduce_parameter_count"}},
	{"name": "max_pages_enforced", "condition": {"operation": "run_report", "page_limit_exceeded": True}, "effect": {"decision": "deny", "reason": "report_page_limit_exceeded", "required_action": "add_filters_or_pagination"}},
	{"name": "deprecated_report_cannot_run", "condition": {"operation": "run_report", "report_state": "deprecated"}, "effect": {"decision": "deny", "reason": "deprecated_report_cannot_be_run", "required_action": "use_current_report_version"}},
	{"name": "archived_report_read_only", "condition": {"operation": "update_report", "report_state": "archived"}, "effect": {"decision": "deny", "reason": "archived_report_is_read_only", "required_action": "create_new_report_version"}},
	{"name": "section_type_supported", "condition": {"operation": "add_section", "section_type_supported": False}, "effect": {"decision": "deny", "reason": "section_type_not_supported", "required_action": "select_supported_section_type"}},
	{"name": "audit_all_runs", "condition": {"operation": "run_report", "audit_enabled": True}, "effect": {"decision": "allow", "reason": "report_run_audited", "required_action": "emit_report_run_event"}},
	{"name": "delete_published_report_requires_archive_first", "condition": {"operation": "delete_report", "report_state": "published"}, "effect": {"decision": "deny", "reason": "published_report_must_be_archived_before_deletion", "required_action": "archive_report_first"}},
	{"name": "schedule_requires_published_report", "condition": {"operation": "create_schedule", "report_state": "draft"}, "effect": {"decision": "deny", "reason": "schedule_requires_published_report", "required_action": "publish_report_before_scheduling"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	return {
		"capability": CAPABILITY_ID, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION,
		"configuration": config,
		"configuration_schema": {"required": ["tenant_id", "ui", "theme"], "properties": {"tenant_id": {"type": "string"}, "ui": {"type": "object"}, "theme": {"type": "object"}}},
		"rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": RULES},
		"ui": {"shell": "apg_python", "requires_theme": True, "template_roots": ["bia/rpt/templates"], "routes": UI_ROUTES},
		"theme": THEME, "streaming": STREAMING, "provides": PROVIDES, "requires": REQUIRES,
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	for rule in RULES:
		if all(context.get(k) == v for k, v in rule["condition"].items()):
			return {"matched_rule": rule["name"], "decision": rule["effect"]["decision"], "reason": rule["effect"]["reason"], "required_action": rule["effect"]["required_action"]}
	return {"matched_rule": None, "decision": "allow", "reason": "no_rule_matched", "required_action": None}
