"""Executable capability contract for APG Dashboard Management (bia_dsh)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "bia_dsh"
CAPABILITY_NAME = "Dashboard Management"
CAPABILITY_VERSION = "1.0.0"
DASHBOARD_EVENT_STREAM = "apg.bia.dsh.lifecycle"

SUPPORTED_WIDGET_TYPES = ["bar_chart", "line_chart", "pie_chart", "donut_chart", "scatter_plot", "heatmap", "table", "kpi_card", "gauge", "treemap", "funnel", "map", "text", "image", "iframe"]
SUPPORTED_LAYOUT_TYPES = ["grid", "freeform", "responsive_grid", "tabbed", "stacked"]
SUPPORTED_DATASOURCE_TYPES = ["metric", "query", "cube", "api_endpoint", "static"]
SUPPORTED_REFRESH_INTERVALS = ["manual", "30s", "1m", "5m", "15m", "30m", "1h", "6h", "24h"]
SUPPORTED_DASHBOARD_STATES = ["draft", "published", "archived", "scheduled"]
SUPPORTED_ACCESS_LEVELS = ["private", "team", "organisation", "public"]
SUPPORTED_SNAPSHOT_FORMATS = ["png", "pdf", "html"]
SUPPORTED_THEME_MODES = ["light", "dark", "auto", "brand"]
SUPPORTED_FILTER_TYPES = ["date_range", "dropdown", "multi_select", "text_search", "slider", "checkbox"]
SUPPORTED_EXPORT_FORMATS = ["png", "pdf", "html", "json"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["dashboard_author", "widget_builder", "layout_reviewer", "access_reviewer", "snapshot_scheduler"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"dashboards": {
		"supported_states": SUPPORTED_DASHBOARD_STATES,
		"supported_access_levels": SUPPORTED_ACCESS_LEVELS,
		"require_owner": True,
		"max_widgets_per_dashboard": 50,
	},
	"widgets": {
		"supported_types": SUPPORTED_WIDGET_TYPES,
		"supported_datasource_types": SUPPORTED_DATASOURCE_TYPES,
		"supported_refresh_intervals": SUPPORTED_REFRESH_INTERVALS,
		"require_datasource": True,
	},
	"layouts": {
		"supported_types": SUPPORTED_LAYOUT_TYPES,
		"default_type": "responsive_grid",
		"max_columns": 24,
	},
	"snapshots": {
		"supported_formats": SUPPORTED_SNAPSHOT_FORMATS,
		"scheduled_snapshots_enabled": True,
		"retention_days": 90,
	},
	"filters": {
		"supported_types": SUPPORTED_FILTER_TYPES,
		"max_filters_per_dashboard": 20,
		"cross_widget_filtering": True,
	},
	"governance": {
		"require_tenant_context": True,
		"policy_attached_for_writes": True,
		"audit_events": True,
		"cross_tenant_dashboard_denied": True,
		"unapproved_public_access_denied": True,
	},
	"observability": {
		"event_stream": DASHBOARD_EVENT_STREAM,
		"stream_processor": "bytewax",
	},
	"ui": {
		"enable_builder": True,
		"enable_viewer": True,
		"enable_snapshots": True,
		"enable_filters": True,
	},
	"theme": {"default_theme": "bia_dsh_dashboards", "allow_tenant_overrides": True},
}

PROVIDES = [
	"dashboard_creation",
	"widget_library",
	"real_time_data_binding",
	"responsive_layout_engine",
	"scheduled_snapshots",
	"cross_widget_filtering",
	"dashboard_sharing",
	"dashboard_export",
	"dashboard_embedding",
]

REQUIRES = ["auth", "audl", "mten", "conf", "schd", "mqeb", "ntfy", "bia_anl"]

UI_ROUTES = [
	{"name": "dashboard_home", "path": "/bia/dsh/", "component": "DashboardHome", "permission": "bia_dsh:view", "nav_group": "Overview"},
	{"name": "dashboard_gallery", "path": "/bia/dsh/gallery", "component": "DashboardGallery", "permission": "bia_dsh:view", "nav_group": "Dashboards"},
	{"name": "dashboard_view", "path": "/bia/dsh/<id>/view", "component": "DashboardViewer", "permission": "bia_dsh:view", "nav_group": "Dashboards"},
	{"name": "dashboard_builder", "path": "/bia/dsh/<id>/build", "component": "DashboardBuilder", "permission": "bia_dsh:edit", "nav_group": "Dashboards"},
	{"name": "dashboard_new", "path": "/bia/dsh/new", "component": "DashboardCreate", "permission": "bia_dsh:create", "nav_group": "Dashboards"},
	{"name": "widget_library", "path": "/bia/dsh/widgets", "component": "WidgetLibrary", "permission": "bia_dsh:view", "nav_group": "Widgets"},
	{"name": "widget_detail", "path": "/bia/dsh/widgets/<id>", "component": "WidgetDetail", "permission": "bia_dsh:view", "nav_group": "Widgets"},
	{"name": "widget_new", "path": "/bia/dsh/widgets/new", "component": "WidgetCreate", "permission": "bia_dsh:edit", "nav_group": "Widgets"},
	{"name": "snapshots", "path": "/bia/dsh/snapshots", "component": "SnapshotManager", "permission": "bia_dsh:snapshots", "nav_group": "Snapshots"},
	{"name": "snapshot_schedule", "path": "/bia/dsh/snapshots/schedule", "component": "SnapshotScheduler", "permission": "bia_dsh:snapshots", "nav_group": "Snapshots"},
	{"name": "filter_manager", "path": "/bia/dsh/<id>/filters", "component": "FilterManager", "permission": "bia_dsh:edit", "nav_group": "Widgets"},
	{"name": "sharing", "path": "/bia/dsh/<id>/share", "component": "DashboardShare", "permission": "bia_dsh:share", "nav_group": "Dashboards"},
	{"name": "audit_log", "path": "/bia/dsh/audit", "component": "DashboardAuditLog", "permission": "bia_dsh:admin", "nav_group": "Governance"},
	{"name": "settings", "path": "/bia/dsh/settings", "component": "DashboardSettings", "permission": "bia_dsh:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "bia_dsh_dashboards",
	"tokens": {
		"color.primary": "#7C3AED",
		"color.accent": "#0EA5E9",
		"color.success": "#16A34A",
		"color.warning": "#CA8A04",
		"color.danger": "#DC2626",
		"surface.canvas": "#F8F7FF",
		"surface.panel": "#FFFFFF",
		"text.primary": "#1E1B4B",
		"text.secondary": "#4C4f6B",
		"border.radius": "8px",
		"density": "comfortable",
	},
	"components": {
		"dashboard": {"icon": "layout-dashboard", "status_indicator": "dashboard-state-chip"},
		"widget": {"icon": "bar-chart-2", "status_indicator": "widget-type-chip"},
		"snapshot": {"icon": "camera", "status_indicator": "snapshot-format-chip"},
		"filter": {"icon": "filter", "status_indicator": "filter-type-chip"},
		"layout": {"icon": "grid", "status_indicator": "layout-type-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": DASHBOARD_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"dashboard_created",
		"dashboard_published",
		"dashboard_archived",
		"widget_added",
		"widget_updated",
		"widget_removed",
		"snapshot_taken",
		"snapshot_scheduled",
		"dashboard_shared",
		"filter_applied",
		"data_refreshed",
	],
	"guardrails": [
		"cross_tenant_dashboard_denied",
		"unapproved_public_access_denied",
		"max_widget_limit_enforced",
		"snapshot_retention_enforced",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "write_policy_required", "required_action": "attach_policy"}},
	{"name": "cross_tenant_dashboard_denied", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_dashboard_not_permitted", "required_action": "restrict_to_tenant"}},
	{"name": "public_access_requires_approval", "condition": {"access_level": "public", "access_approved": False}, "effect": {"decision": "deny", "reason": "public_dashboard_requires_approval", "required_action": "submit_for_approval"}},
	{"name": "widget_limit_enforced", "condition": {"operation": "add_widget", "widget_count_exceeded": True}, "effect": {"decision": "deny", "reason": "max_widget_limit_reached", "required_action": "remove_widget_first"}},
	{"name": "widget_type_supported", "condition": {"operation": "add_widget", "widget_type_supported": False}, "effect": {"decision": "deny", "reason": "widget_type_not_supported", "required_action": "select_supported_widget_type"}},
	{"name": "widget_requires_datasource", "condition": {"operation": "add_widget", "datasource_present": False}, "effect": {"decision": "deny", "reason": "widget_datasource_required", "required_action": "attach_datasource"}},
	{"name": "layout_type_supported", "condition": {"operation": "set_layout", "layout_type_supported": False}, "effect": {"decision": "deny", "reason": "layout_type_not_supported", "required_action": "select_supported_layout_type"}},
	{"name": "snapshot_format_supported", "condition": {"operation": "take_snapshot", "format_supported": False}, "effect": {"decision": "deny", "reason": "snapshot_format_not_supported", "required_action": "select_supported_snapshot_format"}},
	{"name": "snapshot_requires_published", "condition": {"operation": "schedule_snapshot", "dashboard_state": "draft"}, "effect": {"decision": "deny", "reason": "snapshot_requires_published_dashboard", "required_action": "publish_dashboard_first"}},
	{"name": "filter_type_supported", "condition": {"operation": "add_filter", "filter_type_supported": False}, "effect": {"decision": "deny", "reason": "filter_type_not_supported", "required_action": "select_supported_filter_type"}},
	{"name": "filter_limit_enforced", "condition": {"operation": "add_filter", "filter_count_exceeded": True}, "effect": {"decision": "deny", "reason": "max_filter_limit_reached", "required_action": "remove_filter_first"}},
	{"name": "dashboard_owner_required", "condition": {"operation": "create_dashboard", "owner_present": False}, "effect": {"decision": "deny", "reason": "dashboard_owner_required", "required_action": "attach_owner"}},
	{"name": "share_requires_published", "condition": {"operation": "share_dashboard", "dashboard_state": "draft"}, "effect": {"decision": "deny", "reason": "sharing_requires_published_dashboard", "required_action": "publish_dashboard_first"}},
	{"name": "delete_shared_dashboard_requires_owner", "condition": {"operation": "delete_dashboard", "requester_is_owner": False, "access_level": "team"}, "effect": {"decision": "deny", "reason": "only_owner_can_delete_shared_dashboard", "required_action": "transfer_ownership_first"}},
	{"name": "archived_dashboard_cannot_be_published", "condition": {"operation": "publish_dashboard", "dashboard_state": "archived"}, "effect": {"decision": "deny", "reason": "archived_dashboard_cannot_be_republished", "required_action": "create_new_dashboard"}},
	{"name": "refresh_interval_supported", "condition": {"operation": "set_refresh", "refresh_interval_supported": False}, "effect": {"decision": "deny", "reason": "refresh_interval_not_supported", "required_action": "select_supported_refresh_interval"}},
	{"name": "embed_requires_published", "condition": {"operation": "embed_dashboard", "dashboard_state": "draft"}, "effect": {"decision": "deny", "reason": "embedding_requires_published_dashboard", "required_action": "publish_dashboard_first"}},
	{"name": "export_access_check", "condition": {"operation": "export_dashboard", "export_permitted": False}, "effect": {"decision": "deny", "reason": "export_not_permitted", "required_action": "request_export_permission"}},
	{"name": "audit_dashboard_views", "condition": {"operation": "view_dashboard", "audit_enabled": True}, "effect": {"decision": "allow", "reason": "dashboard_view_audited", "required_action": "emit_view_event"}},
	{"name": "snapshot_retention_enforced", "condition": {"operation": "take_snapshot", "retention_exceeded": True}, "effect": {"decision": "deny", "reason": "snapshot_retention_limit_exceeded", "required_action": "delete_old_snapshots_first"}},
	{"name": "real_time_requires_published", "condition": {"operation": "bind_realtime", "dashboard_state": "draft"}, "effect": {"decision": "deny", "reason": "real_time_binding_requires_published_dashboard", "required_action": "publish_dashboard_first"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	return {
		"capability": CAPABILITY_ID,
		"display_name": CAPABILITY_NAME,
		"version": CAPABILITY_VERSION,
		"configuration": config,
		"configuration_schema": {
			"required": ["tenant_id", "ui", "theme"],
			"properties": {
				"tenant_id": {"type": "string"},
				"ui": {"type": "object"},
				"theme": {"type": "object"},
			},
		},
		"rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": RULES},
		"ui": {
			"shell": "apg_python",
			"requires_theme": True,
			"template_roots": ["bia/dsh/templates"],
			"routes": UI_ROUTES,
		},
		"theme": THEME,
		"streaming": STREAMING,
		"provides": PROVIDES,
		"requires": REQUIRES,
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	for rule in RULES:
		cond = rule["condition"]
		if all(context.get(k) == v for k, v in cond.items()):
			return {"matched_rule": rule["name"], "decision": rule["effect"]["decision"],
			        "reason": rule["effect"]["reason"], "required_action": rule["effect"]["required_action"]}
	return {"matched_rule": None, "decision": "allow", "reason": "no_rule_matched", "required_action": None}
