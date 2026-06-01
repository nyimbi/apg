"""Executable capability contract for APG Intelligence Dashboard."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "intel_dashboard"
CAPABILITY_NAME = "Intelligence Dashboard"
CAPABILITY_VERSION = "1.1.0"
DASHBOARD_EVENT_STREAM = "apg.intel.dashboard.lifecycle"

SUPPORTED_AUTHORITY_TYPES = ["mission_order", "legal_mandate", "partner_authority", "consent", "incident_response_authority", "public_interest_authority"]
SUPPORTED_CLASSIFICATIONS = ["unclassified", "confidential", "secret", "top_secret"]
SUPPORTED_WORKSPACE_TYPES = ["operations_center", "threat_watch", "executive_overview", "investigation_room", "incident_room", "partner_view", "field_dashboard"]
SUPPORTED_DASHBOARD_TYPES = ["operational", "strategic", "threat", "incident", "investigative", "executive", "partner"]
SUPPORTED_SOURCE_TYPES = ["capability_summary", "graph_query", "rag_extract", "geospatial_layer", "alert_feed", "reporting_product", "prediction_projection", "threat_assessment"]
SUPPORTED_METRIC_TYPES = ["count", "rate", "risk_score", "trend", "status", "coverage", "latency", "confidence"]
SUPPORTED_WIDGET_TYPES = ["kpi_tile", "trend_chart", "map", "network_graph", "table", "timeline", "watchlist", "status_board"]
SUPPORTED_FILTER_TYPES = ["time_range", "classification", "geography", "source", "risk_level", "owner", "status"]
SUPPORTED_VIEW_TYPES = ["analyst", "supervisor", "executive", "partner", "field", "public_safety"]
SUPPORTED_SHARE_TYPES = ["internal", "partner", "executive", "field_team", "watch_center", "case_team"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["layout_designer", "metric_steward", "source_reviewer", "access_reviewer", "theme_reviewer", "briefing_preparer"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"authorities": {"supported_authority_types": SUPPORTED_AUTHORITY_TYPES, "supported_classifications": SUPPORTED_CLASSIFICATIONS, "approver_required": True, "expiry_required": True, "evidence_required": True},
	"workspaces": {"supported_workspace_types": SUPPORTED_WORKSPACE_TYPES, "supported_classifications": SUPPORTED_CLASSIFICATIONS, "authority_required": True, "evidence_required": True},
	"dashboards": {"supported_dashboard_types": SUPPORTED_DASHBOARD_TYPES, "workspace_required": True, "owner_required": True, "classification_required": True, "evidence_required": True},
	"sources": {"supported_source_types": SUPPORTED_SOURCE_TYPES, "dashboard_required": True, "custodian_required": True, "evidence_required": True},
	"metrics": {"supported_metric_types": SUPPORTED_METRIC_TYPES, "source_required": True, "confidence_required": True, "evidence_required": True},
	"widgets": {"supported_widget_types": SUPPORTED_WIDGET_TYPES, "dashboard_required": True, "metric_required": True, "evidence_required": True},
	"filters": {"supported_filter_types": SUPPORTED_FILTER_TYPES, "dashboard_required": True, "evidence_required": True},
	"views": {"supported_view_types": SUPPORTED_VIEW_TYPES, "dashboard_required": True, "viewer_role_required": True, "evidence_required": True},
	"shares": {"supported_share_types": SUPPORTED_SHARE_TYPES, "dashboard_required": True, "recipient_required": True, "approval_required": True, "evidence_required": True},
	"reviews": {"supported_statuses": SUPPORTED_REVIEW_STATUSES, "reviewer_required": True, "evidence_required": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "name_required": True, "scope_required": True, "human_approval_required_for_privileged_actions": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "lawful_authority_required": True, "cross_tenant_dashboard_denied": True, "uncited_metric_denied": True, "classification_leak_denied": True, "source_tampering_denied": True, "privacy_bypass_denied": True, "autonomous_share_denied": True, "unapproved_public_view_denied": True},
	"observability": {"event_stream": DASHBOARD_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "nlp": "nlpc", "graph": "grph", "rag": "ragn", "geospatial": "geos", "event_stream": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_authorities": True, "enable_workspaces": True, "enable_dashboards": True, "enable_sources": True, "enable_metrics": True, "enable_widgets": True, "enable_filters": True, "enable_views": True, "enable_shares": True, "enable_reviews": True, "enable_agents": True},
	"theme": {"default_theme": "intel_dashboard_control", "allow_tenant_overrides": True},
}

PROVIDES = ["dashboard_authority_workflow", "dashboard_workspace_workflow", "dashboard_composition_workflow", "dashboard_source_workflow", "dashboard_metric_workflow", "dashboard_widget_workflow", "dashboard_filter_workflow", "dashboard_view_workflow", "dashboard_share_workflow", "dashboard_review_workflow", "dashboard_agent_workflow"]
REQUIRES = ["auth", "audl", "ntfy", "nlpc", "grph", "ragn", "geos"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/intel-dashboard/dashboard", "component": "IntelligenceDashboardHome", "permission": "intel_dashboard:view", "nav_group": "Overview"},
	{"name": "authorities", "path": "/intel-dashboard/authorities", "component": "DashboardAuthorityConsole", "permission": "intel_dashboard:authorities", "nav_group": "Governance"},
	{"name": "workspaces", "path": "/intel-dashboard/workspaces", "component": "DashboardWorkspaceConsole", "permission": "intel_dashboard:workspaces", "nav_group": "Planning"},
	{"name": "dashboards", "path": "/intel-dashboard/dashboards", "component": "DashboardComposer", "permission": "intel_dashboard:dashboards", "nav_group": "Composition"},
	{"name": "sources", "path": "/intel-dashboard/sources", "component": "DashboardSourceLedger", "permission": "intel_dashboard:sources", "nav_group": "Data"},
	{"name": "metrics", "path": "/intel-dashboard/metrics", "component": "DashboardMetricLibrary", "permission": "intel_dashboard:metrics", "nav_group": "Data"},
	{"name": "widgets", "path": "/intel-dashboard/widgets", "component": "DashboardWidgetWorkbench", "permission": "intel_dashboard:widgets", "nav_group": "Composition"},
	{"name": "filters", "path": "/intel-dashboard/filters", "component": "DashboardFilterConsole", "permission": "intel_dashboard:filters", "nav_group": "Composition"},
	{"name": "views", "path": "/intel-dashboard/views", "component": "DashboardViewConsole", "permission": "intel_dashboard:views", "nav_group": "Access"},
	{"name": "shares", "path": "/intel-dashboard/shares", "component": "DashboardShareConsole", "permission": "intel_dashboard:shares", "nav_group": "Access"},
	{"name": "reviews", "path": "/intel-dashboard/reviews", "component": "DashboardReviewConsole", "permission": "intel_dashboard:reviews", "nav_group": "Governance"},
	{"name": "agents", "path": "/intel-dashboard/agents", "component": "DashboardAgentWorkbench", "permission": "intel_dashboard:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/intel-dashboard/settings", "component": "DashboardSettings", "permission": "intel_dashboard:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "intel_dashboard_control",
	"tokens": {"color.primary": "#0F766E", "color.accent": "#7C2D12", "color.success": "#166534", "color.warning": "#A16207", "color.danger": "#B91C1C", "surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF", "text.primary": "#111827", "text.secondary": "#4B5563", "border.radius": "8px", "density": "compact"},
	"components": {"authorities": {"icon": "shield-check", "status_indicator": "authority-chip"}, "workspaces": {"icon": "layout-dashboard", "status_indicator": "workspace-chip"}, "dashboards": {"icon": "panels-top-left", "status_indicator": "dashboard-chip"}, "sources": {"icon": "database", "status_indicator": "source-chip"}, "metrics": {"icon": "gauge", "status_indicator": "metric-chip"}, "widgets": {"icon": "chart-no-axes-combined", "status_indicator": "widget-chip"}, "filters": {"icon": "sliders-horizontal", "status_indicator": "filter-chip"}, "views": {"icon": "eye", "status_indicator": "view-chip"}, "shares": {"icon": "send", "status_indicator": "share-chip"}, "reviews": {"icon": "clipboard-check", "status_indicator": "review-chip"}, "agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"}},
}

STREAMING = {"processor": "bytewax", "stream": DASHBOARD_EVENT_STREAM, "key": "tenant_id", "events": ["dashboard_authority_recorded", "dashboard_workspace_recorded", "dashboard_recorded", "dashboard_source_recorded", "dashboard_metric_recorded", "dashboard_widget_recorded", "dashboard_filter_recorded", "dashboard_view_recorded", "dashboard_share_recorded", "dashboard_review_recorded", "dashboard_agent_registered"], "guardrails": ["dashboard_batch_requires_bytewax", "privileged_dashboard_agent_action_requires_human_approval", "uncited_metric_action_denied", "classification_leak_action_denied", "source_tampering_action_denied", "privacy_bypass_action_denied", "autonomous_share_action_denied", "unapproved_public_view_action_denied"]}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "dashboard_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "dashboard_policy_required", "required_action": "attach_dashboard_policy"}},
	{"name": "authority_type_supported", "condition": {"operation": "record_authority", "authority_type_supported": False}, "effect": {"decision": "deny", "reason": "authority_type_not_supported", "required_action": "select_supported_authority_type"}},
	{"name": "authority_scope_required", "condition": {"operation": "record_authority", "scope_present": False}, "effect": {"decision": "deny", "reason": "authority_scope_required", "required_action": "attach_scope_reference"}},
	{"name": "authority_classification_supported", "condition": {"operation": "record_authority", "classification_supported": False}, "effect": {"decision": "deny", "reason": "classification_not_supported", "required_action": "select_supported_classification"}},
	{"name": "authority_approver_required", "condition": {"operation": "record_authority", "approver_present": False}, "effect": {"decision": "deny", "reason": "authority_approver_required", "required_action": "record_approver"}},
	{"name": "authority_expiry_required", "condition": {"operation": "record_authority", "expiry_present": False}, "effect": {"decision": "deny", "reason": "authority_expiry_required", "required_action": "set_expiry"}},
	{"name": "authority_evidence_required", "condition": {"operation": "record_authority", "evidence_present": False}, "effect": {"decision": "deny", "reason": "authority_evidence_required", "required_action": "attach_authority_evidence"}},
	{"name": "workspace_type_supported", "condition": {"operation": "record_workspace", "workspace_type_supported": False}, "effect": {"decision": "deny", "reason": "workspace_type_not_supported", "required_action": "select_supported_workspace_type"}},
	{"name": "workspace_name_required", "condition": {"operation": "record_workspace", "workspace_name_present": False}, "effect": {"decision": "deny", "reason": "workspace_name_required", "required_action": "name_workspace"}},
	{"name": "workspace_classification_supported", "condition": {"operation": "record_workspace", "classification_supported": False}, "effect": {"decision": "deny", "reason": "classification_not_supported", "required_action": "select_supported_classification"}},
	{"name": "workspace_authority_required", "condition": {"operation": "record_workspace", "authority_present": False}, "effect": {"decision": "deny", "reason": "lawful_authority_required", "required_action": "select_authority"}},
	{"name": "workspace_evidence_required", "condition": {"operation": "record_workspace", "evidence_present": False}, "effect": {"decision": "deny", "reason": "workspace_evidence_required", "required_action": "attach_workspace_evidence"}},
	{"name": "dashboard_workspace_required", "condition": {"operation": "record_dashboard", "workspace_present": False}, "effect": {"decision": "deny", "reason": "workspace_required", "required_action": "select_workspace"}},
	{"name": "dashboard_type_supported", "condition": {"operation": "record_dashboard", "dashboard_type_supported": False}, "effect": {"decision": "deny", "reason": "dashboard_type_not_supported", "required_action": "select_supported_dashboard_type"}},
	{"name": "dashboard_title_required", "condition": {"operation": "record_dashboard", "title_present": False}, "effect": {"decision": "deny", "reason": "dashboard_title_required", "required_action": "title_dashboard"}},
	{"name": "dashboard_owner_required", "condition": {"operation": "record_dashboard", "owner_present": False}, "effect": {"decision": "deny", "reason": "dashboard_owner_required", "required_action": "assign_owner"}},
	{"name": "dashboard_classification_supported", "condition": {"operation": "record_dashboard", "classification_supported": False}, "effect": {"decision": "deny", "reason": "classification_not_supported", "required_action": "select_supported_classification"}},
	{"name": "dashboard_evidence_required", "condition": {"operation": "record_dashboard", "evidence_present": False}, "effect": {"decision": "deny", "reason": "dashboard_evidence_required", "required_action": "attach_dashboard_evidence"}},
	{"name": "source_dashboard_required", "condition": {"operation": "record_source", "dashboard_present": False}, "effect": {"decision": "deny", "reason": "dashboard_required", "required_action": "select_dashboard"}},
	{"name": "source_type_supported", "condition": {"operation": "record_source", "source_type_supported": False}, "effect": {"decision": "deny", "reason": "source_type_not_supported", "required_action": "select_supported_source_type"}},
	{"name": "source_reference_required", "condition": {"operation": "record_source", "source_reference_present": False}, "effect": {"decision": "deny", "reason": "source_reference_required", "required_action": "attach_source_reference"}},
	{"name": "source_custodian_required", "condition": {"operation": "record_source", "custodian_present": False}, "effect": {"decision": "deny", "reason": "source_custodian_required", "required_action": "assign_custodian"}},
	{"name": "source_evidence_required", "condition": {"operation": "record_source", "evidence_present": False}, "effect": {"decision": "deny", "reason": "source_evidence_required", "required_action": "attach_source_evidence"}},
	{"name": "metric_source_required", "condition": {"operation": "record_metric", "source_present": False}, "effect": {"decision": "deny", "reason": "source_required", "required_action": "select_source"}},
	{"name": "metric_type_supported", "condition": {"operation": "record_metric", "metric_type_supported": False}, "effect": {"decision": "deny", "reason": "metric_type_not_supported", "required_action": "select_supported_metric_type"}},
	{"name": "metric_reference_required", "condition": {"operation": "record_metric", "metric_reference_present": False}, "effect": {"decision": "deny", "reason": "metric_reference_required", "required_action": "attach_metric_reference"}},
	{"name": "metric_confidence_valid", "condition": {"operation": "record_metric", "confidence_valid": False}, "effect": {"decision": "deny", "reason": "confidence_score_invalid", "required_action": "set_confidence_0_to_1"}},
	{"name": "metric_evidence_required", "condition": {"operation": "record_metric", "evidence_present": False}, "effect": {"decision": "deny", "reason": "metric_evidence_required", "required_action": "attach_metric_evidence"}},
	{"name": "widget_dashboard_required", "condition": {"operation": "record_widget", "dashboard_present": False}, "effect": {"decision": "deny", "reason": "dashboard_required", "required_action": "select_dashboard"}},
	{"name": "widget_metric_required", "condition": {"operation": "record_widget", "metric_present": False}, "effect": {"decision": "deny", "reason": "metric_required", "required_action": "select_metric"}},
	{"name": "widget_type_supported", "condition": {"operation": "record_widget", "widget_type_supported": False}, "effect": {"decision": "deny", "reason": "widget_type_not_supported", "required_action": "select_supported_widget_type"}},
	{"name": "widget_reference_required", "condition": {"operation": "record_widget", "widget_reference_present": False}, "effect": {"decision": "deny", "reason": "widget_reference_required", "required_action": "attach_widget_reference"}},
	{"name": "widget_evidence_required", "condition": {"operation": "record_widget", "evidence_present": False}, "effect": {"decision": "deny", "reason": "widget_evidence_required", "required_action": "attach_widget_evidence"}},
	{"name": "filter_dashboard_required", "condition": {"operation": "record_filter", "dashboard_present": False}, "effect": {"decision": "deny", "reason": "dashboard_required", "required_action": "select_dashboard"}},
	{"name": "filter_type_supported", "condition": {"operation": "record_filter", "filter_type_supported": False}, "effect": {"decision": "deny", "reason": "filter_type_not_supported", "required_action": "select_supported_filter_type"}},
	{"name": "filter_reference_required", "condition": {"operation": "record_filter", "filter_reference_present": False}, "effect": {"decision": "deny", "reason": "filter_reference_required", "required_action": "attach_filter_reference"}},
	{"name": "filter_evidence_required", "condition": {"operation": "record_filter", "evidence_present": False}, "effect": {"decision": "deny", "reason": "filter_evidence_required", "required_action": "attach_filter_evidence"}},
	{"name": "view_dashboard_required", "condition": {"operation": "record_view", "dashboard_present": False}, "effect": {"decision": "deny", "reason": "dashboard_required", "required_action": "select_dashboard"}},
	{"name": "view_type_supported", "condition": {"operation": "record_view", "view_type_supported": False}, "effect": {"decision": "deny", "reason": "view_type_not_supported", "required_action": "select_supported_view_type"}},
	{"name": "view_reference_required", "condition": {"operation": "record_view", "view_reference_present": False}, "effect": {"decision": "deny", "reason": "view_reference_required", "required_action": "attach_view_reference"}},
	{"name": "view_role_required", "condition": {"operation": "record_view", "viewer_role_present": False}, "effect": {"decision": "deny", "reason": "viewer_role_required", "required_action": "attach_viewer_role"}},
	{"name": "view_evidence_required", "condition": {"operation": "record_view", "evidence_present": False}, "effect": {"decision": "deny", "reason": "view_evidence_required", "required_action": "attach_view_evidence"}},
	{"name": "share_dashboard_required", "condition": {"operation": "record_share", "dashboard_present": False}, "effect": {"decision": "deny", "reason": "dashboard_required", "required_action": "select_dashboard"}},
	{"name": "share_type_supported", "condition": {"operation": "record_share", "share_type_supported": False}, "effect": {"decision": "deny", "reason": "share_type_not_supported", "required_action": "select_supported_share_type"}},
	{"name": "share_recipient_required", "condition": {"operation": "record_share", "recipient_present": False}, "effect": {"decision": "deny", "reason": "recipient_reference_required", "required_action": "attach_recipient_reference"}},
	{"name": "share_approval_required", "condition": {"operation": "record_share", "approval_present": False}, "effect": {"decision": "deny", "reason": "share_approval_required", "required_action": "attach_share_approval"}},
	{"name": "share_evidence_required", "condition": {"operation": "record_share", "evidence_present": False}, "effect": {"decision": "deny", "reason": "share_evidence_required", "required_action": "attach_share_evidence"}},
	{"name": "review_status_supported", "condition": {"operation": "record_review", "status_supported": False}, "effect": {"decision": "deny", "reason": "review_status_not_supported", "required_action": "select_supported_status"}},
	{"name": "review_reviewer_required", "condition": {"operation": "record_review", "reviewer_present": False}, "effect": {"decision": "deny", "reason": "reviewer_required", "required_action": "assign_reviewer"}},
	{"name": "review_evidence_required", "condition": {"operation": "record_review", "evidence_present": False}, "effect": {"decision": "deny", "reason": "review_evidence_required", "required_action": "attach_review_evidence"}},
	{"name": "dashboard_batch_requires_bytewax", "condition": {"operation": "dashboard_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_dashboard_batch_to_bytewax"}},
	{"name": "dashboard_agent_runtime_supported", "condition": {"operation": "register_dashboard_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "dashboard_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "dashboard_agent_role_supported", "condition": {"operation": "register_dashboard_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "dashboard_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "dashboard_agent_name_required", "condition": {"operation": "register_dashboard_agent", "agent_name_present": False}, "effect": {"decision": "deny", "reason": "dashboard_agent_name_required", "required_action": "name_dashboard_agent"}},
	{"name": "dashboard_agent_scope_required", "condition": {"operation": "register_dashboard_agent", "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "dashboard_agent_scope_required", "required_action": "bound_dashboard_agent_scope"}},
	{"name": "privileged_dashboard_agent_action_requires_human_approval", "condition": {"operation": "dashboard_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
	{"name": "uncited_metric_action_denied", "condition": {"operation": "dashboard_agent_action", "uncited_metric_scope": True}, "effect": {"decision": "deny", "reason": "uncited_metric_scope_denied", "required_action": "remove_uncited_metric_scope"}},
	{"name": "classification_leak_action_denied", "condition": {"operation": "dashboard_agent_action", "classification_leak_scope": True}, "effect": {"decision": "deny", "reason": "classification_leak_scope_denied", "required_action": "remove_classification_leak_scope"}},
	{"name": "source_tampering_action_denied", "condition": {"operation": "dashboard_agent_action", "source_tampering_scope": True}, "effect": {"decision": "deny", "reason": "source_tampering_scope_denied", "required_action": "remove_source_tampering_scope"}},
	{"name": "privacy_bypass_action_denied", "condition": {"operation": "dashboard_agent_action", "privacy_bypass_scope": True}, "effect": {"decision": "deny", "reason": "privacy_bypass_scope_denied", "required_action": "remove_privacy_bypass_scope"}},
	{"name": "autonomous_share_action_denied", "condition": {"operation": "dashboard_agent_action", "autonomous_share_scope": True}, "effect": {"decision": "deny", "reason": "autonomous_share_scope_denied", "required_action": "remove_autonomous_share_scope"}},
	{"name": "unapproved_public_view_action_denied", "condition": {"operation": "dashboard_agent_action", "unapproved_public_view_scope": True}, "effect": {"decision": "deny", "reason": "unapproved_public_view_scope_denied", "required_action": "remove_unapproved_public_view_scope"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	return {"capability": CAPABILITY_ID, "name": CAPABILITY_NAME, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "provides": list(PROVIDES), "requires": list(REQUIRES), "configuration": configuration, "configuration_schema": {"type": "object", "required": list(configuration), "properties": {key: {"type": "object"} for key in configuration if key != "tenant_id"} | {"tenant_id": {"type": "string", "minLength": 1}}}, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "api_prefix": "/intel-dashboard/api/v1", "requires_theme": True, "view_module": "views.py", "template_roots": ["templates/", "static/"], "routes": deepcopy(UI_ROUTES)}, "theme": deepcopy(THEME), "streaming": deepcopy(STREAMING)}


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

