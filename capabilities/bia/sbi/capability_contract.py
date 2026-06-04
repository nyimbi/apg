"""Executable capability contract for APG Self-Service BI (bia_sbi)."""

from __future__ import annotations
from copy import deepcopy
from typing import Any

CAPABILITY_ID = "bia_sbi"
CAPABILITY_NAME = "Self-Service BI"
CAPABILITY_VERSION = "1.0.0"
SBI_EVENT_STREAM = "apg.bia.sbi.lifecycle"

SUPPORTED_BUILDER_TOOLS = ["drag_drop_chart", "natural_language_query", "guided_wizard", "template_gallery", "sql_editor"]
SUPPORTED_CHART_TYPES = ["bar", "line", "pie", "scatter", "area", "funnel", "heatmap", "table", "kpi", "map", "treemap", "gauge"]
SUPPORTED_DATASOURCE_MODES = ["governed_catalogue", "sandbox", "uploaded_file", "approved_api"]
SUPPORTED_CATALOGUE_STATES = ["draft", "pending_approval", "published", "deprecated", "restricted"]
SUPPORTED_SANDBOX_STATES = ["active", "paused", "expired", "deleted"]
SUPPORTED_NLQ_ENGINES = ["rule_based", "llm_assisted", "hybrid"]
SUPPORTED_ACCESS_LEVELS = ["personal", "team", "published", "embedded"]
SUPPORTED_GOVERNANCE_TIERS = ["open", "governed", "restricted", "classified"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["catalogue_steward", "nlq_analyst", "sandbox_owner", "access_reviewer", "template_curator"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"builder": {"supported_tools": SUPPORTED_BUILDER_TOOLS, "supported_chart_types": SUPPORTED_CHART_TYPES, "max_datasets_per_workspace": 20},
	"datasources": {"supported_modes": SUPPORTED_DATASOURCE_MODES, "default_mode": "governed_catalogue", "require_catalogue_approval": True, "sandbox_row_limit": 500_000},
	"catalogue": {"supported_states": SUPPORTED_CATALOGUE_STATES, "require_owner": True, "require_description": True, "governance_tiers": SUPPORTED_GOVERNANCE_TIERS},
	"sandboxes": {"supported_states": SUPPORTED_SANDBOX_STATES, "max_sandboxes_per_user": 5, "ttl_days": 30, "auto_expire": True},
	"nlq": {"supported_engines": SUPPORTED_NLQ_ENGINES, "default_engine": "hybrid", "require_feedback": False, "audit_queries": True},
	"access": {"supported_levels": SUPPORTED_ACCESS_LEVELS, "default_level": "personal", "require_approval_for_published": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "cross_tenant_access_denied": True, "classified_data_restricted": True},
	"observability": {"event_stream": SBI_EVENT_STREAM, "stream_processor": "bytewax"},
	"theme": {"default_theme": "bia_sbi_selfservice", "allow_tenant_overrides": True},
}

PROVIDES = ["drag_drop_visual_builder", "natural_language_queries", "governed_data_catalogue", "user_sandboxes", "template_gallery", "self_service_chart_creation", "catalogue_governance", "embedded_analytics"]

REQUIRES = ["auth", "audl", "mten", "conf", "nlpc", "mqeb", "ntfy", "bia_anl"]

UI_ROUTES = [
	{"name": "home", "path": "/bia/sbi/", "component": "SBIHome", "permission": "bia_sbi:view", "nav_group": "Overview"},
	{"name": "builder", "path": "/bia/sbi/builder", "component": "VisualBuilder", "permission": "bia_sbi:build", "nav_group": "Builder"},
	{"name": "workspace", "path": "/bia/sbi/workspaces/<id>", "component": "WorkspaceView", "permission": "bia_sbi:build", "nav_group": "Builder"},
	{"name": "nlq", "path": "/bia/sbi/ask", "component": "NaturalLanguageQuery", "permission": "bia_sbi:query", "nav_group": "Query"},
	{"name": "catalogue", "path": "/bia/sbi/catalogue", "component": "DataCatalogue", "permission": "bia_sbi:catalogue", "nav_group": "Catalogue"},
	{"name": "catalogue_detail", "path": "/bia/sbi/catalogue/<id>", "component": "CatalogueEntry", "permission": "bia_sbi:catalogue", "nav_group": "Catalogue"},
	{"name": "sandboxes", "path": "/bia/sbi/sandboxes", "component": "SandboxManager", "permission": "bia_sbi:sandbox", "nav_group": "Sandboxes"},
	{"name": "sandbox_detail", "path": "/bia/sbi/sandboxes/<id>", "component": "SandboxDetail", "permission": "bia_sbi:sandbox", "nav_group": "Sandboxes"},
	{"name": "templates", "path": "/bia/sbi/templates", "component": "TemplateGallery", "permission": "bia_sbi:view", "nav_group": "Templates"},
	{"name": "published", "path": "/bia/sbi/published", "component": "PublishedAnalytics", "permission": "bia_sbi:view", "nav_group": "Published"},
	{"name": "catalogue_approvals", "path": "/bia/sbi/catalogue/approvals", "component": "CatalogueApprovals", "permission": "bia_sbi:admin", "nav_group": "Governance"},
	{"name": "audit_log", "path": "/bia/sbi/audit", "component": "SBIAuditLog", "permission": "bia_sbi:admin", "nav_group": "Governance"},
	{"name": "settings", "path": "/bia/sbi/settings", "component": "SBISettings", "permission": "bia_sbi:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "bia_sbi_selfservice",
	"tokens": {"color.primary": "#0369A1", "color.accent": "#059669", "color.success": "#16A34A", "color.warning": "#B45309", "color.danger": "#DC2626", "surface.canvas": "#F0F9FF", "surface.panel": "#FFFFFF", "text.primary": "#0C1E2F", "text.secondary": "#475569", "border.radius": "10px", "density": "comfortable"},
	"components": {
		"workspace": {"icon": "layout-grid", "status_indicator": "access-level-chip"},
		"catalogue_entry": {"icon": "book-open", "status_indicator": "catalogue-state-chip"},
		"sandbox": {"icon": "box", "status_indicator": "sandbox-state-chip"},
		"nlq_result": {"icon": "message-square", "status_indicator": "nlq-engine-chip"},
		"template": {"icon": "copy", "status_indicator": "chart-type-chip"},
	},
}

STREAMING = {
	"processor": "bytewax", "stream": SBI_EVENT_STREAM, "key": "tenant_id",
	"events": ["workspace_created", "chart_created", "nlq_submitted", "nlq_answered", "catalogue_entry_created", "catalogue_entry_approved", "sandbox_created", "sandbox_expired", "template_used", "analytics_published"],
	"guardrails": ["cross_tenant_access_denied", "classified_data_restricted", "sandbox_row_limit_enforced", "catalogue_approval_required"],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "write_policy_required", "required_action": "attach_policy"}},
	{"name": "cross_tenant_access_denied", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_access_not_permitted", "required_action": "restrict_to_tenant"}},
	{"name": "chart_type_supported", "condition": {"operation": "create_chart", "chart_type_supported": False}, "effect": {"decision": "deny", "reason": "chart_type_not_supported", "required_action": "select_supported_chart_type"}},
	{"name": "datasource_mode_supported", "condition": {"operation": "add_datasource", "datasource_mode_supported": False}, "effect": {"decision": "deny", "reason": "datasource_mode_not_supported", "required_action": "select_supported_datasource_mode"}},
	{"name": "catalogue_approval_required", "condition": {"operation": "add_datasource", "datasource_mode": "governed_catalogue", "catalogue_approved": False}, "effect": {"decision": "deny", "reason": "governed_catalogue_entry_requires_approval", "required_action": "submit_catalogue_entry_for_approval"}},
	{"name": "sandbox_row_limit_enforced", "condition": {"operation": "query_sandbox", "sandbox_rows_exceeded": True}, "effect": {"decision": "deny", "reason": "sandbox_row_limit_exceeded", "required_action": "filter_data_or_use_governed_catalogue"}},
	{"name": "sandbox_limit_per_user_enforced", "condition": {"operation": "create_sandbox", "sandbox_limit_exceeded": True}, "effect": {"decision": "deny", "reason": "sandbox_limit_per_user_exceeded", "required_action": "delete_old_sandbox_first"}},
	{"name": "expired_sandbox_cannot_query", "condition": {"operation": "query_sandbox", "sandbox_state": "expired"}, "effect": {"decision": "deny", "reason": "expired_sandbox_cannot_be_queried", "required_action": "create_new_sandbox"}},
	{"name": "classified_data_restricted", "condition": {"operation": "add_datasource", "governance_tier": "classified", "user_cleared": False}, "effect": {"decision": "deny", "reason": "classified_data_requires_clearance", "required_action": "request_data_clearance"}},
	{"name": "catalogue_state_supported", "condition": {"operation": "create_catalogue_entry", "catalogue_state_supported": False}, "effect": {"decision": "deny", "reason": "catalogue_state_not_supported", "required_action": "select_supported_catalogue_state"}},
	{"name": "catalogue_entry_requires_owner", "condition": {"operation": "create_catalogue_entry", "owner_present": False}, "effect": {"decision": "deny", "reason": "catalogue_entry_requires_owner", "required_action": "attach_owner"}},
	{"name": "catalogue_entry_requires_description", "condition": {"operation": "create_catalogue_entry", "description_present": False}, "effect": {"decision": "deny", "reason": "catalogue_entry_requires_description", "required_action": "add_description"}},
	{"name": "nlq_engine_supported", "condition": {"operation": "submit_nlq", "nlq_engine_supported": False}, "effect": {"decision": "deny", "reason": "nlq_engine_not_supported", "required_action": "select_supported_nlq_engine"}},
	{"name": "published_access_requires_approval", "condition": {"access_level": "published", "access_approved": False}, "effect": {"decision": "deny", "reason": "published_access_requires_approval", "required_action": "submit_for_publication_approval"}},
	{"name": "datasets_per_workspace_limit", "condition": {"operation": "add_datasource", "dataset_limit_exceeded": True}, "effect": {"decision": "deny", "reason": "max_datasets_per_workspace_exceeded", "required_action": "remove_dataset_first"}},
	{"name": "deprecated_catalogue_cannot_be_queried", "condition": {"operation": "query_catalogue", "catalogue_state": "deprecated"}, "effect": {"decision": "deny", "reason": "deprecated_catalogue_entry_cannot_be_queried", "required_action": "use_current_catalogue_entry"}},
	{"name": "sandbox_auto_expire_enforced", "condition": {"operation": "extend_sandbox", "sandbox_ttl_exceeded": True}, "effect": {"decision": "deny", "reason": "sandbox_maximum_ttl_exceeded", "required_action": "create_new_sandbox"}},
	{"name": "audit_nlq_queries", "condition": {"operation": "submit_nlq", "audit_enabled": True}, "effect": {"decision": "allow", "reason": "nlq_query_audited", "required_action": "emit_nlq_submitted_event"}},
	{"name": "builder_tool_supported", "condition": {"operation": "use_builder_tool", "tool_supported": False}, "effect": {"decision": "deny", "reason": "builder_tool_not_supported", "required_action": "select_supported_builder_tool"}},
	{"name": "restricted_catalogue_requires_approval", "condition": {"operation": "add_datasource", "governance_tier": "restricted", "access_approved": False}, "effect": {"decision": "deny", "reason": "restricted_catalogue_data_requires_access_approval", "required_action": "request_access_approval"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	return {
		"capability": CAPABILITY_ID, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION,
		"configuration": config,
		"configuration_schema": {"required": ["tenant_id", "ui", "theme"], "properties": {"tenant_id": {"type": "string"}, "ui": {"type": "object"}, "theme": {"type": "object"}}},
		"rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": RULES},
		"ui": {"shell": "apg_python", "requires_theme": True, "template_roots": ["bia/sbi/templates"], "routes": UI_ROUTES},
		"theme": THEME, "streaming": STREAMING, "provides": PROVIDES, "requires": REQUIRES,
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	for rule in RULES:
		if all(context.get(k) == v for k, v in rule["condition"].items()):
			return {"matched_rule": rule["name"], "decision": rule["effect"]["decision"], "reason": rule["effect"]["reason"], "required_action": rule["effect"]["required_action"]}
	return {"matched_rule": None, "decision": "allow", "reason": "no_rule_matched", "required_action": None}
