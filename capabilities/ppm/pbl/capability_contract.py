"""Executable capability contract for APG Project Baseline Management (pbl)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "ppm_pbl"
CAPABILITY_NAME = "Project Baseline Management"
CAPABILITY_VERSION = "1.0.0"
PBL_EVENT_STREAM = "apg.ppm.pbl.lifecycle"

# ── Supported enum values ────────────────────────────────────────────────────
SUPPORTED_BASELINE_TYPES = ["scope", "schedule", "cost", "quality", "resource", "risk", "integrated"]
SUPPORTED_BASELINE_STATUSES = ["draft", "pending_approval", "approved", "active", "superseded", "archived"]
SUPPORTED_CHANGE_TYPES = ["scope_change", "schedule_change", "cost_change", "resource_change", "risk_reclassification", "emergency_change", "corrective_action"]
SUPPORTED_CHANGE_STATUSES = ["submitted", "under_impact_assessment", "approved", "rejected", "deferred", "implemented", "verified"]
SUPPORTED_CHANGE_PRIORITIES = ["low", "medium", "high", "critical", "emergency"]
SUPPORTED_IMPACT_AREAS = ["scope", "schedule", "cost", "quality", "risk", "resources", "stakeholders", "contracts"]
SUPPORTED_EV_METRICS = ["pv", "ev", "ac", "sv", "cv", "spi", "cpi", "bac", "eac", "etc", "vac", "tcpi"]
SUPPORTED_EV_FORECASTING_METHODS = ["typical_performance", "atypical_performance", "scheduled_completion", "custom_rate"]
SUPPORTED_VARIANCE_THRESHOLDS = ["tight", "standard", "loose", "custom"]
SUPPORTED_APPROVAL_WORKFLOWS = ["single_approver", "parallel_approval", "sequential_approval", "committee_review"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["baseline_custodian", "change_analyst", "ev_calculator", "variance_monitor", "approval_coordinator"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_DOCUMENT_TYPES = ["baseline_plan", "change_request", "impact_assessment", "approval_record", "variance_report", "ev_report"]

PROVIDES = [
	"scope_baseline_management",
	"schedule_baseline_management",
	"cost_baseline_management",
	"change_control_workflow",
	"earned_value_analysis",
	"baseline_variance_tracking",
	"change_impact_assessment",
	"baseline_approval_workflow",
	"integrated_baseline_review",
	"performance_measurement_baseline",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "comp", "mqeb"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/ppm-pbl/dashboard", "component": "PblDashboard", "permission": "ppm_pbl:view", "nav_group": "Overview"},
	{"name": "baselines", "path": "/ppm-pbl/baselines", "component": "BaselineList", "permission": "ppm_pbl:baselines", "nav_group": "Baselines"},
	{"name": "baseline_detail", "path": "/ppm-pbl/baselines/<id>", "component": "BaselineDetail", "permission": "ppm_pbl:baselines", "nav_group": "Baselines"},
	{"name": "scope_baseline", "path": "/ppm-pbl/scope", "component": "ScopeBaselineWorkbench", "permission": "ppm_pbl:scope", "nav_group": "Baselines"},
	{"name": "schedule_baseline", "path": "/ppm-pbl/schedule", "component": "ScheduleBaselineWorkbench", "permission": "ppm_pbl:schedule", "nav_group": "Baselines"},
	{"name": "cost_baseline", "path": "/ppm-pbl/cost", "component": "CostBaselineWorkbench", "permission": "ppm_pbl:cost", "nav_group": "Baselines"},
	{"name": "change_requests", "path": "/ppm-pbl/changes", "component": "ChangeRequestQueue", "permission": "ppm_pbl:changes", "nav_group": "Change Control"},
	{"name": "change_detail", "path": "/ppm-pbl/changes/<id>", "component": "ChangeRequestDetail", "permission": "ppm_pbl:changes", "nav_group": "Change Control"},
	{"name": "impact_assessment", "path": "/ppm-pbl/impact", "component": "ChangeImpactAssessment", "permission": "ppm_pbl:impact", "nav_group": "Change Control"},
	{"name": "earned_value", "path": "/ppm-pbl/ev", "component": "EarnedValueDashboard", "permission": "ppm_pbl:ev", "nav_group": "Performance"},
	{"name": "variance_report", "path": "/ppm-pbl/variance", "component": "VarianceReportView", "permission": "ppm_pbl:reports", "nav_group": "Reports"},
	{"name": "approvals", "path": "/ppm-pbl/approvals", "component": "BaselineApprovalConsole", "permission": "ppm_pbl:approve", "nav_group": "Governance"},
	{"name": "agents", "path": "/ppm-pbl/agents", "component": "PblAgentWorkbench", "permission": "ppm_pbl:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/ppm-pbl/settings", "component": "PblSettings", "permission": "ppm_pbl:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "ppm_pbl_control",
	"tokens": {
		"color.primary": "#0F766E",
		"color.accent": "#1D4ED8",
		"color.success": "#15803D",
		"color.warning": "#B45309",
		"color.danger": "#B91C1C",
		"surface.canvas": "#F8FAFC",
		"surface.panel": "#FFFFFF",
		"text.primary": "#111827",
		"text.secondary": "#4B5563",
		"border.radius": "6px",
		"density": "comfortable",
	},
	"components": {
		"baseline": {"icon": "anchor", "status_indicator": "baseline-status-chip"},
		"change_request": {"icon": "git-branch", "status_indicator": "change-status-chip"},
		"impact_assessment": {"icon": "alert-triangle", "status_indicator": "impact-chip"},
		"earned_value": {"icon": "activity", "status_indicator": "ev-health-chip"},
		"variance": {"icon": "trending-down", "status_indicator": "variance-chip"},
		"approval": {"icon": "clipboard-check", "status_indicator": "approval-status-chip"},
		"scope": {"icon": "list", "status_indicator": "scope-chip"},
		"schedule": {"icon": "calendar", "status_indicator": "schedule-chip"},
		"cost": {"icon": "dollar-sign", "status_indicator": "cost-chip"},
		"agent": {"icon": "bot", "status_indicator": "agent-runtime-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": PBL_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"baseline_created",
		"baseline_approved",
		"baseline_superseded",
		"change_request_submitted",
		"change_impact_assessed",
		"change_request_approved",
		"change_request_rejected",
		"change_implemented",
		"ev_snapshot_taken",
		"variance_threshold_breached",
		"agent_registered",
	],
	"guardrails": [
		"baseline_batch_requires_bytewax",
		"baseline_approval_requires_designated_approver",
		"change_control_bypass_denied",
		"retroactive_baseline_edit_denied",
		"cross_tenant_baseline_access_denied",
		"ev_manipulation_denied",
		"privileged_agent_action_requires_human_approval",
	],
}

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"baselines": {
		"supported_types": SUPPORTED_BASELINE_TYPES,
		"supported_statuses": SUPPORTED_BASELINE_STATUSES,
		"supported_document_types": SUPPORTED_DOCUMENT_TYPES,
		"owner_required": True,
		"approval_required": True,
		"evidence_required": True,
	},
	"change_control": {
		"supported_change_types": SUPPORTED_CHANGE_TYPES,
		"supported_change_statuses": SUPPORTED_CHANGE_STATUSES,
		"supported_priorities": SUPPORTED_CHANGE_PRIORITIES,
		"supported_impact_areas": SUPPORTED_IMPACT_AREAS,
		"supported_approval_workflows": SUPPORTED_APPROVAL_WORKFLOWS,
		"baseline_required": True,
		"impact_assessment_required": True,
		"approval_required": True,
		"evidence_required": True,
	},
	"earned_value": {
		"supported_metrics": SUPPORTED_EV_METRICS,
		"supported_forecasting_methods": SUPPORTED_EV_FORECASTING_METHODS,
		"supported_variance_thresholds": SUPPORTED_VARIANCE_THRESHOLDS,
		"baseline_required": True,
	},
	"agents": {
		"enabled": True,
		"supported_runtimes": SUPPORTED_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_AGENT_ROLES,
		"name_required": True,
		"scope_required": True,
		"human_approval_required_for_privileged_actions": True,
	},
	"governance": {
		"require_tenant_context": True,
		"policy_attached_for_writes": True,
		"audit_events": True,
		"baseline_approval_requires_designated_approver": True,
		"change_control_bypass_denied": True,
		"retroactive_baseline_edit_denied": True,
		"cross_tenant_baseline_access_denied": True,
		"ev_manipulation_denied": True,
	},
	"observability": {"event_stream": PBL_EVENT_STREAM, "stream_processor": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_baselines": True, "enable_change_control": True, "enable_earned_value": True, "enable_variance": True, "enable_approvals": True, "enable_agents": True},
	"theme": {"default_theme": "ppm_pbl_control", "allow_tenant_overrides": True},
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "baseline_policy_required", "required_action": "attach_baseline_policy"}},
	{"name": "baseline_type_supported", "condition": {"operation": "create_baseline", "baseline_type_supported": False}, "effect": {"decision": "deny", "reason": "baseline_type_not_supported", "required_action": "select_supported_baseline_type"}},
	{"name": "baseline_owner_required", "condition": {"operation": "create_baseline", "owner_present": False}, "effect": {"decision": "deny", "reason": "baseline_owner_required", "required_action": "assign_baseline_owner"}},
	{"name": "baseline_approval_required", "condition": {"operation": "create_baseline", "approval_present": False}, "effect": {"decision": "deny", "reason": "baseline_approval_required_on_creation", "required_action": "initiate_approval_workflow"}},
	{"name": "baseline_evidence_required", "condition": {"operation": "create_baseline", "evidence_present": False}, "effect": {"decision": "deny", "reason": "baseline_evidence_required", "required_action": "attach_baseline_evidence"}},
	{"name": "baseline_approval_requires_designated_approver", "condition": {"operation": "approve_baseline", "designated_approver": False}, "effect": {"decision": "deny", "reason": "baseline_approval_requires_designated_approver", "required_action": "assign_designated_approver"}},
	{"name": "retroactive_baseline_edit_denied", "condition": {"operation": "edit_baseline", "retroactive": True}, "effect": {"decision": "deny", "reason": "retroactive_baseline_edit_denied", "required_action": "submit_change_request_instead"}},
	{"name": "change_type_supported", "condition": {"operation": "submit_change_request", "change_type_supported": False}, "effect": {"decision": "deny", "reason": "change_type_not_supported", "required_action": "select_supported_change_type"}},
	{"name": "change_priority_supported", "condition": {"operation": "submit_change_request", "priority_supported": False}, "effect": {"decision": "deny", "reason": "change_priority_not_supported", "required_action": "select_supported_priority"}},
	{"name": "change_baseline_required", "condition": {"operation": "submit_change_request", "baseline_present": False}, "effect": {"decision": "deny", "reason": "baseline_required", "required_action": "select_baseline"}},
	{"name": "change_impact_required", "condition": {"operation": "submit_change_request", "impact_present": False}, "effect": {"decision": "deny", "reason": "impact_assessment_required", "required_action": "complete_impact_assessment"}},
	{"name": "change_approval_required", "condition": {"operation": "implement_change", "approval_present": False}, "effect": {"decision": "deny", "reason": "change_approval_required", "required_action": "obtain_change_approval"}},
	{"name": "change_evidence_required", "condition": {"operation": "submit_change_request", "evidence_present": False}, "effect": {"decision": "deny", "reason": "change_evidence_required", "required_action": "attach_change_evidence"}},
	{"name": "change_control_bypass_denied", "condition": {"change_control_bypass": True}, "effect": {"decision": "deny", "reason": "change_control_bypass_denied", "required_action": "follow_change_control_process"}},
	{"name": "ev_baseline_required", "condition": {"operation": "take_ev_snapshot", "baseline_present": False}, "effect": {"decision": "deny", "reason": "approved_baseline_required_for_ev", "required_action": "select_approved_baseline"}},
	{"name": "ev_forecasting_method_supported", "condition": {"operation": "take_ev_snapshot", "forecasting_method_supported": False}, "effect": {"decision": "deny", "reason": "ev_forecasting_method_not_supported", "required_action": "select_supported_forecasting_method"}},
	{"name": "ev_manipulation_denied", "condition": {"ev_manipulation": True}, "effect": {"decision": "deny", "reason": "ev_manipulation_denied", "required_action": "use_actual_performance_data"}},
	{"name": "impact_area_supported", "condition": {"operation": "assess_change_impact", "impact_area_supported": False}, "effect": {"decision": "deny", "reason": "impact_area_not_supported", "required_action": "select_supported_impact_area"}},
	{"name": "cross_tenant_baseline_access_denied", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_baseline_access_denied", "required_action": "use_own_tenant_context"}},
	{"name": "baseline_batch_requires_bytewax", "condition": {"operation": "baseline_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_baseline_batch_to_bytewax"}},
	{"name": "agent_runtime_supported", "condition": {"operation": "register_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "agent_role_supported", "condition": {"operation": "register_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "agent_name_required", "condition": {"operation": "register_agent", "agent_name_present": False}, "effect": {"decision": "deny", "reason": "agent_name_required", "required_action": "name_agent"}},
	{"name": "agent_scope_required", "condition": {"operation": "register_agent", "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "agent_scope_required", "required_action": "bound_agent_scope"}},
	{"name": "privileged_agent_action_requires_human_approval", "condition": {"operation": "agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
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
			"required": ["tenant_id", "ui", "theme"],
			"properties": {key: {"type": "object"} for key in configuration if key != "tenant_id"} | {"tenant_id": {"type": "string", "minLength": 1}},
		},
		"rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"api_prefix": "/ppm-pbl/api/v1",
			"requires_theme": True,
			"view_module": "views.py",
			"template_roots": ["templates/", "static/"],
			"routes": deepcopy(UI_ROUTES),
		},
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
		if key.endswith("_ne"):
			if context.get(key[:-3]) == expected:
				return False
			continue
		if context.get(key) != expected:
			return False
	return True
