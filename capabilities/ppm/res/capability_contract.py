"""Executable capability contract for APG Resource Management (res)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "ppm_res"
CAPABILITY_NAME = "Resource Management"
CAPABILITY_VERSION = "1.0.0"
RES_EVENT_STREAM = "apg.ppm.res.lifecycle"

# ── Supported enum values ────────────────────────────────────────────────────
SUPPORTED_RESOURCE_TYPES = ["human", "equipment", "material", "facility", "software_license", "subcontractor", "budget_pool"]
SUPPORTED_RESOURCE_STATUSES = ["available", "allocated", "partially_allocated", "over_allocated", "on_leave", "inactive", "retired"]
SUPPORTED_SKILL_PROFICIENCY_LEVELS = ["beginner", "competent", "proficient", "expert", "master"]
SUPPORTED_ALLOCATION_STATUSES = ["proposed", "confirmed", "active", "completed", "cancelled", "on_hold"]
SUPPORTED_DEMAND_HORIZON_TYPES = ["short_term_30d", "medium_term_90d", "long_term_180d", "annual", "multi_year", "rolling"]
SUPPORTED_UTILISATION_BANDS = ["under_utilised", "optimal", "near_capacity", "over_capacity", "critical"]
SUPPORTED_CAPACITY_PLAN_TYPES = ["staffing_plan", "hiring_plan", "contractor_plan", "training_plan", "re_deployment_plan"]
SUPPORTED_MATCHING_ALGORITHMS = ["exact_skill_match", "weighted_skill_score", "availability_first", "cost_optimised", "balanced_load"]
SUPPORTED_LEAVE_TYPES = ["annual_leave", "sick_leave", "public_holiday", "training", "unpaid_leave", "maternity_paternity"]
SUPPORTED_COST_RATE_TYPES = ["standard_cost", "billing_rate", "overtime_rate", "contractor_rate", "blended_rate"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["resource_planner", "skill_matcher", "capacity_forecaster", "utilisation_analyst", "demand_coordinator"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_DEPARTMENT_TYPES = ["engineering", "product", "design", "operations", "finance", "legal", "sales", "marketing", "hr", "executive", "external"]

PROVIDES = [
	"resource_pool_management",
	"skill_matching_engine",
	"capacity_planning",
	"utilisation_tracking",
	"demand_forecasting",
	"resource_allocation_workflow",
	"leave_and_availability_management",
	"cost_rate_management",
	"resource_demand_vs_supply_analysis",
	"hiring_and_contractor_planning",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "schd", "nlpc", "mqeb"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/ppm-res/dashboard", "component": "ResDashboard", "permission": "ppm_res:view", "nav_group": "Overview"},
	{"name": "resource_pool", "path": "/ppm-res/resources", "component": "ResourcePoolList", "permission": "ppm_res:resources", "nav_group": "Resources"},
	{"name": "resource_detail", "path": "/ppm-res/resources/<id>", "component": "ResourceDetail", "permission": "ppm_res:resources", "nav_group": "Resources"},
	{"name": "skills", "path": "/ppm-res/skills", "component": "SkillCatalog", "permission": "ppm_res:skills", "nav_group": "Skills"},
	{"name": "skill_matching", "path": "/ppm-res/skill-match", "component": "SkillMatchingEngine", "permission": "ppm_res:skill_match", "nav_group": "Skills"},
	{"name": "allocations", "path": "/ppm-res/allocations", "component": "AllocationConsole", "permission": "ppm_res:allocations", "nav_group": "Allocations"},
	{"name": "capacity_plan", "path": "/ppm-res/capacity", "component": "CapacityPlanningView", "permission": "ppm_res:capacity", "nav_group": "Planning"},
	{"name": "utilisation", "path": "/ppm-res/utilisation", "component": "UtilisationTracker", "permission": "ppm_res:utilisation", "nav_group": "Analytics"},
	{"name": "demand_forecast", "path": "/ppm-res/demand", "component": "DemandForecastView", "permission": "ppm_res:demand", "nav_group": "Planning"},
	{"name": "availability", "path": "/ppm-res/availability", "component": "AvailabilityCalendar", "permission": "ppm_res:availability", "nav_group": "Scheduling"},
	{"name": "cost_rates", "path": "/ppm-res/rates", "component": "CostRateTable", "permission": "ppm_res:rates", "nav_group": "Finance"},
	{"name": "reports", "path": "/ppm-res/reports", "component": "ResourceReportBuilder", "permission": "ppm_res:reports", "nav_group": "Reports"},
	{"name": "agents", "path": "/ppm-res/agents", "component": "ResAgentWorkbench", "permission": "ppm_res:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/ppm-res/settings", "component": "ResSettings", "permission": "ppm_res:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "ppm_res_control",
	"tokens": {
		"color.primary": "#0369A1",
		"color.accent": "#7C3AED",
		"color.success": "#15803D",
		"color.warning": "#B45309",
		"color.danger": "#B91C1C",
		"surface.canvas": "#F8FAFC",
		"surface.panel": "#FFFFFF",
		"text.primary": "#111827",
		"text.secondary": "#4B5563",
		"border.radius": "8px",
		"density": "comfortable",
	},
	"components": {
		"resource": {"icon": "user", "status_indicator": "resource-status-chip"},
		"skill": {"icon": "award", "status_indicator": "proficiency-chip"},
		"allocation": {"icon": "calendar-check", "status_indicator": "allocation-status-chip"},
		"capacity_plan": {"icon": "layers", "status_indicator": "capacity-type-chip"},
		"utilisation": {"icon": "activity", "status_indicator": "utilisation-band-chip"},
		"demand_forecast": {"icon": "trending-up", "status_indicator": "demand-horizon-chip"},
		"availability": {"icon": "clock", "status_indicator": "availability-chip"},
		"cost_rate": {"icon": "dollar-sign", "status_indicator": "rate-type-chip"},
		"agent": {"icon": "bot", "status_indicator": "agent-runtime-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": RES_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"resource_created",
		"resource_updated",
		"skill_added",
		"allocation_confirmed",
		"allocation_cancelled",
		"capacity_plan_published",
		"utilisation_snapshot_taken",
		"demand_forecast_generated",
		"over_allocation_detected",
		"leave_recorded",
		"cost_rate_updated",
		"agent_registered",
	],
	"guardrails": [
		"resource_batch_requires_bytewax",
		"over_allocation_requires_manager_approval",
		"cross_tenant_resource_access_denied",
		"skill_proficiency_fabrication_denied",
		"cost_rate_change_requires_finance_approval",
		"privileged_agent_action_requires_human_approval",
	],
}

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"resources": {
		"supported_types": SUPPORTED_RESOURCE_TYPES,
		"supported_statuses": SUPPORTED_RESOURCE_STATUSES,
		"supported_departments": SUPPORTED_DEPARTMENT_TYPES,
		"owner_required": True,
		"cost_rate_required": True,
		"evidence_required": True,
	},
	"skills": {
		"supported_proficiency_levels": SUPPORTED_SKILL_PROFICIENCY_LEVELS,
		"resource_required": True,
		"evidence_required": True,
	},
	"allocations": {
		"supported_statuses": SUPPORTED_ALLOCATION_STATUSES,
		"supported_matching_algorithms": SUPPORTED_MATCHING_ALGORITHMS,
		"resource_required": True,
		"project_required": True,
		"approval_required_for_over_allocation": True,
	},
	"capacity": {
		"supported_plan_types": SUPPORTED_CAPACITY_PLAN_TYPES,
		"supported_demand_horizons": SUPPORTED_DEMAND_HORIZON_TYPES,
		"supported_utilisation_bands": SUPPORTED_UTILISATION_BANDS,
	},
	"leave": {
		"supported_types": SUPPORTED_LEAVE_TYPES,
		"resource_required": True,
		"approval_required": True,
	},
	"cost_rates": {
		"supported_types": SUPPORTED_COST_RATE_TYPES,
		"resource_required": True,
		"finance_approval_required": True,
		"effective_date_required": True,
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
		"over_allocation_requires_manager_approval": True,
		"cross_tenant_resource_access_denied": True,
		"skill_proficiency_fabrication_denied": True,
		"cost_rate_change_requires_finance_approval": True,
	},
	"observability": {"event_stream": RES_EVENT_STREAM, "stream_processor": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_resource_pool": True, "enable_skills": True, "enable_allocations": True, "enable_capacity": True, "enable_utilisation": True, "enable_demand": True, "enable_availability": True, "enable_cost_rates": True, "enable_agents": True},
	"theme": {"default_theme": "ppm_res_control", "allow_tenant_overrides": True},
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "resource_policy_required", "required_action": "attach_resource_policy"}},
	{"name": "resource_type_supported", "condition": {"operation": "create_resource", "resource_type_supported": False}, "effect": {"decision": "deny", "reason": "resource_type_not_supported", "required_action": "select_supported_resource_type"}},
	{"name": "resource_status_supported", "condition": {"operation": "create_resource", "status_supported": False}, "effect": {"decision": "deny", "reason": "resource_status_not_supported", "required_action": "select_supported_status"}},
	{"name": "resource_owner_required", "condition": {"operation": "create_resource", "owner_present": False}, "effect": {"decision": "deny", "reason": "resource_owner_required", "required_action": "assign_resource_owner"}},
	{"name": "resource_cost_rate_required", "condition": {"operation": "create_resource", "cost_rate_present": False}, "effect": {"decision": "deny", "reason": "resource_cost_rate_required", "required_action": "set_cost_rate"}},
	{"name": "resource_evidence_required", "condition": {"operation": "create_resource", "evidence_present": False}, "effect": {"decision": "deny", "reason": "resource_evidence_required", "required_action": "attach_resource_evidence"}},
	{"name": "skill_proficiency_supported", "condition": {"operation": "add_skill", "proficiency_supported": False}, "effect": {"decision": "deny", "reason": "skill_proficiency_not_supported", "required_action": "select_supported_proficiency_level"}},
	{"name": "skill_resource_required", "condition": {"operation": "add_skill", "resource_present": False}, "effect": {"decision": "deny", "reason": "resource_required", "required_action": "select_resource"}},
	{"name": "skill_evidence_required", "condition": {"operation": "add_skill", "evidence_present": False}, "effect": {"decision": "deny", "reason": "skill_evidence_required", "required_action": "attach_skill_evidence"}},
	{"name": "skill_proficiency_fabrication_denied", "condition": {"skill_proficiency_fabrication": True}, "effect": {"decision": "deny", "reason": "skill_proficiency_fabrication_denied", "required_action": "use_verified_skill_evidence"}},
	{"name": "allocation_status_supported", "condition": {"operation": "create_allocation", "status_supported": False}, "effect": {"decision": "deny", "reason": "allocation_status_not_supported", "required_action": "select_supported_status"}},
	{"name": "allocation_resource_required", "condition": {"operation": "create_allocation", "resource_present": False}, "effect": {"decision": "deny", "reason": "resource_required", "required_action": "select_resource"}},
	{"name": "allocation_project_required", "condition": {"operation": "create_allocation", "project_present": False}, "effect": {"decision": "deny", "reason": "project_required", "required_action": "select_project"}},
	{"name": "over_allocation_requires_approval", "condition": {"operation": "create_allocation", "over_allocated": True, "manager_approval_present": False}, "effect": {"decision": "deny", "reason": "over_allocation_requires_manager_approval", "required_action": "obtain_manager_approval"}},
	{"name": "matching_algorithm_supported", "condition": {"operation": "match_skills", "matching_algorithm_supported": False}, "effect": {"decision": "deny", "reason": "matching_algorithm_not_supported", "required_action": "select_supported_matching_algorithm"}},
	{"name": "demand_horizon_supported", "condition": {"operation": "forecast_demand", "demand_horizon_supported": False}, "effect": {"decision": "deny", "reason": "demand_horizon_not_supported", "required_action": "select_supported_demand_horizon"}},
	{"name": "leave_type_supported", "condition": {"operation": "record_leave", "leave_type_supported": False}, "effect": {"decision": "deny", "reason": "leave_type_not_supported", "required_action": "select_supported_leave_type"}},
	{"name": "leave_resource_required", "condition": {"operation": "record_leave", "resource_present": False}, "effect": {"decision": "deny", "reason": "resource_required", "required_action": "select_resource"}},
	{"name": "leave_approval_required", "condition": {"operation": "record_leave", "approval_present": False}, "effect": {"decision": "deny", "reason": "leave_approval_required", "required_action": "obtain_leave_approval"}},
	{"name": "cost_rate_type_supported", "condition": {"operation": "set_cost_rate", "rate_type_supported": False}, "effect": {"decision": "deny", "reason": "cost_rate_type_not_supported", "required_action": "select_supported_rate_type"}},
	{"name": "cost_rate_finance_approval_required", "condition": {"operation": "set_cost_rate", "finance_approval_present": False}, "effect": {"decision": "deny", "reason": "cost_rate_change_requires_finance_approval", "required_action": "obtain_finance_approval"}},
	{"name": "cost_rate_effective_date_required", "condition": {"operation": "set_cost_rate", "effective_date_present": False}, "effect": {"decision": "deny", "reason": "effective_date_required", "required_action": "set_effective_date"}},
	{"name": "cross_tenant_resource_access_denied", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_resource_access_denied", "required_action": "use_own_tenant_context"}},
	{"name": "resource_batch_requires_bytewax", "condition": {"operation": "resource_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_resource_batch_to_bytewax"}},
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
			"api_prefix": "/ppm-res/api/v1",
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
