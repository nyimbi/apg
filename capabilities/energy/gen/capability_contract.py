"""Executable capability contract for APG Generation Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "energy_gen"
CAPABILITY_NAME = "Generation Management"
CAPABILITY_VERSION = "1.0.0"
GEN_EVENT_STREAM = "apg.energy.gen.lifecycle"

SUPPORTED_PLANT_TYPES = ["thermal_coal", "thermal_gas", "thermal_oil", "nuclear", "hydro", "pumped_storage", "solar_pv", "wind_onshore", "wind_offshore", "biomass", "geothermal", "diesel_peaker", "gas_peaker", "combined_cycle"]
SUPPORTED_FUEL_TYPES = ["coal", "natural_gas", "oil", "uranium", "water", "solar", "wind", "biomass", "geothermal", "diesel"]
SUPPORTED_DISPATCH_MODES = ["baseload", "load_following", "peaking", "spinning_reserve", "non_spinning_reserve", "black_start", "must_run", "economic_dispatch", "constrained_on", "constrained_off"]
SUPPORTED_PLANT_STATUSES = ["operational", "under_maintenance", "forced_outage", "planned_outage", "mothballed", "decommissioned", "commissioning", "standby"]
SUPPORTED_OUTAGE_TYPES = ["planned_maintenance", "forced_outage", "partial_derating", "fuel_constraint", "environmental_curtailment", "regulatory_hold", "grid_constraint"]
SUPPORTED_OUTAGE_STATUSES = ["scheduled", "in_progress", "completed", "cancelled", "extended"]
SUPPORTED_CAPACITY_UNIT_TYPES = ["mw", "kw", "mva"]
SUPPORTED_ENERGY_UNIT_TYPES = ["mwh", "kwh", "gwh"]
SUPPORTED_KPI_TYPES = ["availability_factor", "capacity_factor", "heat_rate", "efficiency", "forced_outage_rate", "planned_outage_rate", "equivalent_availability_factor", "net_capacity_factor"]
SUPPORTED_SCHEDULE_STATUSES = ["draft", "submitted", "approved", "active", "completed", "revised"]
SUPPORTED_PERFORMANCE_PERIODS = ["hourly", "daily", "weekly", "monthly", "quarterly", "annual"]
SUPPORTED_APPROVAL_STATUSES = ["pending", "approved", "rejected", "escalated"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["dispatch_optimizer", "outage_planner", "kpi_analyst", "capacity_planner", "fuel_manager"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"ui": {"enable_dashboard": True, "enable_plants": True, "enable_dispatch": True, "enable_outages": True, "enable_kpis": True, "enable_capacity": True, "enable_schedules": True, "enable_fuel": True},
	"theme": {"default_theme": "energy_gen_ops", "allow_tenant_overrides": True},
	"plants": {"supported_plant_types": SUPPORTED_PLANT_TYPES, "supported_statuses": SUPPORTED_PLANT_STATUSES, "capacity_unit": "mw", "owner_required": True, "commissioning_date_required": True},
	"dispatch": {"supported_modes": SUPPORTED_DISPATCH_MODES, "schedule_approval_required": True, "supported_schedule_statuses": SUPPORTED_SCHEDULE_STATUSES},
	"outages": {"supported_outage_types": SUPPORTED_OUTAGE_TYPES, "supported_statuses": SUPPORTED_OUTAGE_STATUSES, "approval_required": True, "minimum_notice_hours": 24},
	"kpis": {"supported_kpi_types": SUPPORTED_KPI_TYPES, "supported_periods": SUPPORTED_PERFORMANCE_PERIODS, "auto_calculate": True},
	"capacity": {"supported_unit_types": SUPPORTED_CAPACITY_UNIT_TYPES, "planning_horizon_years": 10},
	"fuel": {"supported_fuel_types": SUPPORTED_FUEL_TYPES, "stock_monitoring": True, "alert_threshold_days": 7},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_dispatch": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "cross_tenant_denied": True, "unapproved_dispatch_denied": True, "unapproved_outage_denied": True},
	"observability": {"event_stream": GEN_EVENT_STREAM, "stream_processor": "bytewax"},
}

PROVIDES = [
	"plant_registry",
	"dispatch_scheduling",
	"outage_management",
	"capacity_planning",
	"generation_kpis",
	"fuel_management",
	"performance_reporting",
	"dispatch_optimization",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "moni", "schd", "mqeb", "comp"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/energy-gen/dashboard", "component": "GenDashboard", "permission": "energy_gen:view", "nav_group": "Overview"},
	{"name": "plants", "path": "/energy-gen/plants", "component": "PlantRegistry", "permission": "energy_gen:plants", "nav_group": "Assets"},
	{"name": "plant_detail", "path": "/energy-gen/plants/<id>", "component": "PlantDetail", "permission": "energy_gen:plants", "nav_group": "Assets"},
	{"name": "dispatch", "path": "/energy-gen/dispatch", "component": "DispatchConsole", "permission": "energy_gen:dispatch", "nav_group": "Operations"},
	{"name": "schedules", "path": "/energy-gen/schedules", "component": "DispatchSchedules", "permission": "energy_gen:dispatch", "nav_group": "Operations"},
	{"name": "outages", "path": "/energy-gen/outages", "component": "OutageManager", "permission": "energy_gen:outages", "nav_group": "Maintenance"},
	{"name": "outage_detail", "path": "/energy-gen/outages/<id>", "component": "OutageDetail", "permission": "energy_gen:outages", "nav_group": "Maintenance"},
	{"name": "kpis", "path": "/energy-gen/kpis", "component": "GenerationKPIs", "permission": "energy_gen:kpis", "nav_group": "Performance"},
	{"name": "capacity", "path": "/energy-gen/capacity", "component": "CapacityPlanner", "permission": "energy_gen:capacity", "nav_group": "Planning"},
	{"name": "fuel", "path": "/energy-gen/fuel", "component": "FuelManagement", "permission": "energy_gen:fuel", "nav_group": "Resources"},
	{"name": "reports", "path": "/energy-gen/reports", "component": "GenerationReports", "permission": "energy_gen:reports", "nav_group": "Reporting"},
	{"name": "agents", "path": "/energy-gen/agents", "component": "GenAgentWorkbench", "permission": "energy_gen:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/energy-gen/settings", "component": "GenSettings", "permission": "energy_gen:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "energy_gen_ops",
	"tokens": {
		"color.primary": "#1D4ED8",
		"color.accent": "#F59E0B",
		"color.success": "#15803D",
		"color.warning": "#D97706",
		"color.danger": "#DC2626",
		"surface.canvas": "#F0F4F8",
		"surface.panel": "#FFFFFF",
		"text.primary": "#1E293B",
		"text.secondary": "#475569",
		"border.radius": "6px",
		"density": "comfortable",
	},
	"components": {
		"plants": {"icon": "zap", "status_indicator": "plant-status-chip"},
		"dispatch": {"icon": "activity", "status_indicator": "dispatch-mode-chip"},
		"outages": {"icon": "alert-triangle", "status_indicator": "outage-status-chip"},
		"kpis": {"icon": "bar-chart-2", "status_indicator": "kpi-trend-chip"},
		"capacity": {"icon": "layers", "status_indicator": "capacity-chip"},
		"fuel": {"icon": "droplet", "status_indicator": "fuel-level-chip"},
		"schedules": {"icon": "calendar", "status_indicator": "schedule-status-chip"},
		"agents": {"icon": "cpu", "status_indicator": "agent-runtime-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": GEN_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"plant_registered", "plant_status_changed", "dispatch_schedule_created",
		"dispatch_schedule_approved", "outage_scheduled", "outage_started",
		"outage_completed", "kpi_calculated", "capacity_plan_updated",
		"fuel_stock_updated", "fuel_alert_triggered", "gen_agent_registered",
	],
	"guardrails": [
		"dispatch_batch_requires_bytewax",
		"unapproved_dispatch_denied",
		"unapproved_outage_denied",
		"privileged_gen_agent_requires_human_approval",
		"cross_tenant_generation_data_denied",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "write_policy_required", "required_action": "attach_write_policy"}},
	{"name": "plant_type_supported", "condition": {"operation": "register_plant", "plant_type_supported": False}, "effect": {"decision": "deny", "reason": "plant_type_not_supported", "required_action": "select_supported_plant_type"}},
	{"name": "plant_capacity_positive", "condition": {"operation": "register_plant", "capacity_positive": False}, "effect": {"decision": "deny", "reason": "capacity_must_be_positive", "required_action": "set_positive_capacity_mw"}},
	{"name": "plant_owner_required", "condition": {"operation": "register_plant", "owner_present": False}, "effect": {"decision": "deny", "reason": "plant_owner_required", "required_action": "assign_plant_owner"}},
	{"name": "plant_commissioning_date_required", "condition": {"operation": "register_plant", "commissioning_date_present": False}, "effect": {"decision": "deny", "reason": "commissioning_date_required", "required_action": "set_commissioning_date"}},
	{"name": "fuel_type_supported", "condition": {"operation": "register_plant", "fuel_type_supported": False}, "effect": {"decision": "deny", "reason": "fuel_type_not_supported", "required_action": "select_supported_fuel_type"}},
	{"name": "dispatch_mode_supported", "condition": {"operation": "create_dispatch_schedule", "dispatch_mode_supported": False}, "effect": {"decision": "deny", "reason": "dispatch_mode_not_supported", "required_action": "select_supported_dispatch_mode"}},
	{"name": "dispatch_plant_exists", "condition": {"operation": "create_dispatch_schedule", "plant_exists": False}, "effect": {"decision": "deny", "reason": "plant_not_found", "required_action": "register_plant_first"}},
	{"name": "dispatch_schedule_approval_required", "condition": {"operation": "activate_dispatch_schedule", "approval_present": False}, "effect": {"decision": "deny", "reason": "dispatch_approval_required", "required_action": "obtain_dispatch_approval"}},
	{"name": "dispatch_mw_within_capacity", "condition": {"operation": "create_dispatch_schedule", "mw_within_capacity": False}, "effect": {"decision": "deny", "reason": "dispatch_mw_exceeds_capacity", "required_action": "reduce_dispatch_mw"}},
	{"name": "outage_type_supported", "condition": {"operation": "schedule_outage", "outage_type_supported": False}, "effect": {"decision": "deny", "reason": "outage_type_not_supported", "required_action": "select_supported_outage_type"}},
	{"name": "outage_plant_exists", "condition": {"operation": "schedule_outage", "plant_exists": False}, "effect": {"decision": "deny", "reason": "plant_not_found", "required_action": "register_plant_first"}},
	{"name": "outage_notice_period", "condition": {"operation": "schedule_outage", "sufficient_notice": False}, "effect": {"decision": "deny", "reason": "insufficient_outage_notice", "required_action": "extend_outage_lead_time"}},
	{"name": "outage_approval_required", "condition": {"operation": "approve_outage", "approver_present": False}, "effect": {"decision": "deny", "reason": "outage_approver_required", "required_action": "assign_outage_approver"}},
	{"name": "outage_overlap_check", "condition": {"operation": "schedule_outage", "outage_overlap": True}, "effect": {"decision": "deny", "reason": "outage_schedule_conflict", "required_action": "resolve_outage_overlap"}},
	{"name": "kpi_period_supported", "condition": {"operation": "calculate_kpi", "period_supported": False}, "effect": {"decision": "deny", "reason": "kpi_period_not_supported", "required_action": "select_supported_period"}},
	{"name": "kpi_type_supported", "condition": {"operation": "calculate_kpi", "kpi_type_supported": False}, "effect": {"decision": "deny", "reason": "kpi_type_not_supported", "required_action": "select_supported_kpi_type"}},
	{"name": "capacity_plan_horizon_valid", "condition": {"operation": "create_capacity_plan", "horizon_valid": False}, "effect": {"decision": "deny", "reason": "planning_horizon_invalid", "required_action": "set_valid_horizon_years"}},
	{"name": "fuel_type_valid_for_plant", "condition": {"operation": "update_fuel_stock", "fuel_type_matches_plant": False}, "effect": {"decision": "deny", "reason": "fuel_type_mismatch", "required_action": "use_correct_fuel_type"}},
	{"name": "fuel_stock_non_negative", "condition": {"operation": "update_fuel_stock", "stock_non_negative": False}, "effect": {"decision": "deny", "reason": "fuel_stock_cannot_be_negative", "required_action": "set_non_negative_stock"}},
	{"name": "cross_tenant_denied", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_access_denied", "required_action": "use_correct_tenant_context"}},
	{"name": "plant_decommission_requires_approval", "condition": {"operation": "decommission_plant", "approval_present": False}, "effect": {"decision": "deny", "reason": "decommission_approval_required", "required_action": "obtain_decommission_approval"}},
	{"name": "dispatch_batch_requires_bytewax", "condition": {"operation": "batch_dispatch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_required_for_batch_dispatch", "required_action": "route_to_bytewax"}},
	{"name": "gen_agent_runtime_supported", "condition": {"operation": "register_gen_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "gen_agent_role_supported", "condition": {"operation": "register_gen_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "privileged_gen_agent_requires_human_approval", "condition": {"operation": "gen_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required_for_privileged_dispatch", "required_action": "record_human_approval"}},
	{"name": "plant_status_transition_valid", "condition": {"operation": "update_plant_status", "status_transition_valid": False}, "effect": {"decision": "deny", "reason": "invalid_plant_status_transition", "required_action": "use_valid_status_transition"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	return {
		"capability": CAPABILITY_ID,
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
			"api_prefix": "/energy-gen/api/v1",
			"requires_theme": True,
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
