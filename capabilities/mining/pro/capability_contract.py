"""Executable capability contract for APG Mine Production Operations."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "mining_pro"
CAPABILITY_NAME = "Mine Production Operations"
CAPABILITY_VERSION = "1.0.0"
PRO_EVENT_STREAM = "apg.mining.pro.lifecycle"

SUPPORTED_SHIFT_TYPES = ["day", "night", "afternoon", "swing", "extended_day", "extended_night"]
SUPPORTED_PRODUCTION_AREAS = ["open_pit", "underground_stope", "underground_development", "stockpile", "rom_pad", "crusher", "process_plant", "waste_dump", "tailings"]
SUPPORTED_MATERIAL_TYPES = ["ore", "waste", "low_grade", "marginal", "mineralised_waste", "development_waste", "overburden", "topsoil"]
SUPPORTED_BLAST_TYPES = ["production", "development", "trim", "pre_split", "cast", "controlled_detonation"]
SUPPORTED_BLAST_STATUSES = ["planned", "designed", "drilled", "charged", "primed", "fired", "cleared", "mucked"]
SUPPORTED_ACTIVITY_TYPES = ["drilling", "blasting", "loading", "hauling", "grading", "dewatering", "support_installation", "surveying", "rehabilitation"]
SUPPORTED_ORE_TRACKING_METHODS = ["survey_volume", "truck_count", "belt_scale", "weighbridge", "density_model"]
SUPPORTED_GRADE_CONTROL_METHODS = ["face_sampling", "chip_sampling", "blast_hole_assay", "sonic_drill", "grade_scanner"]
SUPPORTED_REPORT_STATUSES = ["draft", "submitted", "approved", "rejected", "archived"]
SUPPORTED_SCHEDULE_TYPES = ["short_term_weekly", "medium_term_monthly", "long_term_annual", "life_of_mine"]
SUPPORTED_CUTOFF_PARAMETERS = ["cut_off_grade", "revenue_factor", "mining_cost", "processing_cost", "nsr"]
SUPPORTED_STOCKPILE_TYPES = ["run_of_mine", "crushed", "high_grade", "low_grade", "blended", "product"]
SUPPORTED_DELAY_CATEGORIES = ["equipment_breakdown", "weather", "blasting_exclusion", "safety_hold", "survey_hold", "regulatory_hold", "shift_change", "planned_maintenance"]
SUPPORTED_REVIEW_STATUSES = ["pending", "in_review", "approved", "rejected"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"shifts": {
		"supported_shift_types": SUPPORTED_SHIFT_TYPES,
		"supervisor_sign_off_required": True,
		"actual_vs_plan_required": True,
	},
	"production": {
		"supported_areas": SUPPORTED_PRODUCTION_AREAS,
		"supported_material_types": SUPPORTED_MATERIAL_TYPES,
		"ore_tracking_method_required": True,
	},
	"blasting": {
		"supported_blast_types": SUPPORTED_BLAST_TYPES,
		"supported_statuses": SUPPORTED_BLAST_STATUSES,
		"design_approval_required": True,
		"fire_authority_required": True,
		"post_blast_inspection_required": True,
	},
	"grade_control": {
		"supported_methods": SUPPORTED_GRADE_CONTROL_METHODS,
		"cutoff_parameter_required": True,
		"ore_waste_boundary_approval_required": True,
	},
	"scheduling": {
		"supported_schedule_types": SUPPORTED_SCHEDULE_TYPES,
		"plan_vs_actual_tracking": True,
	},
	"governance": {
		"require_tenant_context": True,
		"policy_attached_for_writes": True,
		"audit_events": True,
		"unapproved_blast_fire_denied": True,
		"cross_tenant_read_denied": True,
		"grade_boundary_bypass_denied": True,
	},
	"observability": {"event_stream": PRO_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "workflow": "wflo", "monitoring": "moni", "scheduler": "schd", "event_stream": "bytewax"},
	"ui": {
		"enable_dashboard": True,
		"enable_shift_reports": True,
		"enable_production_ledger": True,
		"enable_blast_management": True,
		"enable_grade_control": True,
		"enable_scheduling": True,
		"enable_stockpiles": True,
	},
	"theme": {"default_theme": "mining_pro_ops", "allow_tenant_overrides": True},
}

PROVIDES = [
	"shift_report_workflow",
	"production_ledger_management",
	"blast_design_workflow",
	"blast_firing_authorization",
	"ore_tracking_management",
	"grade_control_workflow",
	"production_scheduling",
	"stockpile_inventory_management",
	"delay_recording",
	"production_kpi_reporting",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "moni", "schd", "mqeb"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/mining-pro/dashboard", "component": "ProDashboard", "permission": "mining_pro:view", "nav_group": "Overview"},
	{"name": "shift_reports", "path": "/mining-pro/shifts", "component": "ShiftReportList", "permission": "mining_pro:view", "nav_group": "Shift Operations"},
	{"name": "shift_create", "path": "/mining-pro/shifts/create", "component": "ShiftReportForm", "permission": "mining_pro:write", "nav_group": "Shift Operations"},
	{"name": "shift_detail", "path": "/mining-pro/shifts/:id", "component": "ShiftReportDetail", "permission": "mining_pro:view", "nav_group": "Shift Operations"},
	{"name": "production_ledger", "path": "/mining-pro/production", "component": "ProductionLedger", "permission": "mining_pro:view", "nav_group": "Production"},
	{"name": "ore_tracking", "path": "/mining-pro/ore-tracking", "component": "OreTrackingConsole", "permission": "mining_pro:write", "nav_group": "Production"},
	{"name": "blast_management", "path": "/mining-pro/blasts", "component": "BlastManagementList", "permission": "mining_pro:view", "nav_group": "Blasting"},
	{"name": "blast_create", "path": "/mining-pro/blasts/create", "component": "BlastDesignForm", "permission": "mining_pro:blast_design", "nav_group": "Blasting"},
	{"name": "blast_detail", "path": "/mining-pro/blasts/:id", "component": "BlastDetail", "permission": "mining_pro:view", "nav_group": "Blasting"},
	{"name": "grade_control", "path": "/mining-pro/grade-control", "component": "GradeControlConsole", "permission": "mining_pro:grade_control", "nav_group": "Grade Control"},
	{"name": "stockpiles", "path": "/mining-pro/stockpiles", "component": "StockpileInventory", "permission": "mining_pro:view", "nav_group": "Stockpiles"},
	{"name": "schedule", "path": "/mining-pro/schedule", "component": "ProductionScheduler", "permission": "mining_pro:schedule", "nav_group": "Planning"},
	{"name": "reports", "path": "/mining-pro/reports", "component": "ProductionReportList", "permission": "mining_pro:reports", "nav_group": "Reporting"},
	{"name": "settings", "path": "/mining-pro/settings", "component": "ProSettings", "permission": "mining_pro:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "mining_pro_ops",
	"tokens": {
		"color.primary": "#1D4ED8",
		"color.accent": "#D97706",
		"color.success": "#15803D",
		"color.warning": "#B45309",
		"color.danger": "#DC2626",
		"surface.canvas": "#F1F5F9",
		"surface.panel": "#FFFFFF",
		"text.primary": "#0F172A",
		"text.secondary": "#475569",
		"border.radius": "4px",
		"density": "compact",
	},
	"components": {
		"shift_reports": {"icon": "clock", "status_indicator": "shift-status-chip"},
		"production": {"icon": "trending-up", "status_indicator": "material-type-chip"},
		"blasts": {"icon": "zap", "status_indicator": "blast-status-chip"},
		"grade_control": {"icon": "bar-chart-2", "status_indicator": "grade-flag-chip"},
		"stockpiles": {"icon": "package", "status_indicator": "stockpile-type-chip"},
		"schedule": {"icon": "calendar", "status_indicator": "schedule-type-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": PRO_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"shift_report_submitted",
		"shift_report_approved",
		"production_tonnes_recorded",
		"ore_movement_recorded",
		"blast_designed",
		"blast_fired",
		"blast_cleared",
		"grade_boundary_updated",
		"stockpile_movement_recorded",
		"production_schedule_published",
		"delay_recorded",
	],
	"guardrails": [
		"unapproved_blast_fire_denied",
		"grade_boundary_bypass_denied",
		"cross_tenant_read_denied",
		"unsigned_shift_report_denied",
		"ore_movement_without_tracking_method_denied",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "write_policy_required", "required_action": "attach_write_policy"}},
	{"name": "shift_type_supported", "condition": {"operation": "create_shift_report", "shift_type_supported": False}, "effect": {"decision": "deny", "reason": "shift_type_not_supported", "required_action": "select_supported_shift_type"}},
	{"name": "shift_supervisor_required", "condition": {"operation": "submit_shift_report", "supervisor_present": False}, "effect": {"decision": "deny", "reason": "supervisor_required", "required_action": "assign_supervisor"}},
	{"name": "shift_dates_required", "condition": {"operation": "create_shift_report", "shift_dates_present": False}, "effect": {"decision": "deny", "reason": "shift_dates_required", "required_action": "provide_shift_dates"}},
	{"name": "production_area_supported", "condition": {"operation": "record_production", "production_area_supported": False}, "effect": {"decision": "deny", "reason": "production_area_not_supported", "required_action": "select_supported_area"}},
	{"name": "material_type_supported", "condition": {"operation": "record_production", "material_type_supported": False}, "effect": {"decision": "deny", "reason": "material_type_not_supported", "required_action": "select_supported_material_type"}},
	{"name": "ore_tracking_method_required", "condition": {"operation": "record_ore_movement", "tracking_method_present": False}, "effect": {"decision": "deny", "reason": "ore_tracking_method_required", "required_action": "specify_tracking_method"}},
	{"name": "blast_type_supported", "condition": {"operation": "create_blast", "blast_type_supported": False}, "effect": {"decision": "deny", "reason": "blast_type_not_supported", "required_action": "select_supported_blast_type"}},
	{"name": "blast_design_approval_required", "condition": {"operation": "charge_blast", "blast_design_approved": False}, "effect": {"decision": "deny", "reason": "blast_design_must_be_approved", "required_action": "obtain_blast_design_approval"}},
	{"name": "blast_fire_authority_required", "condition": {"operation": "fire_blast", "fire_authority_present": False}, "effect": {"decision": "deny", "reason": "fire_authority_required", "required_action": "obtain_fire_authority"}},
	{"name": "post_blast_inspection_required", "condition": {"operation": "clear_blast", "post_blast_inspection_done": False}, "effect": {"decision": "deny", "reason": "post_blast_inspection_required", "required_action": "complete_post_blast_inspection"}},
	{"name": "grade_control_method_supported", "condition": {"operation": "update_grade_boundary", "grade_control_method_supported": False}, "effect": {"decision": "deny", "reason": "grade_control_method_not_supported", "required_action": "select_supported_grade_control_method"}},
	{"name": "ore_waste_boundary_approval_required", "condition": {"operation": "update_grade_boundary", "boundary_approved": False}, "effect": {"decision": "deny", "reason": "ore_waste_boundary_approval_required", "required_action": "obtain_boundary_approval"}},
	{"name": "grade_boundary_bypass_denied", "condition": {"operation": "bypass_grade_boundary", "has_override_authority": False}, "effect": {"decision": "deny", "reason": "grade_boundary_bypass_not_permitted", "required_action": "obtain_override_authority"}},
	{"name": "stockpile_type_supported", "condition": {"operation": "create_stockpile", "stockpile_type_supported": False}, "effect": {"decision": "deny", "reason": "stockpile_type_not_supported", "required_action": "select_supported_stockpile_type"}},
	{"name": "schedule_type_supported", "condition": {"operation": "publish_schedule", "schedule_type_supported": False}, "effect": {"decision": "deny", "reason": "schedule_type_not_supported", "required_action": "select_supported_schedule_type"}},
	{"name": "schedule_approval_required", "condition": {"operation": "publish_schedule", "schedule_approved": False}, "effect": {"decision": "deny", "reason": "schedule_approval_required", "required_action": "obtain_schedule_approval"}},
	{"name": "delay_category_required", "condition": {"operation": "record_delay", "delay_category_present": False}, "effect": {"decision": "deny", "reason": "delay_category_required", "required_action": "select_delay_category"}},
	{"name": "cross_tenant_read_denied", "condition": {"operation": "read", "cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_access_denied", "required_action": "use_own_tenant_context"}},
	{"name": "negative_tonnes_denied", "condition": {"operation": "record_production", "tonnes_negative": True}, "effect": {"decision": "deny", "reason": "negative_tonnes_not_permitted", "required_action": "correct_tonnes_value"}},
	{"name": "future_shift_report_denied", "condition": {"operation": "create_shift_report", "shift_in_future": True}, "effect": {"decision": "deny", "reason": "future_shift_reports_not_permitted", "required_action": "use_current_or_past_shift_date"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	"""Return the full capability contract for the given tenant."""
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
				"shifts": {"type": "object"},
				"blasting": {"type": "object"},
				"grade_control": {"type": "object"},
			},
		},
		"rule_engine": {
			"type": "deterministic",
			"default_decision": "allow",
			"rules": RULES,
		},
		"ui": {
			"shell": "apg_python",
			"requires_theme": True,
			"template_roots": ["mining/pro/templates"],
			"routes": UI_ROUTES,
		},
		"theme": THEME,
		"provides": PROVIDES,
		"requires": REQUIRES,
		"streaming": STREAMING,
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate deterministic rules against the given context dict."""
	matched_denials: list[dict[str, Any]] = []
	matched_allows: list[dict[str, Any]] = []

	for rule in RULES:
		condition = rule["condition"]
		all_match = all(context.get(k) == v for k, v in condition.items())
		if all_match:
			effect = rule["effect"]
			entry = {"rule": rule["name"], "effect": effect}
			if effect["decision"] == "deny":
				matched_denials.append(entry)
			else:
				matched_allows.append(entry)

	if matched_denials:
		return {
			"decision": "deny",
			"matched_denials": matched_denials,
			"matched_allows": matched_allows,
			"required_actions": [d["effect"]["required_action"] for d in matched_denials],
		}

	return {
		"decision": "allow",
		"matched_denials": [],
		"matched_allows": matched_allows,
		"required_actions": [],
	}
