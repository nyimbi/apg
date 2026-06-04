"""Executable capability contract for APG Space Planning & Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "realestate_spa"
CAPABILITY_NAME = "Space Planning & Management"
CAPABILITY_VERSION = "1.0.0"
SPA_EVENT_STREAM = "apg.realestate.spa.lifecycle"

SUPPORTED_SPACE_TYPES = ["private_office", "open_plan", "meeting_room", "conference_room", "hot_desk", "collaboration_zone", "quiet_zone", "reception", "amenity", "storage", "server_room", "common_area", "balcony", "terrace"]
SUPPORTED_SPACE_STATUSES = ["available", "occupied", "reserved", "under_fit_out", "decommissioned", "mothballed"]
SUPPORTED_ALLOCATION_TYPES = ["permanent", "hot_desk", "shared", "dedicated", "project_space", "visitor"]
SUPPORTED_MOVE_TYPES = ["internal_move", "inter_floor_move", "inter_building_move", "consolidation", "expansion", "decommission"]
SUPPORTED_MOVE_STATUSES = ["planning", "approved", "scheduled", "in_progress", "completed", "cancelled"]
SUPPORTED_OCCUPANCY_METRICS = ["headcount", "desk_ratio", "sqm_per_person", "utilisation_rate", "peak_occupancy", "average_daily_occupancy", "density_ratio"]
SUPPORTED_FLOOR_PLAN_FORMATS = ["dwg", "dxf", "pdf", "svg", "ifc", "revit", "png"]
SUPPORTED_WORKPLACE_STRATEGIES = ["assigned_seating", "activity_based_working", "hot_desking", "hybrid_booking", "neighbourhood_model", "agile_working"]
SUPPORTED_SENSOR_TYPES = ["occupancy_sensor", "badge_reader", "wifi_probe", "camera_ai", "desk_sensor", "meeting_room_sensor"]
SUPPORTED_DEPARTMENT_TYPES = ["executive", "operations", "technology", "finance", "hr", "legal", "sales", "marketing", "facilities", "shared_services"]
SUPPORTED_BOOKING_TYPES = ["desk", "meeting_room", "parking", "locker", "visitor_pass"]
SUPPORTED_DENSITY_BANDS = ["dense", "standard", "spacious", "executive", "social_distancing"]
SUPPORTED_CHURN_REASONS = ["headcount_growth", "headcount_reduction", "reorganisation", "refurbishment", "lease_change", "cost_optimisation"]
SUPPORTED_AREA_UNITS = ["sqm", "sqft"]
SUPPORTED_APPROVAL_LEVELS = ["department_head", "facilities_manager", "coo", "board"]

PROVIDES = [
	"floor_plan_management",
	"space_allocation_engine",
	"move_management_workflow",
	"occupancy_analytics",
	"workplace_density_planning",
	"space_booking_engine",
	"sensor_integration_bridge",
	"department_space_reporting",
	"space_optimisation_advisor",
	"chargeback_space_accounting",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "moni", "mqeb", "schd"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/realestate/spa/dashboard", "component": "SpaDashboard", "permission": "realestate_spa:view", "nav_group": "Overview"},
	{"name": "floor-plans", "path": "/realestate/spa/floor-plans", "component": "FloorPlanViewer", "permission": "realestate_spa:floor_plans", "nav_group": "Floor Plans"},
	{"name": "spaces", "path": "/realestate/spa/spaces", "component": "SpaceRegistry", "permission": "realestate_spa:spaces", "nav_group": "Spaces"},
	{"name": "allocations", "path": "/realestate/spa/allocations", "component": "SpaceAllocationConsole", "permission": "realestate_spa:allocations", "nav_group": "Allocation"},
	{"name": "moves", "path": "/realestate/spa/moves", "component": "MoveManagementConsole", "permission": "realestate_spa:moves", "nav_group": "Moves"},
	{"name": "bookings", "path": "/realestate/spa/bookings", "component": "SpaceBookingEngine", "permission": "realestate_spa:bookings", "nav_group": "Bookings"},
	{"name": "occupancy", "path": "/realestate/spa/occupancy", "component": "OccupancyAnalyticsDashboard", "permission": "realestate_spa:occupancy", "nav_group": "Analytics"},
	{"name": "density", "path": "/realestate/spa/density", "component": "DensityPlanningConsole", "permission": "realestate_spa:density", "nav_group": "Planning"},
	{"name": "sensors", "path": "/realestate/spa/sensors", "component": "SensorIntegrationConsole", "permission": "realestate_spa:admin", "nav_group": "Integration"},
	{"name": "departments", "path": "/realestate/spa/departments", "component": "DepartmentSpaceView", "permission": "realestate_spa:departments", "nav_group": "Departments"},
	{"name": "chargeback", "path": "/realestate/spa/chargeback", "component": "SpaceChargebackConsole", "permission": "realestate_spa:chargeback", "nav_group": "Financial"},
	{"name": "reports", "path": "/realestate/spa/reports", "component": "SpaceReportBuilder", "permission": "realestate_spa:reports", "nav_group": "Reporting"},
	{"name": "settings", "path": "/realestate/spa/settings", "component": "SpaSettings", "permission": "realestate_spa:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "realestate_spa_workplace",
	"tokens": {
		"color.primary": "#2563EB",
		"color.accent": "#7C3AED",
		"color.success": "#059669",
		"color.warning": "#D97706",
		"color.danger": "#DC2626",
		"surface.canvas": "#EFF6FF",
		"surface.panel": "#FFFFFF",
		"text.primary": "#1E3A8A",
		"text.secondary": "#64748B",
		"border.radius": "10px",
		"density": "comfortable",
	},
	"components": {
		"floor_plans": {"icon": "layout", "status_indicator": "floor-plan-format-chip"},
		"spaces": {"icon": "square", "status_indicator": "space-status-chip"},
		"allocations": {"icon": "user-check", "status_indicator": "allocation-type-chip"},
		"moves": {"icon": "move", "status_indicator": "move-status-chip"},
		"bookings": {"icon": "calendar", "status_indicator": "booking-type-chip"},
		"occupancy": {"icon": "users", "status_indicator": "occupancy-metric-chip"},
		"density": {"icon": "grid", "status_indicator": "density-band-chip"},
		"sensors": {"icon": "radio", "status_indicator": "sensor-type-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": SPA_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"space_registered", "space_status_changed", "space_allocated", "space_deallocated",
		"move_created", "move_completed", "move_cancelled",
		"booking_created", "booking_cancelled",
		"occupancy_data_ingested", "density_threshold_breached",
		"floor_plan_updated", "sensor_reading_received",
		"space_chargeback_calculated", "optimisation_recommendation_generated",
	],
	"guardrails": [
		"space_double_booking_denied",
		"move_requires_approval_above_headcount_threshold",
		"sensor_data_anonymisation_required",
		"density_below_minimum_triggers_alert",
		"chargeback_requires_verified_occupancy_data",
	],
}

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"spaces": {"supported_types": SUPPORTED_SPACE_TYPES, "supported_statuses": SUPPORTED_SPACE_STATUSES, "supported_area_units": SUPPORTED_AREA_UNITS},
	"allocations": {"supported_types": SUPPORTED_ALLOCATION_TYPES},
	"moves": {"supported_types": SUPPORTED_MOVE_TYPES, "supported_statuses": SUPPORTED_MOVE_STATUSES, "large_move_headcount_threshold": 20},
	"bookings": {"supported_types": SUPPORTED_BOOKING_TYPES, "max_advance_booking_days": 90},
	"occupancy": {"supported_metrics": SUPPORTED_OCCUPANCY_METRICS, "sensor_types": SUPPORTED_SENSOR_TYPES, "anonymise_sensor_data": True},
	"density": {"supported_bands": SUPPORTED_DENSITY_BANDS, "workplace_strategies": SUPPORTED_WORKPLACE_STRATEGIES},
	"floor_plans": {"supported_formats": SUPPORTED_FLOOR_PLAN_FORMATS},
	"departments": {"supported_types": SUPPORTED_DEPARTMENT_TYPES},
	"churn": {"supported_reasons": SUPPORTED_CHURN_REASONS},
	"approvals": {"supported_levels": SUPPORTED_APPROVAL_LEVELS},
	"ui": {"enable_dashboard": True, "enable_floor_plans": True, "enable_occupancy": True, "enable_bookings": True},
	"theme": {"default_theme": "realestate_spa_workplace", "allow_tenant_overrides": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "sensor_data_anonymisation": True},
	"observability": {"event_stream": SPA_EVENT_STREAM, "stream_processor": "bytewax"},
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "space_policy_required", "required_action": "attach_space_policy"}},
	{"name": "space_type_supported", "condition": {"operation": "create_space", "space_type_supported": False}, "effect": {"decision": "deny", "reason": "space_type_not_supported", "required_action": "select_supported_space_type"}},
	{"name": "space_requires_floor_plan", "condition": {"operation": "create_space", "floor_plan_linked": False}, "effect": {"decision": "deny", "reason": "floor_plan_required_for_space", "required_action": "link_floor_plan"}},
	{"name": "space_double_booking_denied", "condition": {"operation": "book_space", "space_already_booked": True}, "effect": {"decision": "deny", "reason": "space_double_booking_not_allowed", "required_action": "select_available_space"}},
	{"name": "decommissioned_space_booking_denied", "condition": {"operation": "book_space", "space_status": "decommissioned"}, "effect": {"decision": "deny", "reason": "decommissioned_space_cannot_be_booked", "required_action": "select_available_space"}},
	{"name": "allocation_type_supported", "condition": {"operation": "allocate_space", "allocation_type_supported": False}, "effect": {"decision": "deny", "reason": "allocation_type_not_supported", "required_action": "select_supported_allocation_type"}},
	{"name": "move_type_supported", "condition": {"operation": "create_move", "move_type_supported": False}, "effect": {"decision": "deny", "reason": "move_type_not_supported", "required_action": "select_supported_move_type"}},
	{"name": "large_move_requires_approval", "condition": {"operation": "create_move", "headcount_above_threshold": True, "approved": False}, "effect": {"decision": "deny", "reason": "large_move_requires_management_approval", "required_action": "submit_move_for_approval"}},
	{"name": "booking_type_supported", "condition": {"operation": "create_booking", "booking_type_supported": False}, "effect": {"decision": "deny", "reason": "booking_type_not_supported", "required_action": "select_supported_booking_type"}},
	{"name": "booking_advance_limit_enforced", "condition": {"operation": "create_booking", "booking_too_far_in_advance": True}, "effect": {"decision": "deny", "reason": "booking_exceeds_maximum_advance_booking_period", "required_action": "book_within_advance_limit"}},
	{"name": "sensor_data_must_be_anonymised", "condition": {"operation": "ingest_sensor_data", "data_anonymised": False}, "effect": {"decision": "deny", "reason": "sensor_data_must_be_anonymised_before_ingestion", "required_action": "anonymise_sensor_data"}},
	{"name": "chargeback_requires_verified_data", "condition": {"operation": "calculate_chargeback", "occupancy_data_verified": False}, "effect": {"decision": "deny", "reason": "chargeback_requires_verified_occupancy_data", "required_action": "verify_occupancy_data"}},
	{"name": "floor_plan_format_supported", "condition": {"operation": "upload_floor_plan", "format_supported": False}, "effect": {"decision": "deny", "reason": "floor_plan_format_not_supported", "required_action": "use_supported_floor_plan_format"}},
	{"name": "workplace_strategy_supported", "condition": {"operation": "set_workplace_strategy", "strategy_supported": False}, "effect": {"decision": "deny", "reason": "workplace_strategy_not_supported", "required_action": "select_supported_workplace_strategy"}},
	{"name": "density_band_supported", "condition": {"operation": "set_density_target", "density_band_supported": False}, "effect": {"decision": "deny", "reason": "density_band_not_supported", "required_action": "select_supported_density_band"}},
	{"name": "department_type_supported", "condition": {"operation": "assign_department", "department_type_supported": False}, "effect": {"decision": "deny", "reason": "department_type_not_supported", "required_action": "select_supported_department_type"}},
	{"name": "occupancy_metric_supported", "condition": {"operation": "calculate_occupancy_metric", "metric_supported": False}, "effect": {"decision": "deny", "reason": "occupancy_metric_not_supported", "required_action": "select_supported_occupancy_metric"}},
	{"name": "cross_tenant_space_denied", "condition": {"operation_type": "write", "cross_tenant": True}, "effect": {"decision": "deny", "reason": "cross_tenant_space_management_not_allowed", "required_action": "use_correct_tenant_context"}},
	{"name": "optimisation_requires_occupancy_history", "condition": {"operation": "generate_optimisation", "occupancy_history_weeks": 4, "sufficient_history": False}, "effect": {"decision": "deny", "reason": "insufficient_occupancy_history_for_optimisation", "required_action": "collect_minimum_occupancy_data"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	"""Return the full capability contract for the given tenant."""
	cfg = deepcopy(DEFAULT_CONFIGURATION)
	cfg["tenant_id"] = tenant_id
	return {
		"capability": CAPABILITY_ID,
		"display_name": CAPABILITY_NAME,
		"version": CAPABILITY_VERSION,
		"configuration": cfg,
		"configuration_schema": {
			"required": ["tenant_id", "ui", "theme"],
			"properties": {"tenant_id": {"type": "string"}, "ui": {"type": "object"}, "theme": {"type": "object"}},
		},
		"rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": RULES},
		"ui": {"shell": "apg_python", "requires_theme": True, "template_roots": ["realestate/spa/templates"], "routes": UI_ROUTES},
		"theme": THEME,
		"streaming": STREAMING,
		"provides": PROVIDES,
		"requires": REQUIRES,
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate all rules against context. Returns first denial or allow."""
	for rule in RULES:
		cond = rule["condition"]
		if all(context.get(k) == v for k, v in cond.items()):
			effect = rule["effect"]
			if effect["decision"] == "deny":
				return {"decision": "deny", "rule": rule["name"], "reason": effect["reason"], "required_action": effect.get("required_action")}
	return {"decision": "allow", "rule": None, "reason": None}
