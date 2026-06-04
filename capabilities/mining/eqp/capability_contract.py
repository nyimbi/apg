"""Executable capability contract for APG Equipment & Plant Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "mining_eqp"
CAPABILITY_NAME = "Equipment & Plant Management"
CAPABILITY_VERSION = "1.0.0"
EQP_EVENT_STREAM = "apg.mining.eqp.lifecycle"

SUPPORTED_EQUIPMENT_CLASSES = ["haul_truck", "excavator", "wheel_loader", "drill_rig", "dozer", "grader", "water_cart", "service_truck", "lhd_loader", "underground_truck", "conveyor", "crusher", "mill", "pump", "compressor", "generator", "crane", "forklift", "light_vehicle", "bus"]
SUPPORTED_OWNERSHIP_TYPES = ["owned", "leased", "contracted", "hire", "shared"]
SUPPORTED_MAINTENANCE_TYPES = ["preventive", "corrective", "predictive", "condition_based", "breakdown", "statutory", "rebuild"]
SUPPORTED_MAINTENANCE_STATUSES = ["scheduled", "in_progress", "awaiting_parts", "deferred", "completed", "cancelled"]
SUPPORTED_DISPATCH_STATUSES = ["available", "operating", "standby", "maintenance", "breakdown", "fuelling", "parked", "standby_ready"]
SUPPORTED_FUEL_TYPES = ["diesel", "lpg", "petrol", "electric", "hybrid", "hydrogen"]
SUPPORTED_CONDITION_RATINGS = ["excellent", "good", "fair", "poor", "critical"]
SUPPORTED_LIFECYCLE_STATUSES = ["commissioned", "active", "standby", "decommissioned", "disposed", "sold"]
SUPPORTED_INSPECTION_TYPES = ["pre_shift", "post_shift", "weekly", "monthly", "annual", "statutory", "ad_hoc"]
SUPPORTED_FAULT_SEVERITIES = ["critical", "major", "minor", "cosmetic"]
SUPPORTED_COMPONENT_TYPES = ["engine", "transmission", "hydraulics", "tyres", "brakes", "electrical", "structure", "payload_system", "safety_system", "monitoring_system"]
SUPPORTED_KPI_TYPES = ["physical_availability", "mechanical_availability", "utilisation", "mtbf", "mttr", "fuel_consumption", "tyre_hours", "payload_utilisation", "cycle_time"]
SUPPORTED_DISPATCH_ASSIGNMENT_TYPES = ["fixed", "dynamic", "zone_based", "task_based"]
SUPPORTED_REVIEW_STATUSES = ["pending", "approved", "rejected"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"fleet": {
		"supported_equipment_classes": SUPPORTED_EQUIPMENT_CLASSES,
		"supported_ownership_types": SUPPORTED_OWNERSHIP_TYPES,
		"supported_lifecycle_statuses": SUPPORTED_LIFECYCLE_STATUSES,
		"asset_number_unique": True,
	},
	"maintenance": {
		"supported_types": SUPPORTED_MAINTENANCE_TYPES,
		"supported_statuses": SUPPORTED_MAINTENANCE_STATUSES,
		"pm_schedule_required": True,
		"work_order_approval_required": True,
	},
	"dispatch": {
		"supported_statuses": SUPPORTED_DISPATCH_STATUSES,
		"pre_shift_inspection_required": True,
		"operator_license_check_required": True,
	},
	"fuel": {
		"supported_fuel_types": SUPPORTED_FUEL_TYPES,
		"fuel_docket_required": True,
		"fuel_variance_alert_threshold_pct": 10,
	},
	"kpis": {
		"supported_kpi_types": SUPPORTED_KPI_TYPES,
		"availability_target_pct": 85,
		"utilisation_target_pct": 75,
	},
	"governance": {
		"require_tenant_context": True,
		"policy_attached_for_writes": True,
		"audit_events": True,
		"breakdown_equipment_dispatch_denied": True,
		"cross_tenant_read_denied": True,
		"unlicensed_operator_dispatch_denied": True,
	},
	"observability": {"event_stream": EQP_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "workflow": "wflo", "monitoring": "moni", "scheduler": "schd", "event_stream": "bytewax"},
	"ui": {
		"enable_dashboard": True,
		"enable_fleet_register": True,
		"enable_maintenance": True,
		"enable_dispatch": True,
		"enable_fuel": True,
		"enable_kpis": True,
		"enable_inspections": True,
	},
	"theme": {"default_theme": "mining_eqp_fleet", "allow_tenant_overrides": True},
}

PROVIDES = [
	"fleet_register_management",
	"equipment_lifecycle_tracking",
	"maintenance_work_order_workflow",
	"preventive_maintenance_scheduling",
	"equipment_dispatch_management",
	"fuel_consumption_tracking",
	"equipment_kpi_reporting",
	"pre_shift_inspection_workflow",
	"fault_and_defect_management",
	"tyre_management",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "moni", "schd", "mqeb"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/mining-eqp/dashboard", "component": "EqpDashboard", "permission": "mining_eqp:view", "nav_group": "Overview"},
	{"name": "fleet", "path": "/mining-eqp/fleet", "component": "FleetRegister", "permission": "mining_eqp:view", "nav_group": "Fleet"},
	{"name": "equipment_create", "path": "/mining-eqp/fleet/create", "component": "EquipmentForm", "permission": "mining_eqp:write", "nav_group": "Fleet"},
	{"name": "equipment_detail", "path": "/mining-eqp/fleet/:id", "component": "EquipmentDetail", "permission": "mining_eqp:view", "nav_group": "Fleet"},
	{"name": "maintenance", "path": "/mining-eqp/maintenance", "component": "MaintenanceWorkOrderList", "permission": "mining_eqp:view", "nav_group": "Maintenance"},
	{"name": "maintenance_create", "path": "/mining-eqp/maintenance/create", "component": "WorkOrderForm", "permission": "mining_eqp:maintenance", "nav_group": "Maintenance"},
	{"name": "pm_schedule", "path": "/mining-eqp/maintenance/schedule", "component": "PMScheduler", "permission": "mining_eqp:maintenance", "nav_group": "Maintenance"},
	{"name": "dispatch", "path": "/mining-eqp/dispatch", "component": "DispatchBoard", "permission": "mining_eqp:dispatch", "nav_group": "Dispatch"},
	{"name": "inspections", "path": "/mining-eqp/inspections", "component": "InspectionList", "permission": "mining_eqp:view", "nav_group": "Inspections"},
	{"name": "fuel", "path": "/mining-eqp/fuel", "component": "FuelLedger", "permission": "mining_eqp:view", "nav_group": "Fuel"},
	{"name": "kpis", "path": "/mining-eqp/kpis", "component": "EquipmentKPIDashboard", "permission": "mining_eqp:reports", "nav_group": "KPIs"},
	{"name": "faults", "path": "/mining-eqp/faults", "component": "FaultRegister", "permission": "mining_eqp:view", "nav_group": "Faults"},
	{"name": "reports", "path": "/mining-eqp/reports", "component": "EqpReportList", "permission": "mining_eqp:reports", "nav_group": "Reporting"},
	{"name": "settings", "path": "/mining-eqp/settings", "component": "EqpSettings", "permission": "mining_eqp:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "mining_eqp_fleet",
	"tokens": {
		"color.primary": "#0369A1",
		"color.accent": "#EA580C",
		"color.success": "#16A34A",
		"color.warning": "#CA8A04",
		"color.danger": "#DC2626",
		"surface.canvas": "#F0F9FF",
		"surface.panel": "#FFFFFF",
		"text.primary": "#0C4A6E",
		"text.secondary": "#0369A1",
		"border.radius": "4px",
		"density": "compact",
	},
	"components": {
		"fleet": {"icon": "truck", "status_indicator": "lifecycle-status-chip"},
		"maintenance": {"icon": "wrench", "status_indicator": "maintenance-status-chip"},
		"dispatch": {"icon": "radio", "status_indicator": "dispatch-status-chip"},
		"fuel": {"icon": "fuel", "status_indicator": "fuel-type-chip"},
		"kpis": {"icon": "gauge", "status_indicator": "kpi-trend-chip"},
		"faults": {"icon": "alert-circle", "status_indicator": "fault-severity-chip"},
		"inspections": {"icon": "clipboard-check", "status_indicator": "inspection-status-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": EQP_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"equipment_commissioned",
		"equipment_decommissioned",
		"work_order_created",
		"work_order_completed",
		"equipment_breakdown_recorded",
		"equipment_dispatched",
		"fuel_docket_recorded",
		"pre_shift_inspection_submitted",
		"fault_reported",
		"fault_resolved",
		"pm_schedule_triggered",
		"kpi_threshold_breached",
	],
	"guardrails": [
		"breakdown_equipment_dispatch_denied",
		"unlicensed_operator_dispatch_denied",
		"cross_tenant_read_denied",
		"uninspected_equipment_dispatch_denied",
		"fuel_variance_alert_threshold",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "write_policy_required", "required_action": "attach_write_policy"}},
	{"name": "equipment_class_supported", "condition": {"operation": "register_equipment", "equipment_class_supported": False}, "effect": {"decision": "deny", "reason": "equipment_class_not_supported", "required_action": "select_supported_equipment_class"}},
	{"name": "asset_number_unique", "condition": {"operation": "register_equipment", "asset_number_unique": False}, "effect": {"decision": "deny", "reason": "asset_number_must_be_unique", "required_action": "provide_unique_asset_number"}},
	{"name": "ownership_type_supported", "condition": {"operation": "register_equipment", "ownership_type_supported": False}, "effect": {"decision": "deny", "reason": "ownership_type_not_supported", "required_action": "select_supported_ownership_type"}},
	{"name": "maintenance_type_supported", "condition": {"operation": "create_work_order", "maintenance_type_supported": False}, "effect": {"decision": "deny", "reason": "maintenance_type_not_supported", "required_action": "select_supported_maintenance_type"}},
	{"name": "work_order_approval_required", "condition": {"operation": "execute_work_order", "work_order_approved": False}, "effect": {"decision": "deny", "reason": "work_order_approval_required", "required_action": "obtain_work_order_approval"}},
	{"name": "breakdown_dispatch_denied", "condition": {"operation": "dispatch_equipment", "equipment_status": "breakdown"}, "effect": {"decision": "deny", "reason": "breakdown_equipment_cannot_be_dispatched", "required_action": "resolve_breakdown_first"}},
	{"name": "pre_shift_inspection_required", "condition": {"operation": "dispatch_equipment", "pre_shift_inspection_complete": False}, "effect": {"decision": "deny", "reason": "pre_shift_inspection_required", "required_action": "complete_pre_shift_inspection"}},
	{"name": "unlicensed_operator_dispatch_denied", "condition": {"operation": "dispatch_equipment", "operator_licensed": False}, "effect": {"decision": "deny", "reason": "operator_must_hold_valid_licence", "required_action": "verify_operator_licence"}},
	{"name": "fuel_type_supported", "condition": {"operation": "record_fuel", "fuel_type_supported": False}, "effect": {"decision": "deny", "reason": "fuel_type_not_supported", "required_action": "select_supported_fuel_type"}},
	{"name": "fuel_docket_required", "condition": {"operation": "record_fuel", "fuel_docket_present": False}, "effect": {"decision": "deny", "reason": "fuel_docket_required", "required_action": "attach_fuel_docket"}},
	{"name": "inspection_type_supported", "condition": {"operation": "submit_inspection", "inspection_type_supported": False}, "effect": {"decision": "deny", "reason": "inspection_type_not_supported", "required_action": "select_supported_inspection_type"}},
	{"name": "fault_severity_required", "condition": {"operation": "report_fault", "fault_severity_present": False}, "effect": {"decision": "deny", "reason": "fault_severity_required", "required_action": "specify_fault_severity"}},
	{"name": "critical_fault_work_order_required", "condition": {"operation": "record_critical_fault", "work_order_raised": False}, "effect": {"decision": "deny", "reason": "critical_fault_requires_work_order", "required_action": "raise_work_order"}},
	{"name": "decommission_requires_approval", "condition": {"operation": "decommission_equipment", "decommission_approved": False}, "effect": {"decision": "deny", "reason": "decommissioning_requires_approval", "required_action": "obtain_decommission_approval"}},
	{"name": "kpi_type_supported", "condition": {"operation": "record_kpi", "kpi_type_supported": False}, "effect": {"decision": "deny", "reason": "kpi_type_not_supported", "required_action": "select_supported_kpi_type"}},
	{"name": "cross_tenant_read_denied", "condition": {"operation": "read", "cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_access_denied", "required_action": "use_own_tenant_context"}},
	{"name": "delete_active_equipment_denied", "condition": {"operation": "delete", "equipment_lifecycle_status": "active"}, "effect": {"decision": "deny", "reason": "active_equipment_cannot_be_deleted", "required_action": "decommission_first"}},
	{"name": "pm_schedule_required_for_commissioning", "condition": {"operation": "commission_equipment", "pm_schedule_attached": False}, "effect": {"decision": "deny", "reason": "pm_schedule_required", "required_action": "attach_pm_schedule"}},
	{"name": "negative_fuel_quantity_denied", "condition": {"operation": "record_fuel", "fuel_quantity_negative": True}, "effect": {"decision": "deny", "reason": "negative_fuel_quantity_not_permitted", "required_action": "correct_fuel_quantity"}},
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
				"fleet": {"type": "object"},
				"maintenance": {"type": "object"},
				"dispatch": {"type": "object"},
				"fuel": {"type": "object"},
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
			"template_roots": ["mining/eqp/templates"],
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
