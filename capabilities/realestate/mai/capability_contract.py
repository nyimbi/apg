"""Executable capability contract for APG Facilities Maintenance."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "realestate_mai"
CAPABILITY_NAME = "Facilities Maintenance"
CAPABILITY_VERSION = "1.0.0"
MAI_EVENT_STREAM = "apg.realestate.mai.lifecycle"

SUPPORTED_WORK_ORDER_TYPES = ["preventive", "corrective", "emergency", "predictive", "statutory", "improvement", "inspection", "condition_survey"]
SUPPORTED_WORK_ORDER_STATUSES = ["raised", "assigned", "in_progress", "on_hold", "pending_parts", "completed", "verified", "closed", "cancelled"]
SUPPORTED_ASSET_CATEGORIES = ["hvac", "electrical", "plumbing", "structural", "fire_safety", "lifts_escalators", "access_control", "it_infrastructure", "landscaping", "cleaning", "security"]
SUPPORTED_MAINTENANCE_FREQUENCIES = ["daily", "weekly", "fortnightly", "monthly", "quarterly", "semi_annual", "annual", "biennial", "as_needed"]
SUPPORTED_PRIORITY_LEVELS = ["p1_critical", "p2_high", "p3_medium", "p4_low", "p5_planned"]
SUPPORTED_CONTRACTOR_TYPES = ["specialist_mechanical", "specialist_electrical", "general_maintenance", "cleaning", "landscaping", "security", "it_support", "pest_control"]
SUPPORTED_SLA_TYPES = ["response_time", "resolution_time", "first_time_fix", "availability", "uptime"]
SUPPORTED_INSPECTION_TYPES = ["statutory", "condition", "pre_purchase", "handover", "periodic", "post_repair", "compliance"]
SUPPORTED_ASSET_STATUSES = ["active", "under_maintenance", "decommissioned", "condemned", "awaiting_replacement", "warranty"]
SUPPORTED_DEFECT_SEVERITIES = ["critical", "major", "minor", "cosmetic"]
SUPPORTED_CAFM_INTEGRATIONS = ["archibus", "maximo", "servicemax", "planon", "famis", "corrigo", "building_engines", "custom_api"]
SUPPORTED_PPM_STATUSES = ["scheduled", "in_progress", "completed", "overdue", "deferred", "cancelled"]
SUPPORTED_LIFECYCLE_PHASES = ["new", "operational", "ageing", "end_of_life", "replacement_due", "decommissioned"]
SUPPORTED_COST_TYPES = ["labour", "materials", "subcontract", "equipment_hire", "travel", "overhead"]
SUPPORTED_COMPLIANCE_STANDARDS = ["OSHA", "ISO_55000", "CIBSE", "BS_EN_13306", "local_fire_code", "electrical_safety_regs"]

PROVIDES = [
	"preventive_maintenance_scheduling",
	"work_order_management",
	"contractor_management",
	"asset_lifecycle_tracking",
	"cafm_integration_bridge",
	"sla_monitoring",
	"inspection_management",
	"defect_tracking",
	"maintenance_cost_management",
	"compliance_maintenance_reporting",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "schd", "comp", "mqeb", "moni"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/realestate/mai/dashboard", "component": "MaiDashboard", "permission": "realestate_mai:view", "nav_group": "Overview"},
	{"name": "work-orders", "path": "/realestate/mai/work-orders", "component": "WorkOrderQueue", "permission": "realestate_mai:work_orders", "nav_group": "Operations"},
	{"name": "ppm-schedules", "path": "/realestate/mai/ppm", "component": "PpmScheduleConsole", "permission": "realestate_mai:ppm", "nav_group": "Planning"},
	{"name": "assets", "path": "/realestate/mai/assets", "component": "AssetRegister", "permission": "realestate_mai:assets", "nav_group": "Assets"},
	{"name": "asset-detail", "path": "/realestate/mai/assets/<id>", "component": "AssetDetail", "permission": "realestate_mai:assets", "nav_group": "Assets"},
	{"name": "contractors", "path": "/realestate/mai/contractors", "component": "MaintenanceContractorRegistry", "permission": "realestate_mai:contractors", "nav_group": "Contractors"},
	{"name": "inspections", "path": "/realestate/mai/inspections", "component": "InspectionConsole", "permission": "realestate_mai:inspections", "nav_group": "Quality"},
	{"name": "defects", "path": "/realestate/mai/defects", "component": "DefectTracker", "permission": "realestate_mai:defects", "nav_group": "Quality"},
	{"name": "sla-monitor", "path": "/realestate/mai/sla", "component": "SlaMonitorDashboard", "permission": "realestate_mai:sla", "nav_group": "Performance"},
	{"name": "costs", "path": "/realestate/mai/costs", "component": "MaintenanceCostConsole", "permission": "realestate_mai:costs", "nav_group": "Financial"},
	{"name": "cafm-integration", "path": "/realestate/mai/cafm", "component": "CafmIntegrationConsole", "permission": "realestate_mai:admin", "nav_group": "Integration"},
	{"name": "compliance", "path": "/realestate/mai/compliance", "component": "MaintenanceComplianceConsole", "permission": "realestate_mai:compliance", "nav_group": "Compliance"},
	{"name": "reports", "path": "/realestate/mai/reports", "component": "MaintenanceReportBuilder", "permission": "realestate_mai:reports", "nav_group": "Reporting"},
	{"name": "settings", "path": "/realestate/mai/settings", "component": "MaiSettings", "permission": "realestate_mai:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "realestate_mai_operations",
	"tokens": {
		"color.primary": "#064E3B",
		"color.accent": "#0369A1",
		"color.success": "#14532D",
		"color.warning": "#92400E",
		"color.danger": "#991B1B",
		"surface.canvas": "#ECFDF5",
		"surface.panel": "#FFFFFF",
		"text.primary": "#064E3B",
		"text.secondary": "#374151",
		"border.radius": "6px",
		"density": "compact",
	},
	"components": {
		"work_orders": {"icon": "wrench", "status_indicator": "work-order-status-chip"},
		"ppm_schedules": {"icon": "calendar", "status_indicator": "ppm-status-chip"},
		"assets": {"icon": "cpu", "status_indicator": "asset-status-chip"},
		"contractors": {"icon": "hard-hat", "status_indicator": "contractor-type-chip"},
		"inspections": {"icon": "clipboard-list", "status_indicator": "inspection-type-chip"},
		"defects": {"icon": "alert-triangle", "status_indicator": "defect-severity-chip"},
		"sla": {"icon": "activity", "status_indicator": "sla-breach-chip"},
		"compliance": {"icon": "shield-check", "status_indicator": "compliance-standard-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": MAI_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"work_order_raised", "work_order_assigned", "work_order_completed", "work_order_overdue",
		"ppm_schedule_generated", "ppm_completed", "ppm_overdue",
		"asset_registered", "asset_status_changed", "asset_end_of_life_alert",
		"inspection_completed", "defect_raised", "defect_resolved",
		"sla_breach_detected", "sla_warning_triggered",
		"contractor_registered", "maintenance_cost_posted",
	],
	"guardrails": [
		"p1_work_order_requires_immediate_response",
		"statutory_inspection_overdue_triggers_alert",
		"decommissioned_asset_work_order_denied",
		"contractor_without_insurance_assignment_denied",
		"sla_breach_triggers_escalation",
	],
}

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"work_orders": {"supported_types": SUPPORTED_WORK_ORDER_TYPES, "supported_statuses": SUPPORTED_WORK_ORDER_STATUSES, "supported_priorities": SUPPORTED_PRIORITY_LEVELS},
	"ppm": {"supported_frequencies": SUPPORTED_MAINTENANCE_FREQUENCIES, "supported_statuses": SUPPORTED_PPM_STATUSES, "auto_generate_advance_days": 30},
	"assets": {"supported_categories": SUPPORTED_ASSET_CATEGORIES, "supported_statuses": SUPPORTED_ASSET_STATUSES, "supported_lifecycle_phases": SUPPORTED_LIFECYCLE_PHASES},
	"contractors": {"supported_types": SUPPORTED_CONTRACTOR_TYPES, "insurance_required": True},
	"inspections": {"supported_types": SUPPORTED_INSPECTION_TYPES},
	"defects": {"supported_severities": SUPPORTED_DEFECT_SEVERITIES},
	"sla": {"supported_types": SUPPORTED_SLA_TYPES, "breach_alert_threshold_pct": 80},
	"costs": {"supported_types": SUPPORTED_COST_TYPES},
	"cafm": {"supported_integrations": SUPPORTED_CAFM_INTEGRATIONS, "sync_enabled": False},
	"compliance": {"supported_standards": SUPPORTED_COMPLIANCE_STANDARDS},
	"ui": {"enable_dashboard": True, "enable_work_orders": True, "enable_ppm": True, "enable_sla": True},
	"theme": {"default_theme": "realestate_mai_operations", "allow_tenant_overrides": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True},
	"observability": {"event_stream": MAI_EVENT_STREAM, "stream_processor": "bytewax"},
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "maintenance_policy_required", "required_action": "attach_maintenance_policy"}},
	{"name": "work_order_type_supported", "condition": {"operation": "raise_work_order", "work_order_type_supported": False}, "effect": {"decision": "deny", "reason": "work_order_type_not_supported", "required_action": "select_supported_work_order_type"}},
	{"name": "work_order_requires_asset", "condition": {"operation": "raise_work_order", "asset_present": False}, "effect": {"decision": "deny", "reason": "asset_required_for_work_order", "required_action": "link_asset"}},
	{"name": "decommissioned_asset_work_order_denied", "condition": {"operation": "raise_work_order", "asset_status": "decommissioned"}, "effect": {"decision": "deny", "reason": "cannot_raise_work_order_for_decommissioned_asset", "required_action": "check_asset_status"}},
	{"name": "p1_work_order_requires_immediate_assignment", "condition": {"operation": "raise_work_order", "priority": "p1_critical", "contractor_assigned": False}, "effect": {"decision": "deny", "reason": "p1_critical_work_order_requires_immediate_contractor_assignment", "required_action": "assign_contractor_immediately"}},
	{"name": "work_order_priority_supported", "condition": {"operation": "raise_work_order", "priority_supported": False}, "effect": {"decision": "deny", "reason": "priority_level_not_supported", "required_action": "select_supported_priority"}},
	{"name": "ppm_frequency_supported", "condition": {"operation": "create_ppm_schedule", "frequency_supported": False}, "effect": {"decision": "deny", "reason": "maintenance_frequency_not_supported", "required_action": "select_supported_frequency"}},
	{"name": "ppm_requires_asset", "condition": {"operation": "create_ppm_schedule", "asset_present": False}, "effect": {"decision": "deny", "reason": "asset_required_for_ppm_schedule", "required_action": "link_asset_to_ppm"}},
	{"name": "contractor_without_insurance_denied", "condition": {"operation": "assign_contractor", "contractor_has_valid_insurance": False}, "effect": {"decision": "deny", "reason": "contractor_must_have_valid_insurance", "required_action": "verify_contractor_insurance"}},
	{"name": "contractor_type_supported", "condition": {"operation": "register_contractor", "contractor_type_supported": False}, "effect": {"decision": "deny", "reason": "contractor_type_not_supported", "required_action": "select_supported_contractor_type"}},
	{"name": "inspection_type_supported", "condition": {"operation": "create_inspection", "inspection_type_supported": False}, "effect": {"decision": "deny", "reason": "inspection_type_not_supported", "required_action": "select_supported_inspection_type"}},
	{"name": "statutory_inspection_overdue_triggers_alert", "condition": {"operation": "check_inspection_status", "inspection_type": "statutory", "overdue": True, "alert_sent": False}, "effect": {"decision": "deny", "reason": "statutory_inspection_overdue_alert_mandatory", "required_action": "send_statutory_inspection_alert"}},
	{"name": "defect_severity_supported", "condition": {"operation": "raise_defect", "severity_supported": False}, "effect": {"decision": "deny", "reason": "defect_severity_not_supported", "required_action": "select_supported_severity"}},
	{"name": "sla_type_supported", "condition": {"operation": "create_sla", "sla_type_supported": False}, "effect": {"decision": "deny", "reason": "sla_type_not_supported", "required_action": "select_supported_sla_type"}},
	{"name": "sla_breach_requires_escalation", "condition": {"operation": "update_work_order", "sla_breached": True, "escalated": False}, "effect": {"decision": "deny", "reason": "sla_breach_requires_immediate_escalation", "required_action": "escalate_sla_breach"}},
	{"name": "asset_category_supported", "condition": {"operation": "register_asset", "asset_category_supported": False}, "effect": {"decision": "deny", "reason": "asset_category_not_supported", "required_action": "select_supported_asset_category"}},
	{"name": "cost_type_supported", "condition": {"operation": "post_maintenance_cost", "cost_type_supported": False}, "effect": {"decision": "deny", "reason": "cost_type_not_supported", "required_action": "select_supported_cost_type"}},
	{"name": "work_order_completion_requires_verification", "condition": {"operation": "close_work_order", "verification_complete": False}, "effect": {"decision": "deny", "reason": "work_order_must_be_verified_before_closing", "required_action": "complete_verification"}},
	{"name": "cafm_integration_requires_configuration", "condition": {"operation": "sync_cafm", "cafm_configured": False}, "effect": {"decision": "deny", "reason": "cafm_integration_not_configured", "required_action": "configure_cafm_integration"}},
	{"name": "cross_tenant_maintenance_denied", "condition": {"operation_type": "write", "cross_tenant": True}, "effect": {"decision": "deny", "reason": "cross_tenant_maintenance_not_allowed", "required_action": "use_correct_tenant_context"}},
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
		"ui": {"shell": "apg_python", "requires_theme": True, "template_roots": ["realestate/mai/templates"], "routes": UI_ROUTES},
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
