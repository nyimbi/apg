"""Executable capability contract for APG Vehicle Maintenance."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "transport_mai"
CAPABILITY_NAME = "Vehicle Maintenance"
CAPABILITY_VERSION = "1.0.0"
MAINTENANCE_EVENT_STREAM = "apg.transport.maintenance.lifecycle"

SUPPORTED_MAINTENANCE_TYPES = ["preventive", "corrective", "predictive", "condition_based", "breakdown", "inspection", "mot", "service", "warranty_claim", "recall"]
SUPPORTED_JOB_STATUSES = ["scheduled", "in_progress", "awaiting_parts", "completed", "cancelled", "deferred", "escalated"]
SUPPORTED_PRIORITY_LEVELS = ["critical", "high", "medium", "low", "planned"]
SUPPORTED_WORKSHOP_TYPES = ["in_house", "authorised_dealer", "independent_garage", "mobile_technician", "roadside_assistance", "specialist"]
SUPPORTED_PARTS_CATEGORIES = ["engine", "transmission", "brakes", "suspension", "electrical", "body", "tyres", "fuel_system", "exhaust", "cooling", "steering", "drivetrain", "cabin", "safety_systems"]
SUPPORTED_WARRANTY_TYPES = ["manufacturer", "extended", "third_party", "service_warranty", "parts_warranty"]
SUPPORTED_ROADWORTHINESS_STANDARDS = ["mot_uk", "ncop_kenya", "roadworthy_za", "tuvde", "controle_technique_fr", "iveco_standard", "manufacturer_spec"]
SUPPORTED_INSPECTION_TYPES = ["pre_trip", "post_trip", "periodic", "annual_mot", "regulatory", "insurance", "breakdown_assessment"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["maintenance_scheduler", "parts_manager", "warranty_tracker", "compliance_inspector", "predictive_analyst"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"maintenance": {"supported_types": SUPPORTED_MAINTENANCE_TYPES, "vehicle_required": True, "technician_required": True, "job_card_required": True},
	"jobs": {"supported_statuses": SUPPORTED_JOB_STATUSES, "priority_levels": SUPPORTED_PRIORITY_LEVELS, "estimated_hours_required": True, "actual_hours_tracked": True},
	"workshop": {"types": SUPPORTED_WORKSHOP_TYPES, "capacity_management_enabled": True, "bay_allocation_enabled": True, "technician_skill_matching": True},
	"parts": {"categories": SUPPORTED_PARTS_CATEGORIES, "stock_management_enabled": True, "reorder_alerts_enabled": True, "supplier_management_enabled": True, "warranty_tracking_enabled": True},
	"warranty": {"types": SUPPORTED_WARRANTY_TYPES, "expiry_tracking": True, "claim_workflow_enabled": True, "manufacturer_portal_integration": False},
	"roadworthiness": {"standards": SUPPORTED_ROADWORTHINESS_STANDARDS, "mot_reminder_enabled": True, "compliance_calendar_enabled": True, "fail_dispatch_on_expired": True},
	"inspections": {"types": SUPPORTED_INSPECTION_TYPES, "defect_recording_enabled": True, "driver_walkaround_enabled": True, "digital_signature_required": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "cross_tenant_maintenance_denied": True, "expired_mot_dispatch_denied": True, "unsafe_vehicle_dispatch_denied": True},
	"observability": {"event_stream": MAINTENANCE_EVENT_STREAM, "stream_processor": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_jobs": True, "enable_workshop": True, "enable_parts": True, "enable_warranty": True, "enable_inspections": True},
	"theme": {"default_theme": "transport_maintenance_control", "allow_tenant_overrides": True},
}

PROVIDES = ["preventive_maintenance_schedule_workflow", "workshop_management_workflow", "parts_inventory_workflow", "warranty_tracking_workflow", "roadworthiness_compliance_workflow"]
REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "moni", "comp", "mqeb", "schd"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/transport-maintenance/dashboard", "component": "MaintenanceDashboard", "permission": "transport_mai:view", "nav_group": "Overview"},
	{"name": "jobs", "path": "/transport-maintenance/jobs", "component": "MaintenanceJobConsole", "permission": "transport_mai:jobs", "nav_group": "Jobs"},
	{"name": "job_create", "path": "/transport-maintenance/jobs/create", "component": "MaintenanceJobForm", "permission": "transport_mai:jobs_write", "nav_group": "Jobs"},
	{"name": "workshop", "path": "/transport-maintenance/workshop", "component": "WorkshopConsole", "permission": "transport_mai:workshop", "nav_group": "Workshop"},
	{"name": "parts", "path": "/transport-maintenance/parts", "component": "PartsInventoryConsole", "permission": "transport_mai:parts", "nav_group": "Parts"},
	{"name": "warranty", "path": "/transport-maintenance/warranty", "component": "WarrantyConsole", "permission": "transport_mai:warranty", "nav_group": "Warranty"},
	{"name": "inspections", "path": "/transport-maintenance/inspections", "component": "InspectionConsole", "permission": "transport_mai:inspections", "nav_group": "Compliance"},
	{"name": "roadworthiness", "path": "/transport-maintenance/roadworthiness", "component": "RoadworthinessConsole", "permission": "transport_mai:compliance", "nav_group": "Compliance"},
	{"name": "schedules", "path": "/transport-maintenance/schedules", "component": "MaintenanceScheduleConsole", "permission": "transport_mai:schedules", "nav_group": "Planning"},
	{"name": "reports", "path": "/transport-maintenance/reports", "component": "MaintenanceReportConsole", "permission": "transport_mai:reports", "nav_group": "Reporting"},
	{"name": "agents", "path": "/transport-maintenance/agents", "component": "MaintenanceAgentWorkbench", "permission": "transport_mai:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/transport-maintenance/settings", "component": "MaintenanceSettings", "permission": "transport_mai:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "transport_maintenance_control",
	"tokens": {"color.primary": "#7C3AED", "color.accent": "#0369A1", "color.success": "#166534", "color.warning": "#A16207", "color.danger": "#991B1B", "surface.canvas": "#FAF5FF", "surface.panel": "#FFFFFF", "text.primary": "#0F172A", "text.secondary": "#475569", "border.radius": "8px", "density": "comfortable"},
	"components": {
		"jobs": {"icon": "wrench", "status_indicator": "job-status-chip"},
		"workshop": {"icon": "home", "status_indicator": "workshop-type-chip"},
		"parts": {"icon": "package", "status_indicator": "parts-category-chip"},
		"warranty": {"icon": "shield", "status_indicator": "warranty-type-chip"},
		"inspections": {"icon": "clipboard-check", "status_indicator": "inspection-type-chip"},
		"roadworthiness": {"icon": "check-circle", "status_indicator": "standard-chip"},
		"agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": MAINTENANCE_EVENT_STREAM,
	"key": "tenant_id",
	"events": ["maintenance_job_created", "maintenance_job_completed", "parts_ordered", "warranty_claimed", "inspection_completed", "roadworthiness_certificate_issued", "maintenance_schedule_generated", "maintenance_agent_registered"],
	"guardrails": ["maintenance_batch_requires_bytewax", "expired_mot_dispatch_denied", "unsafe_vehicle_dispatch_denied", "cross_tenant_maintenance_denied", "privileged_maintenance_agent_action_requires_human_approval"],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "maintenance_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "maintenance_policy_required", "required_action": "attach_maintenance_policy"}},
	{"name": "maintenance_type_supported", "condition": {"operation": "create_job", "maintenance_type_supported": False}, "effect": {"decision": "deny", "reason": "maintenance_type_not_supported", "required_action": "select_supported_maintenance_type"}},
	{"name": "job_vehicle_required", "condition": {"operation": "create_job", "vehicle_present": False}, "effect": {"decision": "deny", "reason": "vehicle_required", "required_action": "assign_vehicle"}},
	{"name": "job_technician_required", "condition": {"operation": "create_job", "technician_present": False}, "effect": {"decision": "deny", "reason": "technician_required", "required_action": "assign_technician"}},
	{"name": "job_status_supported", "condition": {"operation": "update_job_status", "status_supported": False}, "effect": {"decision": "deny", "reason": "job_status_not_supported", "required_action": "select_supported_job_status"}},
	{"name": "job_priority_supported", "condition": {"operation": "create_job", "priority_supported": False}, "effect": {"decision": "deny", "reason": "priority_level_not_supported", "required_action": "select_supported_priority"}},
	{"name": "expired_mot_dispatch_denied", "condition": {"operation": "dispatch_vehicle", "mot_expired": True}, "effect": {"decision": "deny", "reason": "expired_mot_dispatch_denied", "required_action": "renew_mot_before_dispatch"}},
	{"name": "unsafe_vehicle_dispatch_denied", "condition": {"operation": "dispatch_vehicle", "vehicle_safe": False}, "effect": {"decision": "deny", "reason": "unsafe_vehicle_dispatch_denied", "required_action": "complete_safety_inspection"}},
	{"name": "workshop_type_supported", "condition": {"operation": "allocate_workshop", "workshop_type_supported": False}, "effect": {"decision": "deny", "reason": "workshop_type_not_supported", "required_action": "select_supported_workshop_type"}},
	{"name": "parts_category_supported", "condition": {"operation": "order_parts", "parts_category_supported": False}, "effect": {"decision": "deny", "reason": "parts_category_not_supported", "required_action": "select_supported_category"}},
	{"name": "parts_quantity_positive", "condition": {"operation": "order_parts", "quantity_positive": False}, "effect": {"decision": "deny", "reason": "parts_quantity_must_be_positive", "required_action": "correct_parts_quantity"}},
	{"name": "warranty_type_supported", "condition": {"operation": "record_warranty", "warranty_type_supported": False}, "effect": {"decision": "deny", "reason": "warranty_type_not_supported", "required_action": "select_supported_warranty_type"}},
	{"name": "inspection_type_supported", "condition": {"operation": "conduct_inspection", "inspection_type_supported": False}, "effect": {"decision": "deny", "reason": "inspection_type_not_supported", "required_action": "select_supported_inspection_type"}},
	{"name": "roadworthiness_standard_supported", "condition": {"operation": "issue_roadworthiness", "standard_supported": False}, "effect": {"decision": "deny", "reason": "roadworthiness_standard_not_supported", "required_action": "select_supported_standard"}},
	{"name": "cross_tenant_maintenance_denied", "condition": {"operation_type": "write", "cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_maintenance_denied", "required_action": "use_tenant_scoped_context"}},
	{"name": "maintenance_batch_requires_bytewax", "condition": {"operation": "maintenance_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_maintenance_batch_to_bytewax"}},
	{"name": "maintenance_agent_runtime_supported", "condition": {"operation": "register_maintenance_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "maintenance_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "maintenance_agent_role_supported", "condition": {"operation": "register_maintenance_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "maintenance_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "privileged_maintenance_agent_action_requires_human_approval", "condition": {"operation": "maintenance_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
	{"name": "inspection_digital_signature_required", "condition": {"operation": "conduct_inspection", "digital_signature_present": False}, "effect": {"decision": "deny", "reason": "digital_signature_required", "required_action": "obtain_digital_signature"}},
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
		"ui": {"shell": "apg_python", "api_prefix": "/transport-maintenance/api/v1", "requires_theme": True, "view_module": "views.py", "template_roots": ["templates/", "static/"], "routes": deepcopy(UI_ROUTES)},
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
