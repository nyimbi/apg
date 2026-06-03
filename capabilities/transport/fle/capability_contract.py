"""Executable capability contract for APG Fleet Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "transport_fle"
CAPABILITY_NAME = "Fleet Management"
CAPABILITY_VERSION = "1.0.0"
FLEET_EVENT_STREAM = "apg.transport.fleet.lifecycle"

SUPPORTED_VEHICLE_TYPES = ["rigid_truck", "articulated_truck", "van", "pickup", "tractor_unit", "trailer", "tanker", "refrigerated_vehicle", "flatbed", "tipper", "minibus", "motorcycle", "electric_vehicle"]
SUPPORTED_VEHICLE_STATUSES = ["active", "inactive", "in_maintenance", "out_of_service", "disposed", "on_hire", "awaiting_inspection"]
SUPPORTED_FUEL_TYPES = ["diesel", "petrol", "cng", "lng", "electric", "hybrid", "hydrogen", "biodiesel"]
SUPPORTED_OWNERSHIP_TYPES = ["owned", "leased", "hired", "contract_hire", "finance_lease", "hire_purchase"]
SUPPORTED_COMPLIANCE_STANDARDS = ["dvla", "c_tpat", "euro6", "euro5", "adr", "gdpr_telematics", "tachograph_regulation", "operator_licence"]
SUPPORTED_TELEMATICS_PROVIDERS = ["samsara", "geotab", "webfleet", "trimble", "verizon_connect", "motive", "lytx", "zubie", "custom"]
SUPPORTED_DRIVER_STATUSES = ["active", "inactive", "on_leave", "suspended", "training", "probation"]
SUPPORTED_LICENCE_CLASSES = ["am", "a1", "a2", "a", "b", "be", "c1", "c1e", "c", "ce", "d1", "d1e", "d", "de"]
SUPPORTED_UTILISATION_METRICS = ["distance_km", "engine_hours", "idle_time_pct", "load_factor_pct", "fuel_efficiency_l100km", "trips_per_day"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["fleet_analyst", "compliance_checker", "telematics_monitor", "driver_manager", "utilisation_optimiser"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"vehicles": {"supported_types": SUPPORTED_VEHICLE_TYPES, "supported_statuses": SUPPORTED_VEHICLE_STATUSES, "fuel_types": SUPPORTED_FUEL_TYPES, "ownership_types": SUPPORTED_OWNERSHIP_TYPES, "registration_required": True, "vin_required": True},
	"compliance": {"standards": SUPPORTED_COMPLIANCE_STANDARDS, "mot_tracking_enabled": True, "operator_licence_tracking": True, "tachograph_enabled": True, "dvla_check_enabled": True, "c_tpat_enabled": False},
	"telematics": {"providers": SUPPORTED_TELEMATICS_PROVIDERS, "real_time_tracking": True, "driver_behaviour_scoring": True, "harsh_event_detection": True, "geofencing_enabled": True},
	"drivers": {"statuses": SUPPORTED_DRIVER_STATUSES, "licence_classes": SUPPORTED_LICENCE_CLASSES, "cpc_tracking_enabled": True, "tacho_card_tracking": True, "hours_of_service_enabled": True},
	"utilisation": {"metrics": SUPPORTED_UTILISATION_METRICS, "reporting_period_days": 30, "benchmarking_enabled": True, "alert_threshold_pct": 60},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "cross_tenant_fleet_denied": True, "unlicenced_driver_dispatch_denied": True, "non_compliant_vehicle_dispatch_denied": True},
	"observability": {"event_stream": FLEET_EVENT_STREAM, "stream_processor": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_vehicles": True, "enable_drivers": True, "enable_compliance": True, "enable_telematics": True, "enable_utilisation": True},
	"theme": {"default_theme": "transport_fleet_control", "allow_tenant_overrides": True},
}

PROVIDES = ["vehicle_lifecycle_workflow", "telematics_integration_workflow", "driver_management_workflow", "fleet_utilisation_analytics_workflow", "fleet_compliance_workflow"]
REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "moni", "comp", "mqeb", "schd"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/transport-fleet/dashboard", "component": "FleetDashboard", "permission": "transport_fle:view", "nav_group": "Overview"},
	{"name": "vehicles", "path": "/transport-fleet/vehicles", "component": "VehicleConsole", "permission": "transport_fle:vehicles", "nav_group": "Vehicles"},
	{"name": "vehicle_create", "path": "/transport-fleet/vehicles/create", "component": "VehicleForm", "permission": "transport_fle:vehicles_write", "nav_group": "Vehicles"},
	{"name": "vehicle_detail", "path": "/transport-fleet/vehicles/<vehicle_id>", "component": "VehicleDetail", "permission": "transport_fle:vehicles", "nav_group": "Vehicles"},
	{"name": "drivers", "path": "/transport-fleet/drivers", "component": "DriverConsole", "permission": "transport_fle:drivers", "nav_group": "Drivers"},
	{"name": "driver_detail", "path": "/transport-fleet/drivers/<driver_id>", "component": "DriverDetail", "permission": "transport_fle:drivers", "nav_group": "Drivers"},
	{"name": "telematics", "path": "/transport-fleet/telematics", "component": "TelematicsConsole", "permission": "transport_fle:telematics", "nav_group": "Telematics"},
	{"name": "compliance", "path": "/transport-fleet/compliance", "component": "FleetComplianceConsole", "permission": "transport_fle:compliance", "nav_group": "Compliance"},
	{"name": "utilisation", "path": "/transport-fleet/utilisation", "component": "FleetUtilisationConsole", "permission": "transport_fle:utilisation", "nav_group": "Analytics"},
	{"name": "reports", "path": "/transport-fleet/reports", "component": "FleetReportConsole", "permission": "transport_fle:reports", "nav_group": "Reporting"},
	{"name": "agents", "path": "/transport-fleet/agents", "component": "FleetAgentWorkbench", "permission": "transport_fle:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/transport-fleet/settings", "component": "FleetSettings", "permission": "transport_fle:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "transport_fleet_control",
	"tokens": {"color.primary": "#047857", "color.accent": "#0369A1", "color.success": "#166534", "color.warning": "#A16207", "color.danger": "#991B1B", "surface.canvas": "#F0FDF4", "surface.panel": "#FFFFFF", "text.primary": "#0F172A", "text.secondary": "#475569", "border.radius": "8px", "density": "comfortable"},
	"components": {
		"vehicles": {"icon": "truck", "status_indicator": "vehicle-status-chip"},
		"drivers": {"icon": "user", "status_indicator": "driver-status-chip"},
		"telematics": {"icon": "radio", "status_indicator": "telematics-provider-chip"},
		"compliance": {"icon": "shield-check", "status_indicator": "compliance-standard-chip"},
		"utilisation": {"icon": "bar-chart-2", "status_indicator": "utilisation-metric-chip"},
		"agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": FLEET_EVENT_STREAM,
	"key": "tenant_id",
	"events": ["vehicle_registered", "vehicle_status_changed", "driver_registered", "driver_status_changed", "telematics_event", "compliance_check_completed", "utilisation_report_generated", "fleet_agent_registered"],
	"guardrails": ["fleet_batch_requires_bytewax", "unlicenced_driver_dispatch_denied", "non_compliant_vehicle_dispatch_denied", "cross_tenant_fleet_denied", "privileged_fleet_agent_action_requires_human_approval"],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "fleet_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "fleet_policy_required", "required_action": "attach_fleet_policy"}},
	{"name": "vehicle_type_supported", "condition": {"operation": "register_vehicle", "vehicle_type_supported": False}, "effect": {"decision": "deny", "reason": "vehicle_type_not_supported", "required_action": "select_supported_vehicle_type"}},
	{"name": "vehicle_registration_required", "condition": {"operation": "register_vehicle", "registration_present": False}, "effect": {"decision": "deny", "reason": "vehicle_registration_required", "required_action": "provide_vehicle_registration"}},
	{"name": "vehicle_vin_required", "condition": {"operation": "register_vehicle", "vin_present": False}, "effect": {"decision": "deny", "reason": "vehicle_vin_required", "required_action": "provide_vin"}},
	{"name": "vehicle_ownership_type_supported", "condition": {"operation": "register_vehicle", "ownership_type_supported": False}, "effect": {"decision": "deny", "reason": "ownership_type_not_supported", "required_action": "select_supported_ownership_type"}},
	{"name": "vehicle_status_supported", "condition": {"operation": "update_vehicle_status", "status_supported": False}, "effect": {"decision": "deny", "reason": "vehicle_status_not_supported", "required_action": "select_supported_vehicle_status"}},
	{"name": "non_compliant_vehicle_dispatch_denied", "condition": {"operation": "dispatch_vehicle", "compliance_check_passed": False}, "effect": {"decision": "deny", "reason": "non_compliant_vehicle_dispatch_denied", "required_action": "resolve_compliance_issues"}},
	{"name": "driver_licence_class_supported", "condition": {"operation": "register_driver", "licence_class_supported": False}, "effect": {"decision": "deny", "reason": "licence_class_not_supported", "required_action": "select_supported_licence_class"}},
	{"name": "driver_status_supported", "condition": {"operation": "update_driver_status", "status_supported": False}, "effect": {"decision": "deny", "reason": "driver_status_not_supported", "required_action": "select_supported_driver_status"}},
	{"name": "unlicenced_driver_dispatch_denied", "condition": {"operation": "assign_driver", "driver_licenced": False}, "effect": {"decision": "deny", "reason": "unlicenced_driver_dispatch_denied", "required_action": "verify_driver_licence"}},
	{"name": "driver_hours_check_required", "condition": {"operation": "assign_driver", "hours_checked": False}, "effect": {"decision": "deny", "reason": "driver_hours_check_required", "required_action": "verify_driver_hours"}},
	{"name": "telematics_provider_supported", "condition": {"operation": "integrate_telematics", "provider_supported": False}, "effect": {"decision": "deny", "reason": "telematics_provider_not_supported", "required_action": "select_supported_provider"}},
	{"name": "compliance_standard_supported", "condition": {"operation": "record_compliance", "standard_supported": False}, "effect": {"decision": "deny", "reason": "compliance_standard_not_supported", "required_action": "select_supported_standard"}},
	{"name": "utilisation_metric_supported", "condition": {"operation": "record_utilisation", "metric_supported": False}, "effect": {"decision": "deny", "reason": "utilisation_metric_not_supported", "required_action": "select_supported_metric"}},
	{"name": "cross_tenant_fleet_denied", "condition": {"operation_type": "write", "cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_fleet_denied", "required_action": "use_tenant_scoped_context"}},
	{"name": "fleet_batch_requires_bytewax", "condition": {"operation": "fleet_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_fleet_batch_to_bytewax"}},
	{"name": "fleet_agent_runtime_supported", "condition": {"operation": "register_fleet_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "fleet_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "fleet_agent_role_supported", "condition": {"operation": "register_fleet_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "fleet_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "privileged_fleet_agent_action_requires_human_approval", "condition": {"operation": "fleet_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
	{"name": "fuel_type_supported", "condition": {"operation": "register_vehicle", "fuel_type_supported": False}, "effect": {"decision": "deny", "reason": "fuel_type_not_supported", "required_action": "select_supported_fuel_type"}},
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
		"ui": {"shell": "apg_python", "api_prefix": "/transport-fleet/api/v1", "requires_theme": True, "view_module": "views.py", "template_roots": ["templates/", "static/"], "routes": deepcopy(UI_ROUTES)},
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
