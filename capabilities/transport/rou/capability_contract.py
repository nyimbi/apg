"""Executable capability contract for APG Route Optimisation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "transport_rou"
CAPABILITY_NAME = "Route Optimisation"
CAPABILITY_VERSION = "1.0.0"
ROUTE_EVENT_STREAM = "apg.transport.route.lifecycle"

SUPPORTED_ROUTE_TYPES = ["single_stop", "multi_stop", "round_trip", "hub_and_spoke", "milk_run", "zone_delivery", "long_haul", "last_mile", "intermodal", "cross_dock"]
SUPPORTED_OPTIMISATION_OBJECTIVES = ["minimize_distance", "minimize_time", "minimize_cost", "minimize_co2", "maximize_load_factor", "balanced", "priority_first", "time_window_compliance"]
SUPPORTED_CONSTRAINT_TYPES = ["time_window", "vehicle_capacity", "driver_hours", "customer_preference", "hazmat_restriction", "height_restriction", "weight_restriction", "access_restriction", "ferry_crossing"]
SUPPORTED_TRAFFIC_PROVIDERS = ["here_maps", "google_maps", "tomtom", "bing_maps", "openstreetmap", "inrix", "custom_traffic_feed"]
SUPPORTED_TRANSPORT_MODES = ["road", "rail", "sea", "air", "intermodal_road_rail", "intermodal_road_sea", "intermodal_all"]
SUPPORTED_REROUTING_TRIGGERS = ["traffic_incident", "road_closure", "weather_event", "vehicle_breakdown", "driver_request", "customer_change", "time_window_risk", "fuel_shortage"]
SUPPORTED_GEOCODING_PROVIDERS = ["google_maps", "here_maps", "nominatim", "arcgis", "mapbox", "postcoder"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["route_planner", "traffic_monitor", "constraint_manager", "multimodal_coordinator", "dynamic_rerouter"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"routes": {"supported_types": SUPPORTED_ROUTE_TYPES, "max_stops_per_route": 200, "origin_required": True, "destination_required": True, "vehicle_required": True},
	"optimisation": {"objectives": SUPPORTED_OPTIMISATION_OBJECTIVES, "default_objective": "minimize_cost", "max_optimisation_seconds": 30, "multi_objective_enabled": True},
	"constraints": {"types": SUPPORTED_CONSTRAINT_TYPES, "time_window_enforcement": True, "capacity_enforcement": True, "hos_enforcement": True},
	"traffic": {"providers": SUPPORTED_TRAFFIC_PROVIDERS, "real_time_enabled": True, "historical_patterns_enabled": True, "incident_alerts_enabled": True},
	"transport_modes": {"supported": SUPPORTED_TRANSPORT_MODES, "multimodal_planning_enabled": True, "mode_cost_comparison_enabled": True},
	"rerouting": {"triggers": SUPPORTED_REROUTING_TRIGGERS, "auto_reroute_enabled": True, "driver_notification_on_reroute": True, "customer_eta_update_on_reroute": True},
	"geocoding": {"providers": SUPPORTED_GEOCODING_PROVIDERS, "address_validation_enabled": True, "what3words_enabled": False},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "cross_tenant_route_denied": True, "unvalidated_address_dispatch_denied": True},
	"observability": {"event_stream": ROUTE_EVENT_STREAM, "stream_processor": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_routes": True, "enable_optimisation": True, "enable_constraints": True, "enable_traffic": True, "enable_multimodal": True},
	"theme": {"default_theme": "transport_route_control", "allow_tenant_overrides": True},
}

PROVIDES = ["multi_stop_route_planning_workflow", "dynamic_rerouting_workflow", "traffic_integration_workflow", "time_window_constraint_workflow", "multimodal_routing_workflow"]
REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "moni", "nlpc", "mqeb", "schd"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/transport-route/dashboard", "component": "RouteDashboard", "permission": "transport_rou:view", "nav_group": "Overview"},
	{"name": "routes", "path": "/transport-route/routes", "component": "RouteConsole", "permission": "transport_rou:routes", "nav_group": "Routes"},
	{"name": "route_create", "path": "/transport-route/routes/create", "component": "RouteForm", "permission": "transport_rou:routes_write", "nav_group": "Routes"},
	{"name": "route_map", "path": "/transport-route/routes/<route_id>/map", "component": "RouteMap", "permission": "transport_rou:routes", "nav_group": "Routes"},
	{"name": "optimisation", "path": "/transport-route/optimisation", "component": "RouteOptimisationConsole", "permission": "transport_rou:optimisation", "nav_group": "Optimisation"},
	{"name": "constraints", "path": "/transport-route/constraints", "component": "ConstraintConsole", "permission": "transport_rou:constraints", "nav_group": "Planning"},
	{"name": "traffic", "path": "/transport-route/traffic", "component": "TrafficIntegrationConsole", "permission": "transport_rou:traffic", "nav_group": "Traffic"},
	{"name": "rerouting", "path": "/transport-route/rerouting", "component": "ReroutingConsole", "permission": "transport_rou:rerouting", "nav_group": "Dynamic"},
	{"name": "multimodal", "path": "/transport-route/multimodal", "component": "MultimodalConsole", "permission": "transport_rou:multimodal", "nav_group": "Multimodal"},
	{"name": "geocoding", "path": "/transport-route/geocoding", "component": "GeocodingConsole", "permission": "transport_rou:geocoding", "nav_group": "Tools"},
	{"name": "reports", "path": "/transport-route/reports", "component": "RouteReportConsole", "permission": "transport_rou:reports", "nav_group": "Reporting"},
	{"name": "agents", "path": "/transport-route/agents", "component": "RouteAgentWorkbench", "permission": "transport_rou:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/transport-route/settings", "component": "RouteSettings", "permission": "transport_rou:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "transport_route_control",
	"tokens": {"color.primary": "#0F766E", "color.accent": "#1D4ED8", "color.success": "#166534", "color.warning": "#A16207", "color.danger": "#991B1B", "surface.canvas": "#F0FDFA", "surface.panel": "#FFFFFF", "text.primary": "#0F172A", "text.secondary": "#475569", "border.radius": "8px", "density": "comfortable"},
	"components": {
		"routes": {"icon": "navigation", "status_indicator": "route-type-chip"},
		"optimisation": {"icon": "zap", "status_indicator": "optimisation-objective-chip"},
		"constraints": {"icon": "filter", "status_indicator": "constraint-type-chip"},
		"traffic": {"icon": "activity", "status_indicator": "traffic-provider-chip"},
		"rerouting": {"icon": "refresh-cw", "status_indicator": "rerouting-trigger-chip"},
		"multimodal": {"icon": "git-merge", "status_indicator": "transport-mode-chip"},
		"agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": ROUTE_EVENT_STREAM,
	"key": "tenant_id",
	"events": ["route_planned", "route_optimised", "route_dispatched", "traffic_incident_detected", "reroute_triggered", "reroute_completed", "constraint_violation_detected", "multimodal_segment_planned", "route_agent_registered"],
	"guardrails": ["route_batch_requires_bytewax", "unvalidated_address_dispatch_denied", "cross_tenant_route_denied", "privileged_route_agent_action_requires_human_approval"],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "route_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "route_policy_required", "required_action": "attach_route_policy"}},
	{"name": "route_type_supported", "condition": {"operation": "plan_route", "route_type_supported": False}, "effect": {"decision": "deny", "reason": "route_type_not_supported", "required_action": "select_supported_route_type"}},
	{"name": "route_origin_required", "condition": {"operation": "plan_route", "origin_present": False}, "effect": {"decision": "deny", "reason": "origin_required", "required_action": "set_origin"}},
	{"name": "route_destination_required", "condition": {"operation": "plan_route", "destination_present": False}, "effect": {"decision": "deny", "reason": "destination_required", "required_action": "set_destination"}},
	{"name": "route_vehicle_required", "condition": {"operation": "plan_route", "vehicle_present": False}, "effect": {"decision": "deny", "reason": "vehicle_required", "required_action": "assign_vehicle"}},
	{"name": "unvalidated_address_dispatch_denied", "condition": {"operation": "plan_route", "address_validated": False}, "effect": {"decision": "deny", "reason": "unvalidated_address_dispatch_denied", "required_action": "validate_all_addresses"}},
	{"name": "optimisation_objective_supported", "condition": {"operation": "optimise_route", "objective_supported": False}, "effect": {"decision": "deny", "reason": "optimisation_objective_not_supported", "required_action": "select_supported_objective"}},
	{"name": "constraint_type_supported", "condition": {"operation": "add_constraint", "constraint_type_supported": False}, "effect": {"decision": "deny", "reason": "constraint_type_not_supported", "required_action": "select_supported_constraint_type"}},
	{"name": "traffic_provider_supported", "condition": {"operation": "integrate_traffic", "provider_supported": False}, "effect": {"decision": "deny", "reason": "traffic_provider_not_supported", "required_action": "select_supported_provider"}},
	{"name": "transport_mode_supported", "condition": {"operation": "plan_route", "transport_mode_supported": False}, "effect": {"decision": "deny", "reason": "transport_mode_not_supported", "required_action": "select_supported_transport_mode"}},
	{"name": "rerouting_trigger_supported", "condition": {"operation": "trigger_reroute", "trigger_supported": False}, "effect": {"decision": "deny", "reason": "rerouting_trigger_not_supported", "required_action": "select_supported_trigger"}},
	{"name": "rerouting_route_required", "condition": {"operation": "trigger_reroute", "route_present": False}, "effect": {"decision": "deny", "reason": "route_reference_required", "required_action": "select_route"}},
	{"name": "geocoding_provider_supported", "condition": {"operation": "geocode_address", "provider_supported": False}, "effect": {"decision": "deny", "reason": "geocoding_provider_not_supported", "required_action": "select_supported_provider"}},
	{"name": "max_stops_exceeded", "condition": {"operation": "plan_route", "stops_exceed_maximum": True}, "effect": {"decision": "deny", "reason": "max_stops_per_route_exceeded", "required_action": "split_route_or_reduce_stops"}},
	{"name": "cross_tenant_route_denied", "condition": {"operation_type": "write", "cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_route_denied", "required_action": "use_tenant_scoped_context"}},
	{"name": "route_batch_requires_bytewax", "condition": {"operation": "route_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_batch_to_bytewax"}},
	{"name": "route_agent_runtime_supported", "condition": {"operation": "register_route_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "route_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "route_agent_role_supported", "condition": {"operation": "register_route_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "route_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "privileged_route_agent_action_requires_human_approval", "condition": {"operation": "route_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
	{"name": "capacity_constraint_violation", "condition": {"operation": "plan_route", "capacity_constraint_violated": True}, "effect": {"decision": "deny", "reason": "vehicle_capacity_constraint_violated", "required_action": "rebalance_load_across_vehicles"}},
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
		"ui": {"shell": "apg_python", "api_prefix": "/transport-route/api/v1", "requires_theme": True, "view_module": "views.py", "template_roots": ["templates/", "static/"], "routes": deepcopy(UI_ROUTES)},
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
