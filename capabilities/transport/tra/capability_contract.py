"""Executable capability contract for APG Asset Tracking."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "transport_tra"
CAPABILITY_NAME = "Asset Tracking"
CAPABILITY_VERSION = "1.0.0"
TRACKING_EVENT_STREAM = "apg.transport.tracking.lifecycle"

SUPPORTED_ASSET_TYPES = ["vehicle", "trailer", "container", "pallet", "ibc", "roll_cage", "equipment", "genset", "reefer_unit", "tanker_body", "swap_body"]
SUPPORTED_TRACKING_TECHNOLOGIES = ["gps", "gnss", "cellular", "bluetooth_ble", "rfid", "nfc", "barcode", "qr_code", "iot_sensor", "satellite", "lora_wan"]
SUPPORTED_MONITORING_TYPES = ["location", "temperature", "humidity", "shock", "light", "door_open", "pressure", "tilt", "co2", "battery", "fuel_level"]
SUPPORTED_GEOFENCE_TYPES = ["circle", "polygon", "corridor", "point_of_interest", "exclusion_zone", "country_border", "depot", "customer_site"]
SUPPORTED_ALERT_TYPES = ["geofence_entry", "geofence_exit", "temperature_breach", "idle_too_long", "harsh_braking", "harsh_acceleration", "speeding", "unauthorised_use", "low_battery", "tamper_detected", "container_opened"]
SUPPORTED_COLD_CHAIN_STANDARDS = ["atp_agreement", "haccp", "gxp", "who_guidelines", "pda_technical_report", "iata_live_animals", "iata_perishables"]
SUPPORTED_CONTAINER_STATUSES = ["available", "loaded", "in_transit", "at_port", "customs_hold", "empty_return", "under_repair", "decommissioned"]
SUPPORTED_UTILISATION_PERIODS = ["daily", "weekly", "monthly", "quarterly", "annual"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["asset_tracker", "geofence_manager", "cold_chain_monitor", "container_tracker", "utilisation_analyst"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"assets": {"supported_types": SUPPORTED_ASSET_TYPES, "tracking_technologies": SUPPORTED_TRACKING_TECHNOLOGIES, "unique_id_required": True, "owner_required": True, "registration_required": True},
	"monitoring": {"types": SUPPORTED_MONITORING_TYPES, "real_time_enabled": True, "data_retention_days": 365, "anomaly_detection_enabled": True},
	"geofencing": {"types": SUPPORTED_GEOFENCE_TYPES, "max_geofences_per_tenant": 500, "entry_exit_alerts": True, "dwell_time_tracking": True},
	"alerts": {"types": SUPPORTED_ALERT_TYPES, "multi_channel_delivery": True, "alert_suppression_enabled": True, "escalation_enabled": True},
	"cold_chain": {"standards": SUPPORTED_COLD_CHAIN_STANDARDS, "continuous_logging_enabled": True, "breach_alert_immediate": True, "certificate_generation_enabled": True},
	"containers": {"statuses": SUPPORTED_CONTAINER_STATUSES, "iso_number_required": True, "seal_number_tracking": True, "detention_tracking_enabled": True},
	"utilisation": {"periods": SUPPORTED_UTILISATION_PERIODS, "idle_threshold_minutes": 30, "utilisation_benchmarking": True, "cost_per_idle_hour_tracking": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "cross_tenant_tracking_denied": True, "tamper_alert_escalation_required": True},
	"observability": {"event_stream": TRACKING_EVENT_STREAM, "stream_processor": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_assets": True, "enable_geofencing": True, "enable_alerts": True, "enable_cold_chain": True, "enable_containers": True},
	"theme": {"default_theme": "transport_tracking_control", "allow_tenant_overrides": True},
}

PROVIDES = ["realtime_gps_tracking_workflow", "geofencing_workflow", "cold_chain_monitoring_workflow", "container_tracking_workflow", "asset_utilisation_workflow"]
REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "moni", "comp", "mqeb", "nlpc"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/transport-tracking/dashboard", "component": "TrackingDashboard", "permission": "transport_tra:view", "nav_group": "Overview"},
	{"name": "live_map", "path": "/transport-tracking/map", "component": "LiveTrackingMap", "permission": "transport_tra:view", "nav_group": "Live"},
	{"name": "assets", "path": "/transport-tracking/assets", "component": "AssetConsole", "permission": "transport_tra:assets", "nav_group": "Assets"},
	{"name": "asset_detail", "path": "/transport-tracking/assets/<asset_id>", "component": "AssetDetail", "permission": "transport_tra:assets", "nav_group": "Assets"},
	{"name": "geofencing", "path": "/transport-tracking/geofencing", "component": "GeofenceConsole", "permission": "transport_tra:geofencing", "nav_group": "Geofencing"},
	{"name": "alerts", "path": "/transport-tracking/alerts", "component": "TrackingAlertConsole", "permission": "transport_tra:alerts", "nav_group": "Alerts"},
	{"name": "cold_chain", "path": "/transport-tracking/cold-chain", "component": "ColdChainConsole", "permission": "transport_tra:cold_chain", "nav_group": "Cold Chain"},
	{"name": "containers", "path": "/transport-tracking/containers", "component": "ContainerConsole", "permission": "transport_tra:containers", "nav_group": "Containers"},
	{"name": "utilisation", "path": "/transport-tracking/utilisation", "component": "AssetUtilisationConsole", "permission": "transport_tra:utilisation", "nav_group": "Analytics"},
	{"name": "reports", "path": "/transport-tracking/reports", "component": "TrackingReportConsole", "permission": "transport_tra:reports", "nav_group": "Reporting"},
	{"name": "agents", "path": "/transport-tracking/agents", "component": "TrackingAgentWorkbench", "permission": "transport_tra:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/transport-tracking/settings", "component": "TrackingSettings", "permission": "transport_tra:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "transport_tracking_control",
	"tokens": {"color.primary": "#0F766E", "color.accent": "#0369A1", "color.success": "#166534", "color.warning": "#A16207", "color.danger": "#991B1B", "surface.canvas": "#F0FDFA", "surface.panel": "#FFFFFF", "text.primary": "#0F172A", "text.secondary": "#475569", "border.radius": "8px", "density": "compact"},
	"components": {
		"assets": {"icon": "map-pin", "status_indicator": "asset-type-chip"},
		"geofencing": {"icon": "hexagon", "status_indicator": "geofence-type-chip"},
		"alerts": {"icon": "bell-ring", "status_indicator": "alert-type-chip"},
		"cold_chain": {"icon": "thermometer", "status_indicator": "cold-chain-standard-chip"},
		"containers": {"icon": "box", "status_indicator": "container-status-chip"},
		"utilisation": {"icon": "pie-chart", "status_indicator": "utilisation-period-chip"},
		"agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": TRACKING_EVENT_STREAM,
	"key": "tenant_id",
	"events": ["asset_registered", "asset_location_updated", "geofence_entered", "geofence_exited", "tracking_alert_raised", "cold_chain_breach_detected", "container_status_changed", "utilisation_report_generated", "tracking_agent_registered"],
	"guardrails": ["tracking_batch_requires_bytewax", "cross_tenant_tracking_denied", "tamper_alert_escalation_required", "privileged_tracking_agent_action_requires_human_approval"],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "tracking_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "tracking_policy_required", "required_action": "attach_tracking_policy"}},
	{"name": "asset_type_supported", "condition": {"operation": "register_asset", "asset_type_supported": False}, "effect": {"decision": "deny", "reason": "asset_type_not_supported", "required_action": "select_supported_asset_type"}},
	{"name": "asset_unique_id_required", "condition": {"operation": "register_asset", "unique_id_present": False}, "effect": {"decision": "deny", "reason": "unique_asset_id_required", "required_action": "provide_unique_asset_id"}},
	{"name": "asset_owner_required", "condition": {"operation": "register_asset", "owner_present": False}, "effect": {"decision": "deny", "reason": "asset_owner_required", "required_action": "assign_asset_owner"}},
	{"name": "tracking_technology_supported", "condition": {"operation": "install_tracker", "technology_supported": False}, "effect": {"decision": "deny", "reason": "tracking_technology_not_supported", "required_action": "select_supported_technology"}},
	{"name": "monitoring_type_supported", "condition": {"operation": "configure_monitoring", "monitoring_type_supported": False}, "effect": {"decision": "deny", "reason": "monitoring_type_not_supported", "required_action": "select_supported_monitoring_type"}},
	{"name": "geofence_type_supported", "condition": {"operation": "create_geofence", "geofence_type_supported": False}, "effect": {"decision": "deny", "reason": "geofence_type_not_supported", "required_action": "select_supported_geofence_type"}},
	{"name": "geofence_area_required", "condition": {"operation": "create_geofence", "area_defined": False}, "effect": {"decision": "deny", "reason": "geofence_area_required", "required_action": "define_geofence_boundary"}},
	{"name": "alert_type_supported", "condition": {"operation": "configure_alert", "alert_type_supported": False}, "effect": {"decision": "deny", "reason": "alert_type_not_supported", "required_action": "select_supported_alert_type"}},
	{"name": "cold_chain_standard_supported", "condition": {"operation": "configure_cold_chain", "standard_supported": False}, "effect": {"decision": "deny", "reason": "cold_chain_standard_not_supported", "required_action": "select_supported_standard"}},
	{"name": "cold_chain_temp_range_required", "condition": {"operation": "configure_cold_chain", "temp_range_defined": False}, "effect": {"decision": "deny", "reason": "temperature_range_required", "required_action": "define_temperature_range"}},
	{"name": "container_iso_number_required", "condition": {"operation": "register_container", "iso_number_present": False}, "effect": {"decision": "deny", "reason": "container_iso_number_required", "required_action": "provide_iso_number"}},
	{"name": "container_status_supported", "condition": {"operation": "update_container_status", "status_supported": False}, "effect": {"decision": "deny", "reason": "container_status_not_supported", "required_action": "select_supported_container_status"}},
	{"name": "tamper_alert_escalation_required", "condition": {"operation": "update_asset_location", "tamper_detected": True}, "effect": {"decision": "deny", "reason": "tamper_alert_requires_escalation", "required_action": "escalate_tamper_alert"}},
	{"name": "utilisation_period_supported", "condition": {"operation": "generate_utilisation_report", "period_supported": False}, "effect": {"decision": "deny", "reason": "utilisation_period_not_supported", "required_action": "select_supported_period"}},
	{"name": "cross_tenant_tracking_denied", "condition": {"operation_type": "write", "cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_tracking_denied", "required_action": "use_tenant_scoped_context"}},
	{"name": "tracking_batch_requires_bytewax", "condition": {"operation": "tracking_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_tracking_batch_to_bytewax"}},
	{"name": "tracking_agent_runtime_supported", "condition": {"operation": "register_tracking_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "tracking_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "tracking_agent_role_supported", "condition": {"operation": "register_tracking_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "tracking_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "privileged_tracking_agent_action_requires_human_approval", "condition": {"operation": "tracking_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
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
		"ui": {"shell": "apg_python", "api_prefix": "/transport-tracking/api/v1", "requires_theme": True, "view_module": "views.py", "template_roots": ["templates/", "static/"], "routes": deepcopy(UI_ROUTES)},
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
