"""Executable capability contract for APG Geo-Spatial Services."""

from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Any


SUPPORTED_LOCATION_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_LOCATION_AGENT_ROLES = ["geofence_reviewer", "privacy_reviewer", "territory_planner", "event_analyst", "edge_operator"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"geofencing": {
		"geofence_owner_required": True,
		"max_vertices_per_polygon": 5000,
		"minimum_accuracy_meters": 50,
		"active_rule_required": True
	},
	"events": {
		"event_source_registered_required": True,
		"location_consent_required": True,
		"event_retention_days": 90,
		"edge_event_ingestion_supported": True
	},
	"analytics": {
		"spatial_index_required": True,
		"predictive_location_enabled": True,
		"territory_overlap_review_required": True,
		"aggregation_privacy_required": True
	},
	"location_agents": {
		"agent_assist_enabled": True,
		"agent_registration_required": True,
		"agent_runtime_required": True,
		"agent_scope_required": True,
		"agent_contribution_disclosure_required": True,
		"supported_runtimes": SUPPORTED_LOCATION_AGENT_RUNTIMES,
		"allowed_roles": SUPPORTED_LOCATION_AGENT_ROLES
	},
	"governance": {
		"require_tenant_context": True,
		"audit_location_events": True,
		"data_residency_policy_required": True,
		"sensitive_location_review_required": True,
		"tenant_isolation_required": True,
		"state_change_reason_required": True,
		"batch_event_stream": "bytewax"
	},
	"observability": {
		"audit_required": True,
		"quality_metrics_required": True,
		"latency_metrics_required": True,
		"agent_activity_required": True,
		"event_stream": "bytewax"
	},
	"adapters": {
		"generated_app_runtime": "service.GeosService",
		"api_helpers": "api.py",
		"view_models": "views.py",
		"event_stream": "bytewax",
		"prediction": "pred",
		"ai_core": "aicr",
		"master_data": "mdm",
		"audit_sink": "audl",
		"notification": "ntfy",
		"edge_runtime": "edge",
		"workflow": "wflo"
	},
	"ui": {
		"enable_map_console": True,
		"enable_geofence_editor": True,
		"enable_event_monitor": True,
		"enable_spatial_analytics": True,
		"enable_agent_panel": True,
		"enable_audit": True,
		"enable_analytics": True
	},
	"theme": {
		"default_theme": "geos_location_intelligence",
		"allow_tenant_overrides": True
	}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": ["tenant_id", "geofencing", "events", "analytics", "location_agents", "governance", "observability", "adapters", "ui", "theme"],
	"properties": {key: {"type": "object"} for key in ["geofencing", "events", "analytics", "location_agents", "governance", "observability", "adapters", "ui", "theme"]} | {
		"tenant_id": {"type": "string", "minLength": 1}
	}
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All geo-spatial operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "location_consent_required", "description": "Location processing requires consent.", "condition": {"operation": "process_location_event", "location_consent_recorded": False}, "effect": {"decision": "deny", "reason": "location_consent_required", "required_action": "record_location_consent"}},
	{"name": "geofence_requires_owner", "description": "Geofences require an accountable owner.", "condition": {"operation": "create_geofence", "geofence_owner_assigned": False}, "effect": {"decision": "deny", "reason": "geofence_owner_required", "required_action": "assign_geofence_owner"}},
	{"name": "event_source_must_be_registered", "description": "Location events require registered sources.", "condition": {"event_source_registered": False, "location_event_received": True}, "effect": {"decision": "deny", "reason": "event_source_registration_required", "required_action": "register_event_source"}},
	{"name": "sensitive_location_requires_review", "description": "Sensitive location processing requires review.", "condition": {"sensitive_location": True, "privacy_review_recorded": False}, "effect": {"decision": "deny", "reason": "sensitive_location_review_required", "required_action": "record_privacy_review"}},
	{"name": "large_polygon_requires_review", "description": "Large geofence polygons require spatial review.", "condition": {"polygon_vertices_gt": 5000, "spatial_review_recorded": False}, "effect": {"decision": "require_review", "reason": "large_polygon_review_required", "required_action": "review_geofence_geometry"}},
	{"name": "data_residency_policy_required", "description": "Event sources require data residency policy.", "condition": {"operation": "register_event_source", "data_residency_policy_present": False}, "effect": {"decision": "deny", "reason": "data_residency_policy_required", "required_action": "attach_data_residency_policy"}},
	{"name": "active_geofence_rule_required", "description": "Geofences require at least one active rule.", "condition": {"operation": "create_geofence", "active_geofence_rule_present": False}, "effect": {"decision": "deny", "reason": "active_geofence_rule_required", "required_action": "activate_geofence_rule"}},
	{"name": "minimum_location_accuracy_required", "description": "Location events must meet minimum accuracy policy.", "condition": {"accuracy_meters_gt": 50}, "effect": {"decision": "deny", "reason": "minimum_accuracy_required", "required_action": "provide_accurate_location"}},
	{"name": "spatial_index_required", "description": "Spatial analytics require an available spatial index.", "condition": {"operation": "run_spatial_analysis", "spatial_index_available": False}, "effect": {"decision": "deny", "reason": "spatial_index_required", "required_action": "build_spatial_index"}},
	{"name": "aggregation_privacy_required", "description": "Spatial analytics require privacy-preserving aggregation.", "condition": {"operation": "run_spatial_analysis", "aggregation_privacy_applied": False}, "effect": {"decision": "deny", "reason": "aggregation_privacy_required", "required_action": "apply_aggregation_privacy"}},
	{"name": "location_agent_requires_registration", "description": "AI location agents must be registered.", "condition": {"location_agent_present": True, "agent_registered": False}, "effect": {"decision": "deny", "reason": "location_agent_registration_required", "required_action": "register_location_agent"}},
	{"name": "location_agent_runtime_supported", "description": "AI location agents must use a supported runtime.", "condition": {"location_agent_present": True, "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "location_agent_runtime_not_supported", "required_action": "choose_supported_location_agent_runtime"}},
	{"name": "location_agent_requires_scope", "description": "AI location agents require explicit scope.", "condition": {"location_agent_present": True, "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "location_agent_scope_required", "required_action": "set_location_agent_scope"}},
	{"name": "location_agent_requires_disclosure", "description": "AI location-agent contributions require disclosure.", "condition": {"location_agent_present": True, "agent_contribution_disclosed": False}, "effect": {"decision": "deny", "reason": "location_agent_disclosure_required", "required_action": "disclose_location_agent"}},
	{"name": "geos_state_change_requires_reason", "description": "Location lifecycle state changes require a reason.", "condition": {"state_change_requested": True, "state_change_reason_present": False}, "effect": {"decision": "deny", "reason": "geos_state_change_reason_required", "required_action": "record_state_change_reason"}},
	{"name": "geos_state_change_requires_audit", "description": "Location lifecycle state changes require audit evidence.", "condition": {"state_change_requested": True, "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "geos_audit_event_required", "required_action": "record_geos_audit_event"}},
	{"name": "cross_tenant_location_access_denied", "description": "Location records may not cross tenant boundaries.", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_location_access_denied", "required_action": "use_tenant_local_context"}},
	{"name": "batch_location_mutation_requires_bytewax", "description": "Batch location mutations must use Bytewax event streams.", "condition": {"operation": "batch_location_mutation", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/geos/dashboard", "component": "GEOSDashboard", "permission": "geos:view", "nav_group": "Overview"},
	{"name": "maps", "path": "/geos/maps", "component": "MapConsole", "permission": "geos:view", "nav_group": "Maps"},
	{"name": "geofences", "path": "/geos/geofences", "component": "GeofenceEditor", "permission": "geos:manage_geofences", "nav_group": "Geofencing"},
	{"name": "events", "path": "/geos/events", "component": "LocationEventMonitor", "permission": "geos:process_events", "nav_group": "Events"},
	{"name": "territories", "path": "/geos/territories", "component": "TerritoryManager", "permission": "geos:manage_geofences", "nav_group": "Planning"},
	{"name": "analytics", "path": "/geos/analytics", "component": "SpatialAnalytics", "permission": "geos:analyze", "nav_group": "Analysis"},
	{"name": "privacy", "path": "/geos/privacy", "component": "LocationPrivacy", "permission": "geos:admin", "nav_group": "Governance"},
	{"name": "agents", "path": "/geos/agents", "component": "LocationAgentPanel", "permission": "geos:admin", "nav_group": "Agents"},
	{"name": "audit", "path": "/geos/audit", "component": "LocationAuditTrail", "permission": "geos:audit", "nav_group": "Governance"},
	{"name": "settings", "path": "/geos/settings", "component": "GEOSSettings", "permission": "geos:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {
	"name": "geos_location_intelligence",
	"tokens": {
		"color.primary": "#214E34",
		"color.accent": "#2B6CB0",
		"color.success": "#2F855A",
		"color.warning": "#B7791F",
		"color.danger": "#C53030",
		"surface.canvas": "#F7F8FA",
		"surface.panel": "#FFFFFF",
		"text.primary": "#172033",
		"text.secondary": "#52606D",
		"border.radius": "8px",
		"density": "compact"
	},
	"components": {
		"map_console": {"icon": "map", "status_indicator": "location-pill", "risk_style": "privacy-band"},
		"geofence_editor": {"visual": "geometry-canvas", "highlight": "vertex-chip"},
		"event_monitor": {"visual": "event-stream", "status_style": "source-chip"},
		"territory_manager": {"visual": "region-grid", "status_style": "overlap-chip"},
		"agent_panel": {"icon": "bot", "status_style": "scope-chip"},
		"audit_timeline": {"icon": "list-todo", "status_style": "governance-chip"}
	}
}

STREAMING: dict[str, Any] = {
	"processor": "bytewax",
	"topic": "apg.geos.lifecycle",
	"state": ["event_sources", "geofences", "location_events", "territories", "analytics", "location_agents", "audit_events"],
	"events": [
		"event_source_registered",
		"geofence_created",
		"location_event_processed",
		"territory_created",
		"spatial_analysis_completed",
		"location_agent_registered",
		"geofence_state_changed",
	],
	"batch_mutation_guardrail": "batch_location_mutation_requires_bytewax",
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable GEOS capability contract."""
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "geos",
		"display_name": "Geo-Spatial Services",
		"provides": ["geofencing", "location_events", "spatial_analytics", "territory_management", "location_prediction", "location_agents"],
		"requires": ["pred", "aicr", "mdm"],
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"view_module": config["adapters"]["view_models"],
			"api_prefix": "/geos/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True
		},
		"theme": deepcopy(THEME),
		"streaming": deepcopy(STREAMING)
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate default GEOS governance rules."""
	matched: list[str] = []
	actions: list[dict[str, Any]] = []
	decision = "allow"
	for rule in RULES:
		if _matches(rule["condition"], context):
			matched.append(rule["name"])
			effect = rule["effect"]
			actions.append(effect)
			if effect["decision"] == "deny":
				decision = "deny"
			elif effect["decision"] == "require_review" and decision != "deny":
				decision = "require_review"
	return {"decision": decision, "matched_rules": matched, "actions": actions, "context": context}


def _matches(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if key.endswith("_lte"):
			actual = context.get(key[:-4])
			if not isinstance(actual, Number) or not actual <= expected:
				return False
		elif key.endswith("_lt"):
			actual = context.get(key[:-3])
			if not isinstance(actual, Number) or not actual < expected:
				return False
		elif key.endswith("_gt"):
			actual = context.get(key[:-3])
			if not isinstance(actual, Number) or not actual > expected:
				return False
		elif key.endswith("_gte"):
			actual = context.get(key[:-4])
			if not isinstance(actual, Number) or not actual >= expected:
				return False
		elif key.endswith("_ne"):
			if context.get(key[:-3]) == expected:
				return False
		elif context.get(key) != expected:
			return False
	return True


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
	for key, value in source.items():
		if isinstance(value, dict) and isinstance(target.get(key), dict):
			_deep_merge(target[key], value)
		else:
			target[key] = value
