"""Executable capability contract for APG Digital Twin Framework."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"twins": {"twin_owner_required": True, "asset_identity_required": True, "topology_mapping_enabled": True, "state_versioning_enabled": True},
	"telemetry": {"authenticated_source_required": True, "iot_ingestion_required": True, "geospatial_context_enabled": True, "vision_signal_supported": True},
	"simulation": {"model_required": True, "calibration_evidence_required": True, "simulation_approval_required": True, "prediction_confidence_threshold": 0.75},
	"governance": {"require_tenant_context": True, "audit_twin_changes": True, "high_risk_prediction_review_required": True, "model_drift_monitoring": True},
	"ui": {"enable_twin_console": True, "enable_model_library": True, "enable_simulation_lab": True, "enable_topology_view": True},
	"theme": {"default_theme": "dtwn_digital_twin_ops", "allow_tenant_overrides": True}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": ["tenant_id", "twins", "telemetry", "simulation", "governance", "ui", "theme"],
	"properties": {key: {"type": "object"} for key in ["twins", "telemetry", "simulation", "governance", "ui", "theme"]} | {"tenant_id": {"type": "string", "minLength": 1}}
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All digital-twin operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "twin_requires_owner", "description": "Digital twins require an accountable owner.", "condition": {"operation": "create_twin", "twin_owner_assigned": False}, "effect": {"decision": "deny", "reason": "twin_owner_required", "required_action": "assign_twin_owner"}},
	{"name": "simulation_requires_model", "description": "Simulations require an approved model.", "condition": {"operation": "run_simulation", "model_present": False}, "effect": {"decision": "deny", "reason": "simulation_model_required", "required_action": "attach_simulation_model"}},
	{"name": "telemetry_requires_authenticated_source", "description": "Telemetry requires authenticated source identity.", "condition": {"telemetry_source_authenticated": False}, "effect": {"decision": "deny", "reason": "telemetry_source_auth_required", "required_action": "authenticate_telemetry_source"}},
	{"name": "simulation_requires_approval", "description": "Production simulation runs require approval.", "condition": {"operation": "run_production_simulation", "approval_recorded": False}, "effect": {"decision": "deny", "reason": "simulation_approval_required", "required_action": "record_simulation_approval"}},
	{"name": "high_risk_prediction_requires_review", "description": "High-risk twin predictions require review.", "condition": {"prediction_risk_score_gt": 0.8, "prediction_review_recorded": False}, "effect": {"decision": "require_review", "reason": "prediction_review_required", "required_action": "review_prediction"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/dtwn/dashboard", "component": "DTWNDashboard", "permission": "dtwn:view", "nav_group": "Overview"},
	{"name": "twins", "path": "/dtwn/twins", "component": "TwinRegistry", "permission": "dtwn:manage_twins", "nav_group": "Twins"},
	{"name": "models", "path": "/dtwn/models", "component": "ModelLibrary", "permission": "dtwn:model", "nav_group": "Models"},
	{"name": "telemetry", "path": "/dtwn/telemetry", "component": "TelemetryFusion", "permission": "dtwn:view", "nav_group": "Signals"},
	{"name": "simulations", "path": "/dtwn/simulations", "component": "SimulationLab", "permission": "dtwn:simulate", "nav_group": "Simulations"},
	{"name": "predictions", "path": "/dtwn/predictions", "component": "TwinPredictions", "permission": "dtwn:view", "nav_group": "Intelligence"},
	{"name": "topology", "path": "/dtwn/topology", "component": "TwinTopology", "permission": "dtwn:view", "nav_group": "Twins"},
	{"name": "settings", "path": "/dtwn/settings", "component": "DTWNSettings", "permission": "dtwn:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {
	"name": "dtwn_digital_twin_ops",
	"tokens": {"color.primary": "#28536B", "color.accent": "#38A169", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F7F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"},
	"components": {"twin_card": {"icon": "box", "status_indicator": "state-pill", "risk_style": "prediction-band"}, "topology_view": {"visual": "asset-graph", "highlight": "dependency-chip"}, "simulation_lab": {"visual": "scenario-timeline", "status_style": "approval-chip"}, "telemetry_panel": {"visual": "signal-grid", "status_style": "source-chip"}}
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {"capability": "dtwn", "display_name": "Digital Twin Framework", "configuration": config, "configuration_schema": CONFIGURATION_SCHEMA, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "view_module": "views.py", "api_prefix": "/dtwn/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True}, "theme": deepcopy(THEME)}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
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
		if key.endswith("_lt"):
			if not context.get(key[:-3], 0) < expected:
				return False
		elif key.endswith("_gt"):
			if not context.get(key[:-3], 0) > expected:
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
