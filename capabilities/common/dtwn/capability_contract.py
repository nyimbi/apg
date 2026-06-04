"""Executable capability contract for APG Digital Twin Framework."""

from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Any

SUPPORTED_TWIN_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_TWIN_AGENT_ROLES = ["twin_designer", "telemetry_reviewer", "simulation_operator", "prediction_reviewer", "incident_reviewer"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"twins": {"twin_owner_required": True, "asset_identity_required": True, "topology_mapping_enabled": True, "state_versioning_enabled": True},
	"telemetry": {"authenticated_source_required": True, "iot_ingestion_required": True, "geospatial_context_enabled": True, "vision_signal_supported": True},
	"simulation": {"model_required": True, "calibration_evidence_required": True, "simulation_approval_required": True, "prediction_confidence_threshold": 0.75},
	"twin_agents": {
		"agent_assist_enabled": True,
		"agent_registration_required": True,
		"agent_runtime_required": True,
		"agent_scope_required": True,
		"agent_contribution_disclosure_required": True,
		"supported_runtimes": SUPPORTED_TWIN_AGENT_RUNTIMES,
		"allowed_roles": SUPPORTED_TWIN_AGENT_ROLES,
	},
	"governance": {"require_tenant_context": True, "audit_twin_changes": True, "high_risk_prediction_review_required": True, "model_drift_monitoring": True, "tenant_isolation_required": True, "state_change_reason_required": True, "batch_event_stream": "bytewax"},
	"observability": {"audit_required": True, "trace_required": True, "telemetry_metrics_required": True, "agent_activity_required": True, "event_stream": "bytewax"},
	"adapters": {"generated_app_runtime": "service.DtwnService", "api_helpers": "api.py", "view_models": "views.py", "event_stream": "bytewax", "prediction": "pred", "iot": "iotd", "geospatial": "geos", "vision": "cvsn", "anomaly": "anom", "edge": "edge", "machine": "mchn", "audit_sink": "audl"},
	"ui": {"enable_twin_console": True, "enable_model_library": True, "enable_simulation_lab": True, "enable_topology_view": True, "enable_agent_panel": True, "enable_audit": True, "enable_analytics": True},
	"theme": {"default_theme": "dtwn_digital_twin_ops", "allow_tenant_overrides": True}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": ["tenant_id", "twins", "telemetry", "simulation", "twin_agents", "governance", "observability", "adapters", "ui", "theme"],
	"properties": {key: {"type": "object"} for key in ["twins", "telemetry", "simulation", "twin_agents", "governance", "observability", "adapters", "ui", "theme"]} | {"tenant_id": {"type": "string", "minLength": 1}}
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All digital-twin operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "twin_requires_owner", "description": "Digital twins require an accountable owner.", "condition": {"operation": "create_twin", "twin_owner_assigned": False}, "effect": {"decision": "deny", "reason": "twin_owner_required", "required_action": "assign_twin_owner"}},
	{"name": "twin_requires_asset_identity", "description": "Digital twins require asset identity.", "condition": {"operation": "create_twin", "asset_identity_present": False}, "effect": {"decision": "deny", "reason": "asset_identity_required", "required_action": "attach_asset_identity"}},
	{"name": "simulation_model_requires_calibration", "description": "Simulation models require calibration evidence.", "condition": {"operation": "register_simulation_model", "calibration_evidence_present": False}, "effect": {"decision": "deny", "reason": "calibration_evidence_required", "required_action": "attach_calibration_evidence"}},
	{"name": "simulation_model_requires_confidence", "description": "Simulation models must meet confidence threshold.", "condition": {"operation": "register_simulation_model", "model_confidence_lt": 0.75}, "effect": {"decision": "deny", "reason": "prediction_confidence_threshold", "required_action": "raise_model_confidence"}},
	{"name": "simulation_requires_model", "description": "Simulations require an approved model.", "condition": {"operation": "run_simulation", "model_present": False}, "effect": {"decision": "deny", "reason": "simulation_model_required", "required_action": "attach_simulation_model"}},
	{"name": "telemetry_requires_authenticated_source", "description": "Telemetry requires authenticated source identity.", "condition": {"telemetry_source_authenticated": False}, "effect": {"decision": "deny", "reason": "telemetry_source_auth_required", "required_action": "authenticate_telemetry_source"}},
	{"name": "telemetry_requires_measurements", "description": "Telemetry ingestion requires measurements.", "condition": {"operation": "ingest_telemetry", "measurement_count_lte": 0}, "effect": {"decision": "deny", "reason": "telemetry_measurements_required", "required_action": "attach_telemetry_measurements"}},
	{"name": "simulation_requires_approval", "description": "Production simulation runs require approval.", "condition": {"operation": "run_production_simulation", "approval_recorded": False}, "effect": {"decision": "deny", "reason": "simulation_approval_required", "required_action": "record_simulation_approval"}},
	{"name": "high_risk_prediction_requires_review", "description": "High-risk twin predictions require review.", "condition": {"prediction_risk_score_gt": 0.8, "prediction_review_recorded": False}, "effect": {"decision": "require_review", "reason": "prediction_review_required", "required_action": "review_prediction"}},
	{"name": "twin_agent_requires_registration", "description": "AI twin agents must be registered.", "condition": {"twin_agent_present": True, "agent_registered": False}, "effect": {"decision": "deny", "reason": "twin_agent_registration_required", "required_action": "register_twin_agent"}},
	{"name": "twin_agent_runtime_supported", "description": "AI twin agents must use a supported runtime.", "condition": {"twin_agent_present": True, "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "twin_agent_runtime_not_supported", "required_action": "choose_supported_twin_agent_runtime"}},
	{"name": "twin_agent_role_supported", "description": "AI twin agents must use a supported role.", "condition": {"twin_agent_present": True, "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "twin_agent_role_not_supported", "required_action": "choose_supported_twin_agent_role"}},
	{"name": "twin_agent_requires_scope", "description": "AI twin agents require explicit scope.", "condition": {"twin_agent_present": True, "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "twin_agent_scope_required", "required_action": "set_twin_agent_scope"}},
	{"name": "twin_agent_requires_disclosure", "description": "AI twin-agent contributions require disclosure.", "condition": {"twin_agent_present": True, "agent_contribution_disclosed": False}, "effect": {"decision": "deny", "reason": "twin_agent_disclosure_required", "required_action": "disclose_twin_agent"}},
	{"name": "dtwn_state_change_requires_reason", "description": "Digital-twin lifecycle state changes require a reason.", "condition": {"state_change_requested": True, "state_change_reason_present": False}, "effect": {"decision": "deny", "reason": "dtwn_state_change_reason_required", "required_action": "record_state_change_reason"}},
	{"name": "dtwn_state_change_requires_audit", "description": "Digital-twin lifecycle state changes require audit evidence.", "condition": {"state_change_requested": True, "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "dtwn_audit_event_required", "required_action": "record_twin_audit_event"}},
	{"name": "cross_tenant_twin_access_denied", "description": "Digital-twin records may not cross tenant boundaries.", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_twin_access_denied", "required_action": "use_tenant_local_context"}},
	{"name": "batch_twin_mutation_requires_bytewax", "description": "Batch twin mutations must use Bytewax event streams.", "condition": {"operation": "batch_twin_mutation", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}},
	{"name": "write_requires_policy", "description": "Digital twin write operations require an explicit authorization policy.", "condition": {"operation_type": "write", "write_policy_present": False}, "effect": {"decision": "deny", "reason": "dtwn_write_policy_required", "required_action": "attach_write_policy"}},
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/dtwn/dashboard", "component": "DTWNDashboard", "permission": "dtwn:view", "nav_group": "Overview"},
	{"name": "twins", "path": "/dtwn/twins", "component": "TwinRegistry", "permission": "dtwn:manage_twins", "nav_group": "Twins"},
	{"name": "models", "path": "/dtwn/models", "component": "ModelLibrary", "permission": "dtwn:model", "nav_group": "Models"},
	{"name": "telemetry", "path": "/dtwn/telemetry", "component": "TelemetryFusion", "permission": "dtwn:view", "nav_group": "Signals"},
	{"name": "simulations", "path": "/dtwn/simulations", "component": "SimulationLab", "permission": "dtwn:simulate", "nav_group": "Simulations"},
	{"name": "predictions", "path": "/dtwn/predictions", "component": "TwinPredictions", "permission": "dtwn:view", "nav_group": "Intelligence"},
	{"name": "topology", "path": "/dtwn/topology", "component": "TwinTopology", "permission": "dtwn:view", "nav_group": "Twins"},
	{"name": "agents", "path": "/dtwn/agents", "component": "TwinAgentPanel", "permission": "dtwn:model", "nav_group": "Agents"},
	{"name": "audit", "path": "/dtwn/audit", "component": "TwinAuditTrail", "permission": "dtwn:audit", "nav_group": "Governance"},
	{"name": "analytics", "path": "/dtwn/analytics", "component": "TwinAnalytics", "permission": "dtwn:view", "nav_group": "Operations"},
	{"name": "settings", "path": "/dtwn/settings", "component": "DTWNSettings", "permission": "dtwn:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {
	"name": "dtwn_digital_twin_ops",
	"tokens": {"color.primary": "#28536B", "color.accent": "#38A169", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F7F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"},
	"components": {"twin_card": {"icon": "box", "status_indicator": "state-pill", "risk_style": "prediction-band"}, "topology_view": {"visual": "asset-graph", "highlight": "dependency-chip"}, "simulation_lab": {"visual": "scenario-timeline", "status_style": "approval-chip"}, "telemetry_panel": {"visual": "signal-grid", "status_style": "source-chip"}, "agent_panel": {"icon": "bot", "status_style": "scope-chip"}, "audit_timeline": {"icon": "list-checks", "status_style": "governance-chip"}}
}

STREAMING: dict[str, Any] = {
	"processor": "bytewax",
	"topic": "apg.dtwn.lifecycle",
	"state": ["twins", "models", "telemetry", "topology", "simulations", "predictions", "twin_agents", "audit_events"],
	"events": ["twin_created", "model_registered", "telemetry_ingested", "topology_linked", "simulation_completed", "prediction_recorded", "prediction_reviewed", "twin_status_changed", "twin_agent_registered"],
	"batch_mutation_guardrail": "batch_twin_mutation_requires_bytewax",
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {"capability": "dtwn", "display_name": "Digital Twin Framework", "provides": ["twin_registry", "simulation_models", "telemetry_fusion", "prediction_workflows", "asset_topology", "twin_agents"], "requires": ["pred", "iotd", "geos", "cvsn"], "configuration": config, "configuration_schema": CONFIGURATION_SCHEMA, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "view_module": config["adapters"]["view_models"], "api_prefix": "/dtwn/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True}, "theme": deepcopy(THEME), "streaming": deepcopy(STREAMING)}


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
