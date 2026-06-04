"""Executable capability contract for APG Grid Operations."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "energy_grd"
CAPABILITY_NAME = "Grid Operations"
CAPABILITY_VERSION = "1.0.0"
GRD_EVENT_STREAM = "apg.energy.grd.lifecycle"

SUPPORTED_GRID_AREAS = ["transmission", "sub_transmission", "distribution", "interconnection", "offshore", "islanded"]
SUPPORTED_STATE_ESTIMATOR_TYPES = ["weighted_least_squares", "extended_kalman_filter", "linear_state_estimation", "three_phase_se", "distribution_se"]
SUPPORTED_CONTINGENCY_TYPES = ["n_minus_1", "n_minus_2", "n_minus_1_1", "common_mode", "extreme_event", "cascading_failure", "voltage_collapse"]
SUPPORTED_CONTINGENCY_STATUSES = ["normal", "alert", "emergency", "extreme_emergency", "restorative"]
SUPPORTED_VOLTAGE_CONTROL_METHODS = ["avr", "tap_changer", "capacitor_switching", "statcom", "svc", "synchronous_condenser", "reactive_injection", "coordinated_voltage_control"]
SUPPORTED_FREQUENCY_CONTROL_METHODS = ["primary_frequency_response", "secondary_frequency_control", "agc", "ufls", "ofgr", "synthetic_inertia", "fast_frequency_response"]
SUPPORTED_MARKET_PRODUCTS = ["energy", "regulation_up", "regulation_down", "spinning_reserve", "non_spinning_reserve", "black_start", "reactive_power", "inertia", "fast_frequency_response"]
SUPPORTED_SETTLEMENT_STATUSES = ["preliminary", "initial", "final", "revised_final", "audited"]
SUPPORTED_ALARM_SEVERITIES = ["informational", "warning", "minor", "major", "critical", "emergency"]
SUPPORTED_ALARM_CATEGORIES = ["thermal_overload", "voltage_violation", "frequency_deviation", "protection_operation", "communication_failure", "equipment_failure", "cybersecurity", "weather_alert"]
SUPPORTED_EMS_FUNCTIONS = ["state_estimation", "contingency_analysis", "optimal_power_flow", "automatic_voltage_regulation", "load_frequency_control", "energy_management", "generation_scheduling"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["state_estimator", "contingency_analyst", "voltage_controller", "frequency_controller", "market_settlement_analyst"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"ui": {"enable_dashboard": True, "enable_state_estimation": True, "enable_contingency": True, "enable_voltage_control": True, "enable_frequency_control": True, "enable_market_settlement": True, "enable_alarms": True, "enable_ems": True},
	"theme": {"default_theme": "energy_grd_ops", "allow_tenant_overrides": True},
	"state_estimation": {"supported_types": SUPPORTED_STATE_ESTIMATOR_TYPES, "run_interval_seconds": 30, "convergence_threshold": 1e-4, "max_iterations": 50},
	"contingency": {"supported_types": SUPPORTED_CONTINGENCY_TYPES, "supported_statuses": SUPPORTED_CONTINGENCY_STATUSES, "auto_run_on_topology_change": True, "n_1_mandatory": True},
	"voltage_control": {"supported_methods": SUPPORTED_VOLTAGE_CONTROL_METHODS, "target_pu": 1.0, "tolerance_pu": 0.05, "auto_control_enabled": True},
	"frequency_control": {"supported_methods": SUPPORTED_FREQUENCY_CONTROL_METHODS, "nominal_hz": 50.0, "deadband_hz": 0.02, "ufls_threshold_hz": 49.0},
	"market": {"supported_products": SUPPORTED_MARKET_PRODUCTS, "supported_settlement_statuses": SUPPORTED_SETTLEMENT_STATUSES, "metered_data_required": True, "bid_offer_required": True},
	"alarms": {"supported_severities": SUPPORTED_ALARM_SEVERITIES, "supported_categories": SUPPORTED_ALARM_CATEGORIES, "auto_acknowledge_informational": True, "critical_requires_acknowledgement": True},
	"ems": {"supported_functions": SUPPORTED_EMS_FUNCTIONS, "real_time_enabled": True, "study_mode_enabled": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_control_actions": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "cross_tenant_denied": True, "unapproved_control_action_denied": True},
	"observability": {"event_stream": GRD_EVENT_STREAM, "stream_processor": "bytewax"},
}

PROVIDES = [
	"real_time_state_estimation",
	"contingency_analysis",
	"voltage_control",
	"frequency_control",
	"market_settlement",
	"grid_alarm_management",
	"ems_function_management",
	"grid_operational_reporting",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "moni", "comp", "mqeb", "schd"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/energy-grd/dashboard", "component": "GrdDashboard", "permission": "energy_grd:view", "nav_group": "Overview"},
	{"name": "state_estimation", "path": "/energy-grd/state-estimation", "component": "StateEstimationConsole", "permission": "energy_grd:state_estimation", "nav_group": "Real-Time"},
	{"name": "contingency", "path": "/energy-grd/contingency", "component": "ContingencyAnalysis", "permission": "energy_grd:contingency", "nav_group": "Analysis"},
	{"name": "contingency_detail", "path": "/energy-grd/contingency/<id>", "component": "ContingencyDetail", "permission": "energy_grd:contingency", "nav_group": "Analysis"},
	{"name": "voltage_control", "path": "/energy-grd/voltage-control", "component": "VoltageControlConsole", "permission": "energy_grd:voltage_control", "nav_group": "Control"},
	{"name": "frequency_control", "path": "/energy-grd/frequency-control", "component": "FrequencyControlConsole", "permission": "energy_grd:frequency_control", "nav_group": "Control"},
	{"name": "market_settlement", "path": "/energy-grd/market-settlement", "component": "MarketSettlementConsole", "permission": "energy_grd:market_settlement", "nav_group": "Market"},
	{"name": "settlement_detail", "path": "/energy-grd/market-settlement/<id>", "component": "SettlementDetail", "permission": "energy_grd:market_settlement", "nav_group": "Market"},
	{"name": "alarms", "path": "/energy-grd/alarms", "component": "GridAlarmConsole", "permission": "energy_grd:alarms", "nav_group": "Monitoring"},
	{"name": "ems", "path": "/energy-grd/ems", "component": "EmsConsole", "permission": "energy_grd:ems", "nav_group": "Systems"},
	{"name": "reports", "path": "/energy-grd/reports", "component": "GridReports", "permission": "energy_grd:reports", "nav_group": "Reporting"},
	{"name": "agents", "path": "/energy-grd/agents", "component": "GrdAgentWorkbench", "permission": "energy_grd:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/energy-grd/settings", "component": "GrdSettings", "permission": "energy_grd:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "energy_grd_ops",
	"tokens": {
		"color.primary": "#1E3A5F",
		"color.accent": "#E63946",
		"color.success": "#2D6A4F",
		"color.warning": "#E9C46A",
		"color.danger": "#E63946",
		"surface.canvas": "#EEF2F7",
		"surface.panel": "#FFFFFF",
		"text.primary": "#0D1B2A",
		"text.secondary": "#1E3A5F",
		"border.radius": "4px",
		"density": "compact",
	},
	"components": {
		"state_estimation": {"icon": "activity", "status_indicator": "se-convergence-chip"},
		"contingency": {"icon": "alert-circle", "status_indicator": "contingency-status-chip"},
		"voltage_control": {"icon": "zap", "status_indicator": "voltage-level-chip"},
		"frequency_control": {"icon": "radio", "status_indicator": "frequency-deviation-chip"},
		"market_settlement": {"icon": "dollar-sign", "status_indicator": "settlement-status-chip"},
		"alarms": {"icon": "bell", "status_indicator": "alarm-severity-chip"},
		"ems": {"icon": "server", "status_indicator": "ems-function-chip"},
		"agents": {"icon": "cpu", "status_indicator": "agent-runtime-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": GRD_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"state_estimation_completed", "contingency_violation_detected", "contingency_cleared",
		"voltage_control_action_taken", "frequency_control_action_taken",
		"market_settlement_preliminary", "market_settlement_final",
		"grid_alarm_raised", "grid_alarm_acknowledged", "grid_alarm_cleared",
		"ems_function_executed", "grd_agent_registered",
	],
	"guardrails": [
		"unapproved_control_action_denied",
		"cross_tenant_grid_data_denied",
		"privileged_grd_agent_requires_human_approval",
		"emergency_control_requires_acknowledgement",
		"market_settlement_metered_data_required",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "write_policy_required", "required_action": "attach_write_policy"}},
	{"name": "se_type_supported", "condition": {"operation": "run_state_estimation", "se_type_supported": False}, "effect": {"decision": "deny", "reason": "state_estimator_type_not_supported", "required_action": "select_supported_se_type"}},
	{"name": "se_network_model_required", "condition": {"operation": "run_state_estimation", "network_model_present": False}, "effect": {"decision": "deny", "reason": "network_model_required_for_state_estimation", "required_action": "load_network_model"}},
	{"name": "se_measurements_required", "condition": {"operation": "run_state_estimation", "measurements_present": False}, "effect": {"decision": "deny", "reason": "scada_measurements_required", "required_action": "acquire_scada_measurements"}},
	{"name": "contingency_type_supported", "condition": {"operation": "run_contingency", "contingency_type_supported": False}, "effect": {"decision": "deny", "reason": "contingency_type_not_supported", "required_action": "select_supported_contingency_type"}},
	{"name": "contingency_base_case_required", "condition": {"operation": "run_contingency", "base_case_converged": False}, "effect": {"decision": "deny", "reason": "converged_base_case_required_for_contingency", "required_action": "converge_base_case_first"}},
	{"name": "voltage_control_method_supported", "condition": {"operation": "apply_voltage_control", "control_method_supported": False}, "effect": {"decision": "deny", "reason": "voltage_control_method_not_supported", "required_action": "select_supported_voltage_control_method"}},
	{"name": "voltage_control_approval_required", "condition": {"operation": "apply_voltage_control", "approval_present": False}, "effect": {"decision": "deny", "reason": "voltage_control_action_requires_approval", "required_action": "obtain_control_approval"}},
	{"name": "frequency_control_method_supported", "condition": {"operation": "apply_frequency_control", "control_method_supported": False}, "effect": {"decision": "deny", "reason": "frequency_control_method_not_supported", "required_action": "select_supported_frequency_control_method"}},
	{"name": "frequency_ufls_threshold_valid", "condition": {"operation": "configure_ufls", "threshold_valid": False}, "effect": {"decision": "deny", "reason": "ufls_threshold_invalid", "required_action": "set_valid_ufls_threshold_hz"}},
	{"name": "market_product_supported", "condition": {"operation": "settle_market_interval", "product_supported": False}, "effect": {"decision": "deny", "reason": "market_product_not_supported", "required_action": "select_supported_market_product"}},
	{"name": "market_metered_data_required", "condition": {"operation": "settle_market_interval", "metered_data_present": False}, "effect": {"decision": "deny", "reason": "metered_data_required_for_settlement", "required_action": "provide_metered_data"}},
	{"name": "market_bid_offer_required", "condition": {"operation": "settle_market_interval", "bid_offer_present": False}, "effect": {"decision": "deny", "reason": "bid_offer_data_required_for_settlement", "required_action": "provide_bid_offer_data"}},
	{"name": "settlement_status_valid", "condition": {"operation": "update_settlement_status", "settlement_status_supported": False}, "effect": {"decision": "deny", "reason": "settlement_status_not_supported", "required_action": "select_supported_settlement_status"}},
	{"name": "alarm_severity_supported", "condition": {"operation": "raise_alarm", "alarm_severity_supported": False}, "effect": {"decision": "deny", "reason": "alarm_severity_not_supported", "required_action": "select_supported_alarm_severity"}},
	{"name": "alarm_category_supported", "condition": {"operation": "raise_alarm", "alarm_category_supported": False}, "effect": {"decision": "deny", "reason": "alarm_category_not_supported", "required_action": "select_supported_alarm_category"}},
	{"name": "critical_alarm_acknowledgement_required", "condition": {"operation": "clear_alarm", "alarm_severity": "critical", "acknowledged": False}, "effect": {"decision": "deny", "reason": "critical_alarm_must_be_acknowledged_before_clearing", "required_action": "acknowledge_alarm_first"}},
	{"name": "ems_function_supported", "condition": {"operation": "execute_ems_function", "ems_function_supported": False}, "effect": {"decision": "deny", "reason": "ems_function_not_supported", "required_action": "select_supported_ems_function"}},
	{"name": "cross_tenant_denied", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_access_denied", "required_action": "use_correct_tenant_context"}},
	{"name": "grd_agent_runtime_supported", "condition": {"operation": "register_grd_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "grd_agent_role_supported", "condition": {"operation": "register_grd_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "privileged_grd_agent_requires_human_approval", "condition": {"operation": "grd_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required_for_grid_control_action", "required_action": "record_human_approval"}},
	{"name": "emergency_control_requires_acknowledgement", "condition": {"operation": "grd_agent_action", "system_in_emergency": True, "alarm_acknowledged": False}, "effect": {"decision": "deny", "reason": "emergency_alarm_must_be_acknowledged", "required_action": "acknowledge_emergency_alarm"}},
	{"name": "n1_contingency_mandatory", "condition": {"operation": "skip_n1_contingency", "n1_bypass_allowed": False}, "effect": {"decision": "deny", "reason": "n_minus_1_contingency_analysis_is_mandatory", "required_action": "run_n1_analysis_first"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	return {
		"capability": CAPABILITY_ID,
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
		"ui": {
			"shell": "apg_python",
			"api_prefix": "/energy-grd/api/v1",
			"requires_theme": True,
			"template_roots": ["templates/", "static/"],
			"routes": deepcopy(UI_ROUTES),
		},
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
