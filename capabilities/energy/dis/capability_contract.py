"""Executable capability contract for APG Distribution Network."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "energy_dis"
CAPABILITY_NAME = "Distribution Network"
CAPABILITY_VERSION = "1.0.0"
DIS_EVENT_STREAM = "apg.energy.dis.lifecycle"

SUPPORTED_NETWORK_ELEMENT_TYPES = ["feeder", "substation", "transformer", "bus", "line", "cable", "capacitor_bank", "recloser", "sectionalizer", "fuse", "switch", "meter_point", "distributed_generator", "load_point"]
SUPPORTED_VOLTAGE_LEVELS = ["lv_230v", "lv_415v", "mv_11kv", "mv_22kv", "mv_33kv", "hv_66kv", "hv_132kv"]
SUPPORTED_FAULT_TYPES = ["phase_to_ground", "phase_to_phase", "three_phase", "broken_conductor", "high_impedance", "overload", "voltage_violation", "equipment_failure"]
SUPPORTED_FAULT_STATUSES = ["detected", "isolated", "under_investigation", "crew_dispatched", "restoring", "restored", "closed"]
SUPPORTED_SWITCHING_OPERATIONS = ["open", "close", "lock_open", "lock_close", "tag_out", "tag_in"]
SUPPORTED_TOPOLOGY_STATUSES = ["energized", "de_energized", "partial", "isolated", "under_maintenance"]
SUPPORTED_OUTAGE_CAUSES = ["weather", "tree_contact", "equipment_failure", "vehicle_accident", "animal_contact", "overload", "planned_maintenance", "vandalism", "unknown"]
SUPPORTED_SCADA_PROTOCOLS = ["dnp3", "iec_61850", "modbus", "iec_104", "iccp", "opc_ua"]
SUPPORTED_RESTORATION_STRATEGIES = ["manual_switching", "auto_reclosing", "load_transfer", "fault_isolation", "sectionalizing", "partial_restoration"]
SUPPORTED_LOAD_BALANCING_MODES = ["manual", "automated", "optimization_based", "predictive"]
SUPPORTED_APPROVAL_STATUSES = ["pending", "approved", "rejected", "escalated"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["fault_detector", "restoration_planner", "topology_analyst", "scada_monitor", "load_optimizer"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"ui": {"enable_dashboard": True, "enable_topology": True, "enable_faults": True, "enable_switching": True, "enable_outages": True, "enable_scada": True, "enable_load_balancing": True},
	"theme": {"default_theme": "energy_dis_ops", "allow_tenant_overrides": True},
	"network": {"supported_element_types": SUPPORTED_NETWORK_ELEMENT_TYPES, "supported_voltage_levels": SUPPORTED_VOLTAGE_LEVELS, "topology_validation": True},
	"faults": {"supported_fault_types": SUPPORTED_FAULT_TYPES, "supported_statuses": SUPPORTED_FAULT_STATUSES, "auto_detect": True, "crew_dispatch_required": True},
	"switching": {"supported_operations": SUPPORTED_SWITCHING_OPERATIONS, "approval_required": True, "switching_order_required": True},
	"outages": {"supported_causes": SUPPORTED_OUTAGE_CAUSES, "supported_restoration_strategies": SUPPORTED_RESTORATION_STRATEGIES, "saidi_tracking": True, "saifi_tracking": True},
	"scada": {"supported_protocols": SUPPORTED_SCADA_PROTOCOLS, "polling_interval_seconds": 30, "real_time_enabled": True},
	"load_balancing": {"supported_modes": SUPPORTED_LOAD_BALANCING_MODES, "voltage_limits": {"min_pu": 0.95, "max_pu": 1.05}},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_switching": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "cross_tenant_denied": True, "unapproved_switching_denied": True},
	"observability": {"event_stream": DIS_EVENT_STREAM, "stream_processor": "bytewax"},
}

PROVIDES = [
	"network_topology_management",
	"fault_detection_and_isolation",
	"outage_restoration",
	"switching_order_management",
	"scada_integration",
	"load_balancing",
	"reliability_kpis",
	"distribution_reporting",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "moni", "schd", "mqeb", "geos"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/energy-dis/dashboard", "component": "DisDashboard", "permission": "energy_dis:view", "nav_group": "Overview"},
	{"name": "topology", "path": "/energy-dis/topology", "component": "NetworkTopology", "permission": "energy_dis:topology", "nav_group": "Network"},
	{"name": "elements", "path": "/energy-dis/elements", "component": "NetworkElements", "permission": "energy_dis:topology", "nav_group": "Network"},
	{"name": "faults", "path": "/energy-dis/faults", "component": "FaultManagement", "permission": "energy_dis:faults", "nav_group": "Operations"},
	{"name": "fault_detail", "path": "/energy-dis/faults/<id>", "component": "FaultDetail", "permission": "energy_dis:faults", "nav_group": "Operations"},
	{"name": "switching", "path": "/energy-dis/switching", "component": "SwitchingOrders", "permission": "energy_dis:switching", "nav_group": "Operations"},
	{"name": "outages", "path": "/energy-dis/outages", "component": "OutageManager", "permission": "energy_dis:outages", "nav_group": "Operations"},
	{"name": "scada", "path": "/energy-dis/scada", "component": "ScadaConsole", "permission": "energy_dis:scada", "nav_group": "Monitoring"},
	{"name": "load_balancing", "path": "/energy-dis/load-balancing", "component": "LoadBalancingConsole", "permission": "energy_dis:load_balancing", "nav_group": "Optimization"},
	{"name": "reliability", "path": "/energy-dis/reliability", "component": "ReliabilityKPIs", "permission": "energy_dis:reports", "nav_group": "Performance"},
	{"name": "reports", "path": "/energy-dis/reports", "component": "DistributionReports", "permission": "energy_dis:reports", "nav_group": "Reporting"},
	{"name": "agents", "path": "/energy-dis/agents", "component": "DisAgentWorkbench", "permission": "energy_dis:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/energy-dis/settings", "component": "DisSettings", "permission": "energy_dis:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "energy_dis_ops",
	"tokens": {
		"color.primary": "#0369A1",
		"color.accent": "#EA580C",
		"color.success": "#166534",
		"color.warning": "#CA8A04",
		"color.danger": "#B91C1C",
		"surface.canvas": "#F1F5F9",
		"surface.panel": "#FFFFFF",
		"text.primary": "#0F172A",
		"text.secondary": "#334155",
		"border.radius": "4px",
		"density": "compact",
	},
	"components": {
		"topology": {"icon": "git-branch", "status_indicator": "topology-status-chip"},
		"faults": {"icon": "alert-octagon", "status_indicator": "fault-severity-chip"},
		"switching": {"icon": "toggle-left", "status_indicator": "switching-status-chip"},
		"outages": {"icon": "zap-off", "status_indicator": "outage-cause-chip"},
		"scada": {"icon": "monitor", "status_indicator": "scada-protocol-chip"},
		"load_balancing": {"icon": "sliders", "status_indicator": "load-balance-chip"},
		"reliability": {"icon": "shield", "status_indicator": "reliability-chip"},
		"agents": {"icon": "cpu", "status_indicator": "agent-runtime-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": DIS_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"network_element_registered", "topology_updated", "fault_detected",
		"fault_isolated", "switching_order_created", "switching_order_approved",
		"switching_operation_executed", "outage_started", "outage_restored",
		"scada_reading_received", "load_balance_adjusted", "reliability_kpi_calculated",
	],
	"guardrails": [
		"unapproved_switching_denied",
		"cross_tenant_network_data_denied",
		"privileged_dis_agent_requires_human_approval",
		"live_network_switching_requires_approval",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "write_policy_required", "required_action": "attach_write_policy"}},
	{"name": "element_type_supported", "condition": {"operation": "register_element", "element_type_supported": False}, "effect": {"decision": "deny", "reason": "element_type_not_supported", "required_action": "select_supported_element_type"}},
	{"name": "voltage_level_supported", "condition": {"operation": "register_element", "voltage_level_supported": False}, "effect": {"decision": "deny", "reason": "voltage_level_not_supported", "required_action": "select_supported_voltage_level"}},
	{"name": "element_feeder_required", "condition": {"operation": "register_element", "feeder_present": False}, "effect": {"decision": "deny", "reason": "feeder_reference_required", "required_action": "assign_feeder_reference"}},
	{"name": "fault_type_supported", "condition": {"operation": "report_fault", "fault_type_supported": False}, "effect": {"decision": "deny", "reason": "fault_type_not_supported", "required_action": "select_supported_fault_type"}},
	{"name": "fault_element_exists", "condition": {"operation": "report_fault", "element_exists": False}, "effect": {"decision": "deny", "reason": "element_not_found", "required_action": "register_element_first"}},
	{"name": "fault_location_required", "condition": {"operation": "report_fault", "location_present": False}, "effect": {"decision": "deny", "reason": "fault_location_required", "required_action": "provide_fault_location"}},
	{"name": "switching_operation_supported", "condition": {"operation": "create_switching_order", "switching_op_supported": False}, "effect": {"decision": "deny", "reason": "switching_operation_not_supported", "required_action": "select_supported_switching_operation"}},
	{"name": "switching_approval_required", "condition": {"operation": "execute_switching", "approval_present": False}, "effect": {"decision": "deny", "reason": "switching_approval_required", "required_action": "obtain_switching_approval"}},
	{"name": "switching_order_required", "condition": {"operation": "execute_switching", "switching_order_present": False}, "effect": {"decision": "deny", "reason": "switching_order_required", "required_action": "create_switching_order_first"}},
	{"name": "outage_cause_supported", "condition": {"operation": "record_outage", "outage_cause_supported": False}, "effect": {"decision": "deny", "reason": "outage_cause_not_supported", "required_action": "select_supported_outage_cause"}},
	{"name": "outage_affected_customers_required", "condition": {"operation": "record_outage", "affected_customers_present": False}, "effect": {"decision": "deny", "reason": "affected_customers_count_required", "required_action": "provide_affected_customers_count"}},
	{"name": "scada_protocol_supported", "condition": {"operation": "configure_scada", "protocol_supported": False}, "effect": {"decision": "deny", "reason": "scada_protocol_not_supported", "required_action": "select_supported_scada_protocol"}},
	{"name": "load_balance_mode_supported", "condition": {"operation": "set_load_balance_mode", "mode_supported": False}, "effect": {"decision": "deny", "reason": "load_balance_mode_not_supported", "required_action": "select_supported_mode"}},
	{"name": "voltage_within_limits", "condition": {"operation": "load_balance_check", "voltage_within_limits": False}, "effect": {"decision": "deny", "reason": "voltage_outside_limits", "required_action": "adjust_load_to_restore_voltage"}},
	{"name": "cross_tenant_denied", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_access_denied", "required_action": "use_correct_tenant_context"}},
	{"name": "restoration_strategy_supported", "condition": {"operation": "initiate_restoration", "strategy_supported": False}, "effect": {"decision": "deny", "reason": "restoration_strategy_not_supported", "required_action": "select_supported_restoration_strategy"}},
	{"name": "live_network_switching_requires_approval", "condition": {"operation": "execute_switching", "network_live": True, "approval_present": False}, "effect": {"decision": "deny", "reason": "live_network_requires_approval", "required_action": "obtain_live_network_approval"}},
	{"name": "dis_agent_runtime_supported", "condition": {"operation": "register_dis_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "dis_agent_role_supported", "condition": {"operation": "register_dis_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "privileged_dis_agent_requires_human_approval", "condition": {"operation": "dis_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required_for_network_switching", "required_action": "record_human_approval"}},
	{"name": "fault_isolation_before_repair", "condition": {"operation": "dispatch_crew", "fault_isolated": False}, "effect": {"decision": "deny", "reason": "fault_must_be_isolated_before_crew_dispatch", "required_action": "isolate_fault_first"}},
	{"name": "scada_heartbeat_required", "condition": {"operation": "process_scada_reading", "heartbeat_valid": False}, "effect": {"decision": "deny", "reason": "scada_heartbeat_expired", "required_action": "reconnect_scada_session"}},
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
			"api_prefix": "/energy-dis/api/v1",
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
