"""Executable capability contract for APG Network Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "telecom_net"
CAPABILITY_NAME = "Network Management"
CAPABILITY_VERSION = "1.0.0"
NET_EVENT_STREAM = "apg.telecom.net.lifecycle"

SUPPORTED_FAULT_SEVERITIES = ["critical", "major", "minor", "warning", "informational"]
SUPPORTED_FAULT_CATEGORIES = ["hardware_failure", "software_fault", "link_down", "congestion", "power_failure", "configuration_error", "security_breach", "capacity_threshold", "clock_failure", "environmental"]
SUPPORTED_ALARM_STATUSES = ["raised", "acknowledged", "cleared", "suppressed", "correlated"]
SUPPORTED_PERFORMANCE_METRICS = ["availability", "latency", "throughput", "packet_loss", "jitter", "error_rate", "utilisation", "call_drop_rate", "handover_success_rate", "rach_success_rate"]
SUPPORTED_CONFIG_CHANGE_TYPES = ["parameter_change", "software_upgrade", "hardware_swap", "topology_change", "policy_update", "rollback", "emergency_change"]
SUPPORTED_CHANGE_STATUSES = ["planned", "approved", "in_progress", "completed", "failed", "rolled_back", "cancelled"]
SUPPORTED_SLA_TYPES = ["availability", "latency", "throughput", "jitter", "resolution_time", "custom"]
SUPPORTED_NOC_SHIFTS = ["morning", "afternoon", "night", "weekend"]
SUPPORTED_ESCALATION_LEVELS = ["tier1", "tier2", "tier3", "vendor", "management"]
SUPPORTED_NETWORK_DOMAINS = ["core", "metro", "access", "backhaul", "ims", "ocs", "data_centre", "enterprise"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["fault_analyst", "performance_analyst", "config_manager", "sla_monitor", "noc_operator"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"faults": {"supported_severities": SUPPORTED_FAULT_SEVERITIES, "supported_categories": SUPPORTED_FAULT_CATEGORIES, "supported_alarm_statuses": SUPPORTED_ALARM_STATUSES, "auto_correlation": True, "ttl_hours": 72},
	"performance": {"supported_metrics": SUPPORTED_PERFORMANCE_METRICS, "collection_interval_seconds": 300, "threshold_alerting": True, "trending_enabled": True},
	"configuration": {"supported_change_types": SUPPORTED_CONFIG_CHANGE_TYPES, "supported_statuses": SUPPORTED_CHANGE_STATUSES, "approval_required": True, "rollback_enabled": True, "change_freeze_enabled": True},
	"sla": {"supported_types": SUPPORTED_SLA_TYPES, "breach_alerting": True, "reporting_enabled": True, "penalty_calculation": True},
	"noc": {"supported_shifts": SUPPORTED_NOC_SHIFTS, "supported_escalation_levels": SUPPORTED_ESCALATION_LEVELS, "handover_notes_required": True},
	"domains": {"supported_domains": SUPPORTED_NETWORK_DOMAINS, "cross_domain_correlation": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "unapproved_config_change_denied": True, "alarm_suppression_requires_approval": True, "cross_tenant_access_denied": True},
	"observability": {"event_stream": NET_EVENT_STREAM, "stream_processor": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_faults": True, "enable_performance": True, "enable_configuration": True, "enable_sla": True, "enable_noc": True, "enable_agents": True},
	"theme": {"default_theme": "telecom_net_control", "allow_tenant_overrides": True},
}

PROVIDES = ["fault_management_workflow", "performance_management_workflow", "configuration_management_workflow", "sla_monitoring_workflow", "noc_operations_workflow", "alarm_correlation_workflow", "change_management_workflow", "net_agent_workflow"]
REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "moni", "mqeb", "schd"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/telecom-net/dashboard", "component": "NetDashboard", "permission": "telecom_net:view", "nav_group": "Overview"},
	{"name": "alarms", "path": "/telecom-net/alarms", "component": "NetAlarmConsole", "permission": "telecom_net:faults", "nav_group": "Fault Management"},
	{"name": "fault_tickets", "path": "/telecom-net/faults", "component": "NetFaultTicketQueue", "permission": "telecom_net:faults", "nav_group": "Fault Management"},
	{"name": "performance", "path": "/telecom-net/performance", "component": "NetPerformanceConsole", "permission": "telecom_net:performance", "nav_group": "Performance"},
	{"name": "config_changes", "path": "/telecom-net/config-changes", "component": "NetChangeConsole", "permission": "telecom_net:config", "nav_group": "Configuration"},
	{"name": "sla", "path": "/telecom-net/sla", "component": "NetSlaConsole", "permission": "telecom_net:sla", "nav_group": "SLA"},
	{"name": "noc_view", "path": "/telecom-net/noc", "component": "NetNocView", "permission": "telecom_net:noc", "nav_group": "NOC"},
	{"name": "topology_view", "path": "/telecom-net/topology", "component": "NetTopologyView", "permission": "telecom_net:view", "nav_group": "Overview"},
	{"name": "correlations", "path": "/telecom-net/correlations", "component": "NetCorrelationConsole", "permission": "telecom_net:faults", "nav_group": "Fault Management"},
	{"name": "escalations", "path": "/telecom-net/escalations", "component": "NetEscalationConsole", "permission": "telecom_net:escalations", "nav_group": "Operations"},
	{"name": "agents", "path": "/telecom-net/agents", "component": "NetAgentWorkbench", "permission": "telecom_net:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/telecom-net/settings", "component": "NetSettings", "permission": "telecom_net:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "telecom_net_control",
	"tokens": {"color.primary": "#1E40AF", "color.accent": "#0891B2", "color.success": "#15803D", "color.warning": "#B45309", "color.danger": "#B91C1C", "surface.canvas": "#0F172A", "surface.panel": "#1E293B", "text.primary": "#F1F5F9", "text.secondary": "#94A3B8", "border.radius": "6px", "density": "compact"},
	"components": {"alarms": {"icon": "bell", "status_indicator": "severity-chip"}, "fault_tickets": {"icon": "alert-octagon", "status_indicator": "fault-status-chip"}, "performance": {"icon": "activity", "status_indicator": "metric-chip"}, "config_changes": {"icon": "settings", "status_indicator": "change-status-chip"}, "sla": {"icon": "target", "status_indicator": "sla-type-chip"}, "noc_view": {"icon": "monitor", "status_indicator": "shift-chip"}, "correlations": {"icon": "git-merge", "status_indicator": "correlation-chip"}, "agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"}},
}

STREAMING = {"processor": "bytewax", "stream": NET_EVENT_STREAM, "key": "tenant_id", "events": ["alarm_raised", "alarm_cleared", "fault_ticket_opened", "fault_ticket_resolved", "performance_threshold_breached", "config_change_approved", "config_change_completed", "sla_breach_detected", "noc_escalation_triggered", "net_agent_registered"], "guardrails": ["net_batch_requires_bytewax", "privileged_net_agent_action_requires_human_approval", "unapproved_config_change_denied", "alarm_suppression_requires_approval", "cross_tenant_access_denied"]}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "net_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "net_policy_required", "required_action": "attach_net_policy"}},
	{"name": "fault_severity_supported", "condition": {"operation": "raise_alarm", "severity_supported": False}, "effect": {"decision": "deny", "reason": "fault_severity_not_supported", "required_action": "select_supported_severity"}},
	{"name": "fault_category_supported", "condition": {"operation": "raise_alarm", "category_supported": False}, "effect": {"decision": "deny", "reason": "fault_category_not_supported", "required_action": "select_supported_category"}},
	{"name": "alarm_ne_required", "condition": {"operation": "raise_alarm", "ne_present": False}, "effect": {"decision": "deny", "reason": "ne_reference_required", "required_action": "set_ne_reference"}},
	{"name": "alarm_status_supported", "condition": {"operation": "update_alarm_status", "alarm_status_supported": False}, "effect": {"decision": "deny", "reason": "alarm_status_not_supported", "required_action": "select_supported_alarm_status"}},
	{"name": "alarm_suppression_requires_approval", "condition": {"operation": "suppress_alarm", "approval_present": False}, "effect": {"decision": "deny", "reason": "alarm_suppression_approval_required", "required_action": "attach_suppression_approval"}},
	{"name": "performance_metric_supported", "condition": {"operation": "record_performance", "metric_type_supported": False}, "effect": {"decision": "deny", "reason": "performance_metric_not_supported", "required_action": "select_supported_metric"}},
	{"name": "config_change_type_supported", "condition": {"operation": "submit_config_change", "change_type_supported": False}, "effect": {"decision": "deny", "reason": "change_type_not_supported", "required_action": "select_supported_change_type"}},
	{"name": "config_change_approval_required", "condition": {"operation": "submit_config_change", "approval_present": False}, "effect": {"decision": "deny", "reason": "config_change_approval_required", "required_action": "attach_change_approval"}},
	{"name": "change_freeze_period_denied", "condition": {"operation": "submit_config_change", "in_freeze_period": True, "emergency_override_present": False}, "effect": {"decision": "deny", "reason": "change_freeze_period_active", "required_action": "obtain_emergency_override"}},
	{"name": "config_change_status_supported", "condition": {"operation": "update_change_status", "status_supported": False}, "effect": {"decision": "deny", "reason": "change_status_not_supported", "required_action": "select_supported_change_status"}},
	{"name": "sla_type_supported", "condition": {"operation": "record_sla", "sla_type_supported": False}, "effect": {"decision": "deny", "reason": "sla_type_not_supported", "required_action": "select_supported_sla_type"}},
	{"name": "noc_shift_supported", "condition": {"operation": "record_noc_handover", "shift_supported": False}, "effect": {"decision": "deny", "reason": "noc_shift_not_supported", "required_action": "select_supported_shift"}},
	{"name": "noc_handover_notes_required", "condition": {"operation": "record_noc_handover", "notes_present": False}, "effect": {"decision": "deny", "reason": "handover_notes_required", "required_action": "add_handover_notes"}},
	{"name": "escalation_level_supported", "condition": {"operation": "escalate_fault", "escalation_level_supported": False}, "effect": {"decision": "deny", "reason": "escalation_level_not_supported", "required_action": "select_supported_escalation_level"}},
	{"name": "unapproved_config_change_denied", "condition": {"operation": "net_agent_action", "unapproved_config_change_scope": True}, "effect": {"decision": "deny", "reason": "unapproved_config_change_scope_denied", "required_action": "remove_unapproved_config_change_scope"}},
	{"name": "cross_tenant_access_denied", "condition": {"operation": "net_agent_action", "cross_tenant_access_scope": True}, "effect": {"decision": "deny", "reason": "cross_tenant_access_denied", "required_action": "remove_cross_tenant_access_scope"}},
	{"name": "net_batch_requires_bytewax", "condition": {"operation": "net_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_net_batch_to_bytewax"}},
	{"name": "net_agent_runtime_supported", "condition": {"operation": "register_net_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "net_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "net_agent_role_supported", "condition": {"operation": "register_net_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "net_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "net_agent_name_required", "condition": {"operation": "register_net_agent", "agent_name_present": False}, "effect": {"decision": "deny", "reason": "net_agent_name_required", "required_action": "name_net_agent"}},
	{"name": "net_agent_scope_required", "condition": {"operation": "register_net_agent", "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "net_agent_scope_required", "required_action": "bound_net_agent_scope"}},
	{"name": "privileged_net_agent_action_requires_human_approval", "condition": {"operation": "net_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	return {"capability": CAPABILITY_ID, "name": CAPABILITY_NAME, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "provides": list(PROVIDES), "requires": list(REQUIRES), "configuration": configuration, "configuration_schema": {"type": "object", "required": list(configuration), "properties": {key: {"type": "object"} for key in configuration if key != "tenant_id"} | {"tenant_id": {"type": "string", "minLength": 1}}}, "rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "api_prefix": "/telecom-net/api/v1", "requires_theme": True, "view_module": "views.py", "template_roots": ["templates/", "static/"], "routes": deepcopy(UI_ROUTES)}, "theme": deepcopy(THEME), "streaming": deepcopy(STREAMING)}


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
