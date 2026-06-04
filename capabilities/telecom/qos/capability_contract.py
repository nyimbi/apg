"""Executable capability contract for APG Quality of Service."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "telecom_qos"
CAPABILITY_NAME = "Quality of Service"
CAPABILITY_VERSION = "1.0.0"
QOS_EVENT_STREAM = "apg.telecom.qos.lifecycle"

SUPPORTED_QOS_CLASSES = ["conversational", "streaming", "interactive", "background", "best_effort", "mission_critical", "iot_low_power", "v2x"]
SUPPORTED_TRAFFIC_TYPES = ["voice", "video_streaming", "video_conferencing", "gaming", "browsing", "bulk_data", "iot_telemetry", "signalling", "oam"]
SUPPORTED_POLICY_TYPES = ["bearer_qos", "apn_ambr", "ue_ambr", "gbr_bearer", "non_gbr_bearer", "traffic_shaping", "traffic_policing", "qos_marking", "dscp_remarking"]
SUPPORTED_ENFORCEMENT_STATUSES = ["active", "inactive", "suspended", "overridden", "degraded", "conflict"]
SUPPORTED_DEGRADATION_CAUSES = ["congestion", "interference", "hardware_fault", "software_bug", "misconfiguration", "overloading", "handover_failure", "weather_impact", "power_issue"]
SUPPORTED_SLA_PARAMETERS = ["max_latency_ms", "min_throughput_mbps", "max_packet_loss_pct", "max_jitter_ms", "min_availability_pct", "max_mos_degradation"]
SUPPORTED_REMEDIATION_TYPES = ["auto_optimise", "traffic_steering", "load_balancing", "bearer_reestablishment", "parameter_tuning", "escalate_noc", "capacity_request"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["qos_policy_manager", "traffic_analyst", "sla_enforcer", "degradation_detector", "root_cause_analyst"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"policies": {"supported_types": SUPPORTED_POLICY_TYPES, "supported_qos_classes": SUPPORTED_QOS_CLASSES, "approval_required": True, "conflict_detection": True},
	"traffic": {"supported_types": SUPPORTED_TRAFFIC_TYPES, "deep_packet_inspection": True, "flow_tracking": True, "classification_required": True},
	"enforcement": {"supported_statuses": SUPPORTED_ENFORCEMENT_STATUSES, "real_time_adjustment": True, "pcrf_integration": True},
	"sla": {"supported_parameters": SUPPORTED_SLA_PARAMETERS, "breach_alerting": True, "measurement_interval_seconds": 60},
	"degradation": {"supported_causes": SUPPORTED_DEGRADATION_CAUSES, "auto_detection": True, "confidence_threshold": 0.85},
	"remediation": {"supported_types": SUPPORTED_REMEDIATION_TYPES, "auto_remediation_enabled": True, "human_approval_for_disruptive": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "unapproved_policy_change_denied": True, "cross_tenant_qos_denied": True, "qos_downgrade_requires_approval": True},
	"observability": {"event_stream": QOS_EVENT_STREAM, "stream_processor": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_policies": True, "enable_traffic": True, "enable_enforcement": True, "enable_sla": True, "enable_degradation": True, "enable_remediation": True, "enable_agents": True},
	"theme": {"default_theme": "telecom_qos_control", "allow_tenant_overrides": True},
}

PROVIDES = ["qos_policy_management_workflow", "traffic_prioritisation_workflow", "sla_enforcement_workflow", "degradation_detection_workflow", "root_cause_analysis_workflow", "auto_remediation_workflow", "qos_reporting_workflow", "qos_agent_workflow"]
REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "moni", "mqeb", "wflo"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/telecom-qos/dashboard", "component": "QosDashboard", "permission": "telecom_qos:view", "nav_group": "Overview"},
	{"name": "policies", "path": "/telecom-qos/policies", "component": "QosPolicyConsole", "permission": "telecom_qos:policies", "nav_group": "Policy"},
	{"name": "policy_detail", "path": "/telecom-qos/policies/<id>", "component": "QosPolicyDetail", "permission": "telecom_qos:policies", "nav_group": "Policy"},
	{"name": "traffic", "path": "/telecom-qos/traffic", "component": "QosTrafficConsole", "permission": "telecom_qos:traffic", "nav_group": "Traffic"},
	{"name": "enforcement", "path": "/telecom-qos/enforcement", "component": "QosEnforcementConsole", "permission": "telecom_qos:enforcement", "nav_group": "Operations"},
	{"name": "sla_monitoring", "path": "/telecom-qos/sla", "component": "QosSlaConsole", "permission": "telecom_qos:sla", "nav_group": "SLA"},
	{"name": "degradation", "path": "/telecom-qos/degradation", "component": "QosDegradationConsole", "permission": "telecom_qos:degradation", "nav_group": "Monitoring"},
	{"name": "root_cause", "path": "/telecom-qos/root-cause", "component": "QosRootCauseConsole", "permission": "telecom_qos:degradation", "nav_group": "Monitoring"},
	{"name": "remediation", "path": "/telecom-qos/remediation", "component": "QosRemediationConsole", "permission": "telecom_qos:remediation", "nav_group": "Operations"},
	{"name": "reports", "path": "/telecom-qos/reports", "component": "QosReportConsole", "permission": "telecom_qos:reports", "nav_group": "Reporting"},
	{"name": "agents", "path": "/telecom-qos/agents", "component": "QosAgentWorkbench", "permission": "telecom_qos:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/telecom-qos/settings", "component": "QosSettings", "permission": "telecom_qos:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "telecom_qos_control",
	"tokens": {"color.primary": "#0891B2", "color.accent": "#7C3AED", "color.success": "#15803D", "color.warning": "#B45309", "color.danger": "#B91C1C", "surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF", "text.primary": "#111827", "text.secondary": "#4B5563", "border.radius": "8px", "density": "compact"},
	"components": {"policies": {"icon": "sliders", "status_indicator": "policy-type-chip"}, "traffic": {"icon": "radio", "status_indicator": "traffic-type-chip"}, "enforcement": {"icon": "shield", "status_indicator": "enforcement-status-chip"}, "sla_monitoring": {"icon": "target", "status_indicator": "sla-param-chip"}, "degradation": {"icon": "trending-down", "status_indicator": "degradation-cause-chip"}, "root_cause": {"icon": "search", "status_indicator": "rca-chip"}, "remediation": {"icon": "tool", "status_indicator": "remediation-type-chip"}, "agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"}},
}

STREAMING = {"processor": "bytewax", "stream": QOS_EVENT_STREAM, "key": "tenant_id", "events": ["qos_policy_activated", "qos_policy_changed", "sla_breach_detected", "degradation_detected", "root_cause_identified", "remediation_triggered", "remediation_completed", "traffic_anomaly_detected", "qos_agent_registered"], "guardrails": ["qos_batch_requires_bytewax", "privileged_qos_agent_action_requires_human_approval", "unapproved_policy_change_denied", "qos_downgrade_requires_approval", "cross_tenant_qos_denied"]}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "qos_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "qos_policy_required", "required_action": "attach_qos_policy"}},
	{"name": "qos_policy_type_supported", "condition": {"operation": "create_qos_policy", "policy_type_supported": False}, "effect": {"decision": "deny", "reason": "qos_policy_type_not_supported", "required_action": "select_supported_policy_type"}},
	{"name": "qos_class_supported", "condition": {"operation": "create_qos_policy", "qos_class_supported": False}, "effect": {"decision": "deny", "reason": "qos_class_not_supported", "required_action": "select_supported_qos_class"}},
	{"name": "qos_policy_approval_required", "condition": {"operation": "create_qos_policy", "approval_present": False}, "effect": {"decision": "deny", "reason": "qos_policy_approval_required", "required_action": "attach_policy_approval"}},
	{"name": "qos_conflict_check_required", "condition": {"operation": "create_qos_policy", "conflict_checked": False}, "effect": {"decision": "deny", "reason": "qos_conflict_check_required", "required_action": "check_policy_conflicts"}},
	{"name": "traffic_type_supported", "condition": {"operation": "classify_traffic", "traffic_type_supported": False}, "effect": {"decision": "deny", "reason": "traffic_type_not_supported", "required_action": "select_supported_traffic_type"}},
	{"name": "traffic_classification_required", "condition": {"operation": "classify_traffic", "classification_present": False}, "effect": {"decision": "deny", "reason": "traffic_classification_required", "required_action": "classify_traffic_type"}},
	{"name": "enforcement_status_supported", "condition": {"operation": "update_enforcement_status", "enforcement_status_supported": False}, "effect": {"decision": "deny", "reason": "enforcement_status_not_supported", "required_action": "select_supported_enforcement_status"}},
	{"name": "sla_parameter_supported", "condition": {"operation": "record_sla_measurement", "sla_parameter_supported": False}, "effect": {"decision": "deny", "reason": "sla_parameter_not_supported", "required_action": "select_supported_sla_parameter"}},
	{"name": "degradation_cause_supported", "condition": {"operation": "record_degradation", "degradation_cause_supported": False}, "effect": {"decision": "deny", "reason": "degradation_cause_not_supported", "required_action": "select_supported_degradation_cause"}},
	{"name": "degradation_confidence_required", "condition": {"operation": "record_degradation", "confidence_present": False}, "effect": {"decision": "deny", "reason": "degradation_confidence_required", "required_action": "set_degradation_confidence"}},
	{"name": "remediation_type_supported", "condition": {"operation": "trigger_remediation", "remediation_type_supported": False}, "effect": {"decision": "deny", "reason": "remediation_type_not_supported", "required_action": "select_supported_remediation_type"}},
	{"name": "disruptive_remediation_requires_approval", "condition": {"operation": "trigger_remediation", "is_disruptive": True, "approval_present": False}, "effect": {"decision": "deny", "reason": "disruptive_remediation_approval_required", "required_action": "attach_remediation_approval"}},
	{"name": "qos_downgrade_requires_approval", "condition": {"operation": "change_qos_policy", "is_downgrade": True, "approval_present": False}, "effect": {"decision": "deny", "reason": "qos_downgrade_approval_required", "required_action": "attach_downgrade_approval"}},
	{"name": "cross_tenant_qos_denied", "condition": {"operation": "qos_agent_action", "cross_tenant_qos_scope": True}, "effect": {"decision": "deny", "reason": "cross_tenant_qos_denied", "required_action": "remove_cross_tenant_qos_scope"}},
	{"name": "unapproved_policy_change_denied", "condition": {"operation": "qos_agent_action", "unapproved_policy_change_scope": True}, "effect": {"decision": "deny", "reason": "unapproved_policy_change_denied", "required_action": "remove_unapproved_policy_change_scope"}},
	{"name": "qos_batch_requires_bytewax", "condition": {"operation": "qos_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_qos_batch_to_bytewax"}},
	{"name": "qos_agent_runtime_supported", "condition": {"operation": "register_qos_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "qos_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "qos_agent_role_supported", "condition": {"operation": "register_qos_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "qos_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "qos_agent_name_required", "condition": {"operation": "register_qos_agent", "agent_name_present": False}, "effect": {"decision": "deny", "reason": "qos_agent_name_required", "required_action": "name_qos_agent"}},
	{"name": "qos_agent_scope_required", "condition": {"operation": "register_qos_agent", "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "qos_agent_scope_required", "required_action": "bound_qos_agent_scope"}},
	{"name": "privileged_qos_agent_action_requires_human_approval", "condition": {"operation": "qos_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	return {"capability": CAPABILITY_ID, "name": CAPABILITY_NAME, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "provides": list(PROVIDES), "requires": list(REQUIRES), "configuration": configuration, "configuration_schema": {"type": "object", "required": list(configuration), "properties": {key: {"type": "object"} for key in configuration if key != "tenant_id"} | {"tenant_id": {"type": "string", "minLength": 1}}}, "rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "api_prefix": "/telecom-qos/api/v1", "requires_theme": True, "view_module": "views.py", "template_roots": ["templates/", "static/"], "routes": deepcopy(UI_ROUTES)}, "theme": deepcopy(THEME), "streaming": deepcopy(STREAMING)}


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
