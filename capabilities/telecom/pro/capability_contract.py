"""Executable capability contract for APG Service Provisioning."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "telecom_pro"
CAPABILITY_NAME = "Service Provisioning"
CAPABILITY_VERSION = "1.0.0"
PRO_EVENT_STREAM = "apg.telecom.pro.lifecycle"

SUPPORTED_WORKFLOW_TYPES = ["service_activation", "service_modification", "service_termination", "number_allocation", "sim_provisioning", "bandwidth_provisioning", "vpn_provisioning", "iot_provisioning", "enterprise_service", "bulk_provisioning"]
SUPPORTED_WORKFLOW_STATUSES = ["queued", "in_progress", "waiting_resource", "waiting_network", "waiting_confirmation", "completed", "failed", "rolled_back", "cancelled"]
SUPPORTED_RESOURCE_TYPES = ["msisdn", "imsi", "ip_address", "bandwidth", "vlan", "tunnel", "apn", "routing_table_entry", "qos_policy", "firewall_rule"]
SUPPORTED_NETWORK_ELEMENTS = ["hlr_hss", "msc_mme", "sgsn_sgw", "ggsn_pgw", "ocs", "pcrf", "smsc", "radius", "tacacs", "dns"]
SUPPORTED_CONFIG_PUSH_METHODS = ["cli_template", "netconf", "restconf", "snmp", "soap_api", "rest_api", "file_transfer"]
SUPPORTED_ACTIVATION_STATUSES = ["not_started", "in_progress", "activated", "partially_activated", "failed", "rolled_back"]
SUPPORTED_ROLLBACK_TRIGGERS = ["manual", "timeout", "network_error", "resource_conflict", "verification_failure", "customer_cancel"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["workflow_orchestrator", "resource_allocator", "config_pusher", "activation_verifier", "rollback_handler"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"workflows": {"supported_types": SUPPORTED_WORKFLOW_TYPES, "supported_statuses": SUPPORTED_WORKFLOW_STATUSES, "timeout_minutes": 60, "retry_enabled": True, "max_retries": 3},
	"resources": {"supported_types": SUPPORTED_RESOURCE_TYPES, "reservation_ttl_minutes": 30, "conflict_detection": True, "auto_release_on_failure": True},
	"network_elements": {"supported_elements": SUPPORTED_NETWORK_ELEMENTS, "health_check_before_push": True, "push_timeout_seconds": 120},
	"config_push": {"supported_methods": SUPPORTED_CONFIG_PUSH_METHODS, "dry_run_enabled": True, "rollback_on_failure": True, "verification_required": True},
	"activation": {"supported_statuses": SUPPORTED_ACTIVATION_STATUSES, "confirmation_required": True, "end_to_end_test": True},
	"rollback": {"supported_triggers": SUPPORTED_ROLLBACK_TRIGGERS, "auto_rollback_enabled": True, "rollback_timeout_minutes": 30},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "unapproved_bulk_provisioning_denied": True, "dry_run_bypass_denied": True, "cross_tenant_provisioning_denied": True},
	"observability": {"event_stream": PRO_EVENT_STREAM, "stream_processor": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_workflows": True, "enable_resources": True, "enable_config_push": True, "enable_activation": True, "enable_rollback": True, "enable_agents": True},
	"theme": {"default_theme": "telecom_pro_control", "allow_tenant_overrides": True},
}

PROVIDES = ["service_activation_workflow", "network_resource_allocation", "configuration_push_workflow", "activation_confirmation_workflow", "rollback_workflow", "bulk_provisioning_workflow", "pro_agent_workflow"]
REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "mqeb", "moni", "schd"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/telecom-pro/dashboard", "component": "ProDashboard", "permission": "telecom_pro:view", "nav_group": "Overview"},
	{"name": "workflows", "path": "/telecom-pro/workflows", "component": "ProWorkflowConsole", "permission": "telecom_pro:workflows", "nav_group": "Provisioning"},
	{"name": "workflow_detail", "path": "/telecom-pro/workflows/<id>", "component": "ProWorkflowDetail", "permission": "telecom_pro:workflows", "nav_group": "Provisioning"},
	{"name": "resources", "path": "/telecom-pro/resources", "component": "ProResourceConsole", "permission": "telecom_pro:resources", "nav_group": "Resources"},
	{"name": "config_push", "path": "/telecom-pro/config-push", "component": "ProConfigPushConsole", "permission": "telecom_pro:config_push", "nav_group": "Configuration"},
	{"name": "network_elements", "path": "/telecom-pro/network-elements", "component": "ProNetworkElementConsole", "permission": "telecom_pro:network_elements", "nav_group": "Configuration"},
	{"name": "activation", "path": "/telecom-pro/activation", "component": "ProActivationConsole", "permission": "telecom_pro:activation", "nav_group": "Provisioning"},
	{"name": "rollback", "path": "/telecom-pro/rollback", "component": "ProRollbackConsole", "permission": "telecom_pro:rollback", "nav_group": "Operations"},
	{"name": "bulk_provisioning", "path": "/telecom-pro/bulk", "component": "ProBulkProvisioningConsole", "permission": "telecom_pro:bulk", "nav_group": "Provisioning"},
	{"name": "verification", "path": "/telecom-pro/verification", "component": "ProVerificationConsole", "permission": "telecom_pro:activation", "nav_group": "Provisioning"},
	{"name": "agents", "path": "/telecom-pro/agents", "component": "ProAgentWorkbench", "permission": "telecom_pro:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/telecom-pro/settings", "component": "ProSettings", "permission": "telecom_pro:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "telecom_pro_control",
	"tokens": {"color.primary": "#0F766E", "color.accent": "#1D4ED8", "color.success": "#15803D", "color.warning": "#B45309", "color.danger": "#B91C1C", "surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF", "text.primary": "#111827", "text.secondary": "#4B5563", "border.radius": "8px", "density": "compact"},
	"components": {"workflows": {"icon": "git-branch", "status_indicator": "workflow-status-chip"}, "resources": {"icon": "package", "status_indicator": "resource-type-chip"}, "config_push": {"icon": "upload-cloud", "status_indicator": "push-method-chip"}, "network_elements": {"icon": "server", "status_indicator": "ne-chip"}, "activation": {"icon": "zap", "status_indicator": "activation-status-chip"}, "rollback": {"icon": "rotate-ccw", "status_indicator": "rollback-trigger-chip"}, "bulk_provisioning": {"icon": "layers", "status_indicator": "bulk-chip"}, "agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"}},
}

STREAMING = {"processor": "bytewax", "stream": PRO_EVENT_STREAM, "key": "tenant_id", "events": ["workflow_queued", "resource_reserved", "config_push_dispatched", "config_push_completed", "service_activated", "activation_confirmed", "workflow_failed", "rollback_triggered", "rollback_completed", "pro_agent_registered"], "guardrails": ["pro_batch_requires_bytewax", "privileged_pro_agent_action_requires_human_approval", "unapproved_bulk_provisioning_denied", "dry_run_bypass_denied", "cross_tenant_provisioning_denied"]}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "pro_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "pro_policy_required", "required_action": "attach_pro_policy"}},
	{"name": "workflow_type_supported", "condition": {"operation": "start_workflow", "workflow_type_supported": False}, "effect": {"decision": "deny", "reason": "workflow_type_not_supported", "required_action": "select_supported_workflow_type"}},
	{"name": "workflow_order_required", "condition": {"operation": "start_workflow", "order_reference_present": False}, "effect": {"decision": "deny", "reason": "order_reference_required", "required_action": "attach_order_reference"}},
	{"name": "workflow_status_supported", "condition": {"operation": "update_workflow_status", "workflow_status_supported": False}, "effect": {"decision": "deny", "reason": "workflow_status_not_supported", "required_action": "select_supported_workflow_status"}},
	{"name": "resource_type_supported", "condition": {"operation": "reserve_resource", "resource_type_supported": False}, "effect": {"decision": "deny", "reason": "resource_type_not_supported", "required_action": "select_supported_resource_type"}},
	{"name": "resource_conflict_check_required", "condition": {"operation": "reserve_resource", "conflict_checked": False}, "effect": {"decision": "deny", "reason": "resource_conflict_check_required", "required_action": "check_resource_conflicts"}},
	{"name": "config_push_method_supported", "condition": {"operation": "push_config", "push_method_supported": False}, "effect": {"decision": "deny", "reason": "config_push_method_not_supported", "required_action": "select_supported_push_method"}},
	{"name": "ne_health_check_required", "condition": {"operation": "push_config", "ne_health_checked": False}, "effect": {"decision": "deny", "reason": "ne_health_check_required", "required_action": "check_ne_health"}},
	{"name": "dry_run_bypass_denied", "condition": {"operation": "push_config", "dry_run_bypassed": True}, "effect": {"decision": "deny", "reason": "dry_run_bypass_denied", "required_action": "enable_dry_run"}},
	{"name": "activation_verification_required", "condition": {"operation": "confirm_activation", "verification_completed": False}, "effect": {"decision": "deny", "reason": "activation_verification_required", "required_action": "complete_activation_verification"}},
	{"name": "activation_status_supported", "condition": {"operation": "update_activation_status", "activation_status_supported": False}, "effect": {"decision": "deny", "reason": "activation_status_not_supported", "required_action": "select_supported_activation_status"}},
	{"name": "rollback_trigger_supported", "condition": {"operation": "trigger_rollback", "rollback_trigger_supported": False}, "effect": {"decision": "deny", "reason": "rollback_trigger_not_supported", "required_action": "select_supported_rollback_trigger"}},
	{"name": "bulk_provisioning_approval_required", "condition": {"operation": "start_bulk_provisioning", "approval_present": False}, "effect": {"decision": "deny", "reason": "bulk_provisioning_approval_required", "required_action": "attach_bulk_approval"}},
	{"name": "cross_tenant_provisioning_denied", "condition": {"operation": "pro_agent_action", "cross_tenant_provisioning_scope": True}, "effect": {"decision": "deny", "reason": "cross_tenant_provisioning_denied", "required_action": "remove_cross_tenant_provisioning_scope"}},
	{"name": "pro_batch_requires_bytewax", "condition": {"operation": "pro_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_pro_batch_to_bytewax"}},
	{"name": "pro_agent_runtime_supported", "condition": {"operation": "register_pro_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "pro_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "pro_agent_role_supported", "condition": {"operation": "register_pro_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "pro_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "pro_agent_name_required", "condition": {"operation": "register_pro_agent", "agent_name_present": False}, "effect": {"decision": "deny", "reason": "pro_agent_name_required", "required_action": "name_pro_agent"}},
	{"name": "pro_agent_scope_required", "condition": {"operation": "register_pro_agent", "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "pro_agent_scope_required", "required_action": "bound_pro_agent_scope"}},
	{"name": "privileged_pro_agent_action_requires_human_approval", "condition": {"operation": "pro_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	return {"capability": CAPABILITY_ID, "name": CAPABILITY_NAME, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "provides": list(PROVIDES), "requires": list(REQUIRES), "configuration": configuration, "configuration_schema": {"type": "object", "required": list(configuration), "properties": {key: {"type": "object"} for key in configuration if key != "tenant_id"} | {"tenant_id": {"type": "string", "minLength": 1}}}, "rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "api_prefix": "/telecom-pro/api/v1", "requires_theme": True, "view_module": "views.py", "template_roots": ["templates/", "static/"], "routes": deepcopy(UI_ROUTES)}, "theme": deepcopy(THEME), "streaming": deepcopy(STREAMING)}


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
