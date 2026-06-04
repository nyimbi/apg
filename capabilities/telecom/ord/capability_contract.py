"""Executable capability contract for APG Order Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "telecom_ord"
CAPABILITY_NAME = "Order Management"
CAPABILITY_VERSION = "1.0.0"
ORD_EVENT_STREAM = "apg.telecom.ord.lifecycle"

SUPPORTED_ORDER_TYPES = ["new_service", "change_service", "terminate_service", "number_portability", "sim_swap", "device_upgrade", "plan_change", "bulk_enterprise", "wholesale_activation", "mvno_onboarding"]
SUPPORTED_ORDER_STATUSES = ["submitted", "validated", "decomposed", "in_progress", "pending_customer", "fallout", "completed", "cancelled", "rejected"]
SUPPORTED_FALLOUT_CATEGORIES = ["validation_failure", "credit_failure", "provisioning_failure", "inventory_exhaustion", "network_error", "customer_data_error", "billing_failure", "regulatory_block", "duplicate_order"]
SUPPORTED_DECOMPOSITION_STATUSES = ["not_started", "in_progress", "completed", "failed"]
SUPPORTED_TASK_TYPES = ["customer_verification", "kyc_check", "credit_check", "inventory_reservation", "number_allocation", "sim_provisioning", "network_provisioning", "billing_setup", "notification_dispatch", "service_activation"]
SUPPORTED_CHANNEL_TYPES = ["retail_store", "web_self_service", "mobile_app", "call_centre", "direct_sales", "partner_api", "dealer", "bulk_upload"]
SUPPORTED_PRIORITY_LEVELS = ["low", "normal", "high", "urgent", "emergency"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["order_validator", "order_decomposer", "fallout_handler", "provisioning_orchestrator", "order_tracker"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"orders": {"supported_types": SUPPORTED_ORDER_TYPES, "supported_statuses": SUPPORTED_ORDER_STATUSES, "supported_channels": SUPPORTED_CHANNEL_TYPES, "supported_priorities": SUPPORTED_PRIORITY_LEVELS, "validation_required": True, "sla_hours": {"low": 72, "normal": 24, "high": 4, "urgent": 2, "emergency": 1}},
	"decomposition": {"supported_statuses": SUPPORTED_DECOMPOSITION_STATUSES, "supported_task_types": SUPPORTED_TASK_TYPES, "parallel_execution": True, "dependency_tracking": True},
	"fallout": {"supported_categories": SUPPORTED_FALLOUT_CATEGORIES, "auto_retry_enabled": True, "max_retries": 3, "escalation_threshold_minutes": 30},
	"provisioning": {"orchestration_enabled": True, "rollback_on_failure": True, "confirmation_required": True},
	"tracking": {"real_time_updates": True, "customer_notifications": True, "partner_webhooks": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "duplicate_order_denied": True, "unapproved_bulk_order_denied": True, "cross_tenant_order_denied": True},
	"observability": {"event_stream": ORD_EVENT_STREAM, "stream_processor": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_orders": True, "enable_decomposition": True, "enable_fallout": True, "enable_provisioning": True, "enable_tracking": True, "enable_agents": True},
	"theme": {"default_theme": "telecom_ord_control", "allow_tenant_overrides": True},
}

PROVIDES = ["order_capture_workflow", "order_validation_workflow", "order_decomposition_workflow", "provisioning_orchestration_workflow", "fallout_management_workflow", "order_tracking_workflow", "number_portability_workflow", "ord_agent_workflow"]
REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "mqeb", "schd", "comp"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/telecom-ord/dashboard", "component": "OrdDashboard", "permission": "telecom_ord:view", "nav_group": "Overview"},
	{"name": "orders", "path": "/telecom-ord/orders", "component": "OrdOrderConsole", "permission": "telecom_ord:orders", "nav_group": "Orders"},
	{"name": "order_detail", "path": "/telecom-ord/orders/<id>", "component": "OrdOrderDetail", "permission": "telecom_ord:orders", "nav_group": "Orders"},
	{"name": "decomposition", "path": "/telecom-ord/decomposition", "component": "OrdDecompositionConsole", "permission": "telecom_ord:decomposition", "nav_group": "Processing"},
	{"name": "tasks", "path": "/telecom-ord/tasks", "component": "OrdTaskQueue", "permission": "telecom_ord:tasks", "nav_group": "Processing"},
	{"name": "fallout", "path": "/telecom-ord/fallout", "component": "OrdFalloutConsole", "permission": "telecom_ord:fallout", "nav_group": "Operations"},
	{"name": "provisioning", "path": "/telecom-ord/provisioning", "component": "OrdProvisioningConsole", "permission": "telecom_ord:provisioning", "nav_group": "Processing"},
	{"name": "portability", "path": "/telecom-ord/portability", "component": "OrdPortabilityConsole", "permission": "telecom_ord:portability", "nav_group": "Special Orders"},
	{"name": "bulk_orders", "path": "/telecom-ord/bulk", "component": "OrdBulkOrderConsole", "permission": "telecom_ord:bulk", "nav_group": "Special Orders"},
	{"name": "tracking", "path": "/telecom-ord/tracking", "component": "OrdTrackingConsole", "permission": "telecom_ord:view", "nav_group": "Overview"},
	{"name": "agents", "path": "/telecom-ord/agents", "component": "OrdAgentWorkbench", "permission": "telecom_ord:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/telecom-ord/settings", "component": "OrdSettings", "permission": "telecom_ord:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "telecom_ord_control",
	"tokens": {"color.primary": "#0F766E", "color.accent": "#7C3AED", "color.success": "#15803D", "color.warning": "#B45309", "color.danger": "#B91C1C", "surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF", "text.primary": "#111827", "text.secondary": "#4B5563", "border.radius": "8px", "density": "compact"},
	"components": {"orders": {"icon": "shopping-cart", "status_indicator": "order-status-chip"}, "decomposition": {"icon": "git-branch", "status_indicator": "decomposition-chip"}, "tasks": {"icon": "check-square", "status_indicator": "task-type-chip"}, "fallout": {"icon": "alert-octagon", "status_indicator": "fallout-category-chip"}, "provisioning": {"icon": "zap", "status_indicator": "provision-chip"}, "portability": {"icon": "phone-forwarded", "status_indicator": "port-chip"}, "bulk_orders": {"icon": "layers", "status_indicator": "bulk-chip"}, "agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"}},
}

STREAMING = {"processor": "bytewax", "stream": ORD_EVENT_STREAM, "key": "tenant_id", "events": ["order_submitted", "order_validated", "order_decomposed", "task_completed", "order_fallout", "order_retry", "provisioning_completed", "order_completed", "order_cancelled", "ord_agent_registered"], "guardrails": ["ord_batch_requires_bytewax", "privileged_ord_agent_action_requires_human_approval", "duplicate_order_denied", "unapproved_bulk_order_denied", "cross_tenant_order_denied"]}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "ord_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "ord_policy_required", "required_action": "attach_ord_policy"}},
	{"name": "order_type_supported", "condition": {"operation": "submit_order", "order_type_supported": False}, "effect": {"decision": "deny", "reason": "order_type_not_supported", "required_action": "select_supported_order_type"}},
	{"name": "order_channel_supported", "condition": {"operation": "submit_order", "channel_supported": False}, "effect": {"decision": "deny", "reason": "order_channel_not_supported", "required_action": "select_supported_channel"}},
	{"name": "order_priority_supported", "condition": {"operation": "submit_order", "priority_supported": False}, "effect": {"decision": "deny", "reason": "order_priority_not_supported", "required_action": "select_supported_priority"}},
	{"name": "duplicate_order_denied", "condition": {"operation": "submit_order", "is_duplicate": True}, "effect": {"decision": "deny", "reason": "duplicate_order_detected", "required_action": "resolve_duplicate"}},
	{"name": "order_customer_required", "condition": {"operation": "submit_order", "customer_present": False}, "effect": {"decision": "deny", "reason": "customer_reference_required", "required_action": "attach_customer_reference"}},
	{"name": "order_status_supported", "condition": {"operation": "update_order_status", "order_status_supported": False}, "effect": {"decision": "deny", "reason": "order_status_not_supported", "required_action": "select_supported_order_status"}},
	{"name": "decomposition_requires_valid_order", "condition": {"operation": "decompose_order", "order_valid": False}, "effect": {"decision": "deny", "reason": "order_must_be_valid_for_decomposition", "required_action": "validate_order_first"}},
	{"name": "task_type_supported", "condition": {"operation": "create_task", "task_type_supported": False}, "effect": {"decision": "deny", "reason": "task_type_not_supported", "required_action": "select_supported_task_type"}},
	{"name": "fallout_category_supported", "condition": {"operation": "record_fallout", "fallout_category_supported": False}, "effect": {"decision": "deny", "reason": "fallout_category_not_supported", "required_action": "select_supported_fallout_category"}},
	{"name": "fallout_resolution_required", "condition": {"operation": "resolve_fallout", "resolution_present": False}, "effect": {"decision": "deny", "reason": "fallout_resolution_required", "required_action": "provide_resolution"}},
	{"name": "portability_msisdn_required", "condition": {"operation": "submit_portability_order", "msisdn_present": False}, "effect": {"decision": "deny", "reason": "msisdn_required_for_portability", "required_action": "provide_msisdn"}},
	{"name": "portability_donor_required", "condition": {"operation": "submit_portability_order", "donor_operator_present": False}, "effect": {"decision": "deny", "reason": "donor_operator_required", "required_action": "identify_donor_operator"}},
	{"name": "bulk_order_approval_required", "condition": {"operation": "submit_bulk_order", "approval_present": False}, "effect": {"decision": "deny", "reason": "bulk_order_approval_required", "required_action": "attach_bulk_order_approval"}},
	{"name": "provisioning_confirmation_required", "condition": {"operation": "confirm_provisioning", "confirmation_present": False}, "effect": {"decision": "deny", "reason": "provisioning_confirmation_required", "required_action": "confirm_provisioning_completion"}},
	{"name": "cross_tenant_order_denied", "condition": {"operation": "ord_agent_action", "cross_tenant_order_scope": True}, "effect": {"decision": "deny", "reason": "cross_tenant_order_denied", "required_action": "remove_cross_tenant_order_scope"}},
	{"name": "ord_batch_requires_bytewax", "condition": {"operation": "ord_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_ord_batch_to_bytewax"}},
	{"name": "ord_agent_runtime_supported", "condition": {"operation": "register_ord_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "ord_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "ord_agent_role_supported", "condition": {"operation": "register_ord_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "ord_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "ord_agent_name_required", "condition": {"operation": "register_ord_agent", "agent_name_present": False}, "effect": {"decision": "deny", "reason": "ord_agent_name_required", "required_action": "name_ord_agent"}},
	{"name": "ord_agent_scope_required", "condition": {"operation": "register_ord_agent", "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "ord_agent_scope_required", "required_action": "bound_ord_agent_scope"}},
	{"name": "privileged_ord_agent_action_requires_human_approval", "condition": {"operation": "ord_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	return {"capability": CAPABILITY_ID, "name": CAPABILITY_NAME, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "provides": list(PROVIDES), "requires": list(REQUIRES), "configuration": configuration, "configuration_schema": {"type": "object", "required": list(configuration), "properties": {key: {"type": "object"} for key in configuration if key != "tenant_id"} | {"tenant_id": {"type": "string", "minLength": 1}}}, "rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "api_prefix": "/telecom-ord/api/v1", "requires_theme": True, "view_module": "views.py", "template_roots": ["templates/", "static/"], "routes": deepcopy(UI_ROUTES)}, "theme": deepcopy(THEME), "streaming": deepcopy(STREAMING)}


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
