"""Executable capability contract for APG Delivery Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "transport_del"
CAPABILITY_NAME = "Delivery Management"
CAPABILITY_VERSION = "1.0.0"
DELIVERY_EVENT_STREAM = "apg.transport.delivery.lifecycle"

SUPPORTED_DELIVERY_TYPES = ["standard", "express", "same_day", "next_day", "scheduled", "attended", "unattended", "locker", "click_and_collect", "white_glove", "bulk"]
SUPPORTED_DELIVERY_STATUSES = ["pending", "assigned", "in_transit", "out_for_delivery", "delivered", "failed", "returned", "cancelled", "rescheduled"]
SUPPORTED_POD_TYPES = ["signature", "photo", "pin_code", "qr_code", "biometric", "safe_place", "neighbour", "locker_drop"]
SUPPORTED_FAILURE_REASONS = ["not_home", "wrong_address", "refused", "damaged", "access_denied", "time_window_missed", "customer_cancelled", "force_majeure"]
SUPPORTED_SLA_TIERS = ["bronze", "silver", "gold", "platinum", "custom"]
SUPPORTED_NOTIFICATION_CHANNELS = ["sms", "email", "push_notification", "whatsapp", "ivr", "app_notification"]
SUPPORTED_RETURN_REASONS = ["customer_request", "delivery_failed", "damaged", "wrong_item", "quality_issue", "refused"]
SUPPORTED_RESCHEDULING_SOURCES = ["customer_portal", "driver_app", "call_center", "auto_reschedule", "depot_operator"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["delivery_planner", "pod_verifier", "sla_monitor", "notification_agent", "failed_delivery_handler"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"deliveries": {"supported_types": SUPPORTED_DELIVERY_TYPES, "supported_statuses": SUPPORTED_DELIVERY_STATUSES, "address_verification_required": True, "time_window_required": True},
	"proof_of_delivery": {"supported_types": SUPPORTED_POD_TYPES, "signature_required_by_default": True, "photo_required_for_exceptions": True, "geo_stamp_required": True},
	"failed_deliveries": {"failure_reasons": SUPPORTED_FAILURE_REASONS, "auto_reschedule_enabled": True, "max_attempts": 3, "return_after_max_attempts": True},
	"sla": {"tiers": SUPPORTED_SLA_TIERS, "breach_alert_enabled": True, "escalation_required_on_breach": True, "reporting_enabled": True},
	"notifications": {"channels": SUPPORTED_NOTIFICATION_CHANNELS, "eta_notification_enabled": True, "delivery_confirmation_enabled": True, "failed_delivery_notification_enabled": True},
	"returns": {"reasons": SUPPORTED_RETURN_REASONS, "rma_required": True, "depot_return_enabled": True},
	"rescheduling": {"sources": SUPPORTED_RESCHEDULING_SOURCES, "customer_self_service_enabled": True, "max_reschedule_count": 3},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "cross_tenant_delivery_denied": True, "pod_falsification_denied": True},
	"observability": {"event_stream": DELIVERY_EVENT_STREAM, "stream_processor": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_deliveries": True, "enable_pod": True, "enable_failed_deliveries": True, "enable_sla": True, "enable_notifications": True},
	"theme": {"default_theme": "transport_delivery_control", "allow_tenant_overrides": True},
}

PROVIDES = ["delivery_planning_workflow", "proof_of_delivery_workflow", "customer_notification_workflow", "failed_delivery_workflow", "sla_tracking_workflow", "delivery_return_workflow"]
REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "moni", "comp", "mqeb", "schd"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/transport-delivery/dashboard", "component": "DeliveryDashboard", "permission": "transport_del:view", "nav_group": "Overview"},
	{"name": "deliveries", "path": "/transport-delivery/deliveries", "component": "DeliveryConsole", "permission": "transport_del:deliveries", "nav_group": "Operations"},
	{"name": "delivery_create", "path": "/transport-delivery/deliveries/create", "component": "DeliveryForm", "permission": "transport_del:deliveries_write", "nav_group": "Operations"},
	{"name": "proof_of_delivery", "path": "/transport-delivery/pod", "component": "PodConsole", "permission": "transport_del:pod", "nav_group": "Evidence"},
	{"name": "failed_deliveries", "path": "/transport-delivery/failed", "component": "FailedDeliveryConsole", "permission": "transport_del:failed", "nav_group": "Exceptions"},
	{"name": "rescheduling", "path": "/transport-delivery/rescheduling", "component": "ReschedulingConsole", "permission": "transport_del:rescheduling", "nav_group": "Exceptions"},
	{"name": "sla_tracking", "path": "/transport-delivery/sla", "component": "SlaTrackingConsole", "permission": "transport_del:sla", "nav_group": "Performance"},
	{"name": "notifications", "path": "/transport-delivery/notifications", "component": "DeliveryNotificationConsole", "permission": "transport_del:notifications", "nav_group": "Communications"},
	{"name": "returns", "path": "/transport-delivery/returns", "component": "DeliveryReturnConsole", "permission": "transport_del:returns", "nav_group": "Returns"},
	{"name": "reports", "path": "/transport-delivery/reports", "component": "DeliveryReportConsole", "permission": "transport_del:reports", "nav_group": "Reporting"},
	{"name": "agents", "path": "/transport-delivery/agents", "component": "DeliveryAgentWorkbench", "permission": "transport_del:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/transport-delivery/settings", "component": "DeliverySettings", "permission": "transport_del:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "transport_delivery_control",
	"tokens": {"color.primary": "#0369A1", "color.accent": "#0891B2", "color.success": "#166534", "color.warning": "#A16207", "color.danger": "#991B1B", "surface.canvas": "#F0F4FF", "surface.panel": "#FFFFFF", "text.primary": "#0F172A", "text.secondary": "#475569", "border.radius": "8px", "density": "comfortable"},
	"components": {
		"deliveries": {"icon": "truck", "status_indicator": "delivery-status-chip"},
		"proof_of_delivery": {"icon": "check-circle", "status_indicator": "pod-type-chip"},
		"failed_deliveries": {"icon": "x-circle", "status_indicator": "failure-reason-chip"},
		"sla": {"icon": "clock", "status_indicator": "sla-tier-chip"},
		"notifications": {"icon": "bell", "status_indicator": "channel-chip"},
		"returns": {"icon": "rotate-ccw", "status_indicator": "return-reason-chip"},
		"agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": DELIVERY_EVENT_STREAM,
	"key": "tenant_id",
	"events": ["delivery_created", "delivery_assigned", "delivery_out_for_delivery", "delivery_completed", "delivery_failed", "pod_recorded", "sla_breached", "delivery_notification_sent", "delivery_returned", "delivery_agent_registered"],
	"guardrails": ["delivery_batch_requires_bytewax", "pod_falsification_denied", "cross_tenant_delivery_denied", "privileged_delivery_agent_action_requires_human_approval"],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "delivery_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "delivery_policy_required", "required_action": "attach_delivery_policy"}},
	{"name": "delivery_type_supported", "condition": {"operation": "create_delivery", "delivery_type_supported": False}, "effect": {"decision": "deny", "reason": "delivery_type_not_supported", "required_action": "select_supported_delivery_type"}},
	{"name": "delivery_address_required", "condition": {"operation": "create_delivery", "address_present": False}, "effect": {"decision": "deny", "reason": "delivery_address_required", "required_action": "provide_delivery_address"}},
	{"name": "delivery_time_window_required", "condition": {"operation": "create_delivery", "time_window_present": False}, "effect": {"decision": "deny", "reason": "delivery_time_window_required", "required_action": "set_delivery_time_window"}},
	{"name": "delivery_recipient_required", "condition": {"operation": "create_delivery", "recipient_present": False}, "effect": {"decision": "deny", "reason": "delivery_recipient_required", "required_action": "attach_recipient_details"}},
	{"name": "pod_type_supported", "condition": {"operation": "record_pod", "pod_type_supported": False}, "effect": {"decision": "deny", "reason": "pod_type_not_supported", "required_action": "select_supported_pod_type"}},
	{"name": "pod_delivery_required", "condition": {"operation": "record_pod", "delivery_present": False}, "effect": {"decision": "deny", "reason": "delivery_reference_required", "required_action": "select_delivery"}},
	{"name": "pod_geo_stamp_required", "condition": {"operation": "record_pod", "geo_stamp_present": False}, "effect": {"decision": "deny", "reason": "geo_stamp_required", "required_action": "provide_geo_stamp"}},
	{"name": "pod_falsification_denied", "condition": {"operation": "record_pod", "pod_falsification_detected": True}, "effect": {"decision": "deny", "reason": "pod_falsification_denied", "required_action": "provide_authentic_pod"}},
	{"name": "failed_delivery_reason_required", "condition": {"operation": "record_failed_delivery", "failure_reason_supported": False}, "effect": {"decision": "deny", "reason": "failure_reason_required", "required_action": "select_failure_reason"}},
	{"name": "failed_delivery_reference_required", "condition": {"operation": "record_failed_delivery", "delivery_present": False}, "effect": {"decision": "deny", "reason": "delivery_reference_required", "required_action": "select_delivery"}},
	{"name": "sla_tier_supported", "condition": {"operation": "set_sla", "sla_tier_supported": False}, "effect": {"decision": "deny", "reason": "sla_tier_not_supported", "required_action": "select_supported_sla_tier"}},
	{"name": "notification_channel_supported", "condition": {"operation": "send_notification", "channel_supported": False}, "effect": {"decision": "deny", "reason": "notification_channel_not_supported", "required_action": "select_supported_channel"}},
	{"name": "notification_recipient_required", "condition": {"operation": "send_notification", "recipient_present": False}, "effect": {"decision": "deny", "reason": "recipient_required", "required_action": "attach_recipient"}},
	{"name": "return_reason_supported", "condition": {"operation": "create_return", "return_reason_supported": False}, "effect": {"decision": "deny", "reason": "return_reason_not_supported", "required_action": "select_supported_return_reason"}},
	{"name": "return_rma_required", "condition": {"operation": "create_return", "rma_present": False}, "effect": {"decision": "deny", "reason": "rma_required", "required_action": "generate_rma"}},
	{"name": "reschedule_source_supported", "condition": {"operation": "reschedule_delivery", "reschedule_source_supported": False}, "effect": {"decision": "deny", "reason": "reschedule_source_not_supported", "required_action": "select_supported_source"}},
	{"name": "max_reschedule_exceeded", "condition": {"operation": "reschedule_delivery", "max_reschedule_exceeded": True}, "effect": {"decision": "deny", "reason": "max_reschedule_count_exceeded", "required_action": "initiate_return_process"}},
	{"name": "cross_tenant_delivery_denied", "condition": {"operation_type": "write", "cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_delivery_denied", "required_action": "use_tenant_scoped_context"}},
	{"name": "delivery_batch_requires_bytewax", "condition": {"operation": "delivery_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_delivery_batch_to_bytewax"}},
	{"name": "delivery_agent_runtime_supported", "condition": {"operation": "register_delivery_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "delivery_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "delivery_agent_role_supported", "condition": {"operation": "register_delivery_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "delivery_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "privileged_delivery_agent_action_requires_human_approval", "condition": {"operation": "delivery_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
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
		"ui": {"shell": "apg_python", "api_prefix": "/transport-delivery/api/v1", "requires_theme": True, "view_module": "views.py", "template_roots": ["templates/", "static/"], "routes": deepcopy(UI_ROUTES)},
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
