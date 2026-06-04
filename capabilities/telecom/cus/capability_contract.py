"""Executable capability contract for APG Customer Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "telecom_cus"
CAPABILITY_NAME = "Customer Management"
CAPABILITY_VERSION = "1.0.0"
CUS_EVENT_STREAM = "apg.telecom.cus.lifecycle"

SUPPORTED_CUSTOMER_TYPES = ["individual", "business", "government", "mvno", "wholesale", "iot_fleet", "household"]
SUPPORTED_KYC_STATUSES = ["not_started", "pending", "under_review", "verified", "rejected", "expired", "suspended"]
SUPPORTED_KYC_DOCUMENT_TYPES = ["national_id", "passport", "driving_licence", "company_registration", "utility_bill", "bank_statement", "tax_certificate"]
SUPPORTED_CUSTOMER_STATUSES = ["prospect", "active", "suspended", "deactivated", "churned", "blacklisted", "deceased"]
SUPPORTED_PLAN_TYPES = ["prepaid", "postpaid", "hybrid", "data_only", "voice_only", "bundle", "enterprise_plan", "wholesale_plan"]
SUPPORTED_SIM_STATUSES = ["provisioned", "active", "suspended", "stolen_blocked", "deregistered", "ported_out", "replaced"]
SUPPORTED_DEVICE_TYPES = ["handset", "sim_card", "modem", "router", "iot_device", "wearable", "tablet", "cpe"]
SUPPORTED_CASE_TYPES = ["complaint", "service_request", "billing_query", "technical_fault", "fraud_report", "portability_request", "general_inquiry"]
SUPPORTED_CASE_STATUSES = ["open", "in_progress", "pending_customer", "escalated", "resolved", "closed", "reopened"]
SUPPORTED_LIFECYCLE_EVENTS = ["onboarded", "plan_changed", "sim_swapped", "sim_blocked", "number_ported", "account_suspended", "account_reactivated", "churned", "deceased_notification"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["kyc_reviewer", "account_manager", "case_handler", "churn_analyst", "provisioning_agent"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"customers": {"supported_types": SUPPORTED_CUSTOMER_TYPES, "supported_statuses": SUPPORTED_CUSTOMER_STATUSES, "kyc_required": True, "msisdn_required": True},
	"kyc": {"supported_statuses": SUPPORTED_KYC_STATUSES, "supported_document_types": SUPPORTED_KYC_DOCUMENT_TYPES, "verification_required": True, "expiry_tracking": True},
	"plans": {"supported_plan_types": SUPPORTED_PLAN_TYPES, "credit_check_for_postpaid": True, "activation_approval": True},
	"sims": {"supported_statuses": SUPPORTED_SIM_STATUSES, "iccid_required": True, "imsi_required": True, "max_sims_per_customer": 10},
	"devices": {"supported_types": SUPPORTED_DEVICE_TYPES, "imei_check": True, "blacklist_check": True},
	"cases": {"supported_types": SUPPORTED_CASE_TYPES, "supported_statuses": SUPPORTED_CASE_STATUSES, "sla_hours": {"complaint": 24, "service_request": 48, "billing_query": 72, "technical_fault": 4}},
	"lifecycle": {"supported_events": SUPPORTED_LIFECYCLE_EVENTS, "audit_trail": True, "notification_on_key_events": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "kyc_bypass_denied": True, "cross_tenant_access_denied": True, "pii_access_requires_approval": True},
	"observability": {"event_stream": CUS_EVENT_STREAM, "stream_processor": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_customers": True, "enable_kyc": True, "enable_plans": True, "enable_sims": True, "enable_devices": True, "enable_cases": True, "enable_lifecycle": True, "enable_agents": True},
	"theme": {"default_theme": "telecom_cus_control", "allow_tenant_overrides": True},
}

PROVIDES = ["customer_lifecycle_workflow", "kyc_workflow", "plan_management_workflow", "sim_management_workflow", "device_management_workflow", "case_tracking_workflow", "customer_360_view", "churn_management_workflow", "cus_agent_workflow"]
REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "nlpc", "mqeb", "comp"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/telecom-cus/dashboard", "component": "CusDashboard", "permission": "telecom_cus:view", "nav_group": "Overview"},
	{"name": "customers", "path": "/telecom-cus/customers", "component": "CusCustomerConsole", "permission": "telecom_cus:customers", "nav_group": "Customers"},
	{"name": "customer_detail", "path": "/telecom-cus/customers/<id>", "component": "CusCustomer360", "permission": "telecom_cus:customers", "nav_group": "Customers"},
	{"name": "kyc", "path": "/telecom-cus/kyc", "component": "CusKycConsole", "permission": "telecom_cus:kyc", "nav_group": "Compliance"},
	{"name": "plans", "path": "/telecom-cus/plans", "component": "CusPlanConsole", "permission": "telecom_cus:plans", "nav_group": "Products"},
	{"name": "sims", "path": "/telecom-cus/sims", "component": "CusSimConsole", "permission": "telecom_cus:sims", "nav_group": "Assets"},
	{"name": "devices", "path": "/telecom-cus/devices", "component": "CusDeviceConsole", "permission": "telecom_cus:devices", "nav_group": "Assets"},
	{"name": "cases", "path": "/telecom-cus/cases", "component": "CusCaseQueue", "permission": "telecom_cus:cases", "nav_group": "Support"},
	{"name": "lifecycle", "path": "/telecom-cus/lifecycle", "component": "CusLifecycleLedger", "permission": "telecom_cus:lifecycle", "nav_group": "Customers"},
	{"name": "churn", "path": "/telecom-cus/churn", "component": "CusChurnConsole", "permission": "telecom_cus:churn", "nav_group": "Retention"},
	{"name": "agents", "path": "/telecom-cus/agents", "component": "CusAgentWorkbench", "permission": "telecom_cus:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/telecom-cus/settings", "component": "CusSettings", "permission": "telecom_cus:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "telecom_cus_control",
	"tokens": {"color.primary": "#7C3AED", "color.accent": "#0891B2", "color.success": "#15803D", "color.warning": "#B45309", "color.danger": "#B91C1C", "surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF", "text.primary": "#111827", "text.secondary": "#4B5563", "border.radius": "8px", "density": "comfortable"},
	"components": {"customers": {"icon": "users", "status_indicator": "customer-status-chip"}, "kyc": {"icon": "shield-check", "status_indicator": "kyc-status-chip"}, "plans": {"icon": "package", "status_indicator": "plan-type-chip"}, "sims": {"icon": "sim-card", "status_indicator": "sim-status-chip"}, "devices": {"icon": "smartphone", "status_indicator": "device-type-chip"}, "cases": {"icon": "message-circle", "status_indicator": "case-status-chip"}, "lifecycle": {"icon": "git-commit", "status_indicator": "lifecycle-event-chip"}, "churn": {"icon": "user-x", "status_indicator": "churn-risk-chip"}, "agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"}},
}

STREAMING = {"processor": "bytewax", "stream": CUS_EVENT_STREAM, "key": "tenant_id", "events": ["customer_onboarded", "kyc_verified", "kyc_rejected", "plan_activated", "plan_changed", "sim_provisioned", "sim_blocked", "case_opened", "case_resolved", "customer_churned", "cus_agent_registered"], "guardrails": ["cus_batch_requires_bytewax", "privileged_cus_agent_action_requires_human_approval", "kyc_bypass_denied", "pii_access_requires_approval", "cross_tenant_access_denied"]}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "cus_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "cus_policy_required", "required_action": "attach_cus_policy"}},
	{"name": "customer_type_supported", "condition": {"operation": "create_customer", "customer_type_supported": False}, "effect": {"decision": "deny", "reason": "customer_type_not_supported", "required_action": "select_supported_customer_type"}},
	{"name": "customer_kyc_required", "condition": {"operation": "create_customer", "kyc_initiated": False}, "effect": {"decision": "deny", "reason": "kyc_required_for_customer_creation", "required_action": "initiate_kyc"}},
	{"name": "customer_msisdn_required", "condition": {"operation": "create_customer", "msisdn_present": False}, "effect": {"decision": "deny", "reason": "msisdn_required", "required_action": "assign_msisdn"}},
	{"name": "kyc_document_type_supported", "condition": {"operation": "submit_kyc_document", "document_type_supported": False}, "effect": {"decision": "deny", "reason": "kyc_document_type_not_supported", "required_action": "select_supported_document_type"}},
	{"name": "kyc_bypass_denied", "condition": {"operation": "submit_kyc_document", "kyc_bypass_scope": True}, "effect": {"decision": "deny", "reason": "kyc_bypass_denied", "required_action": "remove_kyc_bypass_scope"}},
	{"name": "plan_type_supported", "condition": {"operation": "activate_plan", "plan_type_supported": False}, "effect": {"decision": "deny", "reason": "plan_type_not_supported", "required_action": "select_supported_plan_type"}},
	{"name": "postpaid_credit_check_required", "condition": {"operation": "activate_plan", "plan_is_postpaid": True, "credit_check_completed": False}, "effect": {"decision": "deny", "reason": "credit_check_required_for_postpaid", "required_action": "complete_credit_check"}},
	{"name": "sim_iccid_required", "condition": {"operation": "provision_sim", "iccid_present": False}, "effect": {"decision": "deny", "reason": "iccid_required", "required_action": "set_iccid"}},
	{"name": "sim_imsi_required", "condition": {"operation": "provision_sim", "imsi_present": False}, "effect": {"decision": "deny", "reason": "imsi_required", "required_action": "set_imsi"}},
	{"name": "sim_status_supported", "condition": {"operation": "update_sim_status", "sim_status_supported": False}, "effect": {"decision": "deny", "reason": "sim_status_not_supported", "required_action": "select_supported_sim_status"}},
	{"name": "device_imei_check_required", "condition": {"operation": "register_device", "imei_checked": False}, "effect": {"decision": "deny", "reason": "device_imei_check_required", "required_action": "check_imei"}},
	{"name": "device_blacklist_check_required", "condition": {"operation": "register_device", "blacklist_checked": False}, "effect": {"decision": "deny", "reason": "device_blacklist_check_required", "required_action": "check_blacklist"}},
	{"name": "case_type_supported", "condition": {"operation": "open_case", "case_type_supported": False}, "effect": {"decision": "deny", "reason": "case_type_not_supported", "required_action": "select_supported_case_type"}},
	{"name": "case_customer_required", "condition": {"operation": "open_case", "customer_present": False}, "effect": {"decision": "deny", "reason": "customer_required_for_case", "required_action": "associate_customer"}},
	{"name": "case_status_supported", "condition": {"operation": "update_case_status", "case_status_supported": False}, "effect": {"decision": "deny", "reason": "case_status_not_supported", "required_action": "select_supported_case_status"}},
	{"name": "pii_access_requires_approval", "condition": {"operation": "access_pii", "approval_present": False}, "effect": {"decision": "deny", "reason": "pii_access_approval_required", "required_action": "attach_pii_access_approval"}},
	{"name": "cross_tenant_access_denied", "condition": {"operation": "cus_agent_action", "cross_tenant_access_scope": True}, "effect": {"decision": "deny", "reason": "cross_tenant_access_denied", "required_action": "remove_cross_tenant_access_scope"}},
	{"name": "cus_batch_requires_bytewax", "condition": {"operation": "cus_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_cus_batch_to_bytewax"}},
	{"name": "cus_agent_runtime_supported", "condition": {"operation": "register_cus_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "cus_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "cus_agent_role_supported", "condition": {"operation": "register_cus_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "cus_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "cus_agent_name_required", "condition": {"operation": "register_cus_agent", "agent_name_present": False}, "effect": {"decision": "deny", "reason": "cus_agent_name_required", "required_action": "name_cus_agent"}},
	{"name": "cus_agent_scope_required", "condition": {"operation": "register_cus_agent", "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "cus_agent_scope_required", "required_action": "bound_cus_agent_scope"}},
	{"name": "privileged_cus_agent_action_requires_human_approval", "condition": {"operation": "cus_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	return {"capability": CAPABILITY_ID, "name": CAPABILITY_NAME, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "provides": list(PROVIDES), "requires": list(REQUIRES), "configuration": configuration, "configuration_schema": {"type": "object", "required": list(configuration), "properties": {key: {"type": "object"} for key in configuration if key != "tenant_id"} | {"tenant_id": {"type": "string", "minLength": 1}}}, "rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "api_prefix": "/telecom-cus/api/v1", "requires_theme": True, "view_module": "views.py", "template_roots": ["templates/", "static/"], "routes": deepcopy(UI_ROUTES)}, "theme": deepcopy(THEME), "streaming": deepcopy(STREAMING)}


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
