"""Executable capability contract for APG Citizen Services Portal."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "government_csr"
CAPABILITY_NAME = "Citizen Services Portal"
CAPABILITY_VERSION = "1.0.0"
CSR_EVENT_STREAM = "apg.government.csr.lifecycle"

SUPPORTED_SERVICE_TYPES = ["certificate_issuance", "permit_application", "licence_application", "benefit_claim", "registration", "complaint", "payment", "information_request", "renewal", "endorsement"]
SUPPORTED_SUBMISSION_CHANNELS = ["web_portal", "mobile_app", "ussd", "kiosk", "assisted_digital", "api"]
SUPPORTED_APPLICATION_STATUSES = ["draft", "submitted", "acknowledged", "under_review", "additional_info_required", "approved", "rejected", "dispatched", "completed", "cancelled"]
SUPPORTED_PAYMENT_METHODS = ["mpesa", "card", "bank_transfer", "government_payment_gateway", "waiver", "instalment"]
SUPPORTED_PAYMENT_STATUSES = ["pending", "completed", "failed", "refunded", "waived", "partially_paid"]
SUPPORTED_NOTIFICATION_TYPES = ["sms", "email", "push", "in_app", "ussd_push"]
SUPPORTED_VERIFICATION_TYPES = ["identity", "document", "biometric", "otp", "nida", "passport"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["application_router", "payment_verifier", "status_updater", "document_checker", "sla_monitor"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"services": {
		"supported_service_types": SUPPORTED_SERVICE_TYPES,
		"submission_channels": SUPPORTED_SUBMISSION_CHANNELS,
		"citizen_id_required": True,
		"service_fee_configured": True,
		"evidence_required": True,
	},
	"applications": {
		"supported_statuses": SUPPORTED_APPLICATION_STATUSES,
		"service_required": True,
		"citizen_id_required": True,
		"channel_required": True,
	},
	"payments": {
		"supported_methods": SUPPORTED_PAYMENT_METHODS,
		"supported_statuses": SUPPORTED_PAYMENT_STATUSES,
		"application_required": True,
		"receipt_required": True,
		"reconciliation_enabled": True,
	},
	"verifications": {
		"supported_verification_types": SUPPORTED_VERIFICATION_TYPES,
		"application_required": True,
		"evidence_required": True,
	},
	"notifications": {
		"supported_types": SUPPORTED_NOTIFICATION_TYPES,
		"application_required": True,
		"citizen_id_required": True,
	},
	"reviews": {
		"supported_statuses": SUPPORTED_REVIEW_STATUSES,
		"reviewer_required": True,
		"evidence_required": True,
	},
	"agents": {
		"enabled": True,
		"supported_runtimes": SUPPORTED_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_AGENT_ROLES,
		"name_required": True,
		"scope_required": True,
		"human_approval_required_for_privileged_actions": True,
	},
	"governance": {
		"require_tenant_context": True,
		"policy_attached_for_writes": True,
		"audit_events": True,
		"payment_before_processing_enforced": True,
		"citizen_data_privacy_enforced": True,
		"cross_tenant_service_denied": True,
		"unauthenticated_submission_denied": True,
		"duplicate_submission_check_enabled": True,
		"evidence_fabrication_denied": True,
	},
	"observability": {"event_stream": CSR_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {
		"auth": "auth",
		"audit": "audl",
		"notifications": "ntfy",
		"workflow": "wflo",
		"monitoring": "moni",
		"search": "srch",
		"event_stream": "bytewax",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_services": True,
		"enable_applications": True,
		"enable_payments": True,
		"enable_verifications": True,
		"enable_notifications": True,
		"enable_reviews": True,
		"enable_agents": True,
		"enable_analytics": True,
	},
	"theme": {"default_theme": "government_csr_portal", "allow_tenant_overrides": True},
}

PROVIDES = [
	"citizen_self_service_workflow",
	"service_application_workflow",
	"application_status_tracking_workflow",
	"epayment_workflow",
	"document_verification_workflow",
	"service_notification_workflow",
	"service_delivery_analytics_workflow",
	"citizen_review_workflow",
	"citizen_services_agent_workflow",
	"service_catalogue_workflow",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "srch", "moni", "mqeb"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/government-csr/dashboard", "component": "CitizenServicesDashboard", "permission": "government_csr:view", "nav_group": "Overview"},
	{"name": "services", "path": "/government-csr/services", "component": "ServiceCatalogue", "permission": "government_csr:services", "nav_group": "Services"},
	{"name": "apply", "path": "/government-csr/apply", "component": "ServiceApplicationForm", "permission": "government_csr:apply", "nav_group": "Services"},
	{"name": "applications", "path": "/government-csr/applications", "component": "ApplicationTrackingConsole", "permission": "government_csr:applications", "nav_group": "Applications"},
	{"name": "payments", "path": "/government-csr/payments", "component": "PaymentConsole", "permission": "government_csr:payments", "nav_group": "Payments"},
	{"name": "verifications", "path": "/government-csr/verifications", "component": "DocumentVerificationConsole", "permission": "government_csr:verify", "nav_group": "Verification"},
	{"name": "notifications", "path": "/government-csr/notifications", "component": "CitizenNotificationConsole", "permission": "government_csr:notify", "nav_group": "Communications"},
	{"name": "analytics", "path": "/government-csr/analytics", "component": "ServiceDeliveryAnalytics", "permission": "government_csr:analytics", "nav_group": "Reporting"},
	{"name": "reviews", "path": "/government-csr/reviews", "component": "ServiceReviewConsole", "permission": "government_csr:review", "nav_group": "Governance"},
	{"name": "agents", "path": "/government-csr/agents", "component": "CitizenServicesAgentWorkbench", "permission": "government_csr:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/government-csr/settings", "component": "CitizenServicesSettings", "permission": "government_csr:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "government_csr_portal",
	"tokens": {
		"color.primary": "#047857",
		"color.accent": "#2563EB",
		"color.success": "#166534",
		"color.warning": "#A16207",
		"color.danger": "#991B1B",
		"surface.canvas": "#ECFDF5",
		"surface.panel": "#FFFFFF",
		"text.primary": "#064E3B",
		"text.secondary": "#374151",
		"border.radius": "10px",
		"density": "spacious",
	},
	"components": {
		"services": {"icon": "layout-grid", "status_indicator": "service-type-chip"},
		"applications": {"icon": "inbox", "status_indicator": "application-status-chip"},
		"payments": {"icon": "credit-card", "status_indicator": "payment-status-chip"},
		"verifications": {"icon": "shield-check", "status_indicator": "verification-type-chip"},
		"notifications": {"icon": "bell", "status_indicator": "notification-channel-chip"},
		"reviews": {"icon": "star", "status_indicator": "review-status-chip"},
		"agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": CSR_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"service_application_submitted",
		"application_status_updated",
		"payment_completed",
		"payment_failed",
		"document_verified",
		"service_notification_sent",
		"application_approved",
		"application_rejected",
		"citizen_services_agent_registered",
		"service_completed",
	],
	"guardrails": [
		"csr_batch_requires_bytewax",
		"payment_before_processing_enforced",
		"citizen_data_privacy_enforced",
		"cross_tenant_service_denied",
		"unauthenticated_submission_denied",
		"evidence_fabrication_denied",
		"privileged_csr_agent_action_requires_human_approval",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "csr_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "citizen_services_policy_required", "required_action": "attach_csr_policy"}},
	{"name": "service_type_supported", "condition": {"operation": "submit_application", "service_type_supported": False}, "effect": {"decision": "deny", "reason": "service_type_not_supported", "required_action": "select_supported_service_type"}},
	{"name": "submission_citizen_id_required", "condition": {"operation": "submit_application", "citizen_id_present": False}, "effect": {"decision": "deny", "reason": "citizen_id_required", "required_action": "authenticate_citizen"}},
	{"name": "submission_channel_supported", "condition": {"operation": "submit_application", "channel_supported": False}, "effect": {"decision": "deny", "reason": "submission_channel_not_supported", "required_action": "select_supported_channel"}},
	{"name": "unauthenticated_submission_denied", "condition": {"operation": "submit_application", "authenticated": False}, "effect": {"decision": "deny", "reason": "unauthenticated_submission_denied", "required_action": "authenticate_citizen"}},
	{"name": "cross_tenant_service_denied", "condition": {"operation": "submit_application", "cross_tenant": True}, "effect": {"decision": "deny", "reason": "cross_tenant_service_denied", "required_action": "use_tenant_scoped_service"}},
	{"name": "payment_method_supported", "condition": {"operation": "record_payment", "payment_method_supported": False}, "effect": {"decision": "deny", "reason": "payment_method_not_supported", "required_action": "select_supported_payment_method"}},
	{"name": "payment_application_required", "condition": {"operation": "record_payment", "application_present": False}, "effect": {"decision": "deny", "reason": "application_required", "required_action": "select_application"}},
	{"name": "payment_receipt_required", "condition": {"operation": "record_payment", "receipt_present": False}, "effect": {"decision": "deny", "reason": "payment_receipt_required", "required_action": "generate_receipt"}},
	{"name": "verification_type_supported", "condition": {"operation": "verify_document", "verification_type_supported": False}, "effect": {"decision": "deny", "reason": "verification_type_not_supported", "required_action": "select_supported_verification_type"}},
	{"name": "verification_application_required", "condition": {"operation": "verify_document", "application_present": False}, "effect": {"decision": "deny", "reason": "application_required", "required_action": "select_application"}},
	{"name": "verification_evidence_required", "condition": {"operation": "verify_document", "evidence_present": False}, "effect": {"decision": "deny", "reason": "verification_evidence_required", "required_action": "upload_document"}},
	{"name": "notification_type_supported", "condition": {"operation": "send_notification", "notification_type_supported": False}, "effect": {"decision": "deny", "reason": "notification_type_not_supported", "required_action": "select_supported_notification_type"}},
	{"name": "notification_citizen_required", "condition": {"operation": "send_notification", "citizen_id_present": False}, "effect": {"decision": "deny", "reason": "citizen_id_required", "required_action": "provide_citizen_id"}},
	{"name": "application_status_supported", "condition": {"operation": "update_status", "status_supported": False}, "effect": {"decision": "deny", "reason": "application_status_not_supported", "required_action": "select_supported_status"}},
	{"name": "review_status_supported", "condition": {"operation": "record_review", "status_supported": False}, "effect": {"decision": "deny", "reason": "review_status_not_supported", "required_action": "select_supported_status"}},
	{"name": "review_reviewer_required", "condition": {"operation": "record_review", "reviewer_present": False}, "effect": {"decision": "deny", "reason": "reviewer_required", "required_action": "assign_reviewer"}},
	{"name": "csr_batch_requires_bytewax", "condition": {"operation": "csr_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_csr_batch_to_bytewax"}},
	{"name": "csr_agent_runtime_supported", "condition": {"operation": "register_csr_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "csr_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "csr_agent_role_supported", "condition": {"operation": "register_csr_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "csr_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "csr_agent_name_required", "condition": {"operation": "register_csr_agent", "agent_name_present": False}, "effect": {"decision": "deny", "reason": "csr_agent_name_required", "required_action": "name_csr_agent"}},
	{"name": "csr_agent_scope_required", "condition": {"operation": "register_csr_agent", "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "csr_agent_scope_required", "required_action": "bound_csr_agent_scope"}},
	{"name": "privileged_csr_agent_action_requires_human_approval", "condition": {"operation": "csr_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
	{"name": "evidence_fabrication_denied", "condition": {"operation": "csr_agent_action", "evidence_fabrication_scope": True}, "effect": {"decision": "deny", "reason": "evidence_fabrication_denied", "required_action": "remove_evidence_fabrication_scope"}},
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
			"required": list(configuration),
			"properties": {key: {"type": "object"} for key in configuration if key != "tenant_id"} | {"tenant_id": {"type": "string", "minLength": 1}},
		},
		"rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"api_prefix": "/government-csr/api/v1",
			"requires_theme": True,
			"view_module": "views.py",
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
