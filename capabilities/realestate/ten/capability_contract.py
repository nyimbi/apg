"""Executable capability contract for APG Tenant Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "realestate_ten"
CAPABILITY_NAME = "Tenant Management"
CAPABILITY_VERSION = "1.0.0"
TEN_EVENT_STREAM = "apg.realestate.ten.lifecycle"

SUPPORTED_TENANT_TYPES = ["corporate", "sme", "sole_trader", "individual", "government", "ngo", "educational", "healthcare", "retail_brand", "franchise", "co_working_operator"]
SUPPORTED_TENANT_STATUSES = ["prospect", "applicant", "approved", "active", "notice_served", "vacating", "former", "blacklisted"]
SUPPORTED_ONBOARDING_STEPS = ["application_received", "referencing", "credit_check", "right_to_rent", "lease_negotiation", "lease_signing", "deposit_registration", "key_handover", "welcome_pack_sent", "portal_activated"]
SUPPORTED_SERVICE_REQUEST_TYPES = ["maintenance_request", "cleaning_request", "access_request", "parking_request", "delivery_coordination", "visitor_management", "it_support", "noise_complaint", "neighbour_dispute", "general_enquiry"]
SUPPORTED_REQUEST_STATUSES = ["open", "acknowledged", "assigned", "in_progress", "awaiting_tenant", "resolved", "closed", "escalated"]
SUPPORTED_COMMUNICATION_CHANNELS = ["portal", "email", "sms", "whatsapp", "phone", "letter", "in_person"]
SUPPORTED_SATISFACTION_DIMENSIONS = ["maintenance_quality", "response_time", "cleanliness", "facilities", "communication", "value_for_money", "overall"]
SUPPORTED_SCORING_MODELS = ["payment_history", "maintenance_compliance", "communication_rating", "lease_compliance", "renewal_likelihood"]
SUPPORTED_CREDIT_GRADES = ["A", "B", "C", "D", "F"]
SUPPORTED_EVENT_TYPES = ["move_in", "move_out", "lease_signed", "rent_increase", "complaint", "award", "visit", "inspection"]
SUPPORTED_DOCUMENT_TYPES = ["id_document", "company_registration", "financial_statement", "lease_copy", "insurance_certificate", "right_to_rent", "correspondence", "survey"]
SUPPORTED_ESCALATION_TYPES = ["noise_complaint", "rent_arrears", "lease_breach", "property_damage", "anti_social_behaviour", "subletting_unauthorised"]
SUPPORTED_PORTAL_FEATURES = ["service_requests", "documents", "statements", "notices", "maintenance_tracking", "booking", "communication", "satisfaction_survey"]
SUPPORTED_SATISFACTION_RATINGS = [1, 2, 3, 4, 5]
SUPPORTED_APPROVAL_LEVELS = ["property_manager", "asset_manager", "portfolio_director"]

PROVIDES = [
	"tenant_onboarding_workflow",
	"tenant_communication_portal",
	"service_request_management",
	"tenant_scoring_engine",
	"satisfaction_tracking",
	"tenant_document_management",
	"tenant_event_timeline",
	"escalation_management",
	"tenant_performance_reporting",
	"tenant_retention_analytics",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "nlpc", "mqeb", "schd"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/realestate/ten/dashboard", "component": "TenDashboard", "permission": "realestate_ten:view", "nav_group": "Overview"},
	{"name": "tenants", "path": "/realestate/ten/tenants", "component": "TenantRegistry", "permission": "realestate_ten:tenants", "nav_group": "Tenants"},
	{"name": "tenant-detail", "path": "/realestate/ten/tenants/<id>", "component": "TenantDetail", "permission": "realestate_ten:tenants", "nav_group": "Tenants"},
	{"name": "onboarding", "path": "/realestate/ten/onboarding", "component": "TenantOnboardingWorkflow", "permission": "realestate_ten:onboarding", "nav_group": "Onboarding"},
	{"name": "service-requests", "path": "/realestate/ten/service-requests", "component": "ServiceRequestQueue", "permission": "realestate_ten:service_requests", "nav_group": "Services"},
	{"name": "communications", "path": "/realestate/ten/communications", "component": "CommunicationPortal", "permission": "realestate_ten:communications", "nav_group": "Communications"},
	{"name": "satisfaction", "path": "/realestate/ten/satisfaction", "component": "SatisfactionTracker", "permission": "realestate_ten:satisfaction", "nav_group": "Analytics"},
	{"name": "scoring", "path": "/realestate/ten/scoring", "component": "TenantScoringConsole", "permission": "realestate_ten:scoring", "nav_group": "Analytics"},
	{"name": "escalations", "path": "/realestate/ten/escalations", "component": "TenantEscalationConsole", "permission": "realestate_ten:escalations", "nav_group": "Escalations"},
	{"name": "documents", "path": "/realestate/ten/documents", "component": "TenantDocumentConsole", "permission": "realestate_ten:documents", "nav_group": "Documents"},
	{"name": "timeline", "path": "/realestate/ten/timeline", "component": "TenantEventTimeline", "permission": "realestate_ten:timeline", "nav_group": "History"},
	{"name": "retention", "path": "/realestate/ten/retention", "component": "RetentionAnalyticsDashboard", "permission": "realestate_ten:retention", "nav_group": "Analytics"},
	{"name": "reports", "path": "/realestate/ten/reports", "component": "TenantReportBuilder", "permission": "realestate_ten:reports", "nav_group": "Reporting"},
	{"name": "settings", "path": "/realestate/ten/settings", "component": "TenSettings", "permission": "realestate_ten:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "realestate_ten_portal",
	"tokens": {
		"color.primary": "#0F4C75",
		"color.accent": "#1B8A5A",
		"color.success": "#15803D",
		"color.warning": "#B45309",
		"color.danger": "#B91C1C",
		"surface.canvas": "#F0F7FF",
		"surface.panel": "#FFFFFF",
		"text.primary": "#0F172A",
		"text.secondary": "#475569",
		"border.radius": "8px",
		"density": "comfortable",
	},
	"components": {
		"tenants": {"icon": "user-circle", "status_indicator": "tenant-status-chip"},
		"onboarding": {"icon": "clipboard-check", "status_indicator": "onboarding-step-chip"},
		"service_requests": {"icon": "tool", "status_indicator": "request-status-chip"},
		"communications": {"icon": "message-square", "status_indicator": "channel-chip"},
		"satisfaction": {"icon": "star", "status_indicator": "rating-chip"},
		"scoring": {"icon": "award", "status_indicator": "credit-grade-chip"},
		"escalations": {"icon": "alert-triangle", "status_indicator": "escalation-type-chip"},
		"documents": {"icon": "file", "status_indicator": "document-type-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": TEN_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"tenant_registered", "tenant_onboarded", "tenant_activated", "tenant_vacated", "tenant_blacklisted",
		"onboarding_step_completed", "service_request_raised", "service_request_resolved",
		"satisfaction_survey_completed", "tenant_score_updated",
		"escalation_raised", "escalation_resolved",
		"communication_sent", "document_uploaded",
		"retention_risk_flagged",
	],
	"guardrails": [
		"blacklisted_tenant_activation_denied",
		"service_request_sla_breach_triggers_escalation",
		"satisfaction_below_threshold_triggers_account_review",
		"unauthorised_subletting_auto_escalation",
		"tenant_data_access_logged_always",
	],
}

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"tenants": {"supported_types": SUPPORTED_TENANT_TYPES, "supported_statuses": SUPPORTED_TENANT_STATUSES},
	"onboarding": {"supported_steps": SUPPORTED_ONBOARDING_STEPS, "mandatory_steps": ["referencing", "credit_check", "deposit_registration"]},
	"service_requests": {"supported_types": SUPPORTED_SERVICE_REQUEST_TYPES, "supported_statuses": SUPPORTED_REQUEST_STATUSES, "sla_response_hours": {"maintenance_request": 4, "emergency": 1, "general_enquiry": 24}},
	"communications": {"supported_channels": SUPPORTED_COMMUNICATION_CHANNELS, "portal_features": SUPPORTED_PORTAL_FEATURES},
	"satisfaction": {"supported_dimensions": SUPPORTED_SATISFACTION_DIMENSIONS, "ratings": SUPPORTED_SATISFACTION_RATINGS, "low_score_threshold": 3},
	"scoring": {"supported_models": SUPPORTED_SCORING_MODELS, "credit_grades": SUPPORTED_CREDIT_GRADES},
	"escalations": {"supported_types": SUPPORTED_ESCALATION_TYPES},
	"documents": {"supported_types": SUPPORTED_DOCUMENT_TYPES},
	"events": {"supported_types": SUPPORTED_EVENT_TYPES},
	"approvals": {"supported_levels": SUPPORTED_APPROVAL_LEVELS},
	"ui": {"enable_dashboard": True, "enable_portal": True, "enable_scoring": True, "enable_satisfaction": True},
	"theme": {"default_theme": "realestate_ten_portal", "allow_tenant_overrides": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "data_access_always_logged": True},
	"observability": {"event_stream": TEN_EVENT_STREAM, "stream_processor": "bytewax"},
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "tenant_management_policy_required", "required_action": "attach_tenant_policy"}},
	{"name": "tenant_type_supported", "condition": {"operation": "register_tenant", "tenant_type_supported": False}, "effect": {"decision": "deny", "reason": "tenant_type_not_supported", "required_action": "select_supported_tenant_type"}},
	{"name": "blacklisted_tenant_activation_denied", "condition": {"operation": "activate_tenant", "tenant_status": "blacklisted"}, "effect": {"decision": "deny", "reason": "blacklisted_tenant_cannot_be_activated", "required_action": "review_blacklist_status_with_management"}},
	{"name": "activation_requires_completed_onboarding", "condition": {"operation": "activate_tenant", "mandatory_onboarding_complete": False}, "effect": {"decision": "deny", "reason": "mandatory_onboarding_steps_must_be_complete", "required_action": "complete_mandatory_onboarding_steps"}},
	{"name": "service_request_type_supported", "condition": {"operation": "raise_service_request", "request_type_supported": False}, "effect": {"decision": "deny", "reason": "service_request_type_not_supported", "required_action": "select_supported_request_type"}},
	{"name": "service_request_requires_tenant", "condition": {"operation": "raise_service_request", "tenant_linked": False}, "effect": {"decision": "deny", "reason": "tenant_required_for_service_request", "required_action": "link_tenant_to_request"}},
	{"name": "sla_breach_triggers_escalation", "condition": {"operation": "update_service_request", "sla_breached": True, "escalated": False}, "effect": {"decision": "deny", "reason": "sla_breach_requires_automatic_escalation", "required_action": "escalate_service_request"}},
	{"name": "communication_channel_supported", "condition": {"operation": "send_communication", "channel_supported": False}, "effect": {"decision": "deny", "reason": "communication_channel_not_supported", "required_action": "select_supported_channel"}},
	{"name": "satisfaction_rating_valid", "condition": {"operation": "record_satisfaction", "rating_valid": False}, "effect": {"decision": "deny", "reason": "satisfaction_rating_must_be_1_to_5", "required_action": "provide_valid_rating"}},
	{"name": "scoring_model_supported", "condition": {"operation": "calculate_score", "scoring_model_supported": False}, "effect": {"decision": "deny", "reason": "scoring_model_not_supported", "required_action": "select_supported_scoring_model"}},
	{"name": "escalation_type_supported", "condition": {"operation": "raise_escalation", "escalation_type_supported": False}, "effect": {"decision": "deny", "reason": "escalation_type_not_supported", "required_action": "select_supported_escalation_type"}},
	{"name": "document_type_supported", "condition": {"operation": "upload_document", "document_type_supported": False}, "effect": {"decision": "deny", "reason": "document_type_not_supported", "required_action": "select_supported_document_type"}},
	{"name": "tenant_data_access_always_logged", "condition": {"operation": "access_tenant_data", "access_logged": False}, "effect": {"decision": "deny", "reason": "tenant_data_access_must_always_be_logged", "required_action": "enable_access_logging"}},
	{"name": "retention_risk_requires_account_review", "condition": {"operation": "flag_retention_risk", "account_review_scheduled": False}, "effect": {"decision": "deny", "reason": "retention_risk_requires_account_review_scheduling", "required_action": "schedule_account_review"}},
	{"name": "satisfaction_low_score_triggers_review", "condition": {"operation": "record_satisfaction", "score_below_threshold": True, "review_triggered": False}, "effect": {"decision": "deny", "reason": "low_satisfaction_score_must_trigger_account_review", "required_action": "trigger_account_review"}},
	{"name": "credit_grade_supported", "condition": {"operation": "assign_credit_grade", "grade_supported": False}, "effect": {"decision": "deny", "reason": "credit_grade_not_supported", "required_action": "select_supported_credit_grade"}},
	{"name": "portal_feature_supported", "condition": {"operation": "enable_portal_feature", "feature_supported": False}, "effect": {"decision": "deny", "reason": "portal_feature_not_supported", "required_action": "select_supported_portal_feature"}},
	{"name": "onboarding_step_sequence_enforced", "condition": {"operation": "complete_onboarding_step", "prerequisite_steps_complete": False}, "effect": {"decision": "deny", "reason": "prerequisite_onboarding_steps_must_be_completed_first", "required_action": "complete_prerequisite_steps"}},
	{"name": "cross_tenant_tenant_management_denied", "condition": {"operation_type": "write", "cross_tenant": True}, "effect": {"decision": "deny", "reason": "cross_tenant_tenant_management_not_allowed", "required_action": "use_correct_tenant_context"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	"""Return the full capability contract for the given tenant."""
	cfg = deepcopy(DEFAULT_CONFIGURATION)
	cfg["tenant_id"] = tenant_id
	return {
		"capability": CAPABILITY_ID,
		"display_name": CAPABILITY_NAME,
		"version": CAPABILITY_VERSION,
		"configuration": cfg,
		"configuration_schema": {
			"required": ["tenant_id", "ui", "theme"],
			"properties": {"tenant_id": {"type": "string"}, "ui": {"type": "object"}, "theme": {"type": "object"}},
		},
		"rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": RULES},
		"ui": {"shell": "apg_python", "requires_theme": True, "template_roots": ["realestate/ten/templates"], "routes": UI_ROUTES},
		"theme": THEME,
		"streaming": STREAMING,
		"provides": PROVIDES,
		"requires": REQUIRES,
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate all rules against context. Returns first denial or allow."""
	for rule in RULES:
		cond = rule["condition"]
		if all(context.get(k) == v for k, v in cond.items()):
			effect = rule["effect"]
			if effect["decision"] == "deny":
				return {"decision": "deny", "rule": rule["name"], "reason": effect["reason"], "required_action": effect.get("required_action")}
	return {"decision": "allow", "rule": None, "reason": None}
