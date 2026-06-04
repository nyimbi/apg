"""Executable capability contract for APG Licensing & Permits."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "government_lic"
CAPABILITY_NAME = "Licensing and Permits"
CAPABILITY_VERSION = "1.0.0"
LIC_EVENT_STREAM = "apg.government.lic.lifecycle"

SUPPORTED_LICENCE_TYPES = ["business", "professional", "trade", "liquor", "food_hygiene", "pharmacy", "medical_practice", "firearms", "broadcast", "motor_vehicle_dealer", "export_import"]
SUPPORTED_APPLICATION_STATUSES = ["draft", "submitted", "acknowledged", "documents_pending", "under_review", "inspection_required", "approved", "rejected", "appealed", "withdrawn"]
SUPPORTED_LICENCE_STATUSES = ["active", "suspended", "revoked", "expired", "cancelled", "under_renewal"]
SUPPORTED_INSPECTION_TYPES = ["initial", "renewal", "complaint_triggered", "random", "follow_up", "pre_licence"]
SUPPORTED_INSPECTION_OUTCOMES = ["pass", "conditional_pass", "fail", "deferred", "not_applicable"]
SUPPORTED_RENEWAL_TYPES = ["standard", "early", "late", "conditional"]
SUPPORTED_FEE_TYPES = ["application_fee", "licence_fee", "inspection_fee", "penalty", "late_fee", "reinstatement_fee"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["application_processor", "inspection_scheduler", "renewal_notifier", "fee_collector", "revocation_reviewer"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"applications": {
		"supported_licence_types": SUPPORTED_LICENCE_TYPES,
		"supported_statuses": SUPPORTED_APPLICATION_STATUSES,
		"applicant_id_required": True,
		"business_registration_required": True,
		"fee_payment_required": True,
		"evidence_required": True,
	},
	"licences": {
		"supported_statuses": SUPPORTED_LICENCE_STATUSES,
		"approved_application_required": True,
		"expiry_date_required": True,
		"licence_number_required": True,
		"evidence_required": True,
	},
	"inspections": {
		"supported_inspection_types": SUPPORTED_INSPECTION_TYPES,
		"supported_outcomes": SUPPORTED_INSPECTION_OUTCOMES,
		"licence_required": True,
		"inspector_required": True,
		"scheduled_date_required": True,
		"evidence_required": True,
	},
	"renewals": {
		"supported_renewal_types": SUPPORTED_RENEWAL_TYPES,
		"licence_required": True,
		"fee_payment_required": True,
		"inspection_required_flag": True,
		"evidence_required": True,
	},
	"fees": {
		"supported_fee_types": SUPPORTED_FEE_TYPES,
		"application_required": True,
		"receipt_required": True,
		"reconciliation_enabled": True,
	},
	"revocations": {
		"licence_required": True,
		"reason_required": True,
		"approval_required": True,
		"notice_period_enforced": True,
		"evidence_required": True,
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
		"licence_without_payment_denied": True,
		"expired_licence_operation_denied": True,
		"revocation_without_notice_denied": True,
		"inspection_fail_blocks_renewal": True,
		"duplicate_licence_denied": True,
		"evidence_fabrication_denied": True,
	},
	"observability": {"event_stream": LIC_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {
		"auth": "auth",
		"audit": "audl",
		"notifications": "ntfy",
		"workflow": "wflo",
		"scheduling": "schd",
		"compliance": "comp",
		"monitoring": "moni",
		"event_stream": "bytewax",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_applications": True,
		"enable_licences": True,
		"enable_inspections": True,
		"enable_renewals": True,
		"enable_fees": True,
		"enable_revocations": True,
		"enable_reviews": True,
		"enable_agents": True,
	},
	"theme": {"default_theme": "government_lic_control", "allow_tenant_overrides": True},
}

PROVIDES = [
	"licence_application_workflow",
	"licence_issuance_workflow",
	"inspection_scheduling_workflow",
	"licence_renewal_workflow",
	"fee_collection_workflow",
	"licence_revocation_workflow",
	"licensing_review_workflow",
	"licensing_agent_workflow",
	"licence_status_tracking_workflow",
	"compliance_monitoring_workflow",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "schd", "comp", "moni", "mqeb"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/government-lic/dashboard", "component": "LicensingDashboard", "permission": "government_lic:view", "nav_group": "Overview"},
	{"name": "applications", "path": "/government-lic/applications", "component": "LicenceApplicationConsole", "permission": "government_lic:apply", "nav_group": "Applications"},
	{"name": "licences", "path": "/government-lic/licences", "component": "LicenceRegister", "permission": "government_lic:licences", "nav_group": "Licences"},
	{"name": "inspections", "path": "/government-lic/inspections", "component": "InspectionScheduleConsole", "permission": "government_lic:inspect", "nav_group": "Inspections"},
	{"name": "renewals", "path": "/government-lic/renewals", "component": "LicenceRenewalConsole", "permission": "government_lic:renew", "nav_group": "Renewals"},
	{"name": "fees", "path": "/government-lic/fees", "component": "FeeCollectionConsole", "permission": "government_lic:fees", "nav_group": "Payments"},
	{"name": "revocations", "path": "/government-lic/revocations", "component": "LicenceRevocationConsole", "permission": "government_lic:revoke", "nav_group": "Compliance"},
	{"name": "compliance", "path": "/government-lic/compliance", "component": "LicensingComplianceDashboard", "permission": "government_lic:compliance", "nav_group": "Compliance"},
	{"name": "reviews", "path": "/government-lic/reviews", "component": "LicensingReviewConsole", "permission": "government_lic:review", "nav_group": "Governance"},
	{"name": "agents", "path": "/government-lic/agents", "component": "LicensingAgentWorkbench", "permission": "government_lic:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/government-lic/settings", "component": "LicensingSettings", "permission": "government_lic:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "government_lic_control",
	"tokens": {
		"color.primary": "#0369A1",
		"color.accent": "#7E22CE",
		"color.success": "#166534",
		"color.warning": "#A16207",
		"color.danger": "#991B1B",
		"surface.canvas": "#F0F9FF",
		"surface.panel": "#FFFFFF",
		"text.primary": "#0C1A2E",
		"text.secondary": "#475569",
		"border.radius": "8px",
		"density": "comfortable",
	},
	"components": {
		"applications": {"icon": "file-plus", "status_indicator": "application-status-chip"},
		"licences": {"icon": "badge-check", "status_indicator": "licence-status-chip"},
		"inspections": {"icon": "search", "status_indicator": "inspection-outcome-chip"},
		"renewals": {"icon": "refresh-cw", "status_indicator": "renewal-type-chip"},
		"fees": {"icon": "dollar-sign", "status_indicator": "fee-type-chip"},
		"revocations": {"icon": "x-circle", "status_indicator": "revocation-chip"},
		"reviews": {"icon": "clipboard-check", "status_indicator": "review-status-chip"},
		"agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": LIC_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"licence_application_submitted",
		"licence_issued",
		"inspection_scheduled",
		"inspection_outcome_recorded",
		"licence_renewed",
		"fee_collected",
		"licence_suspended",
		"licence_revoked",
		"licensing_agent_registered",
		"licence_expired",
	],
	"guardrails": [
		"lic_batch_requires_bytewax",
		"licence_without_payment_denied",
		"expired_licence_operation_denied",
		"revocation_without_notice_denied",
		"inspection_fail_blocks_renewal",
		"duplicate_licence_denied",
		"evidence_fabrication_denied",
		"privileged_licensing_agent_action_requires_human_approval",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "lic_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "licensing_policy_required", "required_action": "attach_licensing_policy"}},
	{"name": "licence_type_supported", "condition": {"operation": "submit_application", "licence_type_supported": False}, "effect": {"decision": "deny", "reason": "licence_type_not_supported", "required_action": "select_supported_licence_type"}},
	{"name": "application_applicant_required", "condition": {"operation": "submit_application", "applicant_id_present": False}, "effect": {"decision": "deny", "reason": "applicant_id_required", "required_action": "provide_applicant_id"}},
	{"name": "application_fee_required", "condition": {"operation": "submit_application", "fee_paid": False}, "effect": {"decision": "deny", "reason": "application_fee_required", "required_action": "pay_application_fee"}},
	{"name": "application_evidence_required", "condition": {"operation": "submit_application", "evidence_present": False}, "effect": {"decision": "deny", "reason": "application_evidence_required", "required_action": "upload_supporting_documents"}},
	{"name": "licence_approved_application_required", "condition": {"operation": "issue_licence", "approved_application_present": False}, "effect": {"decision": "deny", "reason": "approved_application_required", "required_action": "approve_application_first"}},
	{"name": "licence_number_required", "condition": {"operation": "issue_licence", "licence_number_present": False}, "effect": {"decision": "deny", "reason": "licence_number_required", "required_action": "generate_licence_number"}},
	{"name": "licence_expiry_required", "condition": {"operation": "issue_licence", "expiry_date_present": False}, "effect": {"decision": "deny", "reason": "expiry_date_required", "required_action": "set_expiry_date"}},
	{"name": "inspection_type_supported", "condition": {"operation": "schedule_inspection", "inspection_type_supported": False}, "effect": {"decision": "deny", "reason": "inspection_type_not_supported", "required_action": "select_supported_inspection_type"}},
	{"name": "inspection_licence_required", "condition": {"operation": "schedule_inspection", "licence_present": False}, "effect": {"decision": "deny", "reason": "licence_required", "required_action": "select_licence"}},
	{"name": "inspection_inspector_required", "condition": {"operation": "schedule_inspection", "inspector_present": False}, "effect": {"decision": "deny", "reason": "inspector_required", "required_action": "assign_inspector"}},
	{"name": "inspection_outcome_supported", "condition": {"operation": "record_inspection_outcome", "outcome_supported": False}, "effect": {"decision": "deny", "reason": "inspection_outcome_not_supported", "required_action": "select_supported_outcome"}},
	{"name": "renewal_inspection_fail_blocks", "condition": {"operation": "renew_licence", "last_inspection_failed": True}, "effect": {"decision": "deny", "reason": "inspection_fail_blocks_renewal", "required_action": "pass_inspection_before_renewal"}},
	{"name": "renewal_fee_required", "condition": {"operation": "renew_licence", "renewal_fee_paid": False}, "effect": {"decision": "deny", "reason": "renewal_fee_required", "required_action": "pay_renewal_fee"}},
	{"name": "revocation_reason_required", "condition": {"operation": "revoke_licence", "reason_present": False}, "effect": {"decision": "deny", "reason": "revocation_reason_required", "required_action": "provide_revocation_reason"}},
	{"name": "revocation_approval_required", "condition": {"operation": "revoke_licence", "approval_present": False}, "effect": {"decision": "deny", "reason": "revocation_approval_required", "required_action": "obtain_revocation_approval"}},
	{"name": "revocation_notice_required", "condition": {"operation": "revoke_licence", "notice_served": False}, "effect": {"decision": "deny", "reason": "notice_period_required", "required_action": "serve_revocation_notice"}},
	{"name": "duplicate_licence_denied", "condition": {"operation": "issue_licence", "duplicate_detected": True}, "effect": {"decision": "deny", "reason": "duplicate_licence_denied", "required_action": "resolve_duplicate_licence"}},
	{"name": "fee_type_supported", "condition": {"operation": "collect_fee", "fee_type_supported": False}, "effect": {"decision": "deny", "reason": "fee_type_not_supported", "required_action": "select_supported_fee_type"}},
	{"name": "fee_receipt_required", "condition": {"operation": "collect_fee", "receipt_present": False}, "effect": {"decision": "deny", "reason": "fee_receipt_required", "required_action": "generate_receipt"}},
	{"name": "lic_batch_requires_bytewax", "condition": {"operation": "lic_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_lic_batch_to_bytewax"}},
	{"name": "lic_agent_runtime_supported", "condition": {"operation": "register_lic_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "lic_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "lic_agent_role_supported", "condition": {"operation": "register_lic_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "lic_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "lic_agent_name_required", "condition": {"operation": "register_lic_agent", "agent_name_present": False}, "effect": {"decision": "deny", "reason": "lic_agent_name_required", "required_action": "name_lic_agent"}},
	{"name": "lic_agent_scope_required", "condition": {"operation": "register_lic_agent", "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "lic_agent_scope_required", "required_action": "bound_lic_agent_scope"}},
	{"name": "privileged_licensing_agent_action_requires_human_approval", "condition": {"operation": "lic_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
	{"name": "evidence_fabrication_denied", "condition": {"operation": "lic_agent_action", "evidence_fabrication_scope": True}, "effect": {"decision": "deny", "reason": "evidence_fabrication_denied", "required_action": "remove_evidence_fabrication_scope"}},
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
			"api_prefix": "/government-lic/api/v1",
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
