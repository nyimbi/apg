"""Executable capability contract for APG Permits Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "government_per"
CAPABILITY_NAME = "Permits Management"
CAPABILITY_VERSION = "1.0.0"
PER_EVENT_STREAM = "apg.government.per.lifecycle"

SUPPORTED_PERMIT_TYPES = ["building", "environmental", "occupation", "demolition", "subdivision", "change_of_use", "signage", "road_works", "utility_connection", "water_abstraction", "mining_exploration"]
SUPPORTED_APPLICATION_STATUSES = ["draft", "submitted", "acknowledged", "under_review", "additional_info_required", "site_inspection_required", "conditionally_approved", "approved", "rejected", "appealed", "expired"]
SUPPORTED_PERMIT_STATUSES = ["active", "expired", "revoked", "suspended", "completed", "transferred"]
SUPPORTED_CONDITION_TYPES = ["pre_commencement", "during_construction", "pre_occupation", "ongoing", "financial_security", "environmental_mitigation"]
SUPPORTED_INSPECTION_TYPES = ["site_survey", "foundation", "structural", "plumbing", "electrical", "fire_safety", "final", "compliance"]
SUPPORTED_INSPECTION_OUTCOMES = ["pass", "conditional_pass", "fail", "deferred", "not_reached"]
SUPPORTED_COMPLIANCE_STATUSES = ["compliant", "minor_breach", "major_breach", "enforcement_pending", "enforcement_resolved"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["permit_assessor", "inspection_scheduler", "condition_monitor", "compliance_officer", "fee_collector"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"applications": {
		"supported_permit_types": SUPPORTED_PERMIT_TYPES,
		"supported_statuses": SUPPORTED_APPLICATION_STATUSES,
		"applicant_id_required": True,
		"site_reference_required": True,
		"fee_payment_required": True,
		"evidence_required": True,
	},
	"permits": {
		"supported_statuses": SUPPORTED_PERMIT_STATUSES,
		"approved_application_required": True,
		"permit_number_required": True,
		"expiry_date_required": True,
		"conditions_attached": True,
		"evidence_required": True,
	},
	"conditions": {
		"supported_condition_types": SUPPORTED_CONDITION_TYPES,
		"permit_required": True,
		"due_date_required": True,
		"responsible_party_required": True,
		"evidence_required": True,
	},
	"inspections": {
		"supported_inspection_types": SUPPORTED_INSPECTION_TYPES,
		"supported_outcomes": SUPPORTED_INSPECTION_OUTCOMES,
		"permit_required": True,
		"inspector_required": True,
		"scheduled_date_required": True,
		"evidence_required": True,
	},
	"compliance": {
		"supported_statuses": SUPPORTED_COMPLIANCE_STATUSES,
		"permit_required": True,
		"officer_required": True,
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
		"permit_without_payment_denied": True,
		"construction_before_permit_denied": True,
		"occupation_before_final_inspection_denied": True,
		"condition_breach_triggers_enforcement": True,
		"duplicate_permit_denied": True,
		"evidence_fabrication_denied": True,
	},
	"observability": {"event_stream": PER_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {
		"auth": "auth",
		"audit": "audl",
		"notifications": "ntfy",
		"workflow": "wflo",
		"geospatial": "geos",
		"scheduling": "schd",
		"compliance": "comp",
		"monitoring": "moni",
		"event_stream": "bytewax",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_applications": True,
		"enable_permits": True,
		"enable_conditions": True,
		"enable_inspections": True,
		"enable_compliance": True,
		"enable_reviews": True,
		"enable_agents": True,
	},
	"theme": {"default_theme": "government_per_control", "allow_tenant_overrides": True},
}

PROVIDES = [
	"permit_application_workflow",
	"permit_issuance_workflow",
	"conditional_approval_workflow",
	"inspection_scheduling_workflow",
	"permit_compliance_monitoring_workflow",
	"permit_revocation_workflow",
	"permits_review_workflow",
	"permits_agent_workflow",
	"permit_transfer_workflow",
	"enforcement_action_workflow",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "geos", "schd", "comp", "moni", "mqeb"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/government-per/dashboard", "component": "PermitsDashboard", "permission": "government_per:view", "nav_group": "Overview"},
	{"name": "applications", "path": "/government-per/applications", "component": "PermitApplicationConsole", "permission": "government_per:apply", "nav_group": "Applications"},
	{"name": "permits", "path": "/government-per/permits", "component": "PermitRegister", "permission": "government_per:permits", "nav_group": "Permits"},
	{"name": "conditions", "path": "/government-per/conditions", "component": "PermitConditionsConsole", "permission": "government_per:conditions", "nav_group": "Conditions"},
	{"name": "inspections", "path": "/government-per/inspections", "component": "InspectionScheduleConsole", "permission": "government_per:inspect", "nav_group": "Inspections"},
	{"name": "compliance", "path": "/government-per/compliance", "component": "ComplianceMonitoringConsole", "permission": "government_per:compliance", "nav_group": "Compliance"},
	{"name": "map", "path": "/government-per/map", "component": "PermitSiteMap", "permission": "government_per:view", "nav_group": "Geography"},
	{"name": "enforcement", "path": "/government-per/enforcement", "component": "EnforcementActionConsole", "permission": "government_per:enforce", "nav_group": "Compliance"},
	{"name": "reviews", "path": "/government-per/reviews", "component": "PermitsReviewConsole", "permission": "government_per:review", "nav_group": "Governance"},
	{"name": "agents", "path": "/government-per/agents", "component": "PermitsAgentWorkbench", "permission": "government_per:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/government-per/settings", "component": "PermitsSettings", "permission": "government_per:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "government_per_control",
	"tokens": {
		"color.primary": "#065F46",
		"color.accent": "#D97706",
		"color.success": "#166534",
		"color.warning": "#A16207",
		"color.danger": "#991B1B",
		"surface.canvas": "#ECFDF5",
		"surface.panel": "#FFFFFF",
		"text.primary": "#022C22",
		"text.secondary": "#374151",
		"border.radius": "8px",
		"density": "comfortable",
	},
	"components": {
		"applications": {"icon": "file-plus", "status_indicator": "application-status-chip"},
		"permits": {"icon": "stamp", "status_indicator": "permit-status-chip"},
		"conditions": {"icon": "list-checks", "status_indicator": "condition-type-chip"},
		"inspections": {"icon": "hard-hat", "status_indicator": "inspection-outcome-chip"},
		"compliance": {"icon": "shield", "status_indicator": "compliance-status-chip"},
		"reviews": {"icon": "clipboard-check", "status_indicator": "review-status-chip"},
		"agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": PER_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"permit_application_submitted",
		"permit_issued",
		"permit_condition_recorded",
		"inspection_scheduled",
		"inspection_outcome_recorded",
		"permit_compliance_updated",
		"permit_revoked",
		"enforcement_action_initiated",
		"permits_agent_registered",
		"permit_completed",
	],
	"guardrails": [
		"per_batch_requires_bytewax",
		"permit_without_payment_denied",
		"construction_before_permit_denied",
		"occupation_before_final_inspection_denied",
		"condition_breach_triggers_enforcement",
		"evidence_fabrication_denied",
		"privileged_permits_agent_action_requires_human_approval",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "per_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "permits_policy_required", "required_action": "attach_permits_policy"}},
	{"name": "permit_type_supported", "condition": {"operation": "submit_application", "permit_type_supported": False}, "effect": {"decision": "deny", "reason": "permit_type_not_supported", "required_action": "select_supported_permit_type"}},
	{"name": "application_applicant_required", "condition": {"operation": "submit_application", "applicant_id_present": False}, "effect": {"decision": "deny", "reason": "applicant_id_required", "required_action": "provide_applicant_id"}},
	{"name": "application_site_required", "condition": {"operation": "submit_application", "site_reference_present": False}, "effect": {"decision": "deny", "reason": "site_reference_required", "required_action": "provide_site_reference"}},
	{"name": "application_fee_required", "condition": {"operation": "submit_application", "fee_paid": False}, "effect": {"decision": "deny", "reason": "application_fee_required", "required_action": "pay_application_fee"}},
	{"name": "application_evidence_required", "condition": {"operation": "submit_application", "evidence_present": False}, "effect": {"decision": "deny", "reason": "application_evidence_required", "required_action": "upload_supporting_documents"}},
	{"name": "permit_approved_application_required", "condition": {"operation": "issue_permit", "approved_application_present": False}, "effect": {"decision": "deny", "reason": "approved_application_required", "required_action": "approve_application_first"}},
	{"name": "permit_number_required", "condition": {"operation": "issue_permit", "permit_number_present": False}, "effect": {"decision": "deny", "reason": "permit_number_required", "required_action": "generate_permit_number"}},
	{"name": "permit_expiry_required", "condition": {"operation": "issue_permit", "expiry_date_present": False}, "effect": {"decision": "deny", "reason": "expiry_date_required", "required_action": "set_permit_expiry"}},
	{"name": "condition_type_supported", "condition": {"operation": "record_condition", "condition_type_supported": False}, "effect": {"decision": "deny", "reason": "condition_type_not_supported", "required_action": "select_supported_condition_type"}},
	{"name": "condition_permit_required", "condition": {"operation": "record_condition", "permit_present": False}, "effect": {"decision": "deny", "reason": "permit_required", "required_action": "select_permit"}},
	{"name": "condition_due_date_required", "condition": {"operation": "record_condition", "due_date_present": False}, "effect": {"decision": "deny", "reason": "due_date_required", "required_action": "set_condition_due_date"}},
	{"name": "inspection_type_supported", "condition": {"operation": "schedule_inspection", "inspection_type_supported": False}, "effect": {"decision": "deny", "reason": "inspection_type_not_supported", "required_action": "select_supported_inspection_type"}},
	{"name": "inspection_permit_required", "condition": {"operation": "schedule_inspection", "permit_present": False}, "effect": {"decision": "deny", "reason": "permit_required", "required_action": "select_permit"}},
	{"name": "inspection_inspector_required", "condition": {"operation": "schedule_inspection", "inspector_present": False}, "effect": {"decision": "deny", "reason": "inspector_required", "required_action": "assign_inspector"}},
	{"name": "occupation_final_inspection_required", "condition": {"operation": "grant_occupation", "final_inspection_passed": False}, "effect": {"decision": "deny", "reason": "final_inspection_required_before_occupation", "required_action": "pass_final_inspection"}},
	{"name": "construction_before_permit_denied", "condition": {"operation": "record_commencement", "permit_active": False}, "effect": {"decision": "deny", "reason": "construction_before_permit_denied", "required_action": "obtain_permit_first"}},
	{"name": "compliance_status_supported", "condition": {"operation": "record_compliance", "compliance_status_supported": False}, "effect": {"decision": "deny", "reason": "compliance_status_not_supported", "required_action": "select_supported_compliance_status"}},
	{"name": "duplicate_permit_denied", "condition": {"operation": "issue_permit", "duplicate_detected": True}, "effect": {"decision": "deny", "reason": "duplicate_permit_denied", "required_action": "resolve_duplicate_permit"}},
	{"name": "per_batch_requires_bytewax", "condition": {"operation": "per_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_per_batch_to_bytewax"}},
	{"name": "per_agent_runtime_supported", "condition": {"operation": "register_per_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "per_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "per_agent_role_supported", "condition": {"operation": "register_per_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "per_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "per_agent_name_required", "condition": {"operation": "register_per_agent", "agent_name_present": False}, "effect": {"decision": "deny", "reason": "per_agent_name_required", "required_action": "name_per_agent"}},
	{"name": "per_agent_scope_required", "condition": {"operation": "register_per_agent", "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "per_agent_scope_required", "required_action": "bound_per_agent_scope"}},
	{"name": "privileged_permits_agent_action_requires_human_approval", "condition": {"operation": "per_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
	{"name": "evidence_fabrication_denied", "condition": {"operation": "per_agent_action", "evidence_fabrication_scope": True}, "effect": {"decision": "deny", "reason": "evidence_fabrication_denied", "required_action": "remove_evidence_fabrication_scope"}},
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
			"api_prefix": "/government-per/api/v1",
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
