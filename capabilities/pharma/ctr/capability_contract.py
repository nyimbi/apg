"""Executable capability contract for APG Pharma Clinical Trials Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "pharma_ctr"
CAPABILITY_NAME = "Clinical Trials Management"
CAPABILITY_VERSION = "1.0.0"
CTR_EVENT_STREAM = "apg.pharma.ctr.lifecycle"

SUPPORTED_TRIAL_PHASES = ["phase_1", "phase_1b", "phase_2", "phase_2b", "phase_3", "phase_3b", "phase_4", "expanded_access", "observational"]
SUPPORTED_TRIAL_TYPES = ["interventional", "observational", "expanded_access", "registry", "bioequivalence", "first_in_human", "basket", "umbrella"]
SUPPORTED_SITE_STATUSES = ["pre_selected", "selected", "initiated", "enrolling", "enrollment_complete", "closed", "terminated", "withdrawn"]
SUPPORTED_PATIENT_STATUSES = ["screened", "enrolled", "randomised", "on_treatment", "completed", "withdrawn", "lost_to_follow_up", "screen_failure"]
SUPPORTED_AE_SEVERITIES = ["grade_1", "grade_2", "grade_3", "grade_4", "grade_5"]
SUPPORTED_AE_TYPES = ["adverse_event", "serious_adverse_event", "suspected_unexpected_serious_adverse_reaction", "disease_related_event", "protocol_deviation"]
SUPPORTED_SUBMISSION_TYPES = ["ind", "cta", "protocol_amendment", "annual_report", "safety_report", "final_report", "eudract", "ctis"]
SUPPORTED_RANDOMISATION_METHODS = ["simple", "stratified", "block", "adaptive", "minimisation", "dynamic"]
SUPPORTED_BLINDING_TYPES = ["open_label", "single_blind", "double_blind", "triple_blind"]
SUPPORTED_DATA_STATUSES = ["draft", "pending_query", "query_resolved", "signed_off", "locked", "unlocked"]
SUPPORTED_PROTOCOL_STATUSES = ["draft", "under_review", "approved", "amended", "superseded", "withdrawn"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["site_qualifier", "patient_screener", "ae_reviewer", "data_monitor", "submission_preparer"]
SUPPORTED_REGULATORY_AUTHORITIES = ["fda", "ema", "mhra", "pmda", "health_canada", "tga", "anvisa", "cdsco", "nmpa"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"trials": {"supported_phases": SUPPORTED_TRIAL_PHASES, "supported_types": SUPPORTED_TRIAL_TYPES, "irb_approval_required": True, "sponsor_required": True, "cro_allowed": True},
	"protocols": {"supported_statuses": SUPPORTED_PROTOCOL_STATUSES, "version_control_required": True, "amendment_tracking": True, "irb_review_required": True},
	"sites": {"supported_statuses": SUPPORTED_SITE_STATUSES, "qualification_visit_required": True, "initiation_visit_required": True, "monitoring_plan_required": True},
	"patients": {"supported_statuses": SUPPORTED_PATIENT_STATUSES, "ic_required": True, "eligibility_check_required": True, "randomisation_required": True},
	"randomisation": {"supported_methods": SUPPORTED_RANDOMISATION_METHODS, "supported_blinding": SUPPORTED_BLINDING_TYPES, "ivrs_integration": True, "audit_trail_required": True},
	"adverse_events": {"supported_severities": SUPPORTED_AE_SEVERITIES, "supported_types": SUPPORTED_AE_TYPES, "reporting_timeline_hours": {"sadie": 24, "susar": 15, "ae": 7}, "meddra_coding_required": True},
	"data_management": {"supported_statuses": SUPPORTED_DATA_STATUSES, "edc_integration": True, "query_management": True, "data_lock_required": True, "21cfr11_compliant": True},
	"submissions": {"supported_types": SUPPORTED_SUBMISSION_TYPES, "supported_authorities": SUPPORTED_REGULATORY_AUTHORITIES, "cover_letter_required": True, "dossier_reference_required": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "gcp_compliance_required": True, "irb_approval_required": True, "informed_consent_required": True, "cross_tenant_denied": True},
	"observability": {"event_stream": CTR_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "workflow": "wflo", "compliance": "comp", "nlp": "nlpc", "event_stream": "mqeb"},
	"ui": {"enable_dashboard": True, "enable_trials": True, "enable_protocols": True, "enable_sites": True, "enable_patients": True, "enable_randomisation": True, "enable_adverse_events": True, "enable_data_management": True, "enable_submissions": True},
	"theme": {"default_theme": "pharma_ctr_clinical", "allow_tenant_overrides": True},
}

PROVIDES = [
	"trial_protocol_workflow",
	"site_selection_workflow",
	"patient_randomisation_workflow",
	"adverse_event_workflow",
	"clinical_data_management_workflow",
	"regulatory_submission_workflow",
	"informed_consent_workflow",
	"monitoring_visit_workflow",
	"safety_reporting_workflow",
	"trial_closure_workflow",
]
REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "comp", "nlpc", "mqeb"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/pharma-ctr/dashboard", "component": "CtrDashboard", "permission": "pharma_ctr:view", "nav_group": "Overview"},
	{"name": "trials", "path": "/pharma-ctr/trials", "component": "TrialRegistry", "permission": "pharma_ctr:trials", "nav_group": "Trials"},
	{"name": "trial_detail", "path": "/pharma-ctr/trials/<id>", "component": "TrialDetail", "permission": "pharma_ctr:trials", "nav_group": "Trials"},
	{"name": "protocols", "path": "/pharma-ctr/protocols", "component": "ProtocolWorkbench", "permission": "pharma_ctr:protocols", "nav_group": "Protocols"},
	{"name": "sites", "path": "/pharma-ctr/sites", "component": "SiteManagement", "permission": "pharma_ctr:sites", "nav_group": "Sites"},
	{"name": "patients", "path": "/pharma-ctr/patients", "component": "PatientTracker", "permission": "pharma_ctr:patients", "nav_group": "Patients"},
	{"name": "randomisation", "path": "/pharma-ctr/randomisation", "component": "RandomisationConsole", "permission": "pharma_ctr:randomisation", "nav_group": "Patients"},
	{"name": "adverse_events", "path": "/pharma-ctr/adverse-events", "component": "AdverseEventQueue", "permission": "pharma_ctr:ae", "nav_group": "Safety"},
	{"name": "safety_reports", "path": "/pharma-ctr/safety-reports", "component": "SafetyReportConsole", "permission": "pharma_ctr:safety", "nav_group": "Safety"},
	{"name": "data_management", "path": "/pharma-ctr/data", "component": "DataManagementConsole", "permission": "pharma_ctr:data", "nav_group": "Data"},
	{"name": "submissions", "path": "/pharma-ctr/submissions", "component": "SubmissionTracker", "permission": "pharma_ctr:submissions", "nav_group": "Regulatory"},
	{"name": "monitoring", "path": "/pharma-ctr/monitoring", "component": "MonitoringVisitLog", "permission": "pharma_ctr:monitoring", "nav_group": "Oversight"},
	{"name": "reports", "path": "/pharma-ctr/reports", "component": "ClinicalReports", "permission": "pharma_ctr:reports", "nav_group": "Reporting"},
	{"name": "settings", "path": "/pharma-ctr/settings", "component": "CtrSettings", "permission": "pharma_ctr:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "pharma_ctr_clinical",
	"tokens": {
		"color.primary": "#0F4C81",
		"color.accent": "#00897B",
		"color.success": "#1B5E20",
		"color.warning": "#E65100",
		"color.danger": "#B71C1C",
		"surface.canvas": "#F1F5F9",
		"surface.panel": "#FFFFFF",
		"text.primary": "#0D1B2A",
		"text.secondary": "#455A64",
		"border.radius": "4px",
		"density": "compact",
	},
	"components": {
		"trials": {"icon": "flask", "status_indicator": "trial-phase-chip"},
		"protocols": {"icon": "file-text", "status_indicator": "protocol-status-chip"},
		"sites": {"icon": "building-2", "status_indicator": "site-status-chip"},
		"patients": {"icon": "user-check", "status_indicator": "patient-status-chip"},
		"adverse_events": {"icon": "alert-triangle", "status_indicator": "ae-severity-chip"},
		"submissions": {"icon": "send", "status_indicator": "submission-type-chip"},
		"data_management": {"icon": "database", "status_indicator": "data-status-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": CTR_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"trial_created", "protocol_approved", "site_initiated", "patient_enrolled",
		"patient_randomised", "adverse_event_reported", "susar_reported",
		"data_query_raised", "data_locked", "submission_filed",
		"monitoring_visit_completed", "trial_closed",
	],
	"guardrails": [
		"gcp_compliance_required",
		"informed_consent_required_before_enrolment",
		"irb_approval_required_before_activation",
		"ae_reporting_timeline_enforced",
		"susar_reporting_timeline_enforced",
		"data_integrity_controls_enforced",
		"cross_tenant_trial_data_denied",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "policy_required", "required_action": "attach_policy"}},
	{"name": "trial_phase_supported", "condition": {"operation": "create_trial", "trial_phase_supported": False}, "effect": {"decision": "deny", "reason": "trial_phase_not_supported", "required_action": "select_supported_phase"}},
	{"name": "trial_irb_approval_required", "condition": {"operation": "activate_trial", "irb_approved": False}, "effect": {"decision": "deny", "reason": "irb_approval_required", "required_action": "obtain_irb_approval"}},
	{"name": "trial_sponsor_required", "condition": {"operation": "create_trial", "sponsor_present": False}, "effect": {"decision": "deny", "reason": "sponsor_required", "required_action": "identify_sponsor"}},
	{"name": "protocol_version_required", "condition": {"operation": "approve_protocol", "version_present": False}, "effect": {"decision": "deny", "reason": "protocol_version_required", "required_action": "version_protocol"}},
	{"name": "protocol_irb_review_required", "condition": {"operation": "approve_protocol", "irb_reviewed": False}, "effect": {"decision": "deny", "reason": "irb_review_required", "required_action": "submit_to_irb"}},
	{"name": "site_qualification_required", "condition": {"operation": "initiate_site", "qualification_visit_completed": False}, "effect": {"decision": "deny", "reason": "qualification_visit_required", "required_action": "complete_qualification_visit"}},
	{"name": "site_initiation_required", "condition": {"operation": "enrol_patient", "site_initiated": False}, "effect": {"decision": "deny", "reason": "site_not_initiated", "required_action": "complete_site_initiation"}},
	{"name": "patient_ic_required", "condition": {"operation": "enrol_patient", "informed_consent_obtained": False}, "effect": {"decision": "deny", "reason": "informed_consent_required", "required_action": "obtain_informed_consent"}},
	{"name": "patient_eligibility_required", "condition": {"operation": "enrol_patient", "eligibility_confirmed": False}, "effect": {"decision": "deny", "reason": "eligibility_confirmation_required", "required_action": "confirm_eligibility"}},
	{"name": "randomisation_method_supported", "condition": {"operation": "randomise_patient", "randomisation_method_supported": False}, "effect": {"decision": "deny", "reason": "randomisation_method_not_supported", "required_action": "select_supported_method"}},
	{"name": "ae_meddra_coding_required", "condition": {"operation": "report_ae", "meddra_coded": False}, "effect": {"decision": "deny", "reason": "meddra_coding_required", "required_action": "apply_meddra_coding"}},
	{"name": "ae_severity_supported", "condition": {"operation": "report_ae", "ae_severity_supported": False}, "effect": {"decision": "deny", "reason": "ae_severity_not_supported", "required_action": "select_supported_severity"}},
	{"name": "sadie_24h_reporting_required", "condition": {"operation": "report_ae", "ae_type": "serious_adverse_event", "within_24h": False}, "effect": {"decision": "deny", "reason": "sadie_24h_reporting_required", "required_action": "expedite_ae_report"}},
	{"name": "susar_15d_reporting_required", "condition": {"operation": "report_ae", "ae_type": "suspected_unexpected_serious_adverse_reaction", "within_15d": False}, "effect": {"decision": "deny", "reason": "susar_15d_reporting_required", "required_action": "expedite_susar_report"}},
	{"name": "data_lock_requires_query_resolution", "condition": {"operation": "lock_data", "open_queries_present": True}, "effect": {"decision": "deny", "reason": "open_queries_must_be_resolved", "required_action": "resolve_open_queries"}},
	{"name": "submission_authority_supported", "condition": {"operation": "file_submission", "authority_supported": False}, "effect": {"decision": "deny", "reason": "regulatory_authority_not_supported", "required_action": "select_supported_authority"}},
	{"name": "submission_cover_letter_required", "condition": {"operation": "file_submission", "cover_letter_present": False}, "effect": {"decision": "deny", "reason": "cover_letter_required", "required_action": "attach_cover_letter"}},
	{"name": "cross_tenant_denied", "condition": {"operation_type": "write", "cross_tenant": True}, "effect": {"decision": "deny", "reason": "cross_tenant_operation_denied", "required_action": "use_correct_tenant_context"}},
	{"name": "gcp_compliance_required", "condition": {"operation_type": "write", "gcp_compliant": False}, "effect": {"decision": "deny", "reason": "gcp_compliance_required", "required_action": "ensure_gcp_compliance"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	return {
		"capability": CAPABILITY_ID,
		"display_name": CAPABILITY_NAME,
		"version": CAPABILITY_VERSION,
		"provides": list(PROVIDES),
		"requires": list(REQUIRES),
		"configuration": configuration,
		"configuration_schema": {
			"type": "object",
			"required": ["tenant_id", "ui", "theme"],
			"properties": {
				"tenant_id": {"type": "string", "minLength": 1},
				"ui": {"type": "object"},
				"theme": {"type": "object"},
			},
		},
		"rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"api_prefix": "/pharma-ctr/api/v1",
			"requires_theme": True,
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
