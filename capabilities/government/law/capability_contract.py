"""Executable capability contract for APG Law Enforcement & Justice."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "government_law"
CAPABILITY_NAME = "Law Enforcement and Justice"
CAPABILITY_VERSION = "1.0.0"
LAW_EVENT_STREAM = "apg.government.law.lifecycle"

SUPPORTED_INCIDENT_TYPES = ["theft", "assault", "fraud", "murder", "traffic_offence", "cybercrime", "corruption", "drug_offence", "sexual_offence", "property_offence", "public_order", "missing_person"]
SUPPORTED_DOCKET_STATUSES = ["open", "under_investigation", "forwarded_to_dpp", "charged", "trial_ongoing", "conviction", "acquittal", "withdrawn", "nolle_prosequi", "closed"]
SUPPORTED_EVIDENCE_TYPES = ["physical", "digital", "documentary", "forensic", "witness_statement", "cctv", "dna", "fingerprint", "audio_recording"]
SUPPORTED_CUSTODY_ACTIONS = ["seized", "logged", "transferred", "examined", "returned", "destroyed", "court_submitted"]
SUPPORTED_COURT_TYPES = ["magistrates_court", "high_court", "court_of_appeal", "supreme_court", "anti_corruption_court", "employment_court", "environment_court"]
SUPPORTED_HEARING_TYPES = ["mention", "plea_taking", "hearing", "ruling", "sentencing", "bail_application", "committal"]
SUPPORTED_PROSECUTION_STATUSES = ["dpp_reviewing", "charges_filed", "summons_issued", "warrant_issued", "prosecution_ongoing", "prosecution_closed"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["incident_recorder", "evidence_custodian", "docket_manager", "court_scheduler", "prosecution_tracker"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"incidents": {
		"supported_incident_types": SUPPORTED_INCIDENT_TYPES,
		"ob_number_required": True,
		"reporting_officer_required": True,
		"location_required": True,
		"evidence_required": True,
	},
	"dockets": {
		"supported_statuses": SUPPORTED_DOCKET_STATUSES,
		"incident_required": True,
		"investigating_officer_required": True,
		"evidence_required": True,
	},
	"evidence": {
		"supported_evidence_types": SUPPORTED_EVIDENCE_TYPES,
		"supported_custody_actions": SUPPORTED_CUSTODY_ACTIONS,
		"docket_required": True,
		"custodian_required": True,
		"chain_of_custody_enforced": True,
		"evidence_reference_required": True,
	},
	"court_scheduling": {
		"supported_court_types": SUPPORTED_COURT_TYPES,
		"supported_hearing_types": SUPPORTED_HEARING_TYPES,
		"docket_required": True,
		"court_required": True,
		"hearing_date_required": True,
	},
	"prosecution": {
		"supported_statuses": SUPPORTED_PROSECUTION_STATUSES,
		"docket_required": True,
		"dpp_reference_required": True,
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
		"chain_of_custody_breach_denied": True,
		"evidence_tampering_denied": True,
		"unauthorised_docket_access_denied": True,
		"prosecution_without_dpp_reference_denied": True,
		"cross_jurisdiction_access_restricted": True,
		"evidence_fabrication_denied": True,
	},
	"observability": {"event_stream": LAW_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {
		"auth": "auth",
		"audit": "audl",
		"notifications": "ntfy",
		"workflow": "wflo",
		"geospatial": "geos",
		"scheduling": "schd",
		"monitoring": "moni",
		"event_stream": "bytewax",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_incidents": True,
		"enable_dockets": True,
		"enable_evidence": True,
		"enable_court_scheduling": True,
		"enable_prosecution": True,
		"enable_reviews": True,
		"enable_agents": True,
	},
	"theme": {"default_theme": "government_law_control", "allow_tenant_overrides": True},
}

PROVIDES = [
	"incident_reporting_workflow",
	"docket_management_workflow",
	"evidence_chain_of_custody_workflow",
	"court_scheduling_workflow",
	"prosecution_tracking_workflow",
	"law_enforcement_review_workflow",
	"law_enforcement_agent_workflow",
	"ob_number_generation_workflow",
	"witness_management_workflow",
	"inter_agency_referral_workflow",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "geos", "schd", "moni", "mqeb"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/government-law/dashboard", "component": "LawEnforcementDashboard", "permission": "government_law:view", "nav_group": "Overview"},
	{"name": "incidents", "path": "/government-law/incidents", "component": "IncidentReportingConsole", "permission": "government_law:incidents", "nav_group": "Incidents"},
	{"name": "dockets", "path": "/government-law/dockets", "component": "DocketManagementConsole", "permission": "government_law:dockets", "nav_group": "Investigations"},
	{"name": "evidence", "path": "/government-law/evidence", "component": "EvidenceChainConsole", "permission": "government_law:evidence", "nav_group": "Evidence"},
	{"name": "custody", "path": "/government-law/custody", "component": "CustodyLedger", "permission": "government_law:custody", "nav_group": "Evidence"},
	{"name": "court_scheduling", "path": "/government-law/court-scheduling", "component": "CourtSchedulingConsole", "permission": "government_law:court", "nav_group": "Courts"},
	{"name": "prosecution", "path": "/government-law/prosecution", "component": "ProsecutionTrackingConsole", "permission": "government_law:prosecution", "nav_group": "Prosecution"},
	{"name": "map", "path": "/government-law/map", "component": "CrimeMapView", "permission": "government_law:view", "nav_group": "Intelligence"},
	{"name": "reviews", "path": "/government-law/reviews", "component": "LawEnforcementReviewConsole", "permission": "government_law:review", "nav_group": "Governance"},
	{"name": "agents", "path": "/government-law/agents", "component": "LawEnforcementAgentWorkbench", "permission": "government_law:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/government-law/settings", "component": "LawEnforcementSettings", "permission": "government_law:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "government_law_control",
	"tokens": {
		"color.primary": "#1E3A5F",
		"color.accent": "#C2410C",
		"color.success": "#166534",
		"color.warning": "#A16207",
		"color.danger": "#991B1B",
		"surface.canvas": "#EFF6FF",
		"surface.panel": "#FFFFFF",
		"text.primary": "#0F172A",
		"text.secondary": "#334155",
		"border.radius": "4px",
		"density": "compact",
	},
	"components": {
		"incidents": {"icon": "alert-circle", "status_indicator": "incident-type-chip"},
		"dockets": {"icon": "folder", "status_indicator": "docket-status-chip"},
		"evidence": {"icon": "package", "status_indicator": "evidence-type-chip"},
		"court_scheduling": {"icon": "calendar", "status_indicator": "hearing-type-chip"},
		"prosecution": {"icon": "gavel", "status_indicator": "prosecution-status-chip"},
		"reviews": {"icon": "clipboard-check", "status_indicator": "review-status-chip"},
		"agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": LAW_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"incident_reported",
		"docket_opened",
		"docket_status_changed",
		"evidence_logged",
		"evidence_custody_action_recorded",
		"court_hearing_scheduled",
		"prosecution_status_updated",
		"docket_forwarded_to_dpp",
		"law_enforcement_agent_registered",
		"conviction_recorded",
	],
	"guardrails": [
		"law_batch_requires_bytewax",
		"chain_of_custody_breach_denied",
		"evidence_tampering_denied",
		"prosecution_without_dpp_reference_denied",
		"evidence_fabrication_denied",
		"privileged_law_agent_action_requires_human_approval",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "law_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "law_enforcement_policy_required", "required_action": "attach_law_policy"}},
	{"name": "incident_type_supported", "condition": {"operation": "report_incident", "incident_type_supported": False}, "effect": {"decision": "deny", "reason": "incident_type_not_supported", "required_action": "select_supported_incident_type"}},
	{"name": "incident_ob_number_required", "condition": {"operation": "report_incident", "ob_number_present": False}, "effect": {"decision": "deny", "reason": "ob_number_required", "required_action": "generate_ob_number"}},
	{"name": "incident_reporting_officer_required", "condition": {"operation": "report_incident", "reporting_officer_present": False}, "effect": {"decision": "deny", "reason": "reporting_officer_required", "required_action": "assign_reporting_officer"}},
	{"name": "incident_location_required", "condition": {"operation": "report_incident", "location_present": False}, "effect": {"decision": "deny", "reason": "location_required", "required_action": "provide_incident_location"}},
	{"name": "incident_evidence_required", "condition": {"operation": "report_incident", "evidence_present": False}, "effect": {"decision": "deny", "reason": "incident_evidence_required", "required_action": "attach_incident_evidence"}},
	{"name": "docket_incident_required", "condition": {"operation": "open_docket", "incident_present": False}, "effect": {"decision": "deny", "reason": "incident_required", "required_action": "select_incident"}},
	{"name": "docket_investigating_officer_required", "condition": {"operation": "open_docket", "investigating_officer_present": False}, "effect": {"decision": "deny", "reason": "investigating_officer_required", "required_action": "assign_investigating_officer"}},
	{"name": "docket_status_supported", "condition": {"operation": "update_docket", "docket_status_supported": False}, "effect": {"decision": "deny", "reason": "docket_status_not_supported", "required_action": "select_supported_docket_status"}},
	{"name": "evidence_type_supported", "condition": {"operation": "log_evidence", "evidence_type_supported": False}, "effect": {"decision": "deny", "reason": "evidence_type_not_supported", "required_action": "select_supported_evidence_type"}},
	{"name": "evidence_docket_required", "condition": {"operation": "log_evidence", "docket_present": False}, "effect": {"decision": "deny", "reason": "docket_required", "required_action": "select_docket"}},
	{"name": "evidence_custodian_required", "condition": {"operation": "log_evidence", "custodian_present": False}, "effect": {"decision": "deny", "reason": "custodian_required", "required_action": "assign_custodian"}},
	{"name": "evidence_reference_required", "condition": {"operation": "log_evidence", "evidence_reference_present": False}, "effect": {"decision": "deny", "reason": "evidence_reference_required", "required_action": "attach_evidence_reference"}},
	{"name": "custody_action_supported", "condition": {"operation": "record_custody_action", "custody_action_supported": False}, "effect": {"decision": "deny", "reason": "custody_action_not_supported", "required_action": "select_supported_custody_action"}},
	{"name": "chain_of_custody_breach_denied", "condition": {"operation": "record_custody_action", "chain_intact": False}, "effect": {"decision": "deny", "reason": "chain_of_custody_breach_denied", "required_action": "restore_chain_of_custody"}},
	{"name": "court_type_supported", "condition": {"operation": "schedule_hearing", "court_type_supported": False}, "effect": {"decision": "deny", "reason": "court_type_not_supported", "required_action": "select_supported_court_type"}},
	{"name": "hearing_type_supported", "condition": {"operation": "schedule_hearing", "hearing_type_supported": False}, "effect": {"decision": "deny", "reason": "hearing_type_not_supported", "required_action": "select_supported_hearing_type"}},
	{"name": "hearing_date_required", "condition": {"operation": "schedule_hearing", "hearing_date_present": False}, "effect": {"decision": "deny", "reason": "hearing_date_required", "required_action": "set_hearing_date"}},
	{"name": "prosecution_dpp_reference_required", "condition": {"operation": "record_prosecution", "dpp_reference_present": False}, "effect": {"decision": "deny", "reason": "dpp_reference_required", "required_action": "obtain_dpp_reference"}},
	{"name": "prosecution_status_supported", "condition": {"operation": "record_prosecution", "prosecution_status_supported": False}, "effect": {"decision": "deny", "reason": "prosecution_status_not_supported", "required_action": "select_supported_prosecution_status"}},
	{"name": "law_batch_requires_bytewax", "condition": {"operation": "law_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_law_batch_to_bytewax"}},
	{"name": "law_agent_runtime_supported", "condition": {"operation": "register_law_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "law_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "law_agent_role_supported", "condition": {"operation": "register_law_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "law_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "law_agent_name_required", "condition": {"operation": "register_law_agent", "agent_name_present": False}, "effect": {"decision": "deny", "reason": "law_agent_name_required", "required_action": "name_law_agent"}},
	{"name": "law_agent_scope_required", "condition": {"operation": "register_law_agent", "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "law_agent_scope_required", "required_action": "bound_law_agent_scope"}},
	{"name": "privileged_law_agent_action_requires_human_approval", "condition": {"operation": "law_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
	{"name": "evidence_fabrication_denied", "condition": {"operation": "law_agent_action", "evidence_fabrication_scope": True}, "effect": {"decision": "deny", "reason": "evidence_fabrication_denied", "required_action": "remove_evidence_fabrication_scope"}},
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
			"api_prefix": "/government-law/api/v1",
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
