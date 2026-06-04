"""Executable capability contract for APG Emergency Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "government_eme"
CAPABILITY_NAME = "Emergency Management"
CAPABILITY_VERSION = "1.0.0"
EME_EVENT_STREAM = "apg.government.eme.lifecycle"

SUPPORTED_INCIDENT_TYPES = ["natural_disaster", "industrial_accident", "public_health", "security_threat", "infrastructure_failure", "mass_casualty", "environmental", "cyber_incident", "civil_unrest"]
SUPPORTED_SEVERITY_LEVELS = ["minor", "moderate", "serious", "major", "catastrophic"]
SUPPORTED_INCIDENT_PHASES = ["detection", "notification", "activation", "response", "recovery", "stand_down", "after_action"]
SUPPORTED_RESOURCE_TYPES = ["personnel", "equipment", "vehicle", "medical_supplies", "food_water", "shelter", "communication", "fuel"]
SUPPORTED_AGENCY_TYPES = ["lead_agency", "support_agency", "ngos", "military", "private_sector", "international", "community"]
SUPPORTED_EOC_STATUSES = ["standby", "partial_activation", "full_activation", "demobilising", "closed"]
SUPPORTED_COMMAND_STRUCTURES = ["unified_command", "area_command", "single_incident_command", "multi_agency_coordination"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["incident_coordinator", "resource_allocator", "situation_reporter", "agency_liaison", "aar_analyst"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"incidents": {
		"supported_incident_types": SUPPORTED_INCIDENT_TYPES,
		"supported_severity_levels": SUPPORTED_SEVERITY_LEVELS,
		"supported_phases": SUPPORTED_INCIDENT_PHASES,
		"location_required": True,
		"commander_required": True,
		"evidence_required": True,
	},
	"resources": {
		"supported_resource_types": SUPPORTED_RESOURCE_TYPES,
		"incident_required": True,
		"quantity_required": True,
		"responsible_agency_required": True,
		"evidence_required": True,
	},
	"agencies": {
		"supported_agency_types": SUPPORTED_AGENCY_TYPES,
		"incident_required": True,
		"contact_required": True,
		"role_required": True,
	},
	"eoc": {
		"supported_statuses": SUPPORTED_EOC_STATUSES,
		"supported_command_structures": SUPPORTED_COMMAND_STRUCTURES,
		"activation_authority_required": True,
		"evidence_required": True,
	},
	"situation_reports": {
		"incident_required": True,
		"author_required": True,
		"period_required": True,
		"evidence_required": True,
	},
	"after_action_reviews": {
		"incident_required": True,
		"reviewer_required": True,
		"lessons_required": True,
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
		"unauthorised_eoc_activation_denied": True,
		"resource_over_allocation_denied": True,
		"uncoordinated_agency_response_denied": True,
		"after_action_mandatory_post_incident": True,
		"evidence_fabrication_denied": True,
	},
	"observability": {"event_stream": EME_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {
		"auth": "auth",
		"audit": "audl",
		"notifications": "ntfy",
		"geospatial": "geos",
		"monitoring": "moni",
		"scheduling": "schd",
		"event_stream": "bytewax",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_incidents": True,
		"enable_resources": True,
		"enable_agencies": True,
		"enable_eoc": True,
		"enable_situation_reports": True,
		"enable_after_action_reviews": True,
		"enable_reviews": True,
		"enable_agents": True,
	},
	"theme": {"default_theme": "government_eme_control", "allow_tenant_overrides": True},
}

PROVIDES = [
	"incident_command_workflow",
	"resource_mobilisation_workflow",
	"multi_agency_coordination_workflow",
	"eoc_management_workflow",
	"situation_reporting_workflow",
	"after_action_review_workflow",
	"emergency_review_workflow",
	"emergency_agent_workflow",
	"incident_phase_transition_workflow",
	"resource_demobilisation_workflow",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "geos", "moni", "schd", "mqeb"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/government-eme/dashboard", "component": "EmergencyDashboard", "permission": "government_eme:view", "nav_group": "Overview"},
	{"name": "incidents", "path": "/government-eme/incidents", "component": "IncidentCommandConsole", "permission": "government_eme:incidents", "nav_group": "Incidents"},
	{"name": "resources", "path": "/government-eme/resources", "component": "ResourceMobilisationConsole", "permission": "government_eme:resources", "nav_group": "Resources"},
	{"name": "agencies", "path": "/government-eme/agencies", "component": "AgencyCoordinationConsole", "permission": "government_eme:agencies", "nav_group": "Coordination"},
	{"name": "eoc", "path": "/government-eme/eoc", "component": "EocManagementConsole", "permission": "government_eme:eoc", "nav_group": "Command"},
	{"name": "situation_reports", "path": "/government-eme/situation-reports", "component": "SituationReportConsole", "permission": "government_eme:reports", "nav_group": "Reporting"},
	{"name": "map", "path": "/government-eme/map", "component": "IncidentMapView", "permission": "government_eme:view", "nav_group": "Situational Awareness"},
	{"name": "after_action", "path": "/government-eme/after-action", "component": "AfterActionReviewConsole", "permission": "government_eme:aar", "nav_group": "Learning"},
	{"name": "reviews", "path": "/government-eme/reviews", "component": "EmergencyReviewConsole", "permission": "government_eme:review", "nav_group": "Governance"},
	{"name": "agents", "path": "/government-eme/agents", "component": "EmergencyAgentWorkbench", "permission": "government_eme:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/government-eme/settings", "component": "EmergencySettings", "permission": "government_eme:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "government_eme_control",
	"tokens": {
		"color.primary": "#DC2626",
		"color.accent": "#D97706",
		"color.success": "#166534",
		"color.warning": "#B45309",
		"color.danger": "#7F1D1D",
		"surface.canvas": "#FFF1F2",
		"surface.panel": "#FFFFFF",
		"text.primary": "#1C0A00",
		"text.secondary": "#57534E",
		"border.radius": "4px",
		"density": "compact",
	},
	"components": {
		"incidents": {"icon": "alert-triangle", "status_indicator": "incident-severity-chip"},
		"resources": {"icon": "package", "status_indicator": "resource-type-chip"},
		"agencies": {"icon": "users", "status_indicator": "agency-type-chip"},
		"eoc": {"icon": "radio", "status_indicator": "eoc-status-chip"},
		"situation_reports": {"icon": "file-text", "status_indicator": "sitrep-period-chip"},
		"after_action": {"icon": "book", "status_indicator": "aar-status-chip"},
		"reviews": {"icon": "clipboard-check", "status_indicator": "review-status-chip"},
		"agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": EME_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"incident_declared",
		"incident_phase_transitioned",
		"resource_mobilised",
		"resource_demobilised",
		"agency_activated",
		"eoc_activated",
		"situation_report_filed",
		"incident_stood_down",
		"after_action_review_completed",
		"emergency_agent_registered",
	],
	"guardrails": [
		"eme_batch_requires_bytewax",
		"unauthorised_eoc_activation_denied",
		"resource_over_allocation_denied",
		"after_action_mandatory_post_incident",
		"evidence_fabrication_denied",
		"privileged_emergency_agent_action_requires_human_approval",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "eme_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "emergency_policy_required", "required_action": "attach_emergency_policy"}},
	{"name": "incident_type_supported", "condition": {"operation": "declare_incident", "incident_type_supported": False}, "effect": {"decision": "deny", "reason": "incident_type_not_supported", "required_action": "select_supported_incident_type"}},
	{"name": "incident_severity_supported", "condition": {"operation": "declare_incident", "severity_supported": False}, "effect": {"decision": "deny", "reason": "severity_not_supported", "required_action": "select_supported_severity"}},
	{"name": "incident_location_required", "condition": {"operation": "declare_incident", "location_present": False}, "effect": {"decision": "deny", "reason": "location_required", "required_action": "provide_location"}},
	{"name": "incident_commander_required", "condition": {"operation": "declare_incident", "commander_present": False}, "effect": {"decision": "deny", "reason": "incident_commander_required", "required_action": "assign_incident_commander"}},
	{"name": "incident_evidence_required", "condition": {"operation": "declare_incident", "evidence_present": False}, "effect": {"decision": "deny", "reason": "incident_evidence_required", "required_action": "attach_incident_evidence"}},
	{"name": "resource_type_supported", "condition": {"operation": "mobilise_resource", "resource_type_supported": False}, "effect": {"decision": "deny", "reason": "resource_type_not_supported", "required_action": "select_supported_resource_type"}},
	{"name": "resource_incident_required", "condition": {"operation": "mobilise_resource", "incident_present": False}, "effect": {"decision": "deny", "reason": "incident_required", "required_action": "select_incident"}},
	{"name": "resource_quantity_required", "condition": {"operation": "mobilise_resource", "quantity_present": False}, "effect": {"decision": "deny", "reason": "quantity_required", "required_action": "specify_quantity"}},
	{"name": "resource_over_allocation_denied", "condition": {"operation": "mobilise_resource", "over_allocated": True}, "effect": {"decision": "deny", "reason": "resource_over_allocation_denied", "required_action": "reduce_allocation"}},
	{"name": "agency_type_supported", "condition": {"operation": "activate_agency", "agency_type_supported": False}, "effect": {"decision": "deny", "reason": "agency_type_not_supported", "required_action": "select_supported_agency_type"}},
	{"name": "agency_incident_required", "condition": {"operation": "activate_agency", "incident_present": False}, "effect": {"decision": "deny", "reason": "incident_required", "required_action": "select_incident"}},
	{"name": "agency_contact_required", "condition": {"operation": "activate_agency", "contact_present": False}, "effect": {"decision": "deny", "reason": "contact_required", "required_action": "provide_agency_contact"}},
	{"name": "eoc_status_supported", "condition": {"operation": "update_eoc", "eoc_status_supported": False}, "effect": {"decision": "deny", "reason": "eoc_status_not_supported", "required_action": "select_supported_eoc_status"}},
	{"name": "eoc_activation_authority_required", "condition": {"operation": "update_eoc", "activation_authority_present": False}, "effect": {"decision": "deny", "reason": "activation_authority_required", "required_action": "provide_activation_authority"}},
	{"name": "unauthorised_eoc_activation_denied", "condition": {"operation": "update_eoc", "authorised": False, "eoc_status_supported": True}, "effect": {"decision": "deny", "reason": "unauthorised_eoc_activation_denied", "required_action": "obtain_eoc_activation_authority"}},
	{"name": "sitrep_incident_required", "condition": {"operation": "file_sitrep", "incident_present": False}, "effect": {"decision": "deny", "reason": "incident_required", "required_action": "select_incident"}},
	{"name": "sitrep_author_required", "condition": {"operation": "file_sitrep", "author_present": False}, "effect": {"decision": "deny", "reason": "author_required", "required_action": "assign_author"}},
	{"name": "aar_incident_required", "condition": {"operation": "record_aar", "incident_present": False}, "effect": {"decision": "deny", "reason": "incident_required", "required_action": "select_incident"}},
	{"name": "aar_reviewer_required", "condition": {"operation": "record_aar", "reviewer_present": False}, "effect": {"decision": "deny", "reason": "reviewer_required", "required_action": "assign_reviewer"}},
	{"name": "aar_lessons_required", "condition": {"operation": "record_aar", "lessons_present": False}, "effect": {"decision": "deny", "reason": "lessons_required", "required_action": "document_lessons_learned"}},
	{"name": "eme_batch_requires_bytewax", "condition": {"operation": "eme_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_eme_batch_to_bytewax"}},
	{"name": "eme_agent_runtime_supported", "condition": {"operation": "register_eme_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "eme_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "eme_agent_role_supported", "condition": {"operation": "register_eme_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "eme_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "eme_agent_name_required", "condition": {"operation": "register_eme_agent", "agent_name_present": False}, "effect": {"decision": "deny", "reason": "eme_agent_name_required", "required_action": "name_eme_agent"}},
	{"name": "eme_agent_scope_required", "condition": {"operation": "register_eme_agent", "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "eme_agent_scope_required", "required_action": "bound_eme_agent_scope"}},
	{"name": "privileged_emergency_agent_action_requires_human_approval", "condition": {"operation": "eme_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
	{"name": "evidence_fabrication_denied", "condition": {"operation": "eme_agent_action", "evidence_fabrication_scope": True}, "effect": {"decision": "deny", "reason": "evidence_fabrication_denied", "required_action": "remove_evidence_fabrication_scope"}},
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
			"api_prefix": "/government-eme/api/v1",
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
