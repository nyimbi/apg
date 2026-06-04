"""Executable capability contract for APG Electoral & Civil Registration."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "government_ele"
CAPABILITY_NAME = "Electoral and Civil Registration"
CAPABILITY_VERSION = "1.0.0"
ELE_EVENT_STREAM = "apg.government.ele.lifecycle"

SUPPORTED_REGISTRATION_TYPES = ["voter", "birth", "death", "marriage", "divorce", "adoption", "name_change", "citizenship"]
SUPPORTED_DEDUPLICATION_METHODS = ["biometric_fingerprint", "biometric_iris", "national_id", "passport", "facial_recognition", "demographic"]
SUPPORTED_POLLING_STATION_TYPES = ["ordinary", "special_interest", "diaspora", "prison", "hospital", "mobile"]
SUPPORTED_ELECTION_TYPES = ["presidential", "parliamentary", "gubernatorial", "ward", "by_election", "referendum", "party_primary"]
SUPPORTED_RESULT_STATUSES = ["provisional", "announced", "disputed", "nullified", "confirmed", "gazetted"]
SUPPORTED_CIVIL_EVENT_STATUSES = ["registered", "amended", "late_registration", "correction_applied", "cancelled"]
SUPPORTED_VERIFICATION_STATUSES = ["pending", "verified", "rejected", "deferred", "requires_biometric"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["registration_officer", "deduplication_checker", "results_collator", "civil_registrar", "boundary_analyst"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"registrations": {
		"supported_registration_types": SUPPORTED_REGISTRATION_TYPES,
		"biometric_required": True,
		"national_id_required": True,
		"deduplication_required": True,
		"evidence_required": True,
	},
	"deduplication": {
		"supported_methods": SUPPORTED_DEDUPLICATION_METHODS,
		"primary_method": "biometric_fingerprint",
		"secondary_method": "national_id",
		"duplicate_detection_threshold": 0.95,
		"manual_review_on_threshold": True,
	},
	"polling_stations": {
		"supported_types": SUPPORTED_POLLING_STATION_TYPES,
		"location_required": True,
		"capacity_required": True,
		"officer_assigned_required": True,
		"evidence_required": True,
	},
	"elections": {
		"supported_election_types": SUPPORTED_ELECTION_TYPES,
		"nomination_deadline_required": True,
		"polling_date_required": True,
		"constituency_required": True,
	},
	"results": {
		"supported_statuses": SUPPORTED_RESULT_STATUSES,
		"polling_station_required": True,
		"presiding_officer_required": True,
		"tallied_votes_required": True,
		"evidence_required": True,
	},
	"civil_registry": {
		"supported_event_statuses": SUPPORTED_CIVIL_EVENT_STATUSES,
		"registrar_required": True,
		"witness_required": True,
		"evidence_required": True,
	},
	"verifications": {
		"supported_statuses": SUPPORTED_VERIFICATION_STATUSES,
		"biometric_match_score_required": True,
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
		"duplicate_voter_denied": True,
		"underage_voter_denied": True,
		"result_manipulation_denied": True,
		"unverified_registration_denied": True,
		"cross_constituency_result_denied": True,
		"evidence_fabrication_denied": True,
	},
	"observability": {"event_stream": ELE_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {
		"auth": "auth",
		"audit": "audl",
		"notifications": "ntfy",
		"geospatial": "geos",
		"monitoring": "moni",
		"compliance": "comp",
		"event_stream": "bytewax",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_registrations": True,
		"enable_deduplication": True,
		"enable_polling_stations": True,
		"enable_elections": True,
		"enable_results": True,
		"enable_civil_registry": True,
		"enable_verifications": True,
		"enable_reviews": True,
		"enable_agents": True,
	},
	"theme": {"default_theme": "government_ele_control", "allow_tenant_overrides": True},
}

PROVIDES = [
	"voter_registration_workflow",
	"biometric_deduplication_workflow",
	"polling_station_management_workflow",
	"election_management_workflow",
	"results_collation_workflow",
	"civil_registration_workflow",
	"electoral_verification_workflow",
	"electoral_review_workflow",
	"electoral_agent_workflow",
	"civil_registry_amendment_workflow",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "geos", "comp", "moni", "mqeb"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/government-ele/dashboard", "component": "ElectoralDashboard", "permission": "government_ele:view", "nav_group": "Overview"},
	{"name": "registrations", "path": "/government-ele/registrations", "component": "VoterRegistrationConsole", "permission": "government_ele:register", "nav_group": "Registration"},
	{"name": "deduplication", "path": "/government-ele/deduplication", "component": "BiometricDeduplicationConsole", "permission": "government_ele:deduplicate", "nav_group": "Registration"},
	{"name": "polling_stations", "path": "/government-ele/polling-stations", "component": "PollingStationManager", "permission": "government_ele:stations", "nav_group": "Elections"},
	{"name": "elections", "path": "/government-ele/elections", "component": "ElectionManagementConsole", "permission": "government_ele:elections", "nav_group": "Elections"},
	{"name": "results", "path": "/government-ele/results", "component": "ResultsCollationConsole", "permission": "government_ele:results", "nav_group": "Results"},
	{"name": "civil_registry", "path": "/government-ele/civil-registry", "component": "CivilRegistryConsole", "permission": "government_ele:civil", "nav_group": "Civil Registry"},
	{"name": "verifications", "path": "/government-ele/verifications", "component": "ElectoralVerificationConsole", "permission": "government_ele:verify", "nav_group": "Verification"},
	{"name": "boundaries", "path": "/government-ele/boundaries", "component": "ConstituencyBoundaryMap", "permission": "government_ele:boundaries", "nav_group": "Geography"},
	{"name": "reviews", "path": "/government-ele/reviews", "component": "ElectoralReviewConsole", "permission": "government_ele:review", "nav_group": "Governance"},
	{"name": "agents", "path": "/government-ele/agents", "component": "ElectoralAgentWorkbench", "permission": "government_ele:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/government-ele/settings", "component": "ElectoralSettings", "permission": "government_ele:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "government_ele_control",
	"tokens": {
		"color.primary": "#B45309",
		"color.accent": "#1D4ED8",
		"color.success": "#166534",
		"color.warning": "#92400E",
		"color.danger": "#991B1B",
		"surface.canvas": "#FFFBEB",
		"surface.panel": "#FFFFFF",
		"text.primary": "#1C1917",
		"text.secondary": "#57534E",
		"border.radius": "6px",
		"density": "comfortable",
	},
	"components": {
		"registrations": {"icon": "user-plus", "status_indicator": "registration-type-chip"},
		"deduplication": {"icon": "fingerprint", "status_indicator": "dedup-status-chip"},
		"polling_stations": {"icon": "map-pin", "status_indicator": "station-type-chip"},
		"elections": {"icon": "vote", "status_indicator": "election-type-chip"},
		"results": {"icon": "bar-chart-2", "status_indicator": "result-status-chip"},
		"civil_registry": {"icon": "book-open", "status_indicator": "civil-event-chip"},
		"verifications": {"icon": "shield-check", "status_indicator": "verification-status-chip"},
		"reviews": {"icon": "clipboard-check", "status_indicator": "review-status-chip"},
		"agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": ELE_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"voter_registered",
		"duplicate_detected",
		"duplicate_resolved",
		"polling_station_assigned",
		"election_results_collated",
		"civil_event_registered",
		"voter_verified",
		"result_announced",
		"electoral_agent_registered",
		"civil_event_amended",
	],
	"guardrails": [
		"ele_batch_requires_bytewax",
		"duplicate_voter_denied",
		"underage_voter_denied",
		"result_manipulation_denied",
		"unverified_registration_denied",
		"cross_constituency_result_denied",
		"evidence_fabrication_denied",
		"privileged_electoral_agent_action_requires_human_approval",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "ele_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "electoral_policy_required", "required_action": "attach_electoral_policy"}},
	{"name": "registration_type_supported", "condition": {"operation": "register_voter", "registration_type_supported": False}, "effect": {"decision": "deny", "reason": "registration_type_not_supported", "required_action": "select_supported_registration_type"}},
	{"name": "voter_biometric_required", "condition": {"operation": "register_voter", "biometric_present": False}, "effect": {"decision": "deny", "reason": "biometric_required", "required_action": "capture_biometric"}},
	{"name": "voter_national_id_required", "condition": {"operation": "register_voter", "national_id_present": False}, "effect": {"decision": "deny", "reason": "national_id_required", "required_action": "provide_national_id"}},
	{"name": "voter_deduplication_required", "condition": {"operation": "register_voter", "deduplication_passed": False}, "effect": {"decision": "deny", "reason": "deduplication_required", "required_action": "run_deduplication_check"}},
	{"name": "duplicate_voter_denied", "condition": {"operation": "register_voter", "duplicate_detected": True}, "effect": {"decision": "deny", "reason": "duplicate_voter_denied", "required_action": "resolve_duplicate"}},
	{"name": "underage_voter_denied", "condition": {"operation": "register_voter", "of_voting_age": False}, "effect": {"decision": "deny", "reason": "underage_voter_denied", "required_action": "verify_age"}},
	{"name": "dedup_method_supported", "condition": {"operation": "run_deduplication", "deduplication_method_supported": False}, "effect": {"decision": "deny", "reason": "deduplication_method_not_supported", "required_action": "select_supported_method"}},
	{"name": "polling_station_type_supported", "condition": {"operation": "assign_polling_station", "station_type_supported": False}, "effect": {"decision": "deny", "reason": "polling_station_type_not_supported", "required_action": "select_supported_station_type"}},
	{"name": "polling_station_location_required", "condition": {"operation": "assign_polling_station", "location_present": False}, "effect": {"decision": "deny", "reason": "location_required", "required_action": "provide_location"}},
	{"name": "polling_station_officer_required", "condition": {"operation": "assign_polling_station", "officer_present": False}, "effect": {"decision": "deny", "reason": "presiding_officer_required", "required_action": "assign_presiding_officer"}},
	{"name": "election_type_supported", "condition": {"operation": "create_election", "election_type_supported": False}, "effect": {"decision": "deny", "reason": "election_type_not_supported", "required_action": "select_supported_election_type"}},
	{"name": "election_polling_date_required", "condition": {"operation": "create_election", "polling_date_present": False}, "effect": {"decision": "deny", "reason": "polling_date_required", "required_action": "set_polling_date"}},
	{"name": "result_polling_station_required", "condition": {"operation": "collate_result", "polling_station_present": False}, "effect": {"decision": "deny", "reason": "polling_station_required", "required_action": "select_polling_station"}},
	{"name": "result_presiding_officer_required", "condition": {"operation": "collate_result", "presiding_officer_present": False}, "effect": {"decision": "deny", "reason": "presiding_officer_required", "required_action": "assign_presiding_officer"}},
	{"name": "result_evidence_required", "condition": {"operation": "collate_result", "evidence_present": False}, "effect": {"decision": "deny", "reason": "result_evidence_required", "required_action": "attach_form_34a"}},
	{"name": "result_manipulation_denied", "condition": {"operation": "collate_result", "manipulation_detected": True}, "effect": {"decision": "deny", "reason": "result_manipulation_denied", "required_action": "flag_for_investigation"}},
	{"name": "cross_constituency_result_denied", "condition": {"operation": "collate_result", "cross_constituency": True}, "effect": {"decision": "deny", "reason": "cross_constituency_result_denied", "required_action": "use_correct_constituency"}},
	{"name": "civil_registrar_required", "condition": {"operation": "register_civil_event", "registrar_present": False}, "effect": {"decision": "deny", "reason": "registrar_required", "required_action": "assign_registrar"}},
	{"name": "civil_event_evidence_required", "condition": {"operation": "register_civil_event", "evidence_present": False}, "effect": {"decision": "deny", "reason": "civil_evidence_required", "required_action": "attach_civil_evidence"}},
	{"name": "ele_batch_requires_bytewax", "condition": {"operation": "ele_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_ele_batch_to_bytewax"}},
	{"name": "ele_agent_runtime_supported", "condition": {"operation": "register_ele_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "ele_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "ele_agent_role_supported", "condition": {"operation": "register_ele_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "ele_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "ele_agent_name_required", "condition": {"operation": "register_ele_agent", "agent_name_present": False}, "effect": {"decision": "deny", "reason": "ele_agent_name_required", "required_action": "name_ele_agent"}},
	{"name": "ele_agent_scope_required", "condition": {"operation": "register_ele_agent", "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "ele_agent_scope_required", "required_action": "bound_ele_agent_scope"}},
	{"name": "privileged_electoral_agent_action_requires_human_approval", "condition": {"operation": "ele_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
	{"name": "evidence_fabrication_denied", "condition": {"operation": "ele_agent_action", "evidence_fabrication_scope": True}, "effect": {"decision": "deny", "reason": "evidence_fabrication_denied", "required_action": "remove_evidence_fabrication_scope"}},
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
			"api_prefix": "/government-ele/api/v1",
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
