"""Executable capability contract for APG Mine Safety & Compliance."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "mining_saf"
CAPABILITY_NAME = "Mine Safety & Compliance"
CAPABILITY_VERSION = "1.0.0"
SAF_EVENT_STREAM = "apg.mining.saf.lifecycle"

SUPPORTED_INCIDENT_TYPES = ["fatality", "lost_time_injury", "medical_treatment_injury", "first_aid_injury", "near_miss", "dangerous_occurrence", "environmental_incident", "property_damage", "vehicle_incident", "occupational_illness"]
SUPPORTED_HAZARD_CATEGORIES = ["mechanical", "electrical", "chemical", "radiation", "gravitational", "biological", "ergonomic", "fire_explosion", "confined_space", "ground_instability", "dust_fumes", "noise_vibration"]
SUPPORTED_RISK_RATINGS = ["extreme", "high", "medium", "low", "negligible"]
SUPPORTED_CONSEQUENCE_LEVELS = ["catastrophic", "major", "moderate", "minor", "insignificant"]
SUPPORTED_LIKELIHOOD_LEVELS = ["almost_certain", "likely", "possible", "unlikely", "rare"]
SUPPORTED_PTW_TYPES = ["hot_work", "confined_space_entry", "electrical_isolation", "working_at_height", "excavation", "lifting_operations", "radiation_work", "explosives_handling", "isolation_lockout", "ground_disturbance"]
SUPPORTED_COMPLIANCE_FRAMEWORKS = ["msha", "osha", "nosa", "iso_45001", "ohsas_18001", "local_mining_regulations", "explosives_regulations", "radiation_regulations"]
SUPPORTED_INVESTIGATION_METHODS = ["why_why_analysis", "taproot", "icam", "bowtie", "fault_tree", "5_whys", "fishbone"]
SUPPORTED_CONTROL_TYPES = ["elimination", "substitution", "engineering", "administrative", "ppe"]
SUPPORTED_TRAINING_TYPES = ["induction", "task_specific", "refresher", "emergency_response", "first_aid", "fire_warden", "rescue_team", "statutory_competency"]
SUPPORTED_DRILL_TYPES = ["fire_evacuation", "rescue", "hazmat_spill", "ground_fall", "tailings_breach", "communication_test"]
SUPPORTED_AUDIT_TYPES = ["internal", "external", "regulatory", "third_party", "self_assessment"]
SUPPORTED_REVIEW_STATUSES = ["pending", "in_review", "approved", "rejected", "closed"]
SUPPORTED_CORRECTIVE_ACTION_STATUSES = ["open", "in_progress", "overdue", "closed", "verified"]
SUPPORTED_PPE_TYPES = ["hard_hat", "safety_boots", "hi_vis_vest", "safety_glasses", "hearing_protection", "respirator", "gloves", "safety_harness", "face_shield", "chemical_suit"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"incidents": {
		"supported_types": SUPPORTED_INCIDENT_TYPES,
		"immediate_notification_required": True,
		"regulatory_notification_required": True,
		"investigation_required_for_lti_and_above": True,
	},
	"hazards": {
		"supported_categories": SUPPORTED_HAZARD_CATEGORIES,
		"risk_assessment_required": True,
		"control_measure_required": True,
		"review_frequency_days": 90,
	},
	"risk_register": {
		"supported_risk_ratings": SUPPORTED_RISK_RATINGS,
		"supported_consequence_levels": SUPPORTED_CONSEQUENCE_LEVELS,
		"supported_likelihood_levels": SUPPORTED_LIKELIHOOD_LEVELS,
		"residual_risk_assessment_required": True,
	},
	"permits_to_work": {
		"supported_types": SUPPORTED_PTW_TYPES,
		"issuer_qualification_required": True,
		"site_inspection_required": True,
		"isolation_verification_required": True,
	},
	"compliance": {
		"supported_frameworks": SUPPORTED_COMPLIANCE_FRAMEWORKS,
		"evidence_required": True,
		"periodic_review_required": True,
	},
	"governance": {
		"require_tenant_context": True,
		"policy_attached_for_writes": True,
		"audit_events": True,
		"uninvestigated_lti_escalation_required": True,
		"expired_ptw_access_denied": True,
		"cross_tenant_read_denied": True,
		"open_extreme_risk_stop_work_trigger": True,
	},
	"observability": {"event_stream": SAF_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "workflow": "wflo", "compliance": "comp", "monitoring": "moni", "event_stream": "bytewax"},
	"ui": {
		"enable_dashboard": True,
		"enable_incidents": True,
		"enable_hazards": True,
		"enable_risk_register": True,
		"enable_permits": True,
		"enable_compliance": True,
		"enable_training": True,
		"enable_audits": True,
	},
	"theme": {"default_theme": "mining_saf_control", "allow_tenant_overrides": True},
}

PROVIDES = [
	"incident_reporting_workflow",
	"hazard_identification_workflow",
	"risk_register_management",
	"permit_to_work_workflow",
	"corrective_action_tracking",
	"compliance_register_management",
	"safety_audit_workflow",
	"emergency_drill_management",
	"safety_statistics_reporting",
	"stop_work_authority_workflow",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "comp", "moni", "mqeb"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/mining-saf/dashboard", "component": "SafDashboard", "permission": "mining_saf:view", "nav_group": "Overview"},
	{"name": "incidents", "path": "/mining-saf/incidents", "component": "IncidentList", "permission": "mining_saf:view", "nav_group": "Incidents"},
	{"name": "incident_create", "path": "/mining-saf/incidents/create", "component": "IncidentReportForm", "permission": "mining_saf:write", "nav_group": "Incidents"},
	{"name": "incident_detail", "path": "/mining-saf/incidents/:id", "component": "IncidentDetail", "permission": "mining_saf:view", "nav_group": "Incidents"},
	{"name": "hazards", "path": "/mining-saf/hazards", "component": "HazardRegister", "permission": "mining_saf:view", "nav_group": "Hazards"},
	{"name": "hazard_create", "path": "/mining-saf/hazards/create", "component": "HazardForm", "permission": "mining_saf:write", "nav_group": "Hazards"},
	{"name": "risk_register", "path": "/mining-saf/risk-register", "component": "RiskRegister", "permission": "mining_saf:view", "nav_group": "Risk"},
	{"name": "permits", "path": "/mining-saf/permits", "component": "PermitToWorkList", "permission": "mining_saf:view", "nav_group": "Permits"},
	{"name": "permit_create", "path": "/mining-saf/permits/create", "component": "PermitToWorkForm", "permission": "mining_saf:ptw_issue", "nav_group": "Permits"},
	{"name": "compliance", "path": "/mining-saf/compliance", "component": "ComplianceRegister", "permission": "mining_saf:compliance", "nav_group": "Compliance"},
	{"name": "audits", "path": "/mining-saf/audits", "component": "AuditList", "permission": "mining_saf:audit", "nav_group": "Audits"},
	{"name": "training", "path": "/mining-saf/training", "component": "TrainingMatrix", "permission": "mining_saf:view", "nav_group": "Training"},
	{"name": "statistics", "path": "/mining-saf/statistics", "component": "SafetyStatsDashboard", "permission": "mining_saf:reports", "nav_group": "Reporting"},
	{"name": "settings", "path": "/mining-saf/settings", "component": "SafSettings", "permission": "mining_saf:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "mining_saf_control",
	"tokens": {
		"color.primary": "#7C3AED",
		"color.accent": "#DC2626",
		"color.success": "#15803D",
		"color.warning": "#D97706",
		"color.danger": "#B91C1C",
		"surface.canvas": "#F5F3FF",
		"surface.panel": "#FFFFFF",
		"text.primary": "#1E1B4B",
		"text.secondary": "#4C1D95",
		"border.radius": "6px",
		"density": "comfortable",
	},
	"components": {
		"incidents": {"icon": "alert-triangle", "status_indicator": "incident-severity-chip"},
		"hazards": {"icon": "alert-octagon", "status_indicator": "risk-rating-chip"},
		"risk_register": {"icon": "shield", "status_indicator": "risk-level-chip"},
		"permits": {"icon": "key", "status_indicator": "ptw-status-chip"},
		"compliance": {"icon": "check-square", "status_indicator": "compliance-status-chip"},
		"audits": {"icon": "clipboard-list", "status_indicator": "audit-status-chip"},
		"training": {"icon": "book-open", "status_indicator": "training-status-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": SAF_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"incident_reported",
		"incident_escalated",
		"incident_investigation_opened",
		"incident_closed",
		"hazard_identified",
		"hazard_risk_assessed",
		"risk_register_updated",
		"permit_issued",
		"permit_closed",
		"corrective_action_assigned",
		"corrective_action_overdue",
		"compliance_obligation_due",
		"stop_work_authority_invoked",
		"emergency_drill_completed",
	],
	"guardrails": [
		"expired_ptw_access_denied",
		"uninvestigated_lti_denied",
		"open_extreme_risk_stop_work_trigger",
		"cross_tenant_read_denied",
		"unqualified_ptw_issuer_denied",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "write_policy_required", "required_action": "attach_write_policy"}},
	{"name": "incident_type_supported", "condition": {"operation": "report_incident", "incident_type_supported": False}, "effect": {"decision": "deny", "reason": "incident_type_not_supported", "required_action": "select_supported_incident_type"}},
	{"name": "incident_location_required", "condition": {"operation": "report_incident", "location_present": False}, "effect": {"decision": "deny", "reason": "incident_location_required", "required_action": "provide_incident_location"}},
	{"name": "incident_immediate_notification", "condition": {"operation": "report_incident", "incident_severity": "fatality", "notification_sent": False}, "effect": {"decision": "deny", "reason": "immediate_notification_required_for_fatality", "required_action": "send_immediate_notification"}},
	{"name": "lti_investigation_required", "condition": {"operation": "close_incident", "incident_type": "lost_time_injury", "investigation_complete": False}, "effect": {"decision": "deny", "reason": "investigation_required_for_lti", "required_action": "complete_investigation_first"}},
	{"name": "hazard_category_supported", "condition": {"operation": "identify_hazard", "hazard_category_supported": False}, "effect": {"decision": "deny", "reason": "hazard_category_not_supported", "required_action": "select_supported_hazard_category"}},
	{"name": "hazard_risk_assessment_required", "condition": {"operation": "submit_hazard", "risk_assessment_complete": False}, "effect": {"decision": "deny", "reason": "risk_assessment_required", "required_action": "complete_risk_assessment"}},
	{"name": "hazard_control_measure_required", "condition": {"operation": "submit_hazard", "control_measure_present": False}, "effect": {"decision": "deny", "reason": "control_measure_required", "required_action": "specify_control_measures"}},
	{"name": "extreme_risk_stop_work_trigger", "condition": {"operation": "submit_hazard", "risk_rating": "extreme", "stop_work_invoked": False}, "effect": {"decision": "deny", "reason": "extreme_risk_requires_stop_work_authority", "required_action": "invoke_stop_work_authority"}},
	{"name": "ptw_type_supported", "condition": {"operation": "issue_permit", "ptw_type_supported": False}, "effect": {"decision": "deny", "reason": "ptw_type_not_supported", "required_action": "select_supported_ptw_type"}},
	{"name": "ptw_issuer_qualification_required", "condition": {"operation": "issue_permit", "issuer_qualified": False}, "effect": {"decision": "deny", "reason": "issuer_qualification_required", "required_action": "verify_issuer_qualification"}},
	{"name": "ptw_isolation_verification_required", "condition": {"operation": "issue_permit", "isolation_verified": False}, "effect": {"decision": "deny", "reason": "isolation_verification_required", "required_action": "verify_isolation"}},
	{"name": "expired_ptw_access_denied", "condition": {"operation": "access_with_permit", "permit_expired": True}, "effect": {"decision": "deny", "reason": "expired_permit_not_valid", "required_action": "renew_or_reissue_permit"}},
	{"name": "compliance_framework_supported", "condition": {"operation": "record_obligation", "framework_supported": False}, "effect": {"decision": "deny", "reason": "compliance_framework_not_supported", "required_action": "select_supported_framework"}},
	{"name": "corrective_action_assignee_required", "condition": {"operation": "create_corrective_action", "assignee_present": False}, "effect": {"decision": "deny", "reason": "assignee_required", "required_action": "assign_responsible_person"}},
	{"name": "corrective_action_due_date_required", "condition": {"operation": "create_corrective_action", "due_date_present": False}, "effect": {"decision": "deny", "reason": "due_date_required", "required_action": "set_due_date"}},
	{"name": "audit_type_supported", "condition": {"operation": "create_audit", "audit_type_supported": False}, "effect": {"decision": "deny", "reason": "audit_type_not_supported", "required_action": "select_supported_audit_type"}},
	{"name": "stop_work_investigation_required", "condition": {"operation": "resume_work_after_stop_work", "investigation_complete": False}, "effect": {"decision": "deny", "reason": "investigation_required_before_resuming_work", "required_action": "complete_stop_work_investigation"}},
	{"name": "cross_tenant_read_denied", "condition": {"operation": "read", "cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_access_denied", "required_action": "use_own_tenant_context"}},
	{"name": "risk_rating_supported", "condition": {"operation": "assess_risk", "risk_rating_supported": False}, "effect": {"decision": "deny", "reason": "risk_rating_not_supported", "required_action": "select_supported_risk_rating"}},
	{"name": "delete_closed_incident_denied", "condition": {"operation": "delete", "incident_status": "closed"}, "effect": {"decision": "deny", "reason": "closed_incidents_cannot_be_deleted", "required_action": "archive_instead"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	"""Return the full capability contract for the given tenant."""
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	return {
		"capability": CAPABILITY_ID,
		"display_name": CAPABILITY_NAME,
		"version": CAPABILITY_VERSION,
		"configuration": config,
		"configuration_schema": {
			"required": ["tenant_id", "ui", "theme"],
			"properties": {
				"tenant_id": {"type": "string"},
				"ui": {"type": "object"},
				"theme": {"type": "object"},
				"incidents": {"type": "object"},
				"hazards": {"type": "object"},
				"permits_to_work": {"type": "object"},
				"compliance": {"type": "object"},
			},
		},
		"rule_engine": {
			"type": "deterministic",
			"default_decision": "allow",
			"rules": RULES,
		},
		"ui": {
			"shell": "apg_python",
			"requires_theme": True,
			"template_roots": ["mining/saf/templates"],
			"routes": UI_ROUTES,
		},
		"theme": THEME,
		"provides": PROVIDES,
		"requires": REQUIRES,
		"streaming": STREAMING,
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate deterministic rules against the given context dict."""
	matched_denials: list[dict[str, Any]] = []
	matched_allows: list[dict[str, Any]] = []

	for rule in RULES:
		condition = rule["condition"]
		all_match = all(context.get(k) == v for k, v in condition.items())
		if all_match:
			effect = rule["effect"]
			entry = {"rule": rule["name"], "effect": effect}
			if effect["decision"] == "deny":
				matched_denials.append(entry)
			else:
				matched_allows.append(entry)

	if matched_denials:
		return {
			"decision": "deny",
			"matched_denials": matched_denials,
			"matched_allows": matched_allows,
			"required_actions": [d["effect"]["required_action"] for d in matched_denials],
		}

	return {
		"decision": "allow",
		"matched_denials": [],
		"matched_allows": matched_allows,
		"required_actions": [],
	}
