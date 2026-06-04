"""Executable capability contract for APG Healthcare Regulatory."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "healthcare_reg"
CAPABILITY_NAME = "Healthcare Regulatory"
CAPABILITY_VERSION = "1.0.0"
REG_EVENT_STREAM = "apg.healthcare.reg.lifecycle"

SUPPORTED_LICENSE_TYPES = [
	"facility_operating", "physician", "nurse", "pharmacist", "laboratory",
	"radiation", "controlled_substance_dea", "clinical_trial", "blood_bank",
]
SUPPORTED_ACCREDITATION_BODIES = ["joint_commission", "dnv", "hfap", "cihq", "cap", "aabb", "cms"]
SUPPORTED_ACCREDITATION_STATUSES = ["accredited", "provisional", "conditional", "not_accredited", "under_review", "in_progress"]
SUPPORTED_INCIDENT_TYPES = [
	"sentinel_event", "near_miss", "medication_error", "patient_fall",
	"healthcare_associated_infection", "wrong_site_surgery", "retained_foreign_body",
	"pressure_ulcer", "elopement", "equipment_failure",
]
SUPPORTED_INCIDENT_SEVERITIES = ["minor", "moderate", "serious", "catastrophic"]
SUPPORTED_REPORT_TYPES = [
	"cms_oqr", "cms_iqr", "joint_commission_core", "state_licensing",
	"dea_schedule_ii", "hipaa_breach", "fda_mdr", "meaningful_use",
]
SUPPORTED_COMPLIANCE_FRAMEWORKS = ["hipaa", "hitech", "cms_conditions", "joint_commission", "dea", "fda_21cfr", "state_health_code"]
SUPPORTED_SUBMISSION_STATUSES = ["draft", "submitted", "accepted", "rejected", "resubmit_required", "closed"]
SUPPORTED_AUDIT_TYPES = ["internal", "external", "mock_survey", "for_cause", "routine"]
SUPPORTED_CORRECTIVE_ACTION_STATUSES = ["open", "in_progress", "completed", "verified", "overdue"]
SUPPORTED_AGENT_ROLES = ["regulatory_steward", "incident_reviewer", "submission_reviewer", "audit_reviewer"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"licensing": {"supported_types": SUPPORTED_LICENSE_TYPES, "expiry_warning_days": 90},
	"accreditation": {"supported_bodies": SUPPORTED_ACCREDITATION_BODIES, "supported_statuses": SUPPORTED_ACCREDITATION_STATUSES},
	"incidents": {"supported_types": SUPPORTED_INCIDENT_TYPES, "supported_severities": SUPPORTED_INCIDENT_SEVERITIES, "sentinel_event_notification_hours": 72, "root_cause_analysis_required_for_sentinel": True},
	"submissions": {"supported_types": SUPPORTED_REPORT_TYPES, "supported_statuses": SUPPORTED_SUBMISSION_STATUSES},
	"compliance": {"supported_frameworks": SUPPORTED_COMPLIANCE_FRAMEWORKS},
	"audits": {"supported_types": SUPPORTED_AUDIT_TYPES, "supported_corrective_action_statuses": SUPPORTED_CORRECTIVE_ACTION_STATUSES},
	"governance": {
		"require_tenant_context": True, "policy_attached_for_writes": True,
		"audit_events": True, "hipaa_compliance_required": True,
		"cross_tenant_regulatory_access_denied": True,
		"sentinel_event_notification_required": True,
		"hipaa_breach_requires_notification": True,
		"license_expiry_alert_required": True,
	},
	"observability": {"event_stream": REG_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "workflow": "wflo", "compliance": "comp", "monitoring": "moni", "event_stream": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_licenses": True, "enable_accreditation": True, "enable_incidents": True, "enable_submissions": True, "enable_audits": True, "enable_hipaa": True},
	"theme": {"default_theme": "healthcare_reg_clinical", "allow_tenant_overrides": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
}

PROVIDES = [
	"facility_licensing_management", "accreditation_management",
	"incident_reporting", "hipaa_compliance_tracking",
	"regulatory_submission_management", "audit_management",
	"corrective_action_tracking", "compliance_dashboard",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "comp", "moni", "mqeb"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/healthcare-reg/dashboard", "component": "RegDashboard", "permission": "healthcare_reg:view", "nav_group": "Overview"},
	{"name": "licenses", "path": "/healthcare-reg/licenses", "component": "RegLicenseList", "permission": "healthcare_reg:licenses", "nav_group": "Licensing"},
	{"name": "license_detail", "path": "/healthcare-reg/licenses/<id>", "component": "RegLicenseDetail", "permission": "healthcare_reg:licenses", "nav_group": "Licensing"},
	{"name": "accreditation", "path": "/healthcare-reg/accreditation", "component": "RegAccreditationList", "permission": "healthcare_reg:accreditation", "nav_group": "Accreditation"},
	{"name": "incidents", "path": "/healthcare-reg/incidents", "component": "RegIncidentList", "permission": "healthcare_reg:incidents", "nav_group": "Incidents"},
	{"name": "incident_new", "path": "/healthcare-reg/incidents/new", "component": "RegIncidentForm", "permission": "healthcare_reg:incidents_write", "nav_group": "Incidents"},
	{"name": "incident_detail", "path": "/healthcare-reg/incidents/<id>", "component": "RegIncidentDetail", "permission": "healthcare_reg:incidents", "nav_group": "Incidents"},
	{"name": "submissions", "path": "/healthcare-reg/submissions", "component": "RegSubmissionList", "permission": "healthcare_reg:submissions", "nav_group": "Submissions"},
	{"name": "hipaa", "path": "/healthcare-reg/hipaa", "component": "RegHipaaCompliance", "permission": "healthcare_reg:hipaa", "nav_group": "HIPAA"},
	{"name": "audits", "path": "/healthcare-reg/audits", "component": "RegAuditList", "permission": "healthcare_reg:audits", "nav_group": "Audits"},
	{"name": "corrective_actions", "path": "/healthcare-reg/corrective-actions", "component": "RegCorrectiveActions", "permission": "healthcare_reg:corrective_actions", "nav_group": "Quality"},
	{"name": "agents", "path": "/healthcare-reg/agents", "component": "RegAgentWorkbench", "permission": "healthcare_reg:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/healthcare-reg/settings", "component": "RegSettings", "permission": "healthcare_reg:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "healthcare_reg_clinical",
	"tokens": {
		"color.primary": "#7F1D1D", "color.accent": "#1D4ED8", "color.success": "#166534",
		"color.warning": "#A16207", "color.danger": "#B91C1C",
		"surface.canvas": "#FFF1F2", "surface.panel": "#FFFFFF",
		"text.primary": "#7F1D1D", "text.secondary": "#991B1B",
		"border.radius": "6px", "density": "comfortable",
	},
	"components": {
		"licenses": {"icon": "award", "status_indicator": "license-status-chip"},
		"accreditation": {"icon": "shield-check", "status_indicator": "accreditation-status-chip"},
		"incidents": {"icon": "alert-triangle", "status_indicator": "incident-severity-chip"},
		"submissions": {"icon": "send", "status_indicator": "submission-status-chip"},
		"hipaa": {"icon": "lock", "status_indicator": "hipaa-status-chip"},
		"audits": {"icon": "search", "status_indicator": "audit-type-chip"},
		"corrective_actions": {"icon": "check-square", "status_indicator": "corrective-action-status-chip"},
		"agents": {"icon": "cpu", "status_indicator": "agent-runtime-chip"},
	},
}

STREAMING = {
	"processor": "bytewax", "stream": REG_EVENT_STREAM, "key": "tenant_id",
	"events": [
		"license_added", "license_expiring", "accreditation_status_changed",
		"incident_reported", "sentinel_event_reported", "hipaa_breach_reported",
		"submission_filed", "submission_accepted", "audit_completed",
		"corrective_action_opened", "corrective_action_completed",
	],
	"guardrails": [
		"sentinel_event_notification_required", "hipaa_breach_requires_notification",
		"cross_tenant_regulatory_access_denied", "license_expiry_alert_required",
		"privileged_agent_action_requires_human_approval",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "write_policy_required", "required_action": "attach_write_policy"}},
	{"name": "cross_tenant_regulatory_access_denied", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_regulatory_data_access_prohibited", "required_action": "use_tenant_scoped_query"}},
	{"name": "license_type_supported", "condition": {"operation": "add_license", "license_type_supported": False}, "effect": {"decision": "deny", "reason": "license_type_not_supported", "required_action": "select_supported_license_type"}},
	{"name": "accreditation_body_supported", "condition": {"operation": "add_accreditation", "accreditation_body_supported": False}, "effect": {"decision": "deny", "reason": "accreditation_body_not_supported", "required_action": "select_supported_accreditation_body"}},
	{"name": "accreditation_status_supported", "condition": {"operation": "update_accreditation", "accreditation_status_supported": False}, "effect": {"decision": "deny", "reason": "accreditation_status_not_supported", "required_action": "select_supported_accreditation_status"}},
	{"name": "incident_type_supported", "condition": {"operation": "report_incident", "incident_type_supported": False}, "effect": {"decision": "deny", "reason": "incident_type_not_supported", "required_action": "select_supported_incident_type"}},
	{"name": "incident_severity_supported", "condition": {"operation": "report_incident", "incident_severity_supported": False}, "effect": {"decision": "deny", "reason": "incident_severity_not_supported", "required_action": "select_supported_incident_severity"}},
	{"name": "sentinel_event_requires_rca", "condition": {"operation": "close_incident", "incident_type": "sentinel_event", "rca_completed": False}, "effect": {"decision": "deny", "reason": "root_cause_analysis_required_for_sentinel_event", "required_action": "complete_root_cause_analysis"}},
	{"name": "sentinel_event_notification_required", "condition": {"operation": "report_incident", "incident_type": "sentinel_event", "notification_sent": False}, "effect": {"decision": "warn", "reason": "sentinel_event_notification_required_within_72_hours", "required_action": "send_sentinel_event_notification"}},
	{"name": "hipaa_breach_requires_notification", "condition": {"operation": "report_incident", "incident_type": "hipaa_breach", "breach_notification_sent": False}, "effect": {"decision": "warn", "reason": "hipaa_breach_notification_required", "required_action": "initiate_breach_notification_protocol"}},
	{"name": "report_type_supported", "condition": {"operation": "file_submission", "report_type_supported": False}, "effect": {"decision": "deny", "reason": "report_type_not_supported", "required_action": "select_supported_report_type"}},
	{"name": "submission_status_supported", "condition": {"operation": "update_submission", "submission_status_supported": False}, "effect": {"decision": "deny", "reason": "submission_status_not_supported", "required_action": "select_supported_submission_status"}},
	{"name": "compliance_framework_supported", "condition": {"operation": "create_compliance_record", "framework_supported": False}, "effect": {"decision": "deny", "reason": "compliance_framework_not_supported", "required_action": "select_supported_framework"}},
	{"name": "audit_type_supported", "condition": {"operation": "create_audit", "audit_type_supported": False}, "effect": {"decision": "deny", "reason": "audit_type_not_supported", "required_action": "select_supported_audit_type"}},
	{"name": "corrective_action_status_supported", "condition": {"operation": "update_corrective_action", "corrective_action_status_supported": False}, "effect": {"decision": "deny", "reason": "corrective_action_status_not_supported", "required_action": "select_supported_corrective_action_status"}},
	{"name": "license_expiry_alert_required", "condition": {"operation": "check_license", "days_to_expiry": 90, "alert_sent": False}, "effect": {"decision": "warn", "reason": "license_expiring_within_90_days", "required_action": "initiate_license_renewal"}},
	{"name": "agent_privileged_action_requires_approval", "condition": {"agent_action": True, "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "privileged_agent_action_requires_human_approval", "required_action": "record_human_approval"}},
	{"name": "closed_submission_not_modifiable", "condition": {"operation": "update_submission", "submission_status": "closed"}, "effect": {"decision": "deny", "reason": "closed_submission_cannot_be_modified", "required_action": "create_amendment_submission"}},
	{"name": "overdue_corrective_action_escalation", "condition": {"operation": "check_corrective_action", "corrective_action_status": "overdue"}, "effect": {"decision": "warn", "reason": "corrective_action_overdue", "required_action": "escalate_to_quality_director"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	return {
		"capability": CAPABILITY_ID, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION,
		"configuration": config,
		"configuration_schema": {"required": ["tenant_id", "ui", "theme"], "properties": {"tenant_id": {"type": "string"}, "ui": {"type": "object"}, "theme": {"type": "object"}}},
		"rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": RULES},
		"ui": {"shell": "apg_python", "requires_theme": True, "template_roots": ["healthcare/reg/templates"], "routes": UI_ROUTES},
		"theme": THEME, "streaming": STREAMING, "provides": PROVIDES, "requires": REQUIRES,
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	for rule in RULES:
		cond = rule["condition"]
		if all(context.get(k) == v for k, v in cond.items()):
			effect = rule["effect"]
			return {"rule": rule["name"], "decision": effect["decision"], "reason": effect["reason"], "required_action": effect.get("required_action")}
	return {"rule": None, "decision": "allow", "reason": "no_rule_matched", "required_action": None}
