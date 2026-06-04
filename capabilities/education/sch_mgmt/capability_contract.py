"""Executable capability contract for APG School Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "education_sch_mgmt"
CAPABILITY_NAME = "School Management"
CAPABILITY_VERSION = "1.0.0"
SCH_MGMT_EVENT_STREAM = "apg.education.sch_mgmt.lifecycle"

# --- supported value sets ---------------------------------------------------

SUPPORTED_STUDENT_STATUSES = [
	"prospective", "applied", "enrolled", "active", "suspended",
	"expelled", "graduated", "withdrawn", "alumni", "deferred",
]
SUPPORTED_ADMISSION_STATUSES = [
	"draft", "submitted", "under_review", "shortlisted",
	"offered", "accepted", "rejected", "waitlisted", "deferred",
]
SUPPORTED_FEE_TYPES = [
	"tuition", "registration", "activity", "transport", "boarding",
	"examination", "library", "lab", "uniform", "sports", "miscellaneous",
]
SUPPORTED_FEE_STATUSES = [
	"pending", "partial", "paid", "overdue", "waived", "refunded",
]
SUPPORTED_STAFF_ROLES = [
	"teacher", "hod", "principal", "vice_principal", "counselor",
	"librarian", "admin_staff", "finance_officer", "it_officer", "support_staff",
]
SUPPORTED_STAFF_STATUSES = [
	"active", "on_leave", "suspended", "terminated", "contract",
]
SUPPORTED_DOCUMENT_TYPES = [
	"birth_certificate", "national_id", "passport", "transcript",
	"medical_report", "recommendation_letter", "fee_receipt", "report_card",
]
SUPPORTED_GRADE_LEVELS = [
	"pre_k", "kindergarten", "grade_1", "grade_2", "grade_3", "grade_4",
	"grade_5", "grade_6", "grade_7", "grade_8", "grade_9", "grade_10",
	"grade_11", "grade_12", "form_1", "form_2", "form_3", "form_4",
	"year_1", "year_2", "year_3", "year_4",
]
SUPPORTED_TERM_TYPES = ["semester", "trimester", "quarter", "term", "annual"]
SUPPORTED_EVENT_TYPES = [
	"academic", "examination", "holiday", "extracurricular",
	"parent_meeting", "staff_meeting", "sports", "cultural",
]
SUPPORTED_COMMUNICATION_CHANNELS = [
	"email", "sms", "portal_message", "push_notification", "letter",
]
SUPPORTED_REPORT_TYPES = [
	"academic_transcript", "fee_statement", "attendance_report",
	"staff_roster", "class_list", "progress_report",
]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = [
	"admissions_assistant", "fee_processor", "record_auditor",
	"communication_dispatcher", "report_generator",
]

# --- wiring -----------------------------------------------------------------

PROVIDES = [
	"student_records_workflow",
	"admissions_workflow",
	"fee_management_workflow",
	"parent_portal_workflow",
	"staff_administration_workflow",
	"academic_calendar_workflow",
	"document_management_workflow",
	"communications_workflow",
	"reporting_workflow",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "comp", "mqeb", "schd"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/sch-mgmt/dashboard", "component": "SchMgmtDashboard", "permission": "education_sch_mgmt:view", "nav_group": "Overview"},
	{"name": "students", "path": "/sch-mgmt/students", "component": "StudentRegistry", "permission": "education_sch_mgmt:view_students", "nav_group": "Students"},
	{"name": "student_detail", "path": "/sch-mgmt/students/<student_id>", "component": "StudentProfile", "permission": "education_sch_mgmt:view_students", "nav_group": "Students"},
	{"name": "admissions", "path": "/sch-mgmt/admissions", "component": "AdmissionsConsole", "permission": "education_sch_mgmt:manage_admissions", "nav_group": "Admissions"},
	{"name": "fees", "path": "/sch-mgmt/fees", "component": "FeeManagementConsole", "permission": "education_sch_mgmt:manage_fees", "nav_group": "Finance"},
	{"name": "fee_invoices", "path": "/sch-mgmt/fees/invoices", "component": "FeeInvoiceLedger", "permission": "education_sch_mgmt:manage_fees", "nav_group": "Finance"},
	{"name": "parent_portal", "path": "/sch-mgmt/parent-portal", "component": "ParentPortal", "permission": "education_sch_mgmt:parent_access", "nav_group": "Community"},
	{"name": "staff", "path": "/sch-mgmt/staff", "component": "StaffDirectory", "permission": "education_sch_mgmt:manage_staff", "nav_group": "Human Resources"},
	{"name": "academic_calendar", "path": "/sch-mgmt/calendar", "component": "AcademicCalendar", "permission": "education_sch_mgmt:manage_calendar", "nav_group": "Planning"},
	{"name": "documents", "path": "/sch-mgmt/documents", "component": "DocumentVault", "permission": "education_sch_mgmt:manage_documents", "nav_group": "Records"},
	{"name": "communications", "path": "/sch-mgmt/communications", "component": "CommunicationsHub", "permission": "education_sch_mgmt:send_communications", "nav_group": "Community"},
	{"name": "reports", "path": "/sch-mgmt/reports", "component": "SchoolReports", "permission": "education_sch_mgmt:view_reports", "nav_group": "Insights"},
	{"name": "agents", "path": "/sch-mgmt/agents", "component": "SchMgmtAgentWorkbench", "permission": "education_sch_mgmt:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/sch-mgmt/settings", "component": "SchMgmtSettings", "permission": "education_sch_mgmt:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "sch_mgmt_campus",
	"tokens": {
		"color.primary": "#0F766E",
		"color.accent": "#D97706",
		"color.success": "#166534",
		"color.warning": "#92400E",
		"color.danger": "#991B1B",
		"surface.canvas": "#F0FDFA",
		"surface.panel": "#FFFFFF",
		"text.primary": "#134E4A",
		"text.secondary": "#5EEAD4",
		"border.radius": "8px",
		"density": "comfortable",
	},
	"components": {
		"students": {"icon": "users", "status_indicator": "student-status-chip"},
		"admissions": {"icon": "clipboard-check", "status_indicator": "admission-status-chip"},
		"fees": {"icon": "credit-card", "status_indicator": "fee-status-chip"},
		"staff": {"icon": "briefcase", "status_indicator": "staff-status-chip"},
		"calendar": {"icon": "calendar", "status_indicator": "event-type-chip"},
		"documents": {"icon": "folder", "status_indicator": "document-type-chip"},
		"communications": {"icon": "message-circle", "status_indicator": "channel-chip"},
		"reports": {"icon": "file-text", "status_indicator": "report-type-chip"},
		"agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": SCH_MGMT_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"student_enrolled", "student_status_changed", "admission_submitted",
		"admission_decision_recorded", "fee_invoice_generated", "fee_payment_recorded",
		"staff_record_created", "calendar_event_published", "document_uploaded",
		"communication_dispatched", "report_generated",
	],
	"guardrails": [
		"sch_mgmt_batch_requires_bytewax",
		"fee_waiver_requires_approval",
		"expulsion_requires_approval",
		"document_sharing_requires_consent",
		"privileged_agent_action_requires_human_approval",
		"cross_tenant_record_access_denied",
		"student_data_export_requires_consent",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "sch_mgmt_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "sch_mgmt_policy_required", "required_action": "attach_sch_mgmt_policy"}},
	{"name": "student_status_supported", "condition": {"operation": "update_student_status", "student_status_supported": False}, "effect": {"decision": "deny", "reason": "student_status_not_supported", "required_action": "select_supported_student_status"}},
	{"name": "expulsion_requires_approval", "condition": {"operation": "update_student_status", "new_status": "expelled", "approval_reference_present": False}, "effect": {"decision": "deny", "reason": "expulsion_requires_approver_sign_off", "required_action": "obtain_expulsion_approval"}},
	{"name": "admission_status_supported", "condition": {"operation": "update_admission_status", "admission_status_supported": False}, "effect": {"decision": "deny", "reason": "admission_status_not_supported", "required_action": "select_supported_admission_status"}},
	{"name": "admission_offer_requires_capacity_check", "condition": {"operation": "offer_admission", "capacity_available": False}, "effect": {"decision": "deny", "reason": "admission_offer_requires_available_capacity", "required_action": "verify_class_capacity"}},
	{"name": "fee_type_supported", "condition": {"operation": "create_fee_invoice", "fee_type_supported": False}, "effect": {"decision": "deny", "reason": "fee_type_not_supported", "required_action": "select_supported_fee_type"}},
	{"name": "fee_waiver_requires_approval", "condition": {"operation": "waive_fee", "approval_reference_present": False}, "effect": {"decision": "deny", "reason": "fee_waiver_requires_approver_sign_off", "required_action": "obtain_fee_waiver_approval"}},
	{"name": "fee_refund_requires_approval", "condition": {"operation": "refund_fee", "approval_reference_present": False}, "effect": {"decision": "deny", "reason": "fee_refund_requires_approver_sign_off", "required_action": "obtain_fee_refund_approval"}},
	{"name": "staff_role_supported", "condition": {"operation": "create_staff_record", "staff_role_supported": False}, "effect": {"decision": "deny", "reason": "staff_role_not_supported", "required_action": "select_supported_staff_role"}},
	{"name": "document_type_supported", "condition": {"operation": "upload_document", "document_type_supported": False}, "effect": {"decision": "deny", "reason": "document_type_not_supported", "required_action": "select_supported_document_type"}},
	{"name": "document_sharing_requires_consent", "condition": {"operation": "share_document", "consent_recorded": False}, "effect": {"decision": "deny", "reason": "consent_required_before_sharing_document", "required_action": "record_consent"}},
	{"name": "event_type_supported", "condition": {"operation": "create_calendar_event", "event_type_supported": False}, "effect": {"decision": "deny", "reason": "calendar_event_type_not_supported", "required_action": "select_supported_event_type"}},
	{"name": "communication_channel_supported", "condition": {"operation": "dispatch_communication", "channel_supported": False}, "effect": {"decision": "deny", "reason": "communication_channel_not_supported", "required_action": "select_supported_channel"}},
	{"name": "report_type_supported", "condition": {"operation": "generate_report", "report_type_supported": False}, "effect": {"decision": "deny", "reason": "report_type_not_supported", "required_action": "select_supported_report_type"}},
	{"name": "student_data_export_requires_consent", "condition": {"operation": "export_student_data", "consent_recorded": False}, "effect": {"decision": "deny", "reason": "student_data_export_requires_consent", "required_action": "record_data_export_consent"}},
	{"name": "cross_tenant_record_access_denied", "condition": {"operation": "access_student_record", "record_tenant_matches_requestor_tenant": False}, "effect": {"decision": "deny", "reason": "cross_tenant_record_access_denied", "required_action": "access_within_tenant"}},
	{"name": "term_type_supported", "condition": {"operation": "create_term", "term_type_supported": False}, "effect": {"decision": "deny", "reason": "term_type_not_supported", "required_action": "select_supported_term_type"}},
	{"name": "grade_level_supported", "condition": {"operation": "assign_grade_level", "grade_level_supported": False}, "effect": {"decision": "deny", "reason": "grade_level_not_supported", "required_action": "select_supported_grade_level"}},
	{"name": "privileged_agent_action_requires_human_approval", "condition": {"operation": "agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required_for_privileged_agent_action", "required_action": "record_human_approval"}},
	{"name": "batch_import_requires_bytewax", "condition": {"operation": "batch_import", "event_stream": "bytewax", "item_count_valid": False}, "effect": {"decision": "deny", "reason": "batch_import_requires_bytewax_stream", "required_action": "configure_bytewax_stream"}},
]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"ui": {"enable_dashboard": True, "enable_students": True, "enable_admissions": True, "enable_fees": True, "enable_parent_portal": True, "enable_staff": True, "enable_calendar": True, "enable_documents": True, "enable_communications": True, "enable_reports": True, "enable_agents": True},
	"theme": {"default_theme": "sch_mgmt_campus", "allow_tenant_overrides": True},
	"students": {"supported_statuses": SUPPORTED_STUDENT_STATUSES, "supported_grade_levels": SUPPORTED_GRADE_LEVELS, "expulsion_requires_approval": True},
	"admissions": {"supported_statuses": SUPPORTED_ADMISSION_STATUSES, "capacity_check_required": True},
	"fees": {"supported_types": SUPPORTED_FEE_TYPES, "supported_statuses": SUPPORTED_FEE_STATUSES, "waiver_requires_approval": True, "refund_requires_approval": True},
	"staff": {"supported_roles": SUPPORTED_STAFF_ROLES, "supported_statuses": SUPPORTED_STAFF_STATUSES},
	"calendar": {"supported_event_types": SUPPORTED_EVENT_TYPES, "supported_term_types": SUPPORTED_TERM_TYPES},
	"documents": {"supported_types": SUPPORTED_DOCUMENT_TYPES, "sharing_requires_consent": True},
	"communications": {"supported_channels": SUPPORTED_COMMUNICATION_CHANNELS},
	"reports": {"supported_types": SUPPORTED_REPORT_TYPES},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "cross_tenant_record_access_denied": True, "student_data_export_requires_consent": True},
	"observability": {"event_stream": SCH_MGMT_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "workflow": "wflo", "compliance": "comp", "event_stream": "bytewax", "scheduler": "schd"},
}


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	"""Return the full capability contract for the given tenant."""
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	config["ui"]["tenant_id"] = tenant_id
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
				"students": {"type": "object"},
				"admissions": {"type": "object"},
				"fees": {"type": "object"},
				"staff": {"type": "object"},
				"governance": {"type": "object"},
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
			"template_roots": ["education/sch_mgmt/templates"],
			"routes": UI_ROUTES,
		},
		"theme": THEME,
		"provides": PROVIDES,
		"requires": REQUIRES,
		"streaming": STREAMING,
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate deterministic rules against the provided context dict."""
	for rule in RULES:
		cond = rule["condition"]
		if all(context.get(k) == v for k, v in cond.items()):
			return {
				"matched_rule": rule["name"],
				**rule["effect"],
			}
	return {"matched_rule": None, "decision": "allow", "reason": "no_rule_matched"}
