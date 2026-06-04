"""Executable capability contract for APG Learning Management System."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "education_lms"
CAPABILITY_NAME = "Learning Management System"
CAPABILITY_VERSION = "1.0.0"
LMS_EVENT_STREAM = "apg.education.lms.lifecycle"

# --- supported value sets ---------------------------------------------------

SUPPORTED_COURSE_TYPES = [
	"instructor_led", "self_paced", "blended", "cohort_based",
	"microlearning", "certification", "competency_based", "mooc",
]
SUPPORTED_CONTENT_TYPES = [
	"video", "document", "scorm", "xapi", "quiz", "assignment",
	"discussion", "live_session", "lab", "simulation",
]
SUPPORTED_ENROLMENT_TYPES = [
	"open", "approval_required", "invitation_only",
	"paid", "voucher", "auto_enrol",
]
SUPPORTED_ASSESSMENT_TYPES = [
	"formative_quiz", "summative_exam", "assignment", "project",
	"peer_review", "practical", "oral", "portfolio",
]
SUPPORTED_GRADING_SCHEMES = [
	"percentage", "letter_grade", "pass_fail", "competency",
	"mastery", "rubric", "points", "weighted",
]
SUPPORTED_COMPLETION_CRITERIA = [
	"all_content_viewed", "passing_score", "assignment_submitted",
	"instructor_mark", "time_spent", "peer_review_complete",
]
SUPPORTED_CERTIFICATE_TYPES = [
	"completion", "achievement", "competency", "professional_development",
]
SUPPORTED_LEARNER_STATUSES = [
	"active", "suspended", "withdrawn", "completed", "pending",
]
SUPPORTED_COURSE_STATUSES = [
	"draft", "review", "published", "archived", "retired",
]
SUPPORTED_ENROLMENT_STATUSES = [
	"pending", "active", "completed", "withdrawn", "failed", "on_hold",
]
SUPPORTED_SUBMISSION_STATUSES = [
	"not_started", "in_progress", "submitted", "graded", "returned", "late",
]
SUPPORTED_SCORM_VERSIONS = ["scorm_1_2", "scorm_2004_3rd", "scorm_2004_4th", "xapi_1_0_3"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = [
	"content_reviewer", "assessment_grader", "learner_advisor",
	"analytics_reporter", "compliance_auditor",
]

# --- wiring -----------------------------------------------------------------

PROVIDES = [
	"course_lifecycle_workflow",
	"content_delivery_workflow",
	"enrolment_workflow",
	"assessment_workflow",
	"grading_workflow",
	"certificate_issuance_workflow",
	"learner_analytics_workflow",
	"scorm_xapi_compliance_workflow",
	"learning_path_workflow",
	"cohort_management_workflow",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "nlpc", "moni", "comp", "mqeb", "schd"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/lms/dashboard", "component": "LmsDashboard", "permission": "education_lms:view", "nav_group": "Overview"},
	{"name": "courses", "path": "/lms/courses", "component": "CourseLibrary", "permission": "education_lms:view", "nav_group": "Learning"},
	{"name": "course_create", "path": "/lms/courses/create", "component": "CourseEditor", "permission": "education_lms:manage_courses", "nav_group": "Learning"},
	{"name": "course_detail", "path": "/lms/courses/<course_id>", "component": "CourseDetail", "permission": "education_lms:view", "nav_group": "Learning"},
	{"name": "content_builder", "path": "/lms/courses/<course_id>/content", "component": "ContentBuilder", "permission": "education_lms:manage_content", "nav_group": "Learning"},
	{"name": "enrolments", "path": "/lms/enrolments", "component": "EnrolmentConsole", "permission": "education_lms:manage_enrolments", "nav_group": "Administration"},
	{"name": "assessments", "path": "/lms/assessments", "component": "AssessmentWorkbench", "permission": "education_lms:manage_assessments", "nav_group": "Assessment"},
	{"name": "submissions", "path": "/lms/submissions", "component": "SubmissionQueue", "permission": "education_lms:grade", "nav_group": "Assessment"},
	{"name": "gradebook", "path": "/lms/gradebook", "component": "Gradebook", "permission": "education_lms:grade", "nav_group": "Assessment"},
	{"name": "certificates", "path": "/lms/certificates", "component": "CertificateConsole", "permission": "education_lms:manage_certificates", "nav_group": "Credentials"},
	{"name": "learner_analytics", "path": "/lms/analytics", "component": "LearnerAnalytics", "permission": "education_lms:analytics", "nav_group": "Insights"},
	{"name": "learning_paths", "path": "/lms/paths", "component": "LearningPathEditor", "permission": "education_lms:manage_paths", "nav_group": "Learning"},
	{"name": "agents", "path": "/lms/agents", "component": "LmsAgentWorkbench", "permission": "education_lms:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/lms/settings", "component": "LmsSettings", "permission": "education_lms:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "lms_academy",
	"tokens": {
		"color.primary": "#1D4ED8",
		"color.accent": "#7C3AED",
		"color.success": "#15803D",
		"color.warning": "#B45309",
		"color.danger": "#B91C1C",
		"surface.canvas": "#F0F4FF",
		"surface.panel": "#FFFFFF",
		"text.primary": "#1E293B",
		"text.secondary": "#475569",
		"border.radius": "10px",
		"density": "comfortable",
	},
	"components": {
		"courses": {"icon": "book-open", "status_indicator": "course-status-chip"},
		"content": {"icon": "layers", "status_indicator": "content-type-chip"},
		"enrolments": {"icon": "user-plus", "status_indicator": "enrolment-status-chip"},
		"assessments": {"icon": "clipboard-list", "status_indicator": "assessment-type-chip"},
		"submissions": {"icon": "file-check", "status_indicator": "submission-status-chip"},
		"gradebook": {"icon": "bar-chart-2", "status_indicator": "grade-chip"},
		"certificates": {"icon": "award", "status_indicator": "certificate-type-chip"},
		"learning_paths": {"icon": "git-branch", "status_indicator": "path-status-chip"},
		"agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": LMS_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"course_created", "course_published", "course_archived",
		"content_item_added", "enrolment_recorded", "enrolment_withdrawn",
		"assessment_submitted", "grade_recorded", "certificate_issued",
		"learning_path_assigned", "learner_progress_updated",
	],
	"guardrails": [
		"lms_batch_requires_bytewax",
		"grade_override_requires_approval",
		"certificate_issuance_requires_passing_score",
		"scorm_xapi_requires_compliance_check",
		"privileged_agent_action_requires_human_approval",
		"cross_tenant_enrolment_denied",
		"unapproved_content_publish_denied",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "lms_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "lms_policy_required", "required_action": "attach_lms_policy"}},
	{"name": "course_type_supported", "condition": {"operation": "create_course", "course_type_supported": False}, "effect": {"decision": "deny", "reason": "course_type_not_supported", "required_action": "select_supported_course_type"}},
	{"name": "course_publish_requires_review", "condition": {"operation": "publish_course", "review_approved": False}, "effect": {"decision": "deny", "reason": "course_must_be_reviewed_before_publish", "required_action": "submit_for_review"}},
	{"name": "content_type_supported", "condition": {"operation": "add_content", "content_type_supported": False}, "effect": {"decision": "deny", "reason": "content_type_not_supported", "required_action": "select_supported_content_type"}},
	{"name": "scorm_version_supported", "condition": {"operation": "add_content", "content_type": "scorm", "scorm_version_supported": False}, "effect": {"decision": "deny", "reason": "scorm_version_not_supported", "required_action": "select_supported_scorm_version"}},
	{"name": "enrolment_type_supported", "condition": {"operation": "enrol_learner", "enrolment_type_supported": False}, "effect": {"decision": "deny", "reason": "enrolment_type_not_supported", "required_action": "select_supported_enrolment_type"}},
	{"name": "paid_enrolment_requires_payment_reference", "condition": {"operation": "enrol_learner", "enrolment_type": "paid", "payment_reference_present": False}, "effect": {"decision": "deny", "reason": "payment_reference_required_for_paid_enrolment", "required_action": "provide_payment_reference"}},
	{"name": "cross_tenant_enrolment_denied", "condition": {"operation": "enrol_learner", "course_tenant_matches_learner_tenant": False}, "effect": {"decision": "deny", "reason": "cross_tenant_enrolment_not_allowed", "required_action": "enrol_within_tenant"}},
	{"name": "assessment_type_supported", "condition": {"operation": "create_assessment", "assessment_type_supported": False}, "effect": {"decision": "deny", "reason": "assessment_type_not_supported", "required_action": "select_supported_assessment_type"}},
	{"name": "grading_scheme_supported", "condition": {"operation": "configure_grading", "grading_scheme_supported": False}, "effect": {"decision": "deny", "reason": "grading_scheme_not_supported", "required_action": "select_supported_grading_scheme"}},
	{"name": "grade_override_requires_approval", "condition": {"operation": "override_grade", "approval_reference_present": False}, "effect": {"decision": "deny", "reason": "grade_override_requires_approver_sign_off", "required_action": "obtain_grade_override_approval"}},
	{"name": "certificate_requires_completion", "condition": {"operation": "issue_certificate", "completion_criteria_met": False}, "effect": {"decision": "deny", "reason": "completion_criteria_not_met_for_certificate", "required_action": "meet_completion_criteria"}},
	{"name": "certificate_type_supported", "condition": {"operation": "issue_certificate", "certificate_type_supported": False}, "effect": {"decision": "deny", "reason": "certificate_type_not_supported", "required_action": "select_supported_certificate_type"}},
	{"name": "completion_criterion_supported", "condition": {"operation": "set_completion_criteria", "criterion_supported": False}, "effect": {"decision": "deny", "reason": "completion_criterion_not_supported", "required_action": "select_supported_completion_criterion"}},
	{"name": "learner_status_supported", "condition": {"operation": "update_learner_status", "learner_status_supported": False}, "effect": {"decision": "deny", "reason": "learner_status_not_supported", "required_action": "select_supported_learner_status"}},
	{"name": "course_status_transition_valid", "condition": {"operation": "update_course_status", "status_transition_valid": False}, "effect": {"decision": "deny", "reason": "invalid_course_status_transition", "required_action": "follow_valid_status_transition"}},
	{"name": "submission_late_penalty_requires_policy", "condition": {"operation": "apply_late_penalty", "late_penalty_policy_present": False}, "effect": {"decision": "deny", "reason": "late_penalty_policy_required", "required_action": "attach_late_penalty_policy"}},
	{"name": "privileged_agent_action_requires_human_approval", "condition": {"operation": "agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required_for_privileged_agent_action", "required_action": "record_human_approval"}},
	{"name": "analytics_export_requires_consent", "condition": {"operation": "export_learner_analytics", "consent_recorded": False}, "effect": {"decision": "deny", "reason": "learner_consent_required_for_analytics_export", "required_action": "record_learner_consent"}},
	{"name": "scorm_compliance_check_required", "condition": {"operation": "publish_scorm_content", "compliance_checked": False}, "effect": {"decision": "deny", "reason": "scorm_compliance_check_required_before_publish", "required_action": "run_scorm_compliance_check"}},
	{"name": "batch_import_requires_bytewax", "condition": {"operation": "batch_import", "event_stream": "bytewax", "item_count_valid": False}, "effect": {"decision": "deny", "reason": "batch_import_requires_bytewax_stream", "required_action": "configure_bytewax_stream"}},
]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"ui": {"enable_dashboard": True, "enable_courses": True, "enable_content": True, "enable_enrolments": True, "enable_assessments": True, "enable_gradebook": True, "enable_certificates": True, "enable_analytics": True, "enable_learning_paths": True, "enable_agents": True},
	"theme": {"default_theme": "lms_academy", "allow_tenant_overrides": True},
	"courses": {"supported_types": SUPPORTED_COURSE_TYPES, "supported_statuses": SUPPORTED_COURSE_STATUSES, "review_before_publish": True, "owner_required": True},
	"content": {"supported_types": SUPPORTED_CONTENT_TYPES, "supported_scorm_versions": SUPPORTED_SCORM_VERSIONS, "compliance_check_required": True},
	"enrolments": {"supported_types": SUPPORTED_ENROLMENT_TYPES, "supported_statuses": SUPPORTED_ENROLMENT_STATUSES, "cross_tenant_denied": True},
	"assessments": {"supported_types": SUPPORTED_ASSESSMENT_TYPES, "supported_grading_schemes": SUPPORTED_GRADING_SCHEMES, "supported_completion_criteria": SUPPORTED_COMPLETION_CRITERIA},
	"certificates": {"supported_types": SUPPORTED_CERTIFICATE_TYPES, "completion_required": True},
	"learners": {"supported_statuses": SUPPORTED_LEARNER_STATUSES},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "cross_tenant_enrolment_denied": True, "grade_override_requires_approval": True, "certificate_requires_completion": True, "analytics_export_requires_consent": True},
	"observability": {"event_stream": LMS_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "workflow": "wflo", "nlp": "nlpc", "monitoring": "moni", "compliance": "comp", "event_stream": "bytewax", "scheduler": "schd"},
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
				"courses": {"type": "object"},
				"content": {"type": "object"},
				"enrolments": {"type": "object"},
				"assessments": {"type": "object"},
				"certificates": {"type": "object"},
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
			"template_roots": ["education/lms/templates"],
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
