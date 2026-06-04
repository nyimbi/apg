"""Executable capability contract for HCM Employee Data Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "chr_employee_data_management"
CAPABILITY_NAME = "Employee Data Management"
CAPABILITY_VERSION = "2.2.0"
EMPLOYEE_EVENT_STREAM = "apg.hcm.chr.employee.lifecycle"

SUPPORTED_EMPLOYMENT_TYPES = ["full_time", "part_time", "contractor", "intern", "temporary", "fixed_term", "zero_hours"]
SUPPORTED_EMPLOYMENT_STATUSES = ["draft", "active", "leave", "suspended", "terminated", "alumni", "probation", "notice"]
SUPPORTED_WORK_MODES = ["onsite", "hybrid", "remote", "field", "travelling"]
SUPPORTED_SKILL_LEVELS = ["awareness", "working", "practitioner", "expert", "master"]
SUPPORTED_CERTIFICATION_STATUSES = ["pending", "active", "expired", "revoked", "suspended"]
SUPPORTED_DATA_DOMAINS = ["identity", "employment", "organization", "skills", "certifications", "contacts", "privacy", "compensation"]
SUPPORTED_EMPLOYEE_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_EMPLOYEE_AGENT_ROLES = [
	"profile_steward",
	"data_quality_reviewer",
	"org_design_reviewer",
	"skills_reviewer",
	"compliance_reviewer",
	"onboarding_reviewer",
]
SUPPORTED_HISTORY_EVENTS = [
	"hire", "rehire", "transfer", "promotion", "demotion", "status_change",
	"compensation_change", "termination", "leave_start", "leave_return",
	"department_change", "position_change", "work_mode_change",
]
SUPPORTED_TERMINATION_REASONS = [
	"resignation", "redundancy", "performance", "misconduct",
	"contract_end", "retirement", "death", "mutual_agreement",
]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"employees": {
		"employee_number_required": True,
		"legal_name_required": True,
		"work_email_required": True,
		"department_required": True,
		"position_required": True,
		"manager_required_for_non_executives": True,
		"hire_date_required": True,
		"supported_employment_types": SUPPORTED_EMPLOYMENT_TYPES,
		"supported_statuses": SUPPORTED_EMPLOYMENT_STATUSES,
		"supported_work_modes": SUPPORTED_WORK_MODES,
		"sensitive_change_review_required": True,
		"probation_period_days": 90,
		"notice_period_days": 30,
	},
	"departments": {
		"code_required": True,
		"name_required": True,
		"owner_required": True,
		"cost_center_required": True,
	},
	"positions": {
		"code_required": True,
		"title_required": True,
		"department_required": True,
		"job_level_required": True,
		"headcount_minimum": 0,
		"compensation_band_review_required": True,
	},
	"personal_info": {
		"employee_required": True,
		"effective_date_required": True,
		"country_required": True,
		"privacy_basis_required": True,
	},
	"emergency_contacts": {
		"employee_required": True,
		"name_required": True,
		"relationship_required": True,
		"phone_required": True,
	},
	"employment_history": {
		"employee_required": True,
		"event_type_required": True,
		"effective_date_required": True,
		"reason_required_for_sensitive_events": True,
		"approval_required_for_termination": True,
		"supported_history_events": SUPPORTED_HISTORY_EVENTS,
		"supported_termination_reasons": SUPPORTED_TERMINATION_REASONS,
	},
	"skills": {
		"employee_required": True,
		"skill_required": True,
		"supported_levels": SUPPORTED_SKILL_LEVELS,
		"evidence_required_for_expert": True,
	},
	"certifications": {
		"employee_required": True,
		"name_required": True,
		"issuer_required": True,
		"issued_on_required": True,
		"expires_on_required_for_expiring": True,
		"supported_statuses": SUPPORTED_CERTIFICATION_STATUSES,
	},
	"data_quality": {
		"domain_required": True,
		"supported_domains": SUPPORTED_DATA_DOMAINS,
		"severity_required": True,
		"owner_required_for_high_severity": True,
	},
	"employee_agents": {
		"enabled": True,
		"supported_runtimes": SUPPORTED_EMPLOYEE_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_EMPLOYEE_AGENT_ROLES,
		"max_autonomous_scope": "inspect_prepare_and_recommend",
		"human_approval_required": True,
	},
	"governance": {
		"require_tenant_context": True,
		"policy_attached_for_writes": True,
		"audit_state_changes": True,
		"segregation_of_duties": True,
		"privacy_basis_for_sensitive_data": True,
		"cross_tenant_access_denied": True,
		"privilege_escalation_denied": True,
		"delete_requires_admin_role": True,
		"dual_control_for_mass_updates": True,
		"gdpr_right_to_erasure_workflow": True,
	},
	"observability": {
		"event_stream": EMPLOYEE_EVENT_STREAM,
		"stream_processor": "bytewax",
		"emit_employee_events": True,
		"emit_department_events": True,
		"emit_position_events": True,
		"emit_profile_events": True,
		"emit_skill_events": True,
		"emit_certification_events": True,
		"emit_data_quality_events": True,
		"emit_agent_events": True,
	},
	"adapters": {
		"authorization": "adapter",
		"audit": "adapter",
		"notification": "adapter",
		"workflow": "adapter",
		"document_store": "adapter",
		"payroll": "adapter",
		"benefits": "adapter",
		"identity": "adapter",
		"event_stream": "bytewax",
		"theme": "adapter",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_employees": True,
		"enable_departments": True,
		"enable_positions": True,
		"enable_personal_info": True,
		"enable_contacts": True,
		"enable_history": True,
		"enable_skills": True,
		"enable_certifications": True,
		"enable_quality": True,
		"enable_agents": True,
		"enable_settings": True,
	},
	"theme": {
		"default_theme": "employee_data_control",
		"allow_tenant_overrides": True,
	},
}


PROVIDES = [
	"employee_profile_lifecycle",
	"employee_identity_registry",
	"department_lifecycle",
	"position_lifecycle",
	"employment_history_lifecycle",
	"employee_skill_lifecycle",
	"employee_certification_lifecycle",
	"employee_contact_lifecycle",
	"employee_data_quality_workflow",
	"employee_dashboard_service",
	"employee_agents",
]

REQUIRES = [
	"auth",
	"audl",
	"mten",
	"conf",
	"ntfy",
	"wflo",
	"srch",
	"mdm",
]


UI_ROUTES = [
	{"name": "dashboard", "path": "/hcm/employees/dashboard", "component": "EmployeeDashboard", "permission": "chr_employee_data_management:view", "nav_group": "Overview"},
	{"name": "employees", "path": "/hcm/employees", "component": "EmployeeRegistryWorkbench", "permission": "chr_employee_data_management:manage_employees", "nav_group": "Employees"},
	{"name": "departments", "path": "/hcm/employees/departments", "component": "DepartmentWorkbench", "permission": "chr_employee_data_management:manage_structure", "nav_group": "Organization"},
	{"name": "positions", "path": "/hcm/employees/positions", "component": "PositionWorkbench", "permission": "chr_employee_data_management:manage_structure", "nav_group": "Organization"},
	{"name": "personal_info", "path": "/hcm/employees/personal-info", "component": "PersonalInfoWorkbench", "permission": "chr_employee_data_management:manage_sensitive", "nav_group": "Profiles"},
	{"name": "contacts", "path": "/hcm/employees/contacts", "component": "EmergencyContactWorkbench", "permission": "chr_employee_data_management:manage_employees", "nav_group": "Profiles"},
	{"name": "history", "path": "/hcm/employees/history", "component": "EmploymentHistoryWorkbench", "permission": "chr_employee_data_management:manage_history", "nav_group": "Profiles"},
	{"name": "skills", "path": "/hcm/employees/skills", "component": "EmployeeSkillWorkbench", "permission": "chr_employee_data_management:manage_skills", "nav_group": "Talent"},
	{"name": "certifications", "path": "/hcm/employees/certifications", "component": "EmployeeCertificationWorkbench", "permission": "chr_employee_data_management:manage_certifications", "nav_group": "Talent"},
	{"name": "quality", "path": "/hcm/employees/data-quality", "component": "EmployeeDataQualityWorkbench", "permission": "chr_employee_data_management:govern", "nav_group": "Governance"},
	{"name": "agents", "path": "/hcm/employees/agents", "component": "EmployeeAgentWorkbench", "permission": "chr_employee_data_management:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/hcm/employees/settings", "component": "EmployeeSettings", "permission": "chr_employee_data_management:admin", "nav_group": "Administration"},
]


THEME = {
	"name": "employee_data_control",
	"tokens": {
		"color.primary": "#28536B",
		"color.accent": "#4C6F52",
		"color.success": "#237A57",
		"color.warning": "#B7791F",
		"color.danger": "#B42318",
		"surface.canvas": "#F7F8FA",
		"surface.panel": "#FFFFFF",
		"text.primary": "#172033",
		"text.secondary": "#52606D",
		"border.radius": "8px",
		"density": "compact",
	},
	"components": {
		"employees": {"icon": "id-card", "status_indicator": "employee-pill", "visual": "profile-registry"},
		"departments": {"icon": "building-2", "visual": "org-tree", "status_style": "department-chip"},
		"positions": {"icon": "briefcase", "visual": "headcount-table", "status_style": "position-chip"},
		"personal_info": {"icon": "lock", "visual": "privacy-ledger", "status_style": "privacy-chip"},
		"contacts": {"icon": "phone", "visual": "contact-list", "status_style": "contact-chip"},
		"history": {"icon": "clock-history", "visual": "timeline", "status_style": "event-chip"},
		"skills": {"icon": "star", "visual": "skill-matrix", "status_style": "skill-chip"},
		"certifications": {"icon": "award", "visual": "credential-ledger", "status_style": "credential-chip"},
		"quality": {"icon": "shield-check", "visual": "quality-board", "status_style": "quality-chip"},
		"agents": {"icon": "bot", "visual": "review-lane", "status_style": "agent-chip"},
	},
}


STREAMING = {
	"processor": "bytewax",
	"stream": EMPLOYEE_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"department_created",
		"department_updated",
		"position_created",
		"position_updated",
		"employee_created",
		"employee_status_changed",
		"employee_deleted",
		"personal_info_recorded",
		"emergency_contact_recorded",
		"employment_history_recorded",
		"employee_skill_assigned",
		"employee_skill_removed",
		"employee_certification_assigned",
		"employee_certification_expired",
		"data_quality_issue_recorded",
		"data_quality_issue_resolved",
		"employee_agent_registered",
		"cross_tenant_access_blocked",
		"privilege_escalation_blocked",
	],
	"states": ["draft", "active", "leave", "probation", "notice", "suspended", "terminated", "alumni", "queued", "blocked"],
	"guardrails": [
		"employee_batch_requires_bytewax",
		"employee_event_requires_bytewax",
		"privileged_employee_agent_action_requires_human_approval",
		"cross_tenant_access_denied",
		"privilege_escalation_denied",
	],
}


RULES: list[dict[str, Any]] = [
	# --- Tenant context and write policy (mandatory gates) ---
	{"name": "tenant_context_required", "description": "All employee data operations require tenant context; deny if missing.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "employee_write_requires_policy", "description": "Employee data writes require an attached operation policy.", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "operation_policy_required", "required_action": "attach_operation_policy"}},

	# --- Cross-tenant access prevention ---
	{"name": "cross_tenant_employee_access_denied", "description": "Requests referencing an employee from a different tenant are denied.", "condition": {"operation_type": "read", "employee_tenant_mismatch": True}, "effect": {"decision": "deny", "reason": "cross_tenant_access_denied", "required_action": "use_same_tenant_employee"}},
	{"name": "cross_tenant_department_access_denied", "description": "Requests referencing a department from a different tenant are denied.", "condition": {"operation_type": "write", "department_tenant_mismatch": True}, "effect": {"decision": "deny", "reason": "cross_tenant_access_denied", "required_action": "use_same_tenant_department"}},
	{"name": "cross_tenant_position_access_denied", "description": "Requests referencing a position from a different tenant are denied.", "condition": {"operation_type": "write", "position_tenant_mismatch": True}, "effect": {"decision": "deny", "reason": "cross_tenant_access_denied", "required_action": "use_same_tenant_position"}},

	# --- Privilege escalation prevention ---
	{"name": "self_role_promotion_denied", "description": "Employees cannot promote their own role or permission set.", "condition": {"operation": "update_employee", "self_promotion": True}, "effect": {"decision": "deny", "reason": "self_role_promotion_denied", "required_action": "request_admin_role_change"}},
	{"name": "non_admin_delete_employee_denied", "description": "Deleting an employee record requires admin role.", "condition": {"operation": "delete_employee", "actor_is_admin": False}, "effect": {"decision": "deny", "reason": "admin_role_required_for_delete", "required_action": "elevate_to_admin_role"}},
	{"name": "mass_update_requires_dual_control", "description": "Mass employee updates require a second approver distinct from the initiator.", "condition": {"operation": "mass_update_employees", "dual_control_satisfied": False}, "effect": {"decision": "deny", "reason": "dual_control_required", "required_action": "assign_second_approver"}},

	# --- Department CRUD ---
	{"name": "department_requires_code", "description": "Departments require a code.", "condition": {"operation": "create_department", "code_present": False}, "effect": {"decision": "deny", "reason": "department_code_required", "required_action": "set_department_code"}},
	{"name": "department_requires_name", "description": "Departments require a name.", "condition": {"operation": "create_department", "name_present": False}, "effect": {"decision": "deny", "reason": "department_name_required", "required_action": "set_department_name"}},
	{"name": "department_requires_owner", "description": "Departments require an owner.", "condition": {"operation": "create_department", "owner_present": False}, "effect": {"decision": "deny", "reason": "department_owner_required", "required_action": "assign_department_owner"}},
	{"name": "department_requires_cost_center", "description": "Departments require a cost center.", "condition": {"operation": "create_department", "cost_center_present": False}, "effect": {"decision": "deny", "reason": "cost_center_required", "required_action": "set_cost_center"}},
	{"name": "department_delete_requires_no_active_employees", "description": "Departments with active employees cannot be deleted.", "condition": {"operation": "delete_department", "active_employees_gt": 0}, "effect": {"decision": "deny", "reason": "department_has_active_employees", "required_action": "reassign_employees_before_delete"}},

	# --- Position CRUD ---
	{"name": "position_requires_code", "description": "Positions require a code.", "condition": {"operation": "create_position", "code_present": False}, "effect": {"decision": "deny", "reason": "position_code_required", "required_action": "set_position_code"}},
	{"name": "position_requires_title", "description": "Positions require a title.", "condition": {"operation": "create_position", "title_present": False}, "effect": {"decision": "deny", "reason": "position_title_required", "required_action": "set_position_title"}},
	{"name": "position_requires_department", "description": "Positions require a same-tenant department.", "condition": {"operation": "create_position", "department_present": False}, "effect": {"decision": "deny", "reason": "department_required", "required_action": "select_department"}},
	{"name": "position_requires_job_level", "description": "Positions require a job level.", "condition": {"operation": "create_position", "job_level_present": False}, "effect": {"decision": "deny", "reason": "job_level_required", "required_action": "set_job_level"}},
	{"name": "position_headcount_nonnegative", "description": "Authorized headcount cannot be negative.", "condition": {"operation": "create_position", "authorized_headcount_lt": 0}, "effect": {"decision": "deny", "reason": "authorized_headcount_invalid", "required_action": "set_valid_headcount"}},
	{"name": "compensation_band_requires_review", "description": "Positions with compensation bands require review before creation.", "condition": {"operation": "create_position", "compensation_band_present": True, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "compensation_band_review_required", "required_action": "record_compensation_review"}},

	# --- Employee create ---
	{"name": "employee_requires_number", "description": "Employees require an employee number.", "condition": {"operation": "create_employee", "employee_number_present": False}, "effect": {"decision": "deny", "reason": "employee_number_required", "required_action": "set_employee_number"}},
	{"name": "employee_requires_first_name", "description": "Employees require a first name.", "condition": {"operation": "create_employee", "first_name_present": False}, "effect": {"decision": "deny", "reason": "first_name_required", "required_action": "set_first_name"}},
	{"name": "employee_requires_last_name", "description": "Employees require a last name.", "condition": {"operation": "create_employee", "last_name_present": False}, "effect": {"decision": "deny", "reason": "last_name_required", "required_action": "set_last_name"}},
	{"name": "employee_requires_email", "description": "Employees require a work email.", "condition": {"operation": "create_employee", "work_email_present": False}, "effect": {"decision": "deny", "reason": "work_email_required", "required_action": "set_work_email"}},
	{"name": "employee_email_format", "description": "Employee work email must be valid format.", "condition": {"operation": "create_employee", "work_email_valid": False}, "effect": {"decision": "deny", "reason": "work_email_invalid", "required_action": "set_valid_work_email"}},
	{"name": "employee_requires_department", "description": "Employees require a same-tenant department.", "condition": {"operation": "create_employee", "department_present": False}, "effect": {"decision": "deny", "reason": "department_required", "required_action": "select_department"}},
	{"name": "employee_requires_position", "description": "Employees require a same-tenant position.", "condition": {"operation": "create_employee", "position_present": False}, "effect": {"decision": "deny", "reason": "position_required", "required_action": "select_position"}},
	{"name": "employee_manager_required_for_non_executives", "description": "Non-executive employees require a manager.", "condition": {"operation": "create_employee", "executive": False, "manager_present": False}, "effect": {"decision": "deny", "reason": "manager_required", "required_action": "assign_manager"}},
	{"name": "employee_requires_hire_date", "description": "Employees require a hire date.", "condition": {"operation": "create_employee", "hire_date_present": False}, "effect": {"decision": "deny", "reason": "hire_date_required", "required_action": "set_hire_date"}},
	{"name": "employee_type_supported", "description": "Employment type must be from the supported set.", "condition": {"operation": "create_employee", "employment_type_supported": False}, "effect": {"decision": "deny", "reason": "employment_type_not_supported", "required_action": "select_supported_employment_type"}},
	{"name": "work_mode_supported", "description": "Work mode must be from the supported set.", "condition": {"operation": "create_employee", "work_mode_supported": False}, "effect": {"decision": "deny", "reason": "work_mode_not_supported", "required_action": "select_supported_work_mode"}},
	{"name": "duplicate_employee_number_denied", "description": "Employee number must be unique within the tenant.", "condition": {"operation": "create_employee", "duplicate_employee_number": True}, "effect": {"decision": "deny", "reason": "duplicate_employee_number", "required_action": "assign_unique_employee_number"}},
	{"name": "duplicate_work_email_denied", "description": "Work email must be unique within the tenant.", "condition": {"operation": "create_employee", "duplicate_work_email": True}, "effect": {"decision": "deny", "reason": "duplicate_work_email", "required_action": "assign_unique_work_email"}},

	# --- Employee update / status change ---
	{"name": "sensitive_employee_change_requires_review", "description": "Sensitive employee field changes require review.", "condition": {"operation": "change_employee_status", "sensitive_change": True, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "sensitive_change_review_required", "required_action": "record_sensitive_change_review"}},
	{"name": "employee_status_supported", "description": "Employee status transition must be to a supported value.", "condition": {"operation": "change_employee_status", "status_supported": False}, "effect": {"decision": "deny", "reason": "employee_status_not_supported", "required_action": "select_supported_status"}},
	{"name": "reinstate_requires_approval", "description": "Reinstating a terminated employee requires approval.", "condition": {"operation": "change_employee_status", "from_status": "terminated", "approval_recorded": False}, "effect": {"decision": "require_review", "reason": "reinstatement_approval_required", "required_action": "record_reinstatement_approval"}},

	# --- Personal info ---
	{"name": "personal_info_requires_employee", "description": "Personal info requires an employee reference.", "condition": {"operation": "record_personal_info", "employee_present": False}, "effect": {"decision": "deny", "reason": "employee_required", "required_action": "select_employee"}},
	{"name": "personal_info_requires_effective_date", "description": "Personal info requires an effective date.", "condition": {"operation": "record_personal_info", "effective_date_present": False}, "effect": {"decision": "deny", "reason": "effective_date_required", "required_action": "set_effective_date"}},
	{"name": "personal_info_requires_country", "description": "Personal info requires a country.", "condition": {"operation": "record_personal_info", "country_present": False}, "effect": {"decision": "deny", "reason": "country_required", "required_action": "set_country"}},
	{"name": "personal_info_requires_privacy_basis", "description": "Sensitive personal info requires a privacy basis (GDPR/PDPA).", "condition": {"operation": "record_personal_info", "privacy_basis_present": False}, "effect": {"decision": "deny", "reason": "privacy_basis_required", "required_action": "set_privacy_basis"}},
	{"name": "personal_info_delete_requires_erasure_workflow", "description": "Deleting personal info requires completion of the right-to-erasure workflow.", "condition": {"operation": "delete_personal_info", "erasure_workflow_complete": False}, "effect": {"decision": "deny", "reason": "erasure_workflow_required", "required_action": "complete_erasure_workflow"}},

	# --- Emergency contacts ---
	{"name": "emergency_contact_requires_employee", "description": "Emergency contacts require an employee reference.", "condition": {"operation": "record_emergency_contact", "employee_present": False}, "effect": {"decision": "deny", "reason": "employee_required", "required_action": "select_employee"}},
	{"name": "emergency_contact_requires_name", "description": "Emergency contacts require a name.", "condition": {"operation": "record_emergency_contact", "name_present": False}, "effect": {"decision": "deny", "reason": "contact_name_required", "required_action": "set_contact_name"}},
	{"name": "emergency_contact_requires_relationship", "description": "Emergency contacts require a relationship.", "condition": {"operation": "record_emergency_contact", "relationship_present": False}, "effect": {"decision": "deny", "reason": "relationship_required", "required_action": "set_relationship"}},
	{"name": "emergency_contact_requires_phone", "description": "Emergency contacts require a phone number.", "condition": {"operation": "record_emergency_contact", "phone_present": False}, "effect": {"decision": "deny", "reason": "phone_required", "required_action": "set_phone"}},

	# --- Employment history ---
	{"name": "history_requires_employee", "description": "Employment history requires an employee reference.", "condition": {"operation": "record_employment_history", "employee_present": False}, "effect": {"decision": "deny", "reason": "employee_required", "required_action": "select_employee"}},
	{"name": "history_requires_event_type", "description": "Employment history requires an event type.", "condition": {"operation": "record_employment_history", "event_type_present": False}, "effect": {"decision": "deny", "reason": "event_type_required", "required_action": "set_event_type"}},
	{"name": "history_event_type_supported", "description": "Employment history event type must be from the supported set.", "condition": {"operation": "record_employment_history", "event_type_supported": False}, "effect": {"decision": "deny", "reason": "event_type_not_supported", "required_action": "select_supported_event_type"}},
	{"name": "history_requires_effective_date", "description": "Employment history requires an effective date.", "condition": {"operation": "record_employment_history", "effective_date_present": False}, "effect": {"decision": "deny", "reason": "effective_date_required", "required_action": "set_effective_date"}},
	{"name": "sensitive_history_requires_reason", "description": "Sensitive employment events require a documented reason.", "condition": {"operation": "record_employment_history", "sensitive_event": True, "reason_present": False}, "effect": {"decision": "deny", "reason": "history_reason_required", "required_action": "set_event_reason"}},
	{"name": "termination_requires_approval", "description": "Termination events require manager and HR approval.", "condition": {"operation": "record_employment_history", "termination_event": True, "approval_recorded": False}, "effect": {"decision": "require_review", "reason": "termination_approval_required", "required_action": "record_termination_approval"}},
	{"name": "termination_requires_reason", "description": "Termination events require a supported termination reason.", "condition": {"operation": "record_employment_history", "termination_event": True, "termination_reason_present": False}, "effect": {"decision": "deny", "reason": "termination_reason_required", "required_action": "set_termination_reason"}},
	{"name": "backdated_history_requires_justification", "description": "Backdated employment events require a justification.", "condition": {"operation": "record_employment_history", "backdated": True, "justification_present": False}, "effect": {"decision": "deny", "reason": "backdated_event_justification_required", "required_action": "provide_backdating_justification"}},

	# --- Skills ---
	{"name": "skill_requires_employee", "description": "Employee skills require an employee reference.", "condition": {"operation": "assign_skill", "employee_present": False}, "effect": {"decision": "deny", "reason": "employee_required", "required_action": "select_employee"}},
	{"name": "skill_requires_name", "description": "Employee skills require a skill name.", "condition": {"operation": "assign_skill", "skill_present": False}, "effect": {"decision": "deny", "reason": "skill_required", "required_action": "set_skill"}},
	{"name": "skill_level_supported", "description": "Skill level must be from the supported set.", "condition": {"operation": "assign_skill", "skill_level_supported": False}, "effect": {"decision": "deny", "reason": "skill_level_not_supported", "required_action": "select_supported_skill_level"}},
	{"name": "expert_skill_requires_evidence", "description": "Expert and master skill levels require verifiable evidence.", "condition": {"operation": "assign_skill", "advanced_skill": True, "evidence_present": False}, "effect": {"decision": "require_review", "reason": "skill_evidence_required", "required_action": "record_skill_evidence"}},

	# --- Certifications ---
	{"name": "certification_requires_employee", "description": "Certifications require an employee reference.", "condition": {"operation": "assign_certification", "employee_present": False}, "effect": {"decision": "deny", "reason": "employee_required", "required_action": "select_employee"}},
	{"name": "certification_requires_name", "description": "Certifications require a name.", "condition": {"operation": "assign_certification", "name_present": False}, "effect": {"decision": "deny", "reason": "certification_name_required", "required_action": "set_certification_name"}},
	{"name": "certification_requires_issuer", "description": "Certifications require an issuer.", "condition": {"operation": "assign_certification", "issuer_present": False}, "effect": {"decision": "deny", "reason": "issuer_required", "required_action": "set_issuer"}},
	{"name": "certification_requires_issued_on", "description": "Certifications require an issue date.", "condition": {"operation": "assign_certification", "issued_on_present": False}, "effect": {"decision": "deny", "reason": "issued_on_required", "required_action": "set_issued_on"}},
	{"name": "expiring_certification_requires_expiry", "description": "Expiring certifications require an expiry date.", "condition": {"operation": "assign_certification", "expiring": True, "expires_on_present": False}, "effect": {"decision": "deny", "reason": "expires_on_required", "required_action": "set_expiry_date"}},
	{"name": "certification_status_supported", "description": "Certification status must be from the supported set.", "condition": {"operation": "assign_certification", "certification_status_supported": False}, "effect": {"decision": "deny", "reason": "certification_status_not_supported", "required_action": "select_supported_certification_status"}},
	{"name": "expired_certification_auto_flags_employee", "description": "Certifications past expiry must trigger a data quality flag on the employee.", "condition": {"operation": "assign_certification", "expiry_in_past": True, "quality_flag_created": False}, "effect": {"decision": "require_review", "reason": "expired_certification_requires_quality_flag", "required_action": "create_certification_expiry_quality_flag"}},

	# --- Data quality ---
	{"name": "quality_requires_domain", "description": "Data quality issues require a domain.", "condition": {"operation": "record_data_quality_issue", "domain_present": False}, "effect": {"decision": "deny", "reason": "quality_domain_required", "required_action": "set_quality_domain"}},
	{"name": "quality_domain_supported", "description": "Data quality domain must be from the supported set.", "condition": {"operation": "record_data_quality_issue", "domain_supported": False}, "effect": {"decision": "deny", "reason": "quality_domain_not_supported", "required_action": "select_supported_quality_domain"}},
	{"name": "quality_requires_severity", "description": "Data quality issues require a severity.", "condition": {"operation": "record_data_quality_issue", "severity_present": False}, "effect": {"decision": "deny", "reason": "quality_severity_required", "required_action": "set_quality_severity"}},
	{"name": "high_severity_quality_requires_owner", "description": "High-severity data quality issues require an assigned owner.", "condition": {"operation": "record_data_quality_issue", "high_severity": True, "owner_present": False}, "effect": {"decision": "deny", "reason": "quality_owner_required", "required_action": "assign_quality_owner"}},
	{"name": "quality_resolution_requires_evidence", "description": "Resolving a quality issue requires documented evidence of fix.", "condition": {"operation": "resolve_data_quality_issue", "resolution_evidence_present": False}, "effect": {"decision": "deny", "reason": "resolution_evidence_required", "required_action": "document_resolution_evidence"}},

	# --- Domain-specific governance rules ---
	{"name": "probation_extension_requires_hr_approval", "description": "Extending an employee's probation period requires HR approval.", "condition": {"operation": "extend_probation", "hr_approval_recorded": False}, "effect": {"decision": "deny", "reason": "hr_approval_required_for_probation_extension", "required_action": "obtain_hr_approval"}},
	{"name": "org_chart_depth_limit_enforced", "description": "Org chart depth must not exceed the tenant-configured maximum.", "condition": {"operation": "create_employee", "org_depth_exceeded": True}, "effect": {"decision": "deny", "reason": "org_chart_depth_limit_exceeded", "required_action": "flatten_org_structure"}},
	{"name": "contractor_requires_end_date", "description": "Contractor employment types require a contract end date.", "condition": {"operation": "create_employee", "employment_type": "contractor", "end_date_present": False}, "effect": {"decision": "deny", "reason": "contractor_end_date_required", "required_action": "set_contractor_end_date"}},
	{"name": "salary_change_requires_approval_threshold", "description": "Salary changes exceeding threshold require executive approval.", "condition": {"operation": "update_employee", "salary_change_pct_gt": 20, "executive_approval_recorded": False}, "effect": {"decision": "require_review", "reason": "large_salary_change_requires_executive_approval", "required_action": "obtain_executive_salary_approval"}},
	{"name": "position_overfilled_denied", "description": "Cannot assign an employee to a position that has reached its authorized headcount.", "condition": {"operation": "create_employee", "position_overfilled": True}, "effect": {"decision": "deny", "reason": "position_headcount_exhausted", "required_action": "increase_authorized_headcount_or_choose_different_position"}},
	{"name": "department_headcount_freeze_enforced", "description": "Hiring into a headcount-frozen department is denied.", "condition": {"operation": "create_employee", "department_headcount_frozen": True}, "effect": {"decision": "deny", "reason": "department_headcount_frozen", "required_action": "obtain_headcount_freeze_exception"}},
	{"name": "sensitive_data_export_requires_data_steward", "description": "Exporting sensitive employee data domains requires data steward sign-off.", "condition": {"operation": "export_employee_data", "sensitive_domain": True, "data_steward_approved": False}, "effect": {"decision": "deny", "reason": "data_steward_approval_required_for_sensitive_export", "required_action": "obtain_data_steward_approval"}},
	{"name": "bulk_termination_requires_ceo_approval", "description": "Bulk terminations of 10 or more employees require CEO-level approval.", "condition": {"operation": "bulk_terminate_employees", "count_gte": 10, "ceo_approval_recorded": False}, "effect": {"decision": "deny", "reason": "ceo_approval_required_for_bulk_termination", "required_action": "obtain_ceo_approval"}},
	{"name": "employee_data_access_logged", "description": "All access to sensitive employee data must be logged to the audit trail.", "condition": {"operation_type": "read", "sensitive_domain": True, "audit_logged": False}, "effect": {"decision": "deny", "reason": "sensitive_data_access_must_be_audited", "required_action": "enable_audit_logging"}},
	{"name": "manager_cannot_approve_own_report_termination", "description": "A manager cannot be both initiator and approver of a direct report's termination (SoD).", "condition": {"operation": "record_employment_history", "termination_event": True, "initiator_equals_approver": True}, "effect": {"decision": "deny", "reason": "segregation_of_duties_violation", "required_action": "assign_independent_approver"}},

	# --- Streaming and agents ---
	{"name": "employee_batch_requires_bytewax", "description": "Employee batch operations must be routed through the Bytewax event stream.", "condition": {"operation": "employee_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_employee_batch_to_bytewax"}},
	{"name": "employee_event_requires_bytewax", "description": "Employee lifecycle events must be published to the Bytewax stream.", "condition": {"operation": "employee_event", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_employee_event_to_bytewax"}},
	{"name": "employee_agent_runtime_supported", "description": "Employee agents must use an approved runtime.", "condition": {"operation": "register_employee_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "employee_agent_runtime_not_supported", "required_action": "select_supported_agent_runtime"}},
	{"name": "employee_agent_role_supported", "description": "Employee agents must use an approved role.", "condition": {"operation": "register_employee_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "employee_agent_role_not_supported", "required_action": "select_supported_agent_role"}},
	{"name": "privileged_employee_agent_action_requires_human_approval", "description": "Privileged employee data actions proposed by agents require human approval before execution.", "condition": {"operation": "employee_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
]


def _configuration_schema() -> dict[str, Any]:
	return {
		"type": "object",
		"required": ["tenant_id", "ui", "theme"],
		"properties": {
			"tenant_id": {"type": "string", "minLength": 1},
		} | {
			key: {"type": "object"} for key in DEFAULT_CONFIGURATION if key != "tenant_id"
		},
	}


def _matches_condition(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if key.endswith("_lte"):
			if context.get(key[:-4]) is None or context[key[:-4]] > expected:
				return False
			continue
		if key.endswith("_lt"):
			if context.get(key[:-3]) is None or context[key[:-3]] >= expected:
				return False
			continue
		if key.endswith("_gte"):
			if context.get(key[:-4]) is None or context[key[:-4]] < expected:
				return False
			continue
		if key.endswith("_gt"):
			if context.get(key[:-3]) is None or context[key[:-3]] <= expected:
				return False
			continue
		if key.endswith("_ne"):
			if context.get(key[:-3]) == expected:
				return False
			continue
		if context.get(key) != expected:
			return False
	return True


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	from copy import deepcopy
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	if overrides:
		for key, value in overrides.items():
			if isinstance(value, dict) and isinstance(configuration.get(key), dict):
				configuration[key].update(value)
			else:
				configuration[key] = value

	return {
		"capability": CAPABILITY_ID,
		"display_name": CAPABILITY_NAME,
		"name": CAPABILITY_NAME,
		"version": CAPABILITY_VERSION,
		"configuration": configuration,
		"configuration_schema": _configuration_schema(),
		"provides": PROVIDES,
		"requires": REQUIRES,
		"rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"api_prefix": "/hcm/employees/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"requires_theme": True,
			"view_module": "views.py",
			"template_roots": ["templates/", "static/"],
		},
		"theme": deepcopy(THEME),
		"streaming": deepcopy(STREAMING),
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	contract = get_capability_contract(context.get("tenant_id", "default"))
	matched = [
		rule for rule in contract["rule_engine"]["rules"]
		if _matches_condition(rule["condition"], context)
	]
	decision = "allow"
	for rule in matched:
		rule_decision = rule["effect"]["decision"]
		if rule_decision == "deny":
			decision = "deny"
			break
		if rule_decision == "require_review" and decision == "allow":
			decision = "require_review"
	return {
		"decision": decision,
		"matched_rules": [rule["name"] for rule in matched],
		"effects": [rule["effect"] for rule in matched],
	}
