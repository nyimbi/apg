"""Executable capability contract for HCM Employee Data Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "chr_employee_data_management"
CAPABILITY_NAME = "Employee Data Management"
CAPABILITY_VERSION = "2.1.0"
EMPLOYEE_EVENT_STREAM = "apg.hcm.chr.employee.lifecycle"

SUPPORTED_EMPLOYMENT_TYPES = ["full_time", "part_time", "contractor", "intern", "temporary"]
SUPPORTED_EMPLOYMENT_STATUSES = ["draft", "active", "leave", "suspended", "terminated", "alumni"]
SUPPORTED_WORK_MODES = ["onsite", "hybrid", "remote", "field"]
SUPPORTED_SKILL_LEVELS = ["awareness", "working", "practitioner", "expert", "master"]
SUPPORTED_CERTIFICATION_STATUSES = ["pending", "active", "expired", "revoked"]
SUPPORTED_DATA_DOMAINS = ["identity", "employment", "organization", "skills", "certifications", "contacts", "privacy"]
SUPPORTED_EMPLOYEE_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_EMPLOYEE_AGENT_ROLES = [
	"profile_steward",
	"data_quality_reviewer",
	"org_design_reviewer",
	"skills_reviewer",
	"compliance_reviewer",
	"onboarding_reviewer",
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
	"ntfy",
	"composition_events",
	"composition_config",
	"workflow",
	"document_management",
	"identity_access",
	"privacy_policy",
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
		"departments": {"visual": "org-tree", "status_style": "department-chip"},
		"positions": {"visual": "headcount-table", "status_style": "position-chip"},
		"personal_info": {"visual": "privacy-ledger", "status_style": "privacy-chip"},
		"contacts": {"visual": "contact-list", "status_style": "contact-chip"},
		"history": {"visual": "timeline", "status_style": "event-chip"},
		"skills": {"visual": "skill-matrix", "status_style": "skill-chip"},
		"certifications": {"visual": "credential-ledger", "status_style": "credential-chip"},
		"quality": {"visual": "quality-board", "status_style": "quality-chip"},
		"agents": {"visual": "review-lane", "status_style": "agent-chip"},
	},
}


STREAMING = {
	"processor": "bytewax",
	"stream": EMPLOYEE_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"department_created",
		"position_created",
		"employee_created",
		"employee_status_changed",
		"personal_info_recorded",
		"emergency_contact_recorded",
		"employment_history_recorded",
		"employee_skill_assigned",
		"employee_certification_assigned",
		"data_quality_issue_recorded",
		"employee_agent_registered",
	],
	"states": ["draft", "active", "leave", "suspended", "terminated", "alumni", "queued", "blocked"],
	"guardrails": [
		"employee_batch_requires_bytewax",
		"employee_event_requires_bytewax",
		"privileged_employee_agent_action_requires_human_approval",
	],
}


RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "Employee data operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "employee_write_requires_policy", "description": "Employee data writes require policy attachment.", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "operation_policy_required", "required_action": "attach_operation_policy"}},
	{"name": "department_requires_code", "description": "Departments require code.", "condition": {"operation": "create_department", "code_present": False}, "effect": {"decision": "deny", "reason": "department_code_required", "required_action": "set_department_code"}},
	{"name": "department_requires_name", "description": "Departments require name.", "condition": {"operation": "create_department", "name_present": False}, "effect": {"decision": "deny", "reason": "department_name_required", "required_action": "set_department_name"}},
	{"name": "department_requires_owner", "description": "Departments require owner.", "condition": {"operation": "create_department", "owner_present": False}, "effect": {"decision": "deny", "reason": "department_owner_required", "required_action": "assign_department_owner"}},
	{"name": "department_requires_cost_center", "description": "Departments require cost center.", "condition": {"operation": "create_department", "cost_center_present": False}, "effect": {"decision": "deny", "reason": "cost_center_required", "required_action": "set_cost_center"}},
	{"name": "position_requires_code", "description": "Positions require code.", "condition": {"operation": "create_position", "code_present": False}, "effect": {"decision": "deny", "reason": "position_code_required", "required_action": "set_position_code"}},
	{"name": "position_requires_title", "description": "Positions require title.", "condition": {"operation": "create_position", "title_present": False}, "effect": {"decision": "deny", "reason": "position_title_required", "required_action": "set_position_title"}},
	{"name": "position_requires_department", "description": "Positions require same-tenant department.", "condition": {"operation": "create_position", "department_present": False}, "effect": {"decision": "deny", "reason": "department_required", "required_action": "select_department"}},
	{"name": "position_requires_job_level", "description": "Positions require job level.", "condition": {"operation": "create_position", "job_level_present": False}, "effect": {"decision": "deny", "reason": "job_level_required", "required_action": "set_job_level"}},
	{"name": "position_headcount_nonnegative", "description": "Authorized headcount cannot be negative.", "condition": {"operation": "create_position", "authorized_headcount_lt": 0}, "effect": {"decision": "deny", "reason": "authorized_headcount_invalid", "required_action": "set_valid_headcount"}},
	{"name": "compensation_band_requires_review", "description": "Positions with compensation bands require review.", "condition": {"operation": "create_position", "compensation_band_present": True, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "compensation_band_review_required", "required_action": "record_compensation_review"}},
	{"name": "employee_requires_number", "description": "Employees require employee number.", "condition": {"operation": "create_employee", "employee_number_present": False}, "effect": {"decision": "deny", "reason": "employee_number_required", "required_action": "set_employee_number"}},
	{"name": "employee_requires_first_name", "description": "Employees require first name.", "condition": {"operation": "create_employee", "first_name_present": False}, "effect": {"decision": "deny", "reason": "first_name_required", "required_action": "set_first_name"}},
	{"name": "employee_requires_last_name", "description": "Employees require last name.", "condition": {"operation": "create_employee", "last_name_present": False}, "effect": {"decision": "deny", "reason": "last_name_required", "required_action": "set_last_name"}},
	{"name": "employee_requires_email", "description": "Employees require work email.", "condition": {"operation": "create_employee", "work_email_present": False}, "effect": {"decision": "deny", "reason": "work_email_required", "required_action": "set_work_email"}},
	{"name": "employee_email_format", "description": "Employee work email must be valid.", "condition": {"operation": "create_employee", "work_email_valid": False}, "effect": {"decision": "deny", "reason": "work_email_invalid", "required_action": "set_valid_work_email"}},
	{"name": "employee_requires_department", "description": "Employees require same-tenant department.", "condition": {"operation": "create_employee", "department_present": False}, "effect": {"decision": "deny", "reason": "department_required", "required_action": "select_department"}},
	{"name": "employee_requires_position", "description": "Employees require same-tenant position.", "condition": {"operation": "create_employee", "position_present": False}, "effect": {"decision": "deny", "reason": "position_required", "required_action": "select_position"}},
	{"name": "employee_manager_required_for_non_executives", "description": "Non-executive employees require manager.", "condition": {"operation": "create_employee", "executive": False, "manager_present": False}, "effect": {"decision": "deny", "reason": "manager_required", "required_action": "assign_manager"}},
	{"name": "employee_requires_hire_date", "description": "Employees require hire date.", "condition": {"operation": "create_employee", "hire_date_present": False}, "effect": {"decision": "deny", "reason": "hire_date_required", "required_action": "set_hire_date"}},
	{"name": "employee_type_supported", "description": "Employment type must be supported.", "condition": {"operation": "create_employee", "employment_type_supported": False}, "effect": {"decision": "deny", "reason": "employment_type_not_supported", "required_action": "select_supported_employment_type"}},
	{"name": "work_mode_supported", "description": "Work mode must be supported.", "condition": {"operation": "create_employee", "work_mode_supported": False}, "effect": {"decision": "deny", "reason": "work_mode_not_supported", "required_action": "select_supported_work_mode"}},
	{"name": "sensitive_employee_change_requires_review", "description": "Sensitive employee changes require review.", "condition": {"operation": "change_employee_status", "sensitive_change": True, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "sensitive_change_review_required", "required_action": "record_sensitive_change_review"}},
	{"name": "employee_status_supported", "description": "Employee status must be supported.", "condition": {"operation": "change_employee_status", "status_supported": False}, "effect": {"decision": "deny", "reason": "employee_status_not_supported", "required_action": "select_supported_status"}},
	{"name": "personal_info_requires_employee", "description": "Personal info requires employee.", "condition": {"operation": "record_personal_info", "employee_present": False}, "effect": {"decision": "deny", "reason": "employee_required", "required_action": "select_employee"}},
	{"name": "personal_info_requires_effective_date", "description": "Personal info requires effective date.", "condition": {"operation": "record_personal_info", "effective_date_present": False}, "effect": {"decision": "deny", "reason": "effective_date_required", "required_action": "set_effective_date"}},
	{"name": "personal_info_requires_country", "description": "Personal info requires country.", "condition": {"operation": "record_personal_info", "country_present": False}, "effect": {"decision": "deny", "reason": "country_required", "required_action": "set_country"}},
	{"name": "personal_info_requires_privacy_basis", "description": "Sensitive personal info requires privacy basis.", "condition": {"operation": "record_personal_info", "privacy_basis_present": False}, "effect": {"decision": "deny", "reason": "privacy_basis_required", "required_action": "set_privacy_basis"}},
	{"name": "emergency_contact_requires_employee", "description": "Emergency contacts require employee.", "condition": {"operation": "record_emergency_contact", "employee_present": False}, "effect": {"decision": "deny", "reason": "employee_required", "required_action": "select_employee"}},
	{"name": "emergency_contact_requires_name", "description": "Emergency contacts require name.", "condition": {"operation": "record_emergency_contact", "name_present": False}, "effect": {"decision": "deny", "reason": "contact_name_required", "required_action": "set_contact_name"}},
	{"name": "emergency_contact_requires_relationship", "description": "Emergency contacts require relationship.", "condition": {"operation": "record_emergency_contact", "relationship_present": False}, "effect": {"decision": "deny", "reason": "relationship_required", "required_action": "set_relationship"}},
	{"name": "emergency_contact_requires_phone", "description": "Emergency contacts require phone.", "condition": {"operation": "record_emergency_contact", "phone_present": False}, "effect": {"decision": "deny", "reason": "phone_required", "required_action": "set_phone"}},
	{"name": "history_requires_employee", "description": "Employment history requires employee.", "condition": {"operation": "record_employment_history", "employee_present": False}, "effect": {"decision": "deny", "reason": "employee_required", "required_action": "select_employee"}},
	{"name": "history_requires_event_type", "description": "Employment history requires event type.", "condition": {"operation": "record_employment_history", "event_type_present": False}, "effect": {"decision": "deny", "reason": "event_type_required", "required_action": "set_event_type"}},
	{"name": "history_requires_effective_date", "description": "Employment history requires effective date.", "condition": {"operation": "record_employment_history", "effective_date_present": False}, "effect": {"decision": "deny", "reason": "effective_date_required", "required_action": "set_effective_date"}},
	{"name": "sensitive_history_requires_reason", "description": "Sensitive employment events require reason.", "condition": {"operation": "record_employment_history", "sensitive_event": True, "reason_present": False}, "effect": {"decision": "deny", "reason": "history_reason_required", "required_action": "set_event_reason"}},
	{"name": "termination_requires_approval", "description": "Termination history requires approval.", "condition": {"operation": "record_employment_history", "termination_event": True, "approval_recorded": False}, "effect": {"decision": "require_review", "reason": "termination_approval_required", "required_action": "record_termination_approval"}},
	{"name": "skill_requires_employee", "description": "Employee skills require employee.", "condition": {"operation": "assign_skill", "employee_present": False}, "effect": {"decision": "deny", "reason": "employee_required", "required_action": "select_employee"}},
	{"name": "skill_requires_name", "description": "Employee skills require skill name.", "condition": {"operation": "assign_skill", "skill_present": False}, "effect": {"decision": "deny", "reason": "skill_required", "required_action": "set_skill"}},
	{"name": "skill_level_supported", "description": "Skill level must be supported.", "condition": {"operation": "assign_skill", "skill_level_supported": False}, "effect": {"decision": "deny", "reason": "skill_level_not_supported", "required_action": "select_supported_skill_level"}},
	{"name": "expert_skill_requires_evidence", "description": "Expert and master skills require evidence.", "condition": {"operation": "assign_skill", "advanced_skill": True, "evidence_present": False}, "effect": {"decision": "require_review", "reason": "skill_evidence_required", "required_action": "record_skill_evidence"}},
	{"name": "certification_requires_employee", "description": "Certifications require employee.", "condition": {"operation": "assign_certification", "employee_present": False}, "effect": {"decision": "deny", "reason": "employee_required", "required_action": "select_employee"}},
	{"name": "certification_requires_name", "description": "Certifications require name.", "condition": {"operation": "assign_certification", "name_present": False}, "effect": {"decision": "deny", "reason": "certification_name_required", "required_action": "set_certification_name"}},
	{"name": "certification_requires_issuer", "description": "Certifications require issuer.", "condition": {"operation": "assign_certification", "issuer_present": False}, "effect": {"decision": "deny", "reason": "issuer_required", "required_action": "set_issuer"}},
	{"name": "certification_requires_issued_on", "description": "Certifications require issued date.", "condition": {"operation": "assign_certification", "issued_on_present": False}, "effect": {"decision": "deny", "reason": "issued_on_required", "required_action": "set_issued_on"}},
	{"name": "expiring_certification_requires_expiry", "description": "Expiring certifications require expiry date.", "condition": {"operation": "assign_certification", "expiring": True, "expires_on_present": False}, "effect": {"decision": "deny", "reason": "expires_on_required", "required_action": "set_expiry_date"}},
	{"name": "certification_status_supported", "description": "Certification status must be supported.", "condition": {"operation": "assign_certification", "certification_status_supported": False}, "effect": {"decision": "deny", "reason": "certification_status_not_supported", "required_action": "select_supported_certification_status"}},
	{"name": "quality_requires_domain", "description": "Data-quality issues require domain.", "condition": {"operation": "record_data_quality_issue", "domain_present": False}, "effect": {"decision": "deny", "reason": "quality_domain_required", "required_action": "set_quality_domain"}},
	{"name": "quality_domain_supported", "description": "Data-quality domain must be supported.", "condition": {"operation": "record_data_quality_issue", "domain_supported": False}, "effect": {"decision": "deny", "reason": "quality_domain_not_supported", "required_action": "select_supported_quality_domain"}},
	{"name": "quality_requires_severity", "description": "Data-quality issues require severity.", "condition": {"operation": "record_data_quality_issue", "severity_present": False}, "effect": {"decision": "deny", "reason": "quality_severity_required", "required_action": "set_quality_severity"}},
	{"name": "high_severity_quality_requires_owner", "description": "High-severity data-quality issues require owner.", "condition": {"operation": "record_data_quality_issue", "high_severity": True, "owner_present": False}, "effect": {"decision": "deny", "reason": "quality_owner_required", "required_action": "assign_quality_owner"}},
	{"name": "employee_batch_requires_bytewax", "description": "Employee batches require Bytewax coordination.", "condition": {"operation": "employee_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_employee_batch_to_bytewax"}},
	{"name": "employee_event_requires_bytewax", "description": "Employee events require Bytewax.", "condition": {"operation": "employee_event", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_employee_event_to_bytewax"}},
	{"name": "employee_agent_runtime_supported", "description": "Employee agents must use an approved runtime.", "condition": {"operation": "register_employee_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "employee_agent_runtime_not_supported", "required_action": "select_supported_agent_runtime"}},
	{"name": "employee_agent_role_supported", "description": "Employee agents must use an approved role.", "condition": {"operation": "register_employee_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "employee_agent_role_not_supported", "required_action": "select_supported_agent_role"}},
	{"name": "privileged_employee_agent_action_requires_human_approval", "description": "Privileged employee data actions proposed by agents require human approval.", "condition": {"operation": "employee_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
]


def _configuration_schema() -> dict[str, Any]:
	return {
		"type": "object",
		"required": list(DEFAULT_CONFIGURATION),
		"properties": {
			key: {"type": "object"} for key in DEFAULT_CONFIGURATION if key != "tenant_id"
		} | {"tenant_id": {"type": "string", "minLength": 1}},
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
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
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
