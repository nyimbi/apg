"""Executable capability contract for APG Clinical Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "healthcare_cli"
CAPABILITY_NAME = "Clinical Management"
CAPABILITY_VERSION = "1.0.0"
CLI_EVENT_STREAM = "apg.healthcare.cli.lifecycle"

SUPPORTED_CARE_PLAN_STATUSES = ["draft", "active", "on_hold", "completed", "revoked", "entered_in_error"]
SUPPORTED_PROTOCOL_TYPES = [
	"sepsis_bundle", "stroke_protocol", "mi_protocol", "covid_protocol",
	"fall_prevention", "pressure_ulcer_prevention", "vte_prophylaxis",
	"pain_management", "rapid_response", "code_blue",
]
SUPPORTED_WORKFLOW_STATES = ["pending", "in_progress", "on_hold", "completed", "cancelled", "overdue"]
SUPPORTED_ALERT_PRIORITIES = ["low", "medium", "high", "critical", "advisory"]
SUPPORTED_INTERVENTION_TYPES = [
	"medication", "procedure", "education", "monitoring", "referral",
	"therapy", "nutrition", "social_work", "palliative",
]
SUPPORTED_CARE_TEAM_ROLES = [
	"attending_physician", "resident", "nurse", "pharmacist", "social_worker",
	"physical_therapist", "occupational_therapist", "dietitian", "case_manager",
]
SUPPORTED_HANDOFF_TYPES = ["shift_change", "transfer", "discharge", "referral", "escalation"]
SUPPORTED_DECISION_SUPPORT_TYPES = [
	"drug_dosing", "diagnostic_suggestion", "preventive_care", "sepsis_screening",
	"deterioration_alert", "guideline_reminder", "contraindication_alert",
]
SUPPORTED_ADHERENCE_STATUSES = ["adherent", "partial", "non_adherent", "not_assessed"]
SUPPORTED_AGENT_ROLES = ["clinical_steward", "protocol_reviewer", "care_plan_reviewer", "handoff_reviewer"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"care_plans": {"supported_statuses": SUPPORTED_CARE_PLAN_STATUSES, "multidisciplinary_team_required": True},
	"protocols": {"supported_types": SUPPORTED_PROTOCOL_TYPES, "evidence_required": True, "activation_criteria_required": True},
	"workflows": {"supported_states": SUPPORTED_WORKFLOW_STATES, "overdue_alert_enabled": True},
	"decision_support": {"supported_types": SUPPORTED_DECISION_SUPPORT_TYPES, "real_time_enabled": True},
	"handoffs": {"supported_types": SUPPORTED_HANDOFF_TYPES, "structured_format_required": True},
	"governance": {
		"require_tenant_context": True, "policy_attached_for_writes": True,
		"audit_events": True, "hipaa_phi_protection": True,
		"cross_tenant_care_plan_access_denied": True,
		"protocol_activation_requires_criteria": True,
		"care_plan_requires_team_member": True,
	},
	"observability": {"event_stream": CLI_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "workflow": "wflo", "nlp": "nlpc", "monitoring": "moni", "event_stream": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_care_plans": True, "enable_protocols": True, "enable_workflows": True, "enable_decision_support": True, "enable_handoffs": True},
	"theme": {"default_theme": "healthcare_cli_clinical", "allow_tenant_overrides": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
}

PROVIDES = [
	"care_plan_management", "clinical_workflow_orchestration",
	"protocol_adherence_tracking", "clinical_decision_support",
	"care_team_management", "clinical_handoff_management",
	"intervention_tracking", "deterioration_alerting",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "nlpc", "moni", "mqeb"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/healthcare-cli/dashboard", "component": "CliDashboard", "permission": "healthcare_cli:view", "nav_group": "Overview"},
	{"name": "care_plans", "path": "/healthcare-cli/care-plans", "component": "CliCarePlanList", "permission": "healthcare_cli:care_plans", "nav_group": "Care Plans"},
	{"name": "care_plan_new", "path": "/healthcare-cli/care-plans/new", "component": "CliCarePlanEditor", "permission": "healthcare_cli:care_plans_write", "nav_group": "Care Plans"},
	{"name": "care_plan_detail", "path": "/healthcare-cli/care-plans/<id>", "component": "CliCarePlanDetail", "permission": "healthcare_cli:care_plans", "nav_group": "Care Plans"},
	{"name": "protocols", "path": "/healthcare-cli/protocols", "component": "CliProtocolLibrary", "permission": "healthcare_cli:protocols", "nav_group": "Protocols"},
	{"name": "protocol_detail", "path": "/healthcare-cli/protocols/<id>", "component": "CliProtocolDetail", "permission": "healthcare_cli:protocols", "nav_group": "Protocols"},
	{"name": "workflows", "path": "/healthcare-cli/workflows", "component": "CliWorkflowBoard", "permission": "healthcare_cli:workflows", "nav_group": "Workflows"},
	{"name": "decision_support", "path": "/healthcare-cli/cds", "component": "CliDecisionSupport", "permission": "healthcare_cli:cds", "nav_group": "Decision Support"},
	{"name": "handoffs", "path": "/healthcare-cli/handoffs", "component": "CliHandoffConsole", "permission": "healthcare_cli:handoffs", "nav_group": "Handoffs"},
	{"name": "care_team", "path": "/healthcare-cli/care-team/<patient_id>", "component": "CliCareTeam", "permission": "healthcare_cli:care_team", "nav_group": "Team"},
	{"name": "alerts", "path": "/healthcare-cli/alerts", "component": "CliClinicalAlerts", "permission": "healthcare_cli:alerts", "nav_group": "Alerts"},
	{"name": "agents", "path": "/healthcare-cli/agents", "component": "CliAgentWorkbench", "permission": "healthcare_cli:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/healthcare-cli/settings", "component": "CliSettings", "permission": "healthcare_cli:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "healthcare_cli_clinical",
	"tokens": {
		"color.primary": "#1E3A5F", "color.accent": "#0891B2", "color.success": "#166534",
		"color.warning": "#A16207", "color.danger": "#B91C1C",
		"surface.canvas": "#F0F4F8", "surface.panel": "#FFFFFF",
		"text.primary": "#1E3A5F", "text.secondary": "#4B5563",
		"border.radius": "6px", "density": "comfortable",
	},
	"components": {
		"care_plans": {"icon": "clipboard", "status_indicator": "care-plan-status-chip"},
		"protocols": {"icon": "book", "status_indicator": "protocol-type-chip"},
		"workflows": {"icon": "git-branch", "status_indicator": "workflow-state-chip"},
		"decision_support": {"icon": "cpu", "status_indicator": "cds-type-chip"},
		"handoffs": {"icon": "arrow-right-circle", "status_indicator": "handoff-type-chip"},
		"care_team": {"icon": "users", "status_indicator": "role-chip"},
		"alerts": {"icon": "bell", "status_indicator": "priority-chip"},
		"agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"},
	},
}

STREAMING = {
	"processor": "bytewax", "stream": CLI_EVENT_STREAM, "key": "tenant_id",
	"events": [
		"care_plan_created", "care_plan_activated", "care_plan_completed",
		"protocol_activated", "workflow_state_changed", "intervention_completed",
		"handoff_recorded", "cds_alert_triggered", "deterioration_alert_fired",
	],
	"guardrails": [
		"cross_tenant_care_plan_access_denied", "protocol_activation_requires_criteria",
		"care_plan_requires_team_member", "privileged_agent_action_requires_human_approval",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "write_policy_required", "required_action": "attach_write_policy"}},
	{"name": "cross_tenant_care_plan_access_denied", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_care_plan_access_prohibited", "required_action": "use_tenant_scoped_query"}},
	{"name": "care_plan_status_supported", "condition": {"operation": "update_care_plan", "care_plan_status_supported": False}, "effect": {"decision": "deny", "reason": "care_plan_status_not_supported", "required_action": "select_supported_care_plan_status"}},
	{"name": "care_plan_requires_team_member", "condition": {"operation": "activate_care_plan", "team_member_assigned": False}, "effect": {"decision": "deny", "reason": "care_plan_requires_at_least_one_team_member", "required_action": "assign_care_team_member"}},
	{"name": "protocol_type_supported", "condition": {"operation": "activate_protocol", "protocol_type_supported": False}, "effect": {"decision": "deny", "reason": "protocol_type_not_supported", "required_action": "select_supported_protocol_type"}},
	{"name": "protocol_activation_requires_criteria", "condition": {"operation": "activate_protocol", "activation_criteria_met": False}, "effect": {"decision": "deny", "reason": "protocol_activation_criteria_not_met", "required_action": "verify_activation_criteria"}},
	{"name": "workflow_state_supported", "condition": {"operation": "transition_workflow", "workflow_state_supported": False}, "effect": {"decision": "deny", "reason": "workflow_state_not_supported", "required_action": "select_supported_workflow_state"}},
	{"name": "alert_priority_supported", "condition": {"operation": "create_alert", "alert_priority_supported": False}, "effect": {"decision": "deny", "reason": "alert_priority_not_supported", "required_action": "select_supported_alert_priority"}},
	{"name": "intervention_type_supported", "condition": {"operation": "add_intervention", "intervention_type_supported": False}, "effect": {"decision": "deny", "reason": "intervention_type_not_supported", "required_action": "select_supported_intervention_type"}},
	{"name": "care_team_role_supported", "condition": {"operation": "assign_team_member", "care_team_role_supported": False}, "effect": {"decision": "deny", "reason": "care_team_role_not_supported", "required_action": "select_supported_care_team_role"}},
	{"name": "handoff_type_supported", "condition": {"operation": "record_handoff", "handoff_type_supported": False}, "effect": {"decision": "deny", "reason": "handoff_type_not_supported", "required_action": "select_supported_handoff_type"}},
	{"name": "handoff_requires_structured_format", "condition": {"operation": "record_handoff", "structured_format_used": False}, "effect": {"decision": "deny", "reason": "structured_handoff_format_required", "required_action": "use_structured_handoff_format"}},
	{"name": "cds_type_supported", "condition": {"operation": "create_cds_alert", "cds_type_supported": False}, "effect": {"decision": "deny", "reason": "cds_type_not_supported", "required_action": "select_supported_cds_type"}},
	{"name": "revoked_care_plan_not_editable", "condition": {"operation": "update_care_plan", "care_plan_status": "revoked"}, "effect": {"decision": "deny", "reason": "revoked_care_plan_cannot_be_edited", "required_action": "create_new_care_plan"}},
	{"name": "completed_protocol_not_re_activatable", "condition": {"operation": "activate_protocol", "protocol_status": "completed"}, "effect": {"decision": "deny", "reason": "completed_protocol_cannot_be_reactivated", "required_action": "create_new_protocol_instance"}},
	{"name": "agent_privileged_action_requires_approval", "condition": {"agent_action": True, "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "privileged_agent_action_requires_human_approval", "required_action": "record_human_approval"}},
	{"name": "overdue_workflow_alert", "condition": {"operation": "check_workflow", "workflow_state": "overdue"}, "effect": {"decision": "warn", "reason": "workflow_task_overdue", "required_action": "escalate_to_care_team"}},
	{"name": "adherence_status_supported", "condition": {"operation": "record_adherence", "adherence_status_supported": False}, "effect": {"decision": "deny", "reason": "adherence_status_not_supported", "required_action": "select_supported_adherence_status"}},
	{"name": "decision_support_evidence_required", "condition": {"operation": "create_cds_alert", "evidence_reference_present": False}, "effect": {"decision": "deny", "reason": "cds_evidence_reference_required", "required_action": "attach_evidence_reference"}},
	{"name": "deterioration_alert_patient_required", "condition": {"operation": "fire_deterioration_alert", "patient_id_present": False}, "effect": {"decision": "deny", "reason": "patient_id_required_for_deterioration_alert", "required_action": "specify_patient_id"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	return {
		"capability": CAPABILITY_ID, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION,
		"configuration": config,
		"configuration_schema": {"required": ["tenant_id", "ui", "theme"], "properties": {"tenant_id": {"type": "string"}, "ui": {"type": "object"}, "theme": {"type": "object"}}},
		"rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": RULES},
		"ui": {"shell": "apg_python", "requires_theme": True, "template_roots": ["healthcare/cli/templates"], "routes": UI_ROUTES},
		"theme": THEME, "streaming": STREAMING, "provides": PROVIDES, "requires": REQUIRES,
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	for rule in RULES:
		cond = rule["condition"]
		if all(context.get(k) == v for k, v in cond.items()):
			effect = rule["effect"]
			return {"rule": rule["name"], "decision": effect["decision"], "reason": effect["reason"], "required_action": effect.get("required_action")}
	return {"rule": None, "decision": "allow", "reason": "no_rule_matched", "required_action": None}
