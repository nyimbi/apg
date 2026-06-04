"""Executable capability contract for APG Patient Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "healthcare_pmt"
CAPABILITY_NAME = "Patient Management"
CAPABILITY_VERSION = "1.0.0"
PMT_EVENT_STREAM = "apg.healthcare.pmt.lifecycle"

SUPPORTED_ADMISSION_TYPES = [
	"emergency", "elective", "urgent", "newborn", "trauma",
	"observation", "day_surgery", "psychiatric",
]
SUPPORTED_DISCHARGE_DISPOSITIONS = [
	"home", "home_with_services", "snf", "rehab", "ltac", "hospice",
	"ama", "expired", "transfer", "left_without_treatment",
]
SUPPORTED_BED_STATUSES = ["available", "occupied", "cleaning", "maintenance", "blocked"]
SUPPORTED_APPOINTMENT_TYPES = [
	"new_patient", "follow_up", "annual_wellness", "urgent", "procedure",
	"telehealth", "consultation", "preventive",
]
SUPPORTED_APPOINTMENT_STATUSES = [
	"scheduled", "confirmed", "checked_in", "in_progress",
	"completed", "cancelled", "no_show", "rescheduled",
]
SUPPORTED_PATIENT_STATUSES = ["active", "inactive", "deceased", "merged"]
SUPPORTED_INSURANCE_TYPES = [
	"commercial", "medicare", "medicaid", "self_pay", "workers_comp",
	"tricare", "va", "other_government",
]
SUPPORTED_ADT_EVENT_TYPES = ["admit", "discharge", "transfer", "update", "cancel_admit", "cancel_discharge"]
SUPPORTED_VISIT_TYPES = ["inpatient", "outpatient", "emergency", "observation", "ambulatory", "telemedicine"]
SUPPORTED_BILLING_STATUSES = ["not_billed", "pending", "submitted", "partial_paid", "paid", "denied", "appealed"]
SUPPORTED_GENDER_CODES = ["male", "female", "other", "unknown"]
SUPPORTED_AGENT_ROLES = ["registration_steward", "adt_reviewer", "bed_manager", "billing_reviewer"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"registration": {
		"supported_patient_statuses": SUPPORTED_PATIENT_STATUSES,
		"supported_gender_codes": SUPPORTED_GENDER_CODES,
		"mrn_prefix": "MRN",
		"duplicate_check_enabled": True,
	},
	"adt": {
		"supported_admission_types": SUPPORTED_ADMISSION_TYPES,
		"supported_discharge_dispositions": SUPPORTED_DISCHARGE_DISPOSITIONS,
		"supported_event_types": SUPPORTED_ADT_EVENT_TYPES,
		"supported_visit_types": SUPPORTED_VISIT_TYPES,
	},
	"bed_management": {
		"supported_statuses": SUPPORTED_BED_STATUSES,
		"housekeeping_integration": True,
	},
	"appointments": {
		"supported_types": SUPPORTED_APPOINTMENT_TYPES,
		"supported_statuses": SUPPORTED_APPOINTMENT_STATUSES,
		"reminder_hours_before": [48, 24, 2],
	},
	"billing": {
		"supported_insurance_types": SUPPORTED_INSURANCE_TYPES,
		"supported_billing_statuses": SUPPORTED_BILLING_STATUSES,
	},
	"governance": {
		"require_tenant_context": True,
		"policy_attached_for_writes": True,
		"audit_events": True,
		"hipaa_phi_protection": True,
		"duplicate_mrn_denied": True,
		"cross_tenant_patient_access_denied": True,
		"discharge_requires_physician_order": True,
	},
	"observability": {"event_stream": PMT_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {
		"auth": "auth",
		"audit": "audl",
		"notifications": "ntfy",
		"scheduler": "schd",
		"workflow": "wflo",
		"event_stream": "bytewax",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_registration": True,
		"enable_adt": True,
		"enable_bed_management": True,
		"enable_appointments": True,
		"enable_billing": True,
	},
	"theme": {"default_theme": "healthcare_pmt_clinical", "allow_tenant_overrides": True},
	"agents": {
		"enabled": True,
		"supported_runtimes": SUPPORTED_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_AGENT_ROLES,
		"human_approval_required_for_privileged_actions": True,
	},
}

PROVIDES = [
	"patient_registration",
	"adt_workflow",
	"bed_management",
	"appointment_scheduling",
	"patient_billing",
	"mrn_generation",
	"insurance_verification",
	"patient_search",
	"visit_management",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "schd", "mqeb"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/healthcare-pmt/dashboard", "component": "PmtDashboard", "permission": "healthcare_pmt:view", "nav_group": "Overview"},
	{"name": "patients", "path": "/healthcare-pmt/patients", "component": "PmtPatientList", "permission": "healthcare_pmt:patients", "nav_group": "Patients"},
	{"name": "patient_register", "path": "/healthcare-pmt/patients/register", "component": "PmtPatientRegister", "permission": "healthcare_pmt:patients_write", "nav_group": "Patients"},
	{"name": "patient_detail", "path": "/healthcare-pmt/patients/<id>", "component": "PmtPatientDetail", "permission": "healthcare_pmt:patients", "nav_group": "Patients"},
	{"name": "admissions", "path": "/healthcare-pmt/admissions", "component": "PmtAdmissionConsole", "permission": "healthcare_pmt:adt", "nav_group": "ADT"},
	{"name": "discharges", "path": "/healthcare-pmt/discharges", "component": "PmtDischargeConsole", "permission": "healthcare_pmt:adt", "nav_group": "ADT"},
	{"name": "transfers", "path": "/healthcare-pmt/transfers", "component": "PmtTransferConsole", "permission": "healthcare_pmt:adt", "nav_group": "ADT"},
	{"name": "bed_board", "path": "/healthcare-pmt/beds", "component": "PmtBedBoard", "permission": "healthcare_pmt:beds", "nav_group": "Beds"},
	{"name": "appointments", "path": "/healthcare-pmt/appointments", "component": "PmtAppointmentCalendar", "permission": "healthcare_pmt:appointments", "nav_group": "Scheduling"},
	{"name": "appointment_detail", "path": "/healthcare-pmt/appointments/<id>", "component": "PmtAppointmentDetail", "permission": "healthcare_pmt:appointments", "nav_group": "Scheduling"},
	{"name": "billing", "path": "/healthcare-pmt/billing", "component": "PmtBillingConsole", "permission": "healthcare_pmt:billing", "nav_group": "Billing"},
	{"name": "insurance", "path": "/healthcare-pmt/insurance", "component": "PmtInsuranceConsole", "permission": "healthcare_pmt:billing", "nav_group": "Billing"},
	{"name": "agents", "path": "/healthcare-pmt/agents", "component": "PmtAgentWorkbench", "permission": "healthcare_pmt:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/healthcare-pmt/settings", "component": "PmtSettings", "permission": "healthcare_pmt:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "healthcare_pmt_clinical",
	"tokens": {
		"color.primary": "#1D4ED8",
		"color.accent": "#7C3AED",
		"color.success": "#166534",
		"color.warning": "#A16207",
		"color.danger": "#B91C1C",
		"surface.canvas": "#F8FAFC",
		"surface.panel": "#FFFFFF",
		"text.primary": "#111827",
		"text.secondary": "#4B5563",
		"border.radius": "8px",
		"density": "comfortable",
	},
	"components": {
		"patients": {"icon": "user", "status_indicator": "patient-status-chip"},
		"admissions": {"icon": "log-in", "status_indicator": "admission-type-chip"},
		"discharges": {"icon": "log-out", "status_indicator": "disposition-chip"},
		"beds": {"icon": "layout", "status_indicator": "bed-status-chip"},
		"appointments": {"icon": "calendar", "status_indicator": "appointment-status-chip"},
		"billing": {"icon": "credit-card", "status_indicator": "billing-status-chip"},
		"insurance": {"icon": "shield", "status_indicator": "insurance-type-chip"},
		"agents": {"icon": "cpu", "status_indicator": "agent-runtime-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": PMT_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"patient_registered", "patient_updated", "patient_merged",
		"patient_admitted", "patient_discharged", "patient_transferred",
		"bed_status_changed", "appointment_scheduled", "appointment_updated",
		"billing_record_created",
	],
	"guardrails": [
		"phi_access_requires_authorization",
		"duplicate_mrn_creation_denied",
		"cross_tenant_patient_access_denied",
		"discharge_without_physician_order_denied",
		"privileged_agent_action_requires_human_approval",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "write_policy_required", "required_action": "attach_write_policy"}},
	{"name": "duplicate_mrn_denied", "condition": {"operation": "register_patient", "mrn_exists": True}, "effect": {"decision": "deny", "reason": "duplicate_mrn_exists", "required_action": "merge_or_use_existing_patient"}},
	{"name": "admission_type_supported", "condition": {"operation": "admit_patient", "admission_type_supported": False}, "effect": {"decision": "deny", "reason": "admission_type_not_supported", "required_action": "select_supported_admission_type"}},
	{"name": "discharge_requires_physician_order", "condition": {"operation": "discharge_patient", "physician_order_present": False}, "effect": {"decision": "deny", "reason": "physician_discharge_order_required", "required_action": "obtain_physician_discharge_order"}},
	{"name": "discharge_disposition_supported", "condition": {"operation": "discharge_patient", "disposition_supported": False}, "effect": {"decision": "deny", "reason": "discharge_disposition_not_supported", "required_action": "select_supported_disposition"}},
	{"name": "transfer_requires_receiving_unit", "condition": {"operation": "transfer_patient", "receiving_unit_present": False}, "effect": {"decision": "deny", "reason": "receiving_unit_required", "required_action": "specify_receiving_unit"}},
	{"name": "bed_status_supported", "condition": {"operation": "update_bed_status", "bed_status_supported": False}, "effect": {"decision": "deny", "reason": "bed_status_not_supported", "required_action": "select_supported_bed_status"}},
	{"name": "appointment_type_supported", "condition": {"operation": "schedule_appointment", "appointment_type_supported": False}, "effect": {"decision": "deny", "reason": "appointment_type_not_supported", "required_action": "select_supported_appointment_type"}},
	{"name": "appointment_slot_available", "condition": {"operation": "schedule_appointment", "slot_available": False}, "effect": {"decision": "deny", "reason": "appointment_slot_not_available", "required_action": "select_available_slot"}},
	{"name": "cross_tenant_patient_access_denied", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_patient_access_prohibited", "required_action": "use_tenant_scoped_query"}},
	{"name": "insurance_type_supported", "condition": {"operation": "add_insurance", "insurance_type_supported": False}, "effect": {"decision": "deny", "reason": "insurance_type_not_supported", "required_action": "select_supported_insurance_type"}},
	{"name": "visit_type_supported", "condition": {"operation": "create_visit", "visit_type_supported": False}, "effect": {"decision": "deny", "reason": "visit_type_not_supported", "required_action": "select_supported_visit_type"}},
	{"name": "patient_merge_requires_approval", "condition": {"operation": "merge_patients", "approval_present": False}, "effect": {"decision": "deny", "reason": "patient_merge_approval_required", "required_action": "obtain_merge_approval"}},
	{"name": "inactive_patient_adt_denied", "condition": {"operation": "admit_patient", "patient_status": "inactive"}, "effect": {"decision": "deny", "reason": "inactive_patient_cannot_be_admitted", "required_action": "reactivate_patient"}},
	{"name": "deceased_patient_modification_denied", "condition": {"operation": "update_patient", "patient_status": "deceased"}, "effect": {"decision": "deny", "reason": "deceased_patient_record_is_locked", "required_action": "use_amendment_workflow"}},
	{"name": "billing_status_supported", "condition": {"operation": "update_billing", "billing_status_supported": False}, "effect": {"decision": "deny", "reason": "billing_status_not_supported", "required_action": "select_supported_billing_status"}},
	{"name": "gender_code_supported", "condition": {"operation": "register_patient", "gender_code_supported": False}, "effect": {"decision": "deny", "reason": "gender_code_not_supported", "required_action": "select_supported_gender_code"}},
	{"name": "agent_privileged_action_requires_approval", "condition": {"agent_action": True, "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "privileged_agent_action_requires_human_approval", "required_action": "record_human_approval"}},
	{"name": "adt_event_type_supported", "condition": {"operation": "record_adt_event", "adt_event_type_supported": False}, "effect": {"decision": "deny", "reason": "adt_event_type_not_supported", "required_action": "select_supported_adt_event_type"}},
	{"name": "appointment_cancel_requires_reason", "condition": {"operation": "cancel_appointment", "reason_present": False}, "effect": {"decision": "deny", "reason": "cancellation_reason_required", "required_action": "provide_cancellation_reason"}},
	{"name": "bed_assignment_requires_available_bed", "condition": {"operation": "assign_bed", "bed_status": "occupied"}, "effect": {"decision": "deny", "reason": "bed_not_available", "required_action": "select_available_bed"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
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
				"registration": {"type": "object"},
				"adt": {"type": "object"},
			},
		},
		"rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": RULES},
		"ui": {"shell": "apg_python", "requires_theme": True, "template_roots": ["healthcare/pmt/templates"], "routes": UI_ROUTES},
		"theme": THEME,
		"streaming": STREAMING,
		"provides": PROVIDES,
		"requires": REQUIRES,
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	for rule in RULES:
		cond = rule["condition"]
		if all(context.get(k) == v for k, v in cond.items()):
			effect = rule["effect"]
			return {"rule": rule["name"], "decision": effect["decision"], "reason": effect["reason"], "required_action": effect.get("required_action")}
	return {"rule": None, "decision": "allow", "reason": "no_rule_matched", "required_action": None}
