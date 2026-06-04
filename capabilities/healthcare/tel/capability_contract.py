"""Executable capability contract for APG Telemedicine."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "healthcare_tel"
CAPABILITY_NAME = "Telemedicine"
CAPABILITY_VERSION = "1.0.0"
TEL_EVENT_STREAM = "apg.healthcare.tel.lifecycle"

SUPPORTED_CONSULTATION_TYPES = [
	"video", "audio_only", "asynchronous", "store_and_forward",
	"remote_patient_monitoring", "second_opinion", "urgent_care",
]
SUPPORTED_SESSION_STATUSES = ["scheduled", "waiting", "in_progress", "completed", "cancelled", "no_show", "technical_failure"]
SUPPORTED_PLATFORM_TYPES = ["webrtc", "zoom_healthcare", "doxy_me", "amwell", "teladoc", "custom"]
SUPPORTED_MONITORING_DEVICE_TYPES = [
	"glucometer", "blood_pressure_cuff", "pulse_oximeter", "weight_scale",
	"ecg_patch", "thermometer", "spirometer", "cgm",
]
SUPPORTED_PRESCRIPTION_TRANSMISSION_METHODS = ["surescripts", "epcs", "fax", "print"]
SUPPORTED_BILLING_CODES = [
	"99201", "99202", "99203", "99204", "99205",
	"99211", "99212", "99213", "99214", "99215",
	"G2012", "G2252", "99421", "99422", "99423",
]
SUPPORTED_STATES_TELEMEDICINE = ["interstate_compact", "individual_state_license", "emergency_waiver"]
SUPPORTED_CONSENT_TYPES = ["informed_consent", "hipaa_authorization", "telehealth_specific"]
SUPPORTED_TECHNICAL_REQUIREMENTS = ["bandwidth_check", "device_compatibility", "audio_video_test", "e911_disclosure"]
SUPPORTED_AGENT_ROLES = ["telehealth_steward", "session_reviewer", "prescription_reviewer", "billing_reviewer"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"consultations": {"supported_types": SUPPORTED_CONSULTATION_TYPES, "consent_required": True, "min_session_minutes": 5},
	"sessions": {"supported_statuses": SUPPORTED_SESSION_STATUSES, "supported_platforms": SUPPORTED_PLATFORM_TYPES, "recording_consent_required": True},
	"monitoring": {"supported_device_types": SUPPORTED_MONITORING_DEVICE_TYPES, "alert_threshold_required": True},
	"prescriptions": {"supported_transmission_methods": SUPPORTED_PRESCRIPTION_TRANSMISSION_METHODS, "controlled_substance_requires_in_person": True},
	"billing": {"supported_codes": SUPPORTED_BILLING_CODES, "place_of_service_code": "02"},
	"licensing": {"supported_frameworks": SUPPORTED_STATES_TELEMEDICINE},
	"governance": {
		"require_tenant_context": True, "policy_attached_for_writes": True,
		"audit_events": True, "hipaa_phi_protection": True,
		"cross_tenant_session_access_denied": True,
		"patient_consent_required": True,
		"e911_disclosure_required": True,
		"controlled_substance_telemedicine_restriction": True,
		"recording_requires_consent": True,
	},
	"observability": {"event_stream": TEL_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "workflow": "wflo", "scheduler": "schd", "compliance": "comp", "monitoring": "moni", "event_stream": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_scheduling": True, "enable_sessions": True, "enable_monitoring": True, "enable_prescriptions": True, "enable_billing": True},
	"theme": {"default_theme": "healthcare_tel_clinical", "allow_tenant_overrides": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
}

PROVIDES = [
	"virtual_consultation_booking", "video_session_management",
	"remote_patient_monitoring", "prescription_transmission",
	"telehealth_billing", "patient_consent_management",
	"technical_readiness_check", "asynchronous_consultation",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "schd", "comp", "moni", "mqeb"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/healthcare-tel/dashboard", "component": "TelDashboard", "permission": "healthcare_tel:view", "nav_group": "Overview"},
	{"name": "schedule", "path": "/healthcare-tel/schedule", "component": "TelConsultationSchedule", "permission": "healthcare_tel:schedule", "nav_group": "Scheduling"},
	{"name": "consultation_new", "path": "/healthcare-tel/schedule/new", "component": "TelConsultationBook", "permission": "healthcare_tel:schedule_write", "nav_group": "Scheduling"},
	{"name": "consultation_detail", "path": "/healthcare-tel/schedule/<id>", "component": "TelConsultationDetail", "permission": "healthcare_tel:schedule", "nav_group": "Scheduling"},
	{"name": "sessions", "path": "/healthcare-tel/sessions", "component": "TelSessionList", "permission": "healthcare_tel:sessions", "nav_group": "Sessions"},
	{"name": "session_room", "path": "/healthcare-tel/sessions/<id>/room", "component": "TelSessionRoom", "permission": "healthcare_tel:sessions", "nav_group": "Sessions"},
	{"name": "monitoring", "path": "/healthcare-tel/monitoring", "component": "TelRemoteMonitoring", "permission": "healthcare_tel:monitoring", "nav_group": "Monitoring"},
	{"name": "monitoring_device", "path": "/healthcare-tel/monitoring/<patient_id>", "component": "TelPatientMonitor", "permission": "healthcare_tel:monitoring", "nav_group": "Monitoring"},
	{"name": "prescriptions", "path": "/healthcare-tel/prescriptions", "component": "TelPrescriptionTransmit", "permission": "healthcare_tel:prescriptions", "nav_group": "Prescriptions"},
	{"name": "billing", "path": "/healthcare-tel/billing", "component": "TelBillingConsole", "permission": "healthcare_tel:billing", "nav_group": "Billing"},
	{"name": "consent", "path": "/healthcare-tel/consent", "component": "TelConsentManager", "permission": "healthcare_tel:consent", "nav_group": "Compliance"},
	{"name": "agents", "path": "/healthcare-tel/agents", "component": "TelAgentWorkbench", "permission": "healthcare_tel:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/healthcare-tel/settings", "component": "TelSettings", "permission": "healthcare_tel:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "healthcare_tel_clinical",
	"tokens": {
		"color.primary": "#4338CA", "color.accent": "#0891B2", "color.success": "#166534",
		"color.warning": "#A16207", "color.danger": "#B91C1C",
		"surface.canvas": "#EEF2FF", "surface.panel": "#FFFFFF",
		"text.primary": "#1E1B4B", "text.secondary": "#4338CA",
		"border.radius": "8px", "density": "comfortable",
	},
	"components": {
		"consultations": {"icon": "video", "status_indicator": "consultation-type-chip"},
		"sessions": {"icon": "monitor", "status_indicator": "session-status-chip"},
		"monitoring": {"icon": "activity", "status_indicator": "device-type-chip"},
		"prescriptions": {"icon": "file-text", "status_indicator": "transmission-method-chip"},
		"billing": {"icon": "dollar-sign", "status_indicator": "billing-code-chip"},
		"consent": {"icon": "check-circle", "status_indicator": "consent-type-chip"},
		"agents": {"icon": "cpu", "status_indicator": "agent-runtime-chip"},
	},
}

STREAMING = {
	"processor": "bytewax", "stream": TEL_EVENT_STREAM, "key": "tenant_id",
	"events": [
		"consultation_booked", "consultation_cancelled", "session_started",
		"session_completed", "session_failed", "monitoring_alert_triggered",
		"prescription_transmitted", "consent_obtained", "billing_record_created",
	],
	"guardrails": [
		"patient_consent_required_before_session", "e911_disclosure_required",
		"controlled_substance_telemedicine_restriction",
		"recording_requires_explicit_consent",
		"cross_tenant_session_access_denied",
		"privileged_agent_action_requires_human_approval",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "write_policy_required", "required_action": "attach_write_policy"}},
	{"name": "cross_tenant_session_access_denied", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_telemedicine_session_prohibited", "required_action": "use_tenant_scoped_query"}},
	{"name": "patient_consent_required", "condition": {"operation": "start_session", "patient_consent_obtained": False}, "effect": {"decision": "deny", "reason": "patient_consent_required_before_telemedicine_session", "required_action": "obtain_patient_consent"}},
	{"name": "e911_disclosure_required", "condition": {"operation": "start_session", "e911_disclosure_provided": False}, "effect": {"decision": "deny", "reason": "e911_disclosure_required_before_session", "required_action": "provide_e911_disclosure"}},
	{"name": "consultation_type_supported", "condition": {"operation": "book_consultation", "consultation_type_supported": False}, "effect": {"decision": "deny", "reason": "consultation_type_not_supported", "required_action": "select_supported_consultation_type"}},
	{"name": "session_status_supported", "condition": {"operation": "update_session", "session_status_supported": False}, "effect": {"decision": "deny", "reason": "session_status_not_supported", "required_action": "select_supported_session_status"}},
	{"name": "platform_type_supported", "condition": {"operation": "create_session", "platform_type_supported": False}, "effect": {"decision": "deny", "reason": "platform_type_not_supported", "required_action": "select_supported_platform"}},
	{"name": "controlled_substance_telemedicine_restriction", "condition": {"operation": "transmit_prescription", "drug_schedule": "schedule_ii", "in_person_visit_completed": False}, "effect": {"decision": "deny", "reason": "schedule_ii_prescribing_requires_in_person_visit", "required_action": "conduct_in_person_visit_first"}},
	{"name": "prescription_transmission_method_supported", "condition": {"operation": "transmit_prescription", "transmission_method_supported": False}, "effect": {"decision": "deny", "reason": "prescription_transmission_method_not_supported", "required_action": "select_supported_transmission_method"}},
	{"name": "billing_code_supported", "condition": {"operation": "create_billing_record", "billing_code_supported": False}, "effect": {"decision": "deny", "reason": "billing_code_not_supported_for_telemedicine", "required_action": "select_supported_telehealth_billing_code"}},
	{"name": "recording_requires_consent", "condition": {"operation": "start_recording", "recording_consent_obtained": False}, "effect": {"decision": "deny", "reason": "recording_consent_required", "required_action": "obtain_recording_consent"}},
	{"name": "monitoring_device_type_supported", "condition": {"operation": "enroll_monitoring_device", "device_type_supported": False}, "effect": {"decision": "deny", "reason": "monitoring_device_type_not_supported", "required_action": "select_supported_device_type"}},
	{"name": "consent_type_supported", "condition": {"operation": "record_consent", "consent_type_supported": False}, "effect": {"decision": "deny", "reason": "consent_type_not_supported", "required_action": "select_supported_consent_type"}},
	{"name": "cancelled_consultation_not_startable", "condition": {"operation": "start_session", "consultation_status": "cancelled"}, "effect": {"decision": "deny", "reason": "cancelled_consultation_cannot_be_started", "required_action": "rebook_consultation"}},
	{"name": "no_show_requires_rebooking", "condition": {"operation": "rebook_consultation", "previous_status": "no_show"}, "effect": {"decision": "warn", "reason": "patient_had_previous_no_show", "required_action": "confirm_patient_availability"}},
	{"name": "agent_privileged_action_requires_approval", "condition": {"agent_action": True, "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "privileged_agent_action_requires_human_approval", "required_action": "record_human_approval"}},
	{"name": "technical_readiness_check_required", "condition": {"operation": "start_session", "technical_check_completed": False}, "effect": {"decision": "warn", "reason": "technical_readiness_check_recommended", "required_action": "complete_technical_readiness_check"}},
	{"name": "monitoring_alert_threshold_required", "condition": {"operation": "enroll_monitoring_device", "alert_threshold_configured": False}, "effect": {"decision": "deny", "reason": "alert_threshold_required_for_remote_monitoring", "required_action": "configure_alert_thresholds"}},
	{"name": "cross_tenant_session_denied", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_telemedicine_data_prohibited", "required_action": "use_tenant_scoped_session"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	return {
		"capability": CAPABILITY_ID, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION,
		"configuration": config,
		"configuration_schema": {"required": ["tenant_id", "ui", "theme"], "properties": {"tenant_id": {"type": "string"}, "ui": {"type": "object"}, "theme": {"type": "object"}}},
		"rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": RULES},
		"ui": {"shell": "apg_python", "requires_theme": True, "template_roots": ["healthcare/tel/templates"], "routes": UI_ROUTES},
		"theme": THEME, "streaming": STREAMING, "provides": PROVIDES, "requires": REQUIRES,
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	for rule in RULES:
		cond = rule["condition"]
		if all(context.get(k) == v for k, v in cond.items()):
			effect = rule["effect"]
			return {"rule": rule["name"], "decision": effect["decision"], "reason": effect["reason"], "required_action": effect.get("required_action")}
	return {"rule": None, "decision": "allow", "reason": "no_rule_matched", "required_action": None}
