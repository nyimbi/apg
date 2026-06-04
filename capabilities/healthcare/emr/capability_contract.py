"""Executable capability contract for APG Electronic Medical Records."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "healthcare_emr"
CAPABILITY_NAME = "Electronic Medical Records"
CAPABILITY_VERSION = "1.0.0"
EMR_EVENT_STREAM = "apg.healthcare.emr.lifecycle"

SUPPORTED_NOTE_TYPES = [
	"soap_note", "progress_note", "discharge_summary", "operative_note",
	"consultation_note", "nursing_note", "procedure_note", "psychiatric_note",
	"history_physical", "emergency_note",
]
SUPPORTED_PROBLEM_STATUSES = ["active", "inactive", "resolved", "chronic", "episodic"]
SUPPORTED_MEDICATION_STATUSES = ["active", "discontinued", "on_hold", "completed", "entered_in_error"]
SUPPORTED_ALLERGY_TYPES = ["drug", "food", "environmental", "contrast", "latex", "other"]
SUPPORTED_ALLERGY_SEVERITIES = ["mild", "moderate", "severe", "life_threatening"]
SUPPORTED_FHIR_RESOURCE_TYPES = [
	"Patient", "Encounter", "Condition", "MedicationRequest", "AllergyIntolerance",
	"Observation", "DiagnosticReport", "Procedure", "DocumentReference", "CarePlan",
]
SUPPORTED_ICD10_CHAPTERS = [
	"A00-B99", "C00-D49", "D50-D89", "E00-E89", "F01-F99",
	"G00-G99", "H00-H59", "H60-H95", "I00-I99", "J00-J99",
	"K00-K95", "L00-L99", "M00-M99", "N00-N99", "O00-O9A",
]
SUPPORTED_ENCOUNTER_STATUSES = ["planned", "arrived", "triaged", "in_progress", "on_leave", "finished", "cancelled"]
SUPPORTED_VITAL_TYPES = [
	"blood_pressure", "heart_rate", "respiratory_rate", "temperature",
	"oxygen_saturation", "weight", "height", "bmi", "pain_scale",
]
SUPPORTED_RECONCILIATION_STATUSES = ["pending", "reconciled", "discrepancy_noted", "escalated"]
SUPPORTED_AGENT_ROLES = ["emr_steward", "note_reviewer", "coding_reviewer", "reconciliation_reviewer"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"notes": {"supported_note_types": SUPPORTED_NOTE_TYPES, "co_signature_required": True, "addendum_allowed": True},
	"problems": {"supported_statuses": SUPPORTED_PROBLEM_STATUSES, "icd10_required": True},
	"medications": {"supported_statuses": SUPPORTED_MEDICATION_STATUSES, "reconciliation_on_admission": True},
	"allergies": {"supported_types": SUPPORTED_ALLERGY_TYPES, "supported_severities": SUPPORTED_ALLERGY_SEVERITIES},
	"fhir": {"version": "R4", "supported_resources": SUPPORTED_FHIR_RESOURCE_TYPES, "export_enabled": True},
	"vitals": {"supported_types": SUPPORTED_VITAL_TYPES},
	"encounters": {"supported_statuses": SUPPORTED_ENCOUNTER_STATUSES},
	"governance": {
		"require_tenant_context": True, "policy_attached_for_writes": True,
		"audit_events": True, "hipaa_phi_protection": True,
		"cross_tenant_record_access_denied": True, "note_amendment_requires_original": True,
		"icd10_required_for_problem": True, "medication_allergy_check_required": True,
		"deceased_record_locked": True,
	},
	"observability": {"event_stream": EMR_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "nlp": "nlpc", "workflow": "wflo", "event_stream": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_chart": True, "enable_notes": True, "enable_problems": True, "enable_medications": True, "enable_allergies": True, "enable_vitals": True, "enable_fhir_export": True},
	"theme": {"default_theme": "healthcare_emr_clinical", "allow_tenant_overrides": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
}

PROVIDES = [
	"patient_chart_management", "clinical_note_authoring", "problem_list_management",
	"medication_reconciliation", "allergy_tracking", "vital_signs_recording",
	"fhir_r4_export", "icd10_coding", "encounter_management",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "nlpc", "wflo", "mqeb"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/healthcare-emr/dashboard", "component": "EmrDashboard", "permission": "healthcare_emr:view", "nav_group": "Overview"},
	{"name": "chart", "path": "/healthcare-emr/chart/<patient_id>", "component": "EmrPatientChart", "permission": "healthcare_emr:chart", "nav_group": "Chart"},
	{"name": "notes", "path": "/healthcare-emr/notes", "component": "EmrNoteList", "permission": "healthcare_emr:notes", "nav_group": "Documentation"},
	{"name": "note_new", "path": "/healthcare-emr/notes/new", "component": "EmrNoteEditor", "permission": "healthcare_emr:notes_write", "nav_group": "Documentation"},
	{"name": "note_detail", "path": "/healthcare-emr/notes/<id>", "component": "EmrNoteDetail", "permission": "healthcare_emr:notes", "nav_group": "Documentation"},
	{"name": "problems", "path": "/healthcare-emr/problems/<patient_id>", "component": "EmrProblemList", "permission": "healthcare_emr:problems", "nav_group": "Clinical"},
	{"name": "medications", "path": "/healthcare-emr/medications/<patient_id>", "component": "EmrMedicationList", "permission": "healthcare_emr:medications", "nav_group": "Clinical"},
	{"name": "allergies", "path": "/healthcare-emr/allergies/<patient_id>", "component": "EmrAllergyList", "permission": "healthcare_emr:allergies", "nav_group": "Clinical"},
	{"name": "vitals", "path": "/healthcare-emr/vitals/<patient_id>", "component": "EmrVitalSigns", "permission": "healthcare_emr:vitals", "nav_group": "Clinical"},
	{"name": "encounters", "path": "/healthcare-emr/encounters/<patient_id>", "component": "EmrEncounterList", "permission": "healthcare_emr:encounters", "nav_group": "Encounters"},
	{"name": "reconciliation", "path": "/healthcare-emr/reconciliation/<patient_id>", "component": "EmrMedReconciliation", "permission": "healthcare_emr:reconciliation", "nav_group": "Clinical"},
	{"name": "fhir_export", "path": "/healthcare-emr/fhir-export", "component": "EmrFhirExport", "permission": "healthcare_emr:fhir", "nav_group": "Interoperability"},
	{"name": "agents", "path": "/healthcare-emr/agents", "component": "EmrAgentWorkbench", "permission": "healthcare_emr:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/healthcare-emr/settings", "component": "EmrSettings", "permission": "healthcare_emr:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "healthcare_emr_clinical",
	"tokens": {
		"color.primary": "#0F766E", "color.accent": "#0369A1", "color.success": "#166534",
		"color.warning": "#A16207", "color.danger": "#B91C1C",
		"surface.canvas": "#F0FDFA", "surface.panel": "#FFFFFF",
		"text.primary": "#134E4A", "text.secondary": "#0F766E",
		"border.radius": "6px", "density": "comfortable",
	},
	"components": {
		"chart": {"icon": "file-medical", "status_indicator": "chart-status-chip"},
		"notes": {"icon": "file-text", "status_indicator": "note-type-chip"},
		"problems": {"icon": "list-checks", "status_indicator": "problem-status-chip"},
		"medications": {"icon": "pill", "status_indicator": "medication-status-chip"},
		"allergies": {"icon": "alert-triangle", "status_indicator": "allergy-severity-chip"},
		"vitals": {"icon": "activity", "status_indicator": "vital-trend-chip"},
		"encounters": {"icon": "calendar-check", "status_indicator": "encounter-status-chip"},
		"fhir": {"icon": "share-2", "status_indicator": "fhir-resource-chip"},
		"agents": {"icon": "cpu", "status_indicator": "agent-runtime-chip"},
	},
}

STREAMING = {
	"processor": "bytewax", "stream": EMR_EVENT_STREAM, "key": "tenant_id",
	"events": [
		"note_created", "note_amended", "problem_added", "problem_resolved",
		"medication_prescribed", "medication_discontinued", "allergy_recorded",
		"vital_recorded", "encounter_opened", "encounter_closed", "fhir_export_generated",
	],
	"guardrails": [
		"phi_access_requires_authorization", "cross_tenant_record_access_denied",
		"note_amendment_must_preserve_original", "icd10_required_for_problem",
		"medication_allergy_check_required", "privileged_agent_action_requires_human_approval",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "write_policy_required", "required_action": "attach_write_policy"}},
	{"name": "cross_tenant_record_access_denied", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_phi_access_prohibited", "required_action": "use_tenant_scoped_query"}},
	{"name": "note_type_supported", "condition": {"operation": "create_note", "note_type_supported": False}, "effect": {"decision": "deny", "reason": "note_type_not_supported", "required_action": "select_supported_note_type"}},
	{"name": "note_amendment_requires_original", "condition": {"operation": "amend_note", "original_note_present": False}, "effect": {"decision": "deny", "reason": "original_note_required_for_amendment", "required_action": "reference_original_note"}},
	{"name": "problem_requires_icd10", "condition": {"operation": "add_problem", "icd10_code_present": False}, "effect": {"decision": "deny", "reason": "icd10_code_required_for_problem", "required_action": "assign_icd10_code"}},
	{"name": "problem_status_supported", "condition": {"operation": "update_problem", "problem_status_supported": False}, "effect": {"decision": "deny", "reason": "problem_status_not_supported", "required_action": "select_supported_problem_status"}},
	{"name": "medication_allergy_check_required", "condition": {"operation": "prescribe_medication", "allergy_check_performed": False}, "effect": {"decision": "deny", "reason": "allergy_check_required_before_prescribing", "required_action": "perform_allergy_check"}},
	{"name": "medication_status_supported", "condition": {"operation": "update_medication", "medication_status_supported": False}, "effect": {"decision": "deny", "reason": "medication_status_not_supported", "required_action": "select_supported_medication_status"}},
	{"name": "allergy_type_supported", "condition": {"operation": "record_allergy", "allergy_type_supported": False}, "effect": {"decision": "deny", "reason": "allergy_type_not_supported", "required_action": "select_supported_allergy_type"}},
	{"name": "allergy_severity_supported", "condition": {"operation": "record_allergy", "allergy_severity_supported": False}, "effect": {"decision": "deny", "reason": "allergy_severity_not_supported", "required_action": "select_supported_allergy_severity"}},
	{"name": "vital_type_supported", "condition": {"operation": "record_vital", "vital_type_supported": False}, "effect": {"decision": "deny", "reason": "vital_type_not_supported", "required_action": "select_supported_vital_type"}},
	{"name": "encounter_status_supported", "condition": {"operation": "update_encounter", "encounter_status_supported": False}, "effect": {"decision": "deny", "reason": "encounter_status_not_supported", "required_action": "select_supported_encounter_status"}},
	{"name": "deceased_record_locked", "condition": {"operation": "update_chart", "patient_deceased": True}, "effect": {"decision": "deny", "reason": "deceased_patient_chart_is_locked", "required_action": "use_amendment_workflow"}},
	{"name": "fhir_export_requires_phi_consent", "condition": {"operation": "fhir_export", "phi_consent_present": False}, "effect": {"decision": "deny", "reason": "phi_consent_required_for_fhir_export", "required_action": "obtain_phi_consent"}},
	{"name": "reconciliation_status_supported", "condition": {"operation": "update_reconciliation", "reconciliation_status_supported": False}, "effect": {"decision": "deny", "reason": "reconciliation_status_not_supported", "required_action": "select_supported_reconciliation_status"}},
	{"name": "note_cosignature_required", "condition": {"operation": "finalize_note", "cosignature_required": True, "cosignature_present": False}, "effect": {"decision": "deny", "reason": "co_signature_required_to_finalize_note", "required_action": "obtain_co_signature"}},
	{"name": "agent_privileged_action_requires_approval", "condition": {"agent_action": True, "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "privileged_agent_action_requires_human_approval", "required_action": "record_human_approval"}},
	{"name": "medication_reconciliation_on_admission", "condition": {"operation": "admit_patient", "med_reconciliation_performed": False}, "effect": {"decision": "warn", "reason": "medication_reconciliation_recommended_on_admission", "required_action": "perform_medication_reconciliation"}},
	{"name": "fhir_resource_type_supported", "condition": {"operation": "fhir_export", "resource_type_supported": False}, "effect": {"decision": "deny", "reason": "fhir_resource_type_not_supported", "required_action": "select_supported_fhir_resource_type"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	return {
		"capability": CAPABILITY_ID, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION,
		"configuration": config,
		"configuration_schema": {"required": ["tenant_id", "ui", "theme"], "properties": {"tenant_id": {"type": "string"}, "ui": {"type": "object"}, "theme": {"type": "object"}}},
		"rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": RULES},
		"ui": {"shell": "apg_python", "requires_theme": True, "template_roots": ["healthcare/emr/templates"], "routes": UI_ROUTES},
		"theme": THEME, "streaming": STREAMING, "provides": PROVIDES, "requires": REQUIRES,
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	for rule in RULES:
		cond = rule["condition"]
		if all(context.get(k) == v for k, v in cond.items()):
			effect = rule["effect"]
			return {"rule": rule["name"], "decision": effect["decision"], "reason": effect["reason"], "required_action": effect.get("required_action")}
	return {"rule": None, "decision": "allow", "reason": "no_rule_matched", "required_action": None}
