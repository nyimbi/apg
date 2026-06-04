"""Executable capability contract for APG Laboratory Information System."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "healthcare_lab"
CAPABILITY_NAME = "Laboratory Information System"
CAPABILITY_VERSION = "1.0.0"
LAB_EVENT_STREAM = "apg.healthcare.lab.lifecycle"

SUPPORTED_ORDER_STATUSES = ["pending", "collected", "processing", "resulted", "verified", "cancelled", "on_hold"]
SUPPORTED_SPECIMEN_TYPES = [
	"blood_venous", "blood_arterial", "urine_random", "urine_24h", "csf",
	"stool", "sputum", "swab_throat", "swab_wound", "biopsy_tissue",
	"pleural_fluid", "synovial_fluid", "bone_marrow",
]
SUPPORTED_TEST_CATEGORIES = [
	"hematology", "chemistry", "microbiology", "immunology", "urinalysis",
	"coagulation", "toxicology", "serology", "molecular_diagnostics", "pathology",
]
SUPPORTED_RESULT_STATUSES = ["preliminary", "final", "corrected", "cancelled", "entered_in_error"]
SUPPORTED_CRITICAL_VALUE_SEVERITIES = ["critical_high", "critical_low", "panic_value"]
SUPPORTED_QC_STATUSES = ["passed", "failed", "pending_review", "repeated", "accepted"]
SUPPORTED_COLLECTION_PRIORITIES = ["routine", "stat", "asap", "timed"]
SUPPORTED_REJECTION_REASONS = [
	"hemolyzed", "lipemic", "insufficient_volume", "wrong_tube", "clotted",
	"incorrect_patient_id", "temperature_excursion", "unlabeled",
]
SUPPORTED_INSTRUMENT_STATUSES = ["online", "offline", "maintenance", "calibrating", "qc_hold"]
SUPPORTED_RESULT_UNITS = ["mg/dL", "g/dL", "mmol/L", "mEq/L", "U/L", "IU/L", "cells/uL", "%", "ng/mL", "pg/mL"]
SUPPORTED_AGENT_ROLES = ["lab_steward", "result_reviewer", "qc_reviewer", "critical_value_reviewer"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"orders": {"supported_statuses": SUPPORTED_ORDER_STATUSES, "supported_priorities": SUPPORTED_COLLECTION_PRIORITIES, "stat_turnaround_minutes": 60},
	"specimens": {"supported_types": SUPPORTED_SPECIMEN_TYPES, "supported_rejection_reasons": SUPPORTED_REJECTION_REASONS, "chain_of_custody_required": True},
	"tests": {"supported_categories": SUPPORTED_TEST_CATEGORIES},
	"results": {"supported_statuses": SUPPORTED_RESULT_STATUSES, "critical_value_notification_required": True, "critical_value_acknowledgement_required": True},
	"qc": {"supported_statuses": SUPPORTED_QC_STATUSES, "westgard_rules_enabled": True, "qc_frequency_hours": 8},
	"instruments": {"supported_statuses": SUPPORTED_INSTRUMENT_STATUSES, "calibration_tracking": True},
	"governance": {
		"require_tenant_context": True, "policy_attached_for_writes": True,
		"audit_events": True, "hipaa_phi_protection": True,
		"cross_tenant_result_access_denied": True,
		"critical_value_notification_required": True,
		"result_amendment_requires_original": True,
		"specimen_rejection_reason_required": True,
	},
	"observability": {"event_stream": LAB_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "workflow": "wflo", "monitoring": "moni", "event_stream": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_orders": True, "enable_specimens": True, "enable_results": True, "enable_qc": True, "enable_instruments": True, "enable_critical_values": True},
	"theme": {"default_theme": "healthcare_lab_clinical", "allow_tenant_overrides": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
}

PROVIDES = [
	"lab_order_management", "specimen_tracking", "result_entry_verification",
	"critical_value_alerting", "qc_management", "instrument_management",
	"lis_integration", "reference_range_evaluation", "lab_reporting",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "moni", "mqeb"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/healthcare-lab/dashboard", "component": "LabDashboard", "permission": "healthcare_lab:view", "nav_group": "Overview"},
	{"name": "orders", "path": "/healthcare-lab/orders", "component": "LabOrderQueue", "permission": "healthcare_lab:orders", "nav_group": "Orders"},
	{"name": "order_new", "path": "/healthcare-lab/orders/new", "component": "LabOrderEntry", "permission": "healthcare_lab:orders_write", "nav_group": "Orders"},
	{"name": "order_detail", "path": "/healthcare-lab/orders/<id>", "component": "LabOrderDetail", "permission": "healthcare_lab:orders", "nav_group": "Orders"},
	{"name": "specimens", "path": "/healthcare-lab/specimens", "component": "LabSpecimenTracker", "permission": "healthcare_lab:specimens", "nav_group": "Specimens"},
	{"name": "specimen_detail", "path": "/healthcare-lab/specimens/<id>", "component": "LabSpecimenDetail", "permission": "healthcare_lab:specimens", "nav_group": "Specimens"},
	{"name": "results", "path": "/healthcare-lab/results", "component": "LabResultWorkbench", "permission": "healthcare_lab:results", "nav_group": "Results"},
	{"name": "result_entry", "path": "/healthcare-lab/results/entry", "component": "LabResultEntry", "permission": "healthcare_lab:results_write", "nav_group": "Results"},
	{"name": "critical_values", "path": "/healthcare-lab/critical-values", "component": "LabCriticalValues", "permission": "healthcare_lab:critical_values", "nav_group": "Alerts"},
	{"name": "qc", "path": "/healthcare-lab/qc", "component": "LabQCConsole", "permission": "healthcare_lab:qc", "nav_group": "Quality"},
	{"name": "instruments", "path": "/healthcare-lab/instruments", "component": "LabInstrumentPanel", "permission": "healthcare_lab:instruments", "nav_group": "Equipment"},
	{"name": "agents", "path": "/healthcare-lab/agents", "component": "LabAgentWorkbench", "permission": "healthcare_lab:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/healthcare-lab/settings", "component": "LabSettings", "permission": "healthcare_lab:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "healthcare_lab_clinical",
	"tokens": {
		"color.primary": "#7C3AED", "color.accent": "#0891B2", "color.success": "#166534",
		"color.warning": "#A16207", "color.danger": "#B91C1C",
		"surface.canvas": "#FAF5FF", "surface.panel": "#FFFFFF",
		"text.primary": "#3B0764", "text.secondary": "#6B21A8",
		"border.radius": "6px", "density": "compact",
	},
	"components": {
		"orders": {"icon": "clipboard-list", "status_indicator": "order-status-chip"},
		"specimens": {"icon": "droplet", "status_indicator": "specimen-status-chip"},
		"results": {"icon": "bar-chart", "status_indicator": "result-status-chip"},
		"critical_values": {"icon": "alert-octagon", "status_indicator": "critical-severity-chip"},
		"qc": {"icon": "check-square", "status_indicator": "qc-status-chip"},
		"instruments": {"icon": "cpu", "status_indicator": "instrument-status-chip"},
		"agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"},
	},
}

STREAMING = {
	"processor": "bytewax", "stream": LAB_EVENT_STREAM, "key": "tenant_id",
	"events": [
		"order_created", "order_cancelled", "specimen_collected", "specimen_rejected",
		"result_entered", "result_verified", "result_amended", "critical_value_flagged",
		"critical_value_acknowledged", "qc_run_completed", "instrument_status_changed",
	],
	"guardrails": [
		"cross_tenant_result_access_denied", "critical_value_notification_required",
		"result_amendment_must_preserve_original", "specimen_rejection_reason_required",
		"privileged_agent_action_requires_human_approval",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "write_policy_required", "required_action": "attach_write_policy"}},
	{"name": "cross_tenant_result_access_denied", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_lab_result_access_prohibited", "required_action": "use_tenant_scoped_query"}},
	{"name": "order_status_supported", "condition": {"operation": "update_order", "order_status_supported": False}, "effect": {"decision": "deny", "reason": "order_status_not_supported", "required_action": "select_supported_order_status"}},
	{"name": "specimen_type_supported", "condition": {"operation": "collect_specimen", "specimen_type_supported": False}, "effect": {"decision": "deny", "reason": "specimen_type_not_supported", "required_action": "select_supported_specimen_type"}},
	{"name": "specimen_rejection_reason_required", "condition": {"operation": "reject_specimen", "rejection_reason_present": False}, "effect": {"decision": "deny", "reason": "rejection_reason_required", "required_action": "specify_rejection_reason"}},
	{"name": "rejection_reason_supported", "condition": {"operation": "reject_specimen", "rejection_reason_supported": False}, "effect": {"decision": "deny", "reason": "rejection_reason_not_supported", "required_action": "select_supported_rejection_reason"}},
	{"name": "result_status_supported", "condition": {"operation": "update_result", "result_status_supported": False}, "effect": {"decision": "deny", "reason": "result_status_not_supported", "required_action": "select_supported_result_status"}},
	{"name": "result_amendment_requires_original", "condition": {"operation": "amend_result", "original_result_present": False}, "effect": {"decision": "deny", "reason": "original_result_required_for_amendment", "required_action": "reference_original_result"}},
	{"name": "critical_value_notification_required", "condition": {"operation": "verify_result", "critical_value": True, "notification_sent": False}, "effect": {"decision": "deny", "reason": "critical_value_notification_required_before_verify", "required_action": "send_critical_value_notification"}},
	{"name": "critical_value_acknowledgement_required", "condition": {"operation": "close_critical_value", "acknowledgement_present": False}, "effect": {"decision": "deny", "reason": "critical_value_acknowledgement_required", "required_action": "obtain_critical_value_acknowledgement"}},
	{"name": "qc_status_supported", "condition": {"operation": "update_qc", "qc_status_supported": False}, "effect": {"decision": "deny", "reason": "qc_status_not_supported", "required_action": "select_supported_qc_status"}},
	{"name": "qc_hold_blocks_result_release", "condition": {"operation": "verify_result", "instrument_qc_status": "qc_hold"}, "effect": {"decision": "deny", "reason": "instrument_on_qc_hold", "required_action": "resolve_qc_hold_before_releasing_results"}},
	{"name": "instrument_status_supported", "condition": {"operation": "update_instrument", "instrument_status_supported": False}, "effect": {"decision": "deny", "reason": "instrument_status_not_supported", "required_action": "select_supported_instrument_status"}},
	{"name": "test_category_supported", "condition": {"operation": "create_order", "test_category_supported": False}, "effect": {"decision": "deny", "reason": "test_category_not_supported", "required_action": "select_supported_test_category"}},
	{"name": "collection_priority_supported", "condition": {"operation": "create_order", "collection_priority_supported": False}, "effect": {"decision": "deny", "reason": "collection_priority_not_supported", "required_action": "select_supported_collection_priority"}},
	{"name": "specimen_required_for_result", "condition": {"operation": "enter_result", "specimen_present": False}, "effect": {"decision": "deny", "reason": "specimen_required_before_result_entry", "required_action": "collect_specimen_first"}},
	{"name": "cancelled_order_not_collectable", "condition": {"operation": "collect_specimen", "order_status": "cancelled"}, "effect": {"decision": "deny", "reason": "cannot_collect_specimen_for_cancelled_order", "required_action": "reorder_test"}},
	{"name": "agent_privileged_action_requires_approval", "condition": {"agent_action": True, "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "privileged_agent_action_requires_human_approval", "required_action": "record_human_approval"}},
	{"name": "stat_order_turnaround_warning", "condition": {"operation": "verify_result", "stat_order_overdue": True}, "effect": {"decision": "warn", "reason": "stat_order_turnaround_exceeded", "required_action": "escalate_to_lab_supervisor"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	return {
		"capability": CAPABILITY_ID, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION,
		"configuration": config,
		"configuration_schema": {"required": ["tenant_id", "ui", "theme"], "properties": {"tenant_id": {"type": "string"}, "ui": {"type": "object"}, "theme": {"type": "object"}}},
		"rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": RULES},
		"ui": {"shell": "apg_python", "requires_theme": True, "template_roots": ["healthcare/lab/templates"], "routes": UI_ROUTES},
		"theme": THEME, "streaming": STREAMING, "provides": PROVIDES, "requires": REQUIRES,
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	for rule in RULES:
		cond = rule["condition"]
		if all(context.get(k) == v for k, v in cond.items()):
			effect = rule["effect"]
			return {"rule": rule["name"], "decision": effect["decision"], "reason": effect["reason"], "required_action": effect.get("required_action")}
	return {"rule": None, "decision": "allow", "reason": "no_rule_matched", "required_action": None}
