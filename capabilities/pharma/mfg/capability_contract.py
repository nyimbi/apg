"""Executable capability contract for APG Pharma Manufacturing."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "pharma_mfg"
CAPABILITY_NAME = "Pharmaceutical Manufacturing"
CAPABILITY_VERSION = "1.0.0"
MFG_EVENT_STREAM = "apg.pharma.mfg.lifecycle"

SUPPORTED_BATCH_STATUSES = ["planned", "in_process", "intermediate_hold", "awaiting_qc", "qc_passed", "qc_failed", "released", "rejected", "recalled", "destroyed"]
SUPPORTED_MANUFACTURING_TYPES = ["bulk_drug_substance", "drug_product", "finished_dose", "sterile_fill_finish", "oral_solid_dose", "biologics", "gene_therapy", "device_combination"]
SUPPORTED_EQUIPMENT_STATUSES = ["operational", "under_qualification", "qualified", "in_maintenance", "out_of_service", "retired"]
SUPPORTED_QUALIFICATION_TYPES = ["iq", "oq", "pq", "requalification", "commissioning", "periodic_review"]
SUPPORTED_DEVIATION_TYPES = ["process_deviation", "equipment_deviation", "material_deviation", "environmental_deviation", "utility_deviation", "procedure_deviation"]
SUPPORTED_DEVIATION_SEVERITIES = ["minor", "major", "critical"]
SUPPORTED_YIELD_TYPES = ["theoretical", "actual", "percentage", "step_yield", "overall_yield", "reconciliation"]
SUPPORTED_GMP_FRAMEWORKS = ["21cfr_part_210", "21cfr_part_211", "eu_gmp_annex_1", "eu_gmp_annex_2", "ich_q7", "ich_q10", "pic_s", "who_gmp"]
SUPPORTED_CLEANING_STATUSES = ["dirty", "cleaning_in_progress", "cleaned", "validated", "cleared_for_use"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["batch_reviewer", "equipment_monitor", "yield_analyst", "deviation_classifier", "gmp_auditor"]
SUPPORTED_MATERIAL_STATUSES = ["quarantine", "released", "rejected", "on_hold", "dispensed", "returned"]
SUPPORTED_LINE_STATUSES = ["available", "running", "changeover", "cleaning", "maintenance", "qualification", "shutdown"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"batches": {"supported_statuses": SUPPORTED_BATCH_STATUSES, "batch_number_required": True, "master_formula_required": True, "qp_release_required": True, "electronic_batch_record": True},
	"manufacturing_types": {"supported_types": SUPPORTED_MANUFACTURING_TYPES, "process_validation_required": True, "cleaning_validation_required": True},
	"equipment": {"supported_statuses": SUPPORTED_EQUIPMENT_STATUSES, "supported_qualification_types": SUPPORTED_QUALIFICATION_TYPES, "calibration_required": True, "maintenance_plan_required": True, "requalification_trigger_months": 12},
	"deviations": {"supported_types": SUPPORTED_DEVIATION_TYPES, "supported_severities": SUPPORTED_DEVIATION_SEVERITIES, "investigation_required": True, "capa_required_for_major_critical": True, "reporting_timeline_hours": {"critical": 24, "major": 72, "minor": 168}},
	"yield_management": {"supported_types": SUPPORTED_YIELD_TYPES, "reconciliation_required": True, "yield_variance_threshold_pct": 2.0, "investigation_trigger_pct": 5.0},
	"gmp": {"supported_frameworks": SUPPORTED_GMP_FRAMEWORKS, "self_inspection_required": True, "change_control_required": True, "document_control_required": True, "21cfr11_electronic_records": True},
	"materials": {"supported_statuses": SUPPORTED_MATERIAL_STATUSES, "vendor_qualification_required": True, "incoming_qc_required": True, "expiry_management": True},
	"lines": {"supported_statuses": SUPPORTED_LINE_STATUSES, "line_clearance_required": True, "cleaning_verification_required": True, "environmental_monitoring": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "gmp_compliance_required": True, "qp_release_required": True, "electronic_signature_required": True, "cross_tenant_denied": True},
	"observability": {"event_stream": MFG_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "workflow": "wflo", "compliance": "comp", "monitoring": "moni", "scheduler": "schd", "event_stream": "mqeb"},
	"ui": {"enable_dashboard": True, "enable_batches": True, "enable_equipment": True, "enable_deviations": True, "enable_yield": True, "enable_gmp": True, "enable_materials": True, "enable_lines": True},
	"theme": {"default_theme": "pharma_mfg_plant", "allow_tenant_overrides": True},
}

PROVIDES = [
	"batch_record_management_workflow",
	"manufacturing_execution_workflow",
	"equipment_qualification_workflow",
	"yield_management_workflow",
	"deviation_management_workflow",
	"gmp_compliance_workflow",
	"material_management_workflow",
	"line_clearance_workflow",
	"cleaning_validation_workflow",
	"qp_release_workflow",
]
REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "comp", "moni", "schd", "mqeb"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/pharma-mfg/dashboard", "component": "MfgDashboard", "permission": "pharma_mfg:view", "nav_group": "Overview"},
	{"name": "batches", "path": "/pharma-mfg/batches", "component": "BatchRegistry", "permission": "pharma_mfg:batches", "nav_group": "Production"},
	{"name": "batch_detail", "path": "/pharma-mfg/batches/<id>", "component": "BatchRecordDetail", "permission": "pharma_mfg:batches", "nav_group": "Production"},
	{"name": "batch_record", "path": "/pharma-mfg/batches/<id>/ebr", "component": "ElectronicBatchRecord", "permission": "pharma_mfg:ebr", "nav_group": "Production"},
	{"name": "lines", "path": "/pharma-mfg/lines", "component": "ProductionLines", "permission": "pharma_mfg:lines", "nav_group": "Production"},
	{"name": "equipment", "path": "/pharma-mfg/equipment", "component": "EquipmentRegistry", "permission": "pharma_mfg:equipment", "nav_group": "Equipment"},
	{"name": "qualification", "path": "/pharma-mfg/equipment/qualification", "component": "QualificationConsole", "permission": "pharma_mfg:qualification", "nav_group": "Equipment"},
	{"name": "materials", "path": "/pharma-mfg/materials", "component": "MaterialManagement", "permission": "pharma_mfg:materials", "nav_group": "Materials"},
	{"name": "deviations", "path": "/pharma-mfg/deviations", "component": "DeviationQueue", "permission": "pharma_mfg:deviations", "nav_group": "Quality"},
	{"name": "yield", "path": "/pharma-mfg/yield", "component": "YieldDashboard", "permission": "pharma_mfg:yield", "nav_group": "Analytics"},
	{"name": "gmp_compliance", "path": "/pharma-mfg/gmp", "component": "GmpComplianceConsole", "permission": "pharma_mfg:gmp", "nav_group": "Compliance"},
	{"name": "environmental", "path": "/pharma-mfg/environmental", "component": "EnvironmentalMonitoring", "permission": "pharma_mfg:environmental", "nav_group": "Compliance"},
	{"name": "reports", "path": "/pharma-mfg/reports", "component": "ManufacturingReports", "permission": "pharma_mfg:reports", "nav_group": "Reporting"},
	{"name": "settings", "path": "/pharma-mfg/settings", "component": "MfgSettings", "permission": "pharma_mfg:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "pharma_mfg_plant",
	"tokens": {
		"color.primary": "#374151",
		"color.accent": "#2563EB",
		"color.success": "#15803D",
		"color.warning": "#D97706",
		"color.danger": "#DC2626",
		"surface.canvas": "#F3F4F6",
		"surface.panel": "#FFFFFF",
		"text.primary": "#111827",
		"text.secondary": "#6B7280",
		"border.radius": "4px",
		"density": "compact",
	},
	"components": {
		"batches": {"icon": "layers", "status_indicator": "batch-status-chip"},
		"equipment": {"icon": "settings", "status_indicator": "equipment-status-chip"},
		"deviations": {"icon": "alert-circle", "status_indicator": "deviation-severity-chip"},
		"materials": {"icon": "box", "status_indicator": "material-status-chip"},
		"lines": {"icon": "activity", "status_indicator": "line-status-chip"},
		"yield": {"icon": "trending-up", "status_indicator": "yield-type-chip"},
		"gmp": {"icon": "shield", "status_indicator": "gmp-framework-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": MFG_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"batch_started", "batch_completed", "batch_released", "batch_rejected",
		"equipment_qualified", "equipment_out_of_service",
		"deviation_raised", "deviation_closed",
		"yield_reconciled", "yield_variance_exceeded",
		"gmp_deviation_critical", "line_clearance_completed",
		"qp_release_signed",
	],
	"guardrails": [
		"qp_release_required_before_distribution",
		"gmp_compliance_required",
		"electronic_signature_required_for_release",
		"deviation_investigation_required",
		"yield_reconciliation_required",
		"equipment_qualification_required_before_use",
		"cross_tenant_batch_data_denied",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "policy_required", "required_action": "attach_policy"}},
	{"name": "batch_master_formula_required", "condition": {"operation": "create_batch", "master_formula_present": False}, "effect": {"decision": "deny", "reason": "master_formula_required", "required_action": "attach_master_formula"}},
	{"name": "batch_number_required", "condition": {"operation": "create_batch", "batch_number_present": False}, "effect": {"decision": "deny", "reason": "batch_number_required", "required_action": "assign_batch_number"}},
	{"name": "batch_status_supported", "condition": {"operation": "update_batch_status", "batch_status_supported": False}, "effect": {"decision": "deny", "reason": "batch_status_not_supported", "required_action": "select_supported_batch_status"}},
	{"name": "qp_release_required", "condition": {"operation": "release_batch", "qp_signed": False}, "effect": {"decision": "deny", "reason": "qp_release_signature_required", "required_action": "obtain_qp_signature"}},
	{"name": "electronic_signature_required", "condition": {"operation": "release_batch", "electronic_signature_present": False}, "effect": {"decision": "deny", "reason": "electronic_signature_required", "required_action": "apply_electronic_signature"}},
	{"name": "equipment_qualification_required", "condition": {"operation": "use_equipment", "equipment_qualified": False}, "effect": {"decision": "deny", "reason": "equipment_qualification_required", "required_action": "complete_qualification"}},
	{"name": "equipment_calibration_required", "condition": {"operation": "use_equipment", "calibration_current": False}, "effect": {"decision": "deny", "reason": "calibration_required", "required_action": "perform_calibration"}},
	{"name": "qualification_type_supported", "condition": {"operation": "record_qualification", "qualification_type_supported": False}, "effect": {"decision": "deny", "reason": "qualification_type_not_supported", "required_action": "select_supported_qualification_type"}},
	{"name": "line_clearance_required", "condition": {"operation": "start_batch", "line_cleared": False}, "effect": {"decision": "deny", "reason": "line_clearance_required", "required_action": "complete_line_clearance"}},
	{"name": "deviation_type_supported", "condition": {"operation": "raise_deviation", "deviation_type_supported": False}, "effect": {"decision": "deny", "reason": "deviation_type_not_supported", "required_action": "select_supported_deviation_type"}},
	{"name": "deviation_investigation_required", "condition": {"operation": "close_deviation", "investigation_completed": False}, "effect": {"decision": "deny", "reason": "investigation_required", "required_action": "complete_investigation"}},
	{"name": "critical_deviation_24h_reporting", "condition": {"operation": "raise_deviation", "severity": "critical", "within_24h": False}, "effect": {"decision": "deny", "reason": "critical_deviation_24h_required", "required_action": "expedite_deviation_report"}},
	{"name": "yield_reconciliation_required", "condition": {"operation": "close_batch", "yield_reconciled": False}, "effect": {"decision": "deny", "reason": "yield_reconciliation_required", "required_action": "complete_yield_reconciliation"}},
	{"name": "material_vendor_qualification_required", "condition": {"operation": "receive_material", "vendor_qualified": False}, "effect": {"decision": "deny", "reason": "vendor_qualification_required", "required_action": "qualify_vendor"}},
	{"name": "material_incoming_qc_required", "condition": {"operation": "release_material", "incoming_qc_completed": False}, "effect": {"decision": "deny", "reason": "incoming_qc_required", "required_action": "complete_incoming_qc"}},
	{"name": "gmp_framework_supported", "condition": {"operation": "register_gmp_compliance", "gmp_framework_supported": False}, "effect": {"decision": "deny", "reason": "gmp_framework_not_supported", "required_action": "select_supported_framework"}},
	{"name": "cleaning_verification_required", "condition": {"operation": "start_batch", "cleaning_verified": False}, "effect": {"decision": "deny", "reason": "cleaning_verification_required", "required_action": "verify_cleaning"}},
	{"name": "cross_tenant_denied", "condition": {"operation_type": "write", "cross_tenant": True}, "effect": {"decision": "deny", "reason": "cross_tenant_operation_denied", "required_action": "use_correct_tenant_context"}},
	{"name": "manufacturing_type_supported", "condition": {"operation": "create_batch", "manufacturing_type_supported": False}, "effect": {"decision": "deny", "reason": "manufacturing_type_not_supported", "required_action": "select_supported_manufacturing_type"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	return {
		"capability": CAPABILITY_ID,
		"display_name": CAPABILITY_NAME,
		"version": CAPABILITY_VERSION,
		"provides": list(PROVIDES),
		"requires": list(REQUIRES),
		"configuration": configuration,
		"configuration_schema": {
			"type": "object",
			"required": ["tenant_id", "ui", "theme"],
			"properties": {
				"tenant_id": {"type": "string", "minLength": 1},
				"ui": {"type": "object"},
				"theme": {"type": "object"},
			},
		},
		"rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"api_prefix": "/pharma-mfg/api/v1",
			"requires_theme": True,
			"template_roots": ["templates/", "static/"],
			"routes": deepcopy(UI_ROUTES),
		},
		"theme": deepcopy(THEME),
		"streaming": deepcopy(STREAMING),
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	actions: list[dict[str, Any]] = []
	for rule in RULES:
		if _matches(rule["condition"], context):
			actions.append(rule["effect"] | {"rule": rule["name"]})
	if not actions:
		return {"decision": "allow", "actions": [], "context": dict(context)}
	return {"decision": "deny", "actions": actions, "context": dict(context)}


def _matches(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if key.endswith("_ne"):
			if context.get(key[:-3]) == expected:
				return False
			continue
		if context.get(key) != expected:
			return False
	return True
