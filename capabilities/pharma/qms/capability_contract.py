"""Executable capability contract for APG Pharma Quality Management System."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "pharma_qms"
CAPABILITY_NAME = "Quality Management System"
CAPABILITY_VERSION = "1.0.0"
QMS_EVENT_STREAM = "apg.pharma.qms.lifecycle"

SUPPORTED_CHANGE_TYPES = ["minor", "major", "critical", "emergency", "administrative", "regulatory_driven", "process_improvement"]
SUPPORTED_CHANGE_STATUSES = ["draft", "impact_assessment", "approval_pending", "approved", "implementation", "effectiveness_check", "closed", "withdrawn"]
SUPPORTED_CAPA_TYPES = ["corrective_action", "preventive_action", "correction", "risk_reduction"]
SUPPORTED_CAPA_STATUSES = ["open", "in_progress", "effectiveness_check", "closed_effective", "closed_ineffective", "overdue"]
SUPPORTED_DEVIATION_TYPES = ["process_deviation", "product_deviation", "equipment_deviation", "documentation_deviation", "environmental_deviation", "personnel_deviation"]
SUPPORTED_DEVIATION_STATUSES = ["open", "under_investigation", "root_cause_identified", "capa_raised", "closed", "recurring"]
SUPPORTED_DOCUMENT_TYPES = ["sop", "work_instruction", "form", "policy", "specification", "validation_protocol", "batch_record", "analytical_method", "risk_assessment"]
SUPPORTED_DOCUMENT_STATUSES = ["draft", "under_review", "approved", "effective", "superseded", "obsolete", "withdrawn"]
SUPPORTED_AUDIT_TYPES = ["internal", "supplier", "regulatory", "customer", "certification", "mock_audit", "unannounced"]
SUPPORTED_AUDIT_STATUSES = ["planned", "in_progress", "audit_report_pending", "findings_raised", "capa_in_progress", "closed"]
SUPPORTED_VALIDATION_TYPES = ["process_validation", "cleaning_validation", "computer_validation", "analytical_method_validation", "equipment_qualification", "facility_qualification"]
SUPPORTED_VALIDATION_STATUSES = ["planned", "protocol_approved", "execution", "report_pending", "approved", "revalidation_required"]
SUPPORTED_RISK_LEVELS = ["low", "medium", "high", "critical"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["change_reviewer", "capa_tracker", "audit_preparer", "document_controller", "risk_assessor"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"change_control": {"supported_types": SUPPORTED_CHANGE_TYPES, "supported_statuses": SUPPORTED_CHANGE_STATUSES, "impact_assessment_required": True, "risk_assessment_required": True, "approval_required": True, "effectiveness_check_required": True},
	"capa": {"supported_types": SUPPORTED_CAPA_TYPES, "supported_statuses": SUPPORTED_CAPA_STATUSES, "root_cause_required": True, "effectiveness_check_required": True, "overdue_escalation_days": 30},
	"deviations": {"supported_types": SUPPORTED_DEVIATION_TYPES, "supported_statuses": SUPPORTED_DEVIATION_STATUSES, "investigation_required": True, "capa_threshold_severity": "major", "reporting_timeline_hours": {"critical": 24, "major": 72}},
	"documents": {"supported_types": SUPPORTED_DOCUMENT_TYPES, "supported_statuses": SUPPORTED_DOCUMENT_STATUSES, "review_cycle_required": True, "approval_required": True, "version_control_required": True, "periodic_review_months": 24},
	"audits": {"supported_types": SUPPORTED_AUDIT_TYPES, "supported_statuses": SUPPORTED_AUDIT_STATUSES, "audit_plan_required": True, "findings_tracking_required": True, "capa_linkage_required": True},
	"validation": {"supported_types": SUPPORTED_VALIDATION_TYPES, "supported_statuses": SUPPORTED_VALIDATION_STATUSES, "protocol_approval_required": True, "report_approval_required": True, "revalidation_trigger_tracking": True},
	"risk": {"supported_levels": SUPPORTED_RISK_LEVELS, "risk_matrix_required": True, "mitigation_required_above": "medium", "residual_risk_assessment_required": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "gmp_compliance_required": True, "electronic_signature_required": True, "change_control_required_for_gmp_changes": True, "cross_tenant_denied": True},
	"observability": {"event_stream": QMS_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "workflow": "wflo", "compliance": "comp", "scheduler": "schd", "event_stream": "mqeb"},
	"ui": {"enable_dashboard": True, "enable_change_control": True, "enable_capa": True, "enable_deviations": True, "enable_documents": True, "enable_audits": True, "enable_validation": True, "enable_risk": True},
	"theme": {"default_theme": "pharma_qms_quality", "allow_tenant_overrides": True},
}

PROVIDES = [
	"change_control_workflow",
	"capa_management_workflow",
	"deviation_management_workflow",
	"document_control_workflow",
	"audit_management_workflow",
	"validation_lifecycle_workflow",
	"risk_management_workflow",
	"quality_metrics_workflow",
	"supplier_quality_workflow",
	"qms_review_workflow",
]
REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "comp", "schd", "mqeb"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/pharma-qms/dashboard", "component": "QmsDashboard", "permission": "pharma_qms:view", "nav_group": "Overview"},
	{"name": "change_control", "path": "/pharma-qms/change-control", "component": "ChangeControlQueue", "permission": "pharma_qms:change_control", "nav_group": "Change Control"},
	{"name": "change_detail", "path": "/pharma-qms/change-control/<id>", "component": "ChangeControlDetail", "permission": "pharma_qms:change_control", "nav_group": "Change Control"},
	{"name": "capa", "path": "/pharma-qms/capa", "component": "CapaManagement", "permission": "pharma_qms:capa", "nav_group": "CAPA"},
	{"name": "capa_detail", "path": "/pharma-qms/capa/<id>", "component": "CapaDetail", "permission": "pharma_qms:capa", "nav_group": "CAPA"},
	{"name": "deviations", "path": "/pharma-qms/deviations", "component": "DeviationQueue", "permission": "pharma_qms:deviations", "nav_group": "Deviations"},
	{"name": "documents", "path": "/pharma-qms/documents", "component": "DocumentController", "permission": "pharma_qms:documents", "nav_group": "Document Control"},
	{"name": "document_detail", "path": "/pharma-qms/documents/<id>", "component": "DocumentDetail", "permission": "pharma_qms:documents", "nav_group": "Document Control"},
	{"name": "audits", "path": "/pharma-qms/audits", "component": "AuditManagement", "permission": "pharma_qms:audits", "nav_group": "Audits"},
	{"name": "audit_detail", "path": "/pharma-qms/audits/<id>", "component": "AuditDetail", "permission": "pharma_qms:audits", "nav_group": "Audits"},
	{"name": "validation", "path": "/pharma-qms/validation", "component": "ValidationRegistry", "permission": "pharma_qms:validation", "nav_group": "Validation"},
	{"name": "risk", "path": "/pharma-qms/risk", "component": "RiskRegister", "permission": "pharma_qms:risk", "nav_group": "Risk"},
	{"name": "metrics", "path": "/pharma-qms/metrics", "component": "QualityMetrics", "permission": "pharma_qms:metrics", "nav_group": "Analytics"},
	{"name": "settings", "path": "/pharma-qms/settings", "component": "QmsSettings", "permission": "pharma_qms:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "pharma_qms_quality",
	"tokens": {
		"color.primary": "#1E40AF",
		"color.accent": "#0F766E",
		"color.success": "#15803D",
		"color.warning": "#B45309",
		"color.danger": "#B91C1C",
		"surface.canvas": "#EFF6FF",
		"surface.panel": "#FFFFFF",
		"text.primary": "#1E1B4B",
		"text.secondary": "#374151",
		"border.radius": "4px",
		"density": "compact",
	},
	"components": {
		"change_control": {"icon": "git-branch", "status_indicator": "change-type-chip"},
		"capa": {"icon": "check-square", "status_indicator": "capa-status-chip"},
		"deviations": {"icon": "alert-triangle", "status_indicator": "deviation-type-chip"},
		"documents": {"icon": "file-text", "status_indicator": "document-status-chip"},
		"audits": {"icon": "clipboard-check", "status_indicator": "audit-type-chip"},
		"validation": {"icon": "shield", "status_indicator": "validation-type-chip"},
		"risk": {"icon": "thermometer", "status_indicator": "risk-level-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": QMS_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"change_initiated", "change_approved", "change_implemented",
		"capa_raised", "capa_closed", "capa_overdue",
		"deviation_raised", "deviation_closed",
		"document_approved", "document_superseded", "document_periodic_review_due",
		"audit_completed", "audit_finding_raised",
		"validation_approved", "validation_revalidation_required",
	],
	"guardrails": [
		"change_control_required_for_gmp_impact",
		"capa_required_for_major_critical_deviations",
		"document_approval_required",
		"audit_finding_capa_linkage_required",
		"validation_protocol_approval_required",
		"electronic_signature_required",
		"cross_tenant_qms_data_denied",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "policy_required", "required_action": "attach_policy"}},
	{"name": "change_type_supported", "condition": {"operation": "initiate_change", "change_type_supported": False}, "effect": {"decision": "deny", "reason": "change_type_not_supported", "required_action": "select_supported_change_type"}},
	{"name": "change_impact_assessment_required", "condition": {"operation": "approve_change", "impact_assessed": False}, "effect": {"decision": "deny", "reason": "impact_assessment_required", "required_action": "complete_impact_assessment"}},
	{"name": "change_risk_assessment_required", "condition": {"operation": "approve_change", "risk_assessed": False}, "effect": {"decision": "deny", "reason": "risk_assessment_required", "required_action": "complete_risk_assessment"}},
	{"name": "change_approval_required", "condition": {"operation": "implement_change", "approved": False}, "effect": {"decision": "deny", "reason": "change_approval_required", "required_action": "obtain_change_approval"}},
	{"name": "change_effectiveness_check_required", "condition": {"operation": "close_change", "effectiveness_checked": False}, "effect": {"decision": "deny", "reason": "effectiveness_check_required", "required_action": "complete_effectiveness_check"}},
	{"name": "capa_type_supported", "condition": {"operation": "create_capa", "capa_type_supported": False}, "effect": {"decision": "deny", "reason": "capa_type_not_supported", "required_action": "select_supported_capa_type"}},
	{"name": "capa_root_cause_required", "condition": {"operation": "close_capa", "root_cause_identified": False}, "effect": {"decision": "deny", "reason": "root_cause_identification_required", "required_action": "identify_root_cause"}},
	{"name": "capa_effectiveness_check_required", "condition": {"operation": "close_capa", "effectiveness_checked": False}, "effect": {"decision": "deny", "reason": "capa_effectiveness_check_required", "required_action": "complete_capa_effectiveness_check"}},
	{"name": "deviation_type_supported", "condition": {"operation": "raise_deviation", "deviation_type_supported": False}, "effect": {"decision": "deny", "reason": "deviation_type_not_supported", "required_action": "select_supported_deviation_type"}},
	{"name": "deviation_investigation_required", "condition": {"operation": "close_deviation", "investigated": False}, "effect": {"decision": "deny", "reason": "investigation_required", "required_action": "complete_investigation"}},
	{"name": "critical_deviation_24h_reporting", "condition": {"operation": "raise_deviation", "severity": "critical", "within_24h": False}, "effect": {"decision": "deny", "reason": "critical_deviation_24h_required", "required_action": "expedite_deviation_report"}},
	{"name": "document_type_supported", "condition": {"operation": "create_document", "document_type_supported": False}, "effect": {"decision": "deny", "reason": "document_type_not_supported", "required_action": "select_supported_document_type"}},
	{"name": "document_approval_required", "condition": {"operation": "make_document_effective", "approved": False}, "effect": {"decision": "deny", "reason": "document_approval_required", "required_action": "obtain_document_approval"}},
	{"name": "document_version_control_required", "condition": {"operation": "update_document", "version_incremented": False}, "effect": {"decision": "deny", "reason": "version_control_required", "required_action": "increment_version"}},
	{"name": "audit_plan_required", "condition": {"operation": "start_audit", "audit_plan_present": False}, "effect": {"decision": "deny", "reason": "audit_plan_required", "required_action": "create_audit_plan"}},
	{"name": "audit_finding_capa_required", "condition": {"operation": "close_audit", "findings_have_capa": False}, "effect": {"decision": "deny", "reason": "capa_required_for_audit_findings", "required_action": "raise_capa_for_findings"}},
	{"name": "validation_protocol_approval_required", "condition": {"operation": "execute_validation", "protocol_approved": False}, "effect": {"decision": "deny", "reason": "validation_protocol_approval_required", "required_action": "approve_validation_protocol"}},
	{"name": "validation_report_approval_required", "condition": {"operation": "complete_validation", "report_approved": False}, "effect": {"decision": "deny", "reason": "validation_report_approval_required", "required_action": "approve_validation_report"}},
	{"name": "cross_tenant_denied", "condition": {"operation_type": "write", "cross_tenant": True}, "effect": {"decision": "deny", "reason": "cross_tenant_operation_denied", "required_action": "use_correct_tenant_context"}},
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
			"api_prefix": "/pharma-qms/api/v1",
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
