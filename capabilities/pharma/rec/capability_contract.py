"""Executable capability contract for APG Pharma Regulatory Compliance."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "pharma_rec"
CAPABILITY_NAME = "Regulatory Compliance"
CAPABILITY_VERSION = "1.0.0"
REC_EVENT_STREAM = "apg.pharma.rec.lifecycle"

SUPPORTED_REGULATORY_FRAMEWORKS = ["21cfr_part_11", "21cfr_part_210", "21cfr_part_211", "eu_gmp", "ich_q10", "ich_q9", "iso_13485", "mdr_eu", "pic_s", "who_gmp", "uk_gmp", "tga_gmp"]
SUPPORTED_SUBMISSION_TYPES = ["nda", "bla", "anda", "maa", "nma", "cta", "pma", "510k", "de_novo", "humanitarian_device", "annual_report", "cbee", "psur", "dsur"]
SUPPORTED_AUDIT_TYPES = ["gmp_inspection", "fda_inspection", "ema_inspection", "unannounced_inspection", "voluntary_audit", "supplier_audit", "systems_audit"]
SUPPORTED_LABEL_CHANGE_TYPES = ["labeling_change_prior_approval", "labeling_change_cbel", "annual_report_labeling", "regional_adaptation", "safety_update"]
SUPPORTED_PMS_TYPES = ["post_market_surveillance", "post_approval_study", "risk_management_plan", "rems", "eur", "periodic_benefit_risk_evaluation"]
SUPPORTED_INTEL_TYPES = ["regulatory_intelligence", "guidance_document", "draft_guidance", "industry_communication", "regulatory_change", "enforcement_action"]
SUPPORTED_COMMITMENT_STATUSES = ["open", "in_progress", "submitted", "fulfilled", "overdue", "waived"]
SUPPORTED_INSPECTION_OUTCOMES = ["no_action_indicated", "voluntary_action_indicated", "official_action_indicated", "warning_letter", "import_alert", "consent_decree"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["compliance_monitor", "inspection_preparer", "label_reviewer", "pms_analyst", "regulatory_intel_analyst"]
SUPPORTED_REGULATORY_REGIONS = ["us_fda", "eu_ema", "uk_mhra", "japan_pmda", "canada_health", "australia_tga", "brazil_anvisa", "india_cdsco", "china_nmpa", "gulf_gcc"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"compliance_frameworks": {"supported_frameworks": SUPPORTED_REGULATORY_FRAMEWORKS, "gap_assessment_required": True, "implementation_plan_required": True, "periodic_review_months": 12},
	"submissions": {"supported_types": SUPPORTED_SUBMISSION_TYPES, "supported_regions": SUPPORTED_REGULATORY_REGIONS, "dossier_reference_required": True, "submission_tracking_required": True, "commitment_tracking_required": True},
	"audits_inspections": {"supported_types": SUPPORTED_AUDIT_TYPES, "supported_outcomes": SUPPORTED_INSPECTION_OUTCOMES, "response_timeline_days": {"warning_letter": 30, "official_action_indicated": 15, "voluntary_action_indicated": 60}, "capa_required_for_findings": True},
	"labeling": {"supported_change_types": SUPPORTED_LABEL_CHANGE_TYPES, "version_control_required": True, "translations_required": True, "qp_approval_required": True, "artwork_approval_required": True},
	"pms": {"supported_types": SUPPORTED_PMS_TYPES, "protocol_required": True, "report_required": True, "signal_integration_required": True},
	"regulatory_intelligence": {"supported_types": SUPPORTED_INTEL_TYPES, "supported_regions": SUPPORTED_REGULATORY_REGIONS, "impact_assessment_required": True, "dissemination_required": True},
	"commitments": {"supported_statuses": SUPPORTED_COMMITMENT_STATUSES, "milestone_tracking_required": True, "overdue_escalation_days": 14, "regulatory_authority_tracking": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "all_frameworks_tracked": True, "inspection_readiness_required": True, "cross_tenant_denied": True},
	"observability": {"event_stream": REC_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "workflow": "wflo", "compliance": "comp", "nlp": "nlpc", "event_stream": "mqeb"},
	"ui": {"enable_dashboard": True, "enable_compliance": True, "enable_submissions": True, "enable_audits": True, "enable_labeling": True, "enable_pms": True, "enable_intel": True, "enable_commitments": True},
	"theme": {"default_theme": "pharma_rec_compliance", "allow_tenant_overrides": True},
}

PROVIDES = [
	"regulatory_compliance_monitoring_workflow",
	"inspection_readiness_workflow",
	"label_management_workflow",
	"post_market_surveillance_workflow",
	"regulatory_intelligence_workflow",
	"commitment_tracking_workflow",
	"compliance_gap_assessment_workflow",
	"inspection_response_workflow",
	"regulatory_change_impact_workflow",
	"compliance_audit_workflow",
]
REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "comp", "nlpc", "mqeb"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/pharma-rec/dashboard", "component": "RecDashboard", "permission": "pharma_rec:view", "nav_group": "Overview"},
	{"name": "compliance_register", "path": "/pharma-rec/compliance", "component": "ComplianceRegister", "permission": "pharma_rec:compliance", "nav_group": "Compliance"},
	{"name": "gap_assessment", "path": "/pharma-rec/compliance/gap", "component": "GapAssessment", "permission": "pharma_rec:gap_assessment", "nav_group": "Compliance"},
	{"name": "inspections", "path": "/pharma-rec/inspections", "component": "InspectionManagement", "permission": "pharma_rec:inspections", "nav_group": "Inspections"},
	{"name": "inspection_detail", "path": "/pharma-rec/inspections/<id>", "component": "InspectionDetail", "permission": "pharma_rec:inspections", "nav_group": "Inspections"},
	{"name": "labeling", "path": "/pharma-rec/labeling", "component": "LabelManagement", "permission": "pharma_rec:labeling", "nav_group": "Labeling"},
	{"name": "pms", "path": "/pharma-rec/pms", "component": "PostMarketSurveillance", "permission": "pharma_rec:pms", "nav_group": "Post-Market"},
	{"name": "regulatory_intel", "path": "/pharma-rec/intelligence", "component": "RegulatoryIntelligence", "permission": "pharma_rec:intel", "nav_group": "Intelligence"},
	{"name": "commitments", "path": "/pharma-rec/commitments", "component": "CommitmentTracker", "permission": "pharma_rec:commitments", "nav_group": "Commitments"},
	{"name": "submissions", "path": "/pharma-rec/submissions", "component": "SubmissionCompliance", "permission": "pharma_rec:submissions", "nav_group": "Submissions"},
	{"name": "reports", "path": "/pharma-rec/reports", "component": "ComplianceReports", "permission": "pharma_rec:reports", "nav_group": "Reporting"},
	{"name": "audit_trail", "path": "/pharma-rec/audit", "component": "ComplianceAuditTrail", "permission": "pharma_rec:audit", "nav_group": "Audit"},
	{"name": "settings", "path": "/pharma-rec/settings", "component": "RecSettings", "permission": "pharma_rec:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "pharma_rec_compliance",
	"tokens": {
		"color.primary": "#1D4ED8",
		"color.accent": "#0F766E",
		"color.success": "#15803D",
		"color.warning": "#92400E",
		"color.danger": "#991B1B",
		"surface.canvas": "#F0F4FF",
		"surface.panel": "#FFFFFF",
		"text.primary": "#1E1B4B",
		"text.secondary": "#4B5563",
		"border.radius": "6px",
		"density": "comfortable",
	},
	"components": {
		"compliance_register": {"icon": "check-circle", "status_indicator": "framework-chip"},
		"inspections": {"icon": "search", "status_indicator": "inspection-outcome-chip"},
		"labeling": {"icon": "tag", "status_indicator": "label-change-type-chip"},
		"pms": {"icon": "activity", "status_indicator": "pms-type-chip"},
		"regulatory_intel": {"icon": "globe", "status_indicator": "intel-type-chip"},
		"commitments": {"icon": "calendar-check", "status_indicator": "commitment-status-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": REC_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"compliance_gap_identified", "inspection_announced", "inspection_completed",
		"warning_letter_received", "inspection_response_submitted",
		"label_change_approved", "label_updated",
		"pms_report_submitted", "commitment_fulfilled", "commitment_overdue",
		"regulatory_change_detected", "impact_assessment_required",
	],
	"guardrails": [
		"inspection_response_timeline_enforced",
		"commitment_milestone_tracking_required",
		"label_version_control_required",
		"pms_protocol_approval_required",
		"regulatory_intel_dissemination_required",
		"compliance_gap_capa_required",
		"cross_tenant_compliance_data_denied",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "policy_required", "required_action": "attach_policy"}},
	{"name": "compliance_framework_supported", "condition": {"operation": "register_compliance", "framework_supported": False}, "effect": {"decision": "deny", "reason": "framework_not_supported", "required_action": "select_supported_framework"}},
	{"name": "compliance_gap_implementation_plan_required", "condition": {"operation": "close_gap", "implementation_plan_present": False}, "effect": {"decision": "deny", "reason": "implementation_plan_required", "required_action": "create_implementation_plan"}},
	{"name": "inspection_type_supported", "condition": {"operation": "record_inspection", "inspection_type_supported": False}, "effect": {"decision": "deny", "reason": "inspection_type_not_supported", "required_action": "select_supported_inspection_type"}},
	{"name": "inspection_outcome_supported", "condition": {"operation": "record_inspection_outcome", "outcome_supported": False}, "effect": {"decision": "deny", "reason": "inspection_outcome_not_supported", "required_action": "select_supported_outcome"}},
	{"name": "warning_letter_30d_response", "condition": {"operation": "respond_to_inspection", "outcome": "warning_letter", "within_30d": False}, "effect": {"decision": "deny", "reason": "warning_letter_30d_response_required", "required_action": "expedite_inspection_response"}},
	{"name": "inspection_capa_required", "condition": {"operation": "close_inspection", "findings_have_capa": False}, "effect": {"decision": "deny", "reason": "capa_required_for_findings", "required_action": "raise_capa_for_findings"}},
	{"name": "label_change_type_supported", "condition": {"operation": "initiate_label_change", "change_type_supported": False}, "effect": {"decision": "deny", "reason": "label_change_type_not_supported", "required_action": "select_supported_change_type"}},
	{"name": "label_qp_approval_required", "condition": {"operation": "approve_label", "qp_approved": False}, "effect": {"decision": "deny", "reason": "qp_approval_required_for_label", "required_action": "obtain_qp_approval"}},
	{"name": "label_version_control_required", "condition": {"operation": "update_label", "version_incremented": False}, "effect": {"decision": "deny", "reason": "label_version_control_required", "required_action": "increment_label_version"}},
	{"name": "pms_protocol_required", "condition": {"operation": "start_pms", "protocol_present": False}, "effect": {"decision": "deny", "reason": "pms_protocol_required", "required_action": "create_pms_protocol"}},
	{"name": "commitment_milestone_required", "condition": {"operation": "create_commitment", "milestone_present": False}, "effect": {"decision": "deny", "reason": "commitment_milestone_required", "required_action": "define_milestones"}},
	{"name": "commitment_overdue_escalation", "condition": {"operation": "check_commitment", "overdue": True, "escalated": False}, "effect": {"decision": "deny", "reason": "overdue_commitment_escalation_required", "required_action": "escalate_overdue_commitment"}},
	{"name": "regulatory_intel_impact_assessment_required", "condition": {"operation": "record_regulatory_change", "impact_assessed": False}, "effect": {"decision": "deny", "reason": "impact_assessment_required", "required_action": "complete_impact_assessment"}},
	{"name": "submission_region_supported", "condition": {"operation": "track_submission", "region_supported": False}, "effect": {"decision": "deny", "reason": "regulatory_region_not_supported", "required_action": "select_supported_region"}},
	{"name": "cross_tenant_denied", "condition": {"operation_type": "write", "cross_tenant": True}, "effect": {"decision": "deny", "reason": "cross_tenant_operation_denied", "required_action": "use_correct_tenant_context"}},
	{"name": "pms_type_supported", "condition": {"operation": "create_pms", "pms_type_supported": False}, "effect": {"decision": "deny", "reason": "pms_type_not_supported", "required_action": "select_supported_pms_type"}},
	{"name": "intel_type_supported", "condition": {"operation": "record_intel", "intel_type_supported": False}, "effect": {"decision": "deny", "reason": "intel_type_not_supported", "required_action": "select_supported_intel_type"}},
	{"name": "inspection_readiness_required", "condition": {"operation": "confirm_inspection_ready", "readiness_assessment_done": False}, "effect": {"decision": "deny", "reason": "inspection_readiness_assessment_required", "required_action": "complete_readiness_assessment"}},
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
			"api_prefix": "/pharma-rec/api/v1",
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
