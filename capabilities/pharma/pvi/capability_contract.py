"""Executable capability contract for APG Pharma Pharmacovigilance."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "pharma_pvi"
CAPABILITY_NAME = "Pharmacovigilance"
CAPABILITY_VERSION = "1.0.0"
PVI_EVENT_STREAM = "apg.pharma.pvi.lifecycle"

SUPPORTED_AE_SOURCES = ["spontaneous", "clinical_trial", "literature", "regulatory_authority", "health_authority", "patient_support_programme", "market_research", "social_media", "healthcare_professional", "patient"]
SUPPORTED_CASE_TYPES = ["adverse_event", "serious_adverse_event", "adverse_drug_reaction", "susar", "lack_of_efficacy", "off_label_use", "overdose", "misuse", "abuse", "occupational_exposure"]
SUPPORTED_SERIOUSNESS_CRITERIA = ["death", "life_threatening", "hospitalisation", "prolonged_hospitalisation", "congenital_anomaly", "medically_significant"]
SUPPORTED_CAUSALITY_ASSESSMENTS = ["certain", "probable", "possible", "unlikely", "conditional", "unassessable"]
SUPPORTED_CASE_STATUSES = ["new", "in_progress", "pending_follow_up", "closed_valid", "closed_invalid", "duplicate", "nullified"]
SUPPORTED_SIGNAL_TYPES = ["new_safety_signal", "strengthened_signal", "weakened_signal", "refuted_signal", "closed_signal"]
SUPPORTED_PSUR_TYPES = ["psur", "pbrer", "dsur", "asr", "par"]
SUPPORTED_REGULATORY_DATABASES = ["eudravigilance", "vaers", "fda_faers", "mhra_yellow_card", "who_vigibase", "Health_Canada_mdr", "tga_daen"]
SUPPORTED_REPORTING_TIMELINES = ["7day_expedited", "15day_expedited", "30day_psur", "90day_periodic", "annual"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["ae_triager", "case_processor", "signal_detector", "literature_screener", "psur_compiler"]
SUPPORTED_MEDDRA_LEVELS = ["soc", "hlgt", "hlt", "pt", "llt"]
SUPPORTED_FOLLOW_UP_TYPES = ["requested", "received", "lost_to_follow_up", "not_required"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"adverse_events": {"supported_sources": SUPPORTED_AE_SOURCES, "supported_types": SUPPORTED_CASE_TYPES, "supported_seriousness": SUPPORTED_SERIOUSNESS_CRITERIA, "meddra_required": True, "narrative_required": True, "causality_required": True},
	"case_processing": {"supported_statuses": SUPPORTED_CASE_STATUSES, "duplicate_check_required": True, "medical_review_required": True, "qc_required": True, "reporting_timelines": {"7day_expedited": 7, "15day_expedited": 15}},
	"signals": {"supported_types": SUPPORTED_SIGNAL_TYPES, "disproportionality_analysis": True, "literature_review_required": True, "clinical_review_required": True, "phvwp_submission": True},
	"psur": {"supported_types": SUPPORTED_PSUR_TYPES, "ibrd_required": True, "signal_evaluation_required": True, "benefit_risk_required": True, "submission_timeline_days": 60},
	"regulatory_reporting": {"supported_databases": SUPPORTED_REGULATORY_DATABASES, "supported_timelines": SUPPORTED_REPORTING_TIMELINES, "e2b_r3_required": True, "cover_letter_required": True},
	"literature": {"database_screening_required": True, "screening_frequency_days": 7, "deduplication_required": True, "relevance_assessment_required": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "gdp_compliance_required": False, "icsr_submission_required": True, "signal_tracking_required": True, "cross_tenant_denied": True, "meddra_coding_required": True},
	"observability": {"event_stream": PVI_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "workflow": "wflo", "compliance": "comp", "nlp": "nlpc", "event_stream": "mqeb"},
	"ui": {"enable_dashboard": True, "enable_case_intake": True, "enable_case_processing": True, "enable_signals": True, "enable_psur": True, "enable_regulatory_reporting": True, "enable_literature": True},
	"theme": {"default_theme": "pharma_pvi_safety", "allow_tenant_overrides": True},
}

PROVIDES = [
	"adverse_event_collection_workflow",
	"case_processing_workflow",
	"signal_detection_workflow",
	"psur_generation_workflow",
	"regulatory_reporting_workflow",
	"literature_screening_workflow",
	"benefit_risk_assessment_workflow",
	"follow_up_management_workflow",
	"duplicate_detection_workflow",
	"meddra_coding_workflow",
]
REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "comp", "nlpc", "mqeb"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/pharma-pvi/dashboard", "component": "PviDashboard", "permission": "pharma_pvi:view", "nav_group": "Overview"},
	{"name": "case_intake", "path": "/pharma-pvi/cases/intake", "component": "CaseIntakeForm", "permission": "pharma_pvi:cases", "nav_group": "Cases"},
	{"name": "cases", "path": "/pharma-pvi/cases", "component": "CaseQueue", "permission": "pharma_pvi:cases", "nav_group": "Cases"},
	{"name": "case_detail", "path": "/pharma-pvi/cases/<id>", "component": "CaseDetail", "permission": "pharma_pvi:cases", "nav_group": "Cases"},
	{"name": "follow_up", "path": "/pharma-pvi/cases/follow-up", "component": "FollowUpQueue", "permission": "pharma_pvi:follow_up", "nav_group": "Cases"},
	{"name": "signals", "path": "/pharma-pvi/signals", "component": "SignalManagement", "permission": "pharma_pvi:signals", "nav_group": "Signal Detection"},
	{"name": "signal_detail", "path": "/pharma-pvi/signals/<id>", "component": "SignalDetail", "permission": "pharma_pvi:signals", "nav_group": "Signal Detection"},
	{"name": "literature", "path": "/pharma-pvi/literature", "component": "LiteratureScreening", "permission": "pharma_pvi:literature", "nav_group": "Literature"},
	{"name": "psur", "path": "/pharma-pvi/psur", "component": "PsurWorkbench", "permission": "pharma_pvi:psur", "nav_group": "Periodic Reports"},
	{"name": "regulatory_reporting", "path": "/pharma-pvi/reporting", "component": "RegulatoryReportingConsole", "permission": "pharma_pvi:reporting", "nav_group": "Reporting"},
	{"name": "submissions", "path": "/pharma-pvi/submissions", "component": "SubmissionTracker", "permission": "pharma_pvi:submissions", "nav_group": "Reporting"},
	{"name": "metrics", "path": "/pharma-pvi/metrics", "component": "PviMetrics", "permission": "pharma_pvi:metrics", "nav_group": "Analytics"},
	{"name": "audit_trail", "path": "/pharma-pvi/audit", "component": "PviAuditTrail", "permission": "pharma_pvi:audit", "nav_group": "Compliance"},
	{"name": "settings", "path": "/pharma-pvi/settings", "component": "PviSettings", "permission": "pharma_pvi:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "pharma_pvi_safety",
	"tokens": {
		"color.primary": "#7C3AED",
		"color.accent": "#DC2626",
		"color.success": "#15803D",
		"color.warning": "#D97706",
		"color.danger": "#991B1B",
		"surface.canvas": "#FAF5FF",
		"surface.panel": "#FFFFFF",
		"text.primary": "#1E1B4B",
		"text.secondary": "#4B5563",
		"border.radius": "6px",
		"density": "compact",
	},
	"components": {
		"cases": {"icon": "file-warning", "status_indicator": "case-status-chip"},
		"signals": {"icon": "radar", "status_indicator": "signal-type-chip"},
		"literature": {"icon": "book-open", "status_indicator": "literature-status-chip"},
		"psur": {"icon": "file-text", "status_indicator": "psur-type-chip"},
		"regulatory_reporting": {"icon": "send", "status_indicator": "reporting-timeline-chip"},
		"follow_up": {"icon": "clock", "status_indicator": "follow-up-type-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": PVI_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"ae_received", "case_created", "case_processed", "case_closed",
		"duplicate_detected", "follow_up_requested", "follow_up_received",
		"signal_detected", "signal_evaluated", "signal_closed",
		"literature_match_found", "psur_submitted", "icsr_submitted",
		"7day_report_filed", "15day_report_filed",
	],
	"guardrails": [
		"7day_expedited_reporting_enforced",
		"15day_expedited_reporting_enforced",
		"meddra_coding_required",
		"duplicate_check_required",
		"medical_review_required_for_serious",
		"e2b_r3_format_required",
		"cross_tenant_case_data_denied",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "policy_required", "required_action": "attach_policy"}},
	{"name": "ae_source_supported", "condition": {"operation": "create_case", "ae_source_supported": False}, "effect": {"decision": "deny", "reason": "ae_source_not_supported", "required_action": "select_supported_ae_source"}},
	{"name": "case_type_supported", "condition": {"operation": "create_case", "case_type_supported": False}, "effect": {"decision": "deny", "reason": "case_type_not_supported", "required_action": "select_supported_case_type"}},
	{"name": "meddra_coding_required", "condition": {"operation": "process_case", "meddra_coded": False}, "effect": {"decision": "deny", "reason": "meddra_coding_required", "required_action": "apply_meddra_coding"}},
	{"name": "narrative_required", "condition": {"operation": "process_case", "narrative_present": False}, "effect": {"decision": "deny", "reason": "case_narrative_required", "required_action": "write_case_narrative"}},
	{"name": "causality_required", "condition": {"operation": "process_case", "causality_assessed": False}, "effect": {"decision": "deny", "reason": "causality_assessment_required", "required_action": "assess_causality"}},
	{"name": "duplicate_check_required", "condition": {"operation": "process_case", "duplicate_check_done": False}, "effect": {"decision": "deny", "reason": "duplicate_check_required", "required_action": "perform_duplicate_check"}},
	{"name": "medical_review_required_for_serious", "condition": {"operation": "close_case", "case_serious": True, "medical_reviewed": False}, "effect": {"decision": "deny", "reason": "medical_review_required", "required_action": "obtain_medical_review"}},
	{"name": "7day_expedited_reporting_required", "condition": {"operation": "submit_icsr", "case_type": "susar", "within_7d": False}, "effect": {"decision": "deny", "reason": "7day_expedited_required", "required_action": "expedite_icsr_submission"}},
	{"name": "15day_expedited_reporting_required", "condition": {"operation": "submit_icsr", "case_serious": True, "within_15d": False}, "effect": {"decision": "deny", "reason": "15day_expedited_required", "required_action": "expedite_icsr_submission"}},
	{"name": "e2b_r3_format_required", "condition": {"operation": "submit_icsr", "e2b_r3_formatted": False}, "effect": {"decision": "deny", "reason": "e2b_r3_format_required", "required_action": "format_as_e2b_r3"}},
	{"name": "regulatory_database_supported", "condition": {"operation": "submit_icsr", "regulatory_database_supported": False}, "effect": {"decision": "deny", "reason": "regulatory_database_not_supported", "required_action": "select_supported_database"}},
	{"name": "signal_type_supported", "condition": {"operation": "create_signal", "signal_type_supported": False}, "effect": {"decision": "deny", "reason": "signal_type_not_supported", "required_action": "select_supported_signal_type"}},
	{"name": "signal_clinical_review_required", "condition": {"operation": "close_signal", "clinical_reviewed": False}, "effect": {"decision": "deny", "reason": "clinical_review_required", "required_action": "obtain_clinical_review"}},
	{"name": "psur_ibrd_required", "condition": {"operation": "create_psur", "ibrd_attached": False}, "effect": {"decision": "deny", "reason": "ibrd_required", "required_action": "attach_ibrd"}},
	{"name": "psur_benefit_risk_required", "condition": {"operation": "submit_psur", "benefit_risk_assessed": False}, "effect": {"decision": "deny", "reason": "benefit_risk_assessment_required", "required_action": "complete_benefit_risk_assessment"}},
	{"name": "literature_screening_frequency_required", "condition": {"operation": "close_literature_cycle", "screening_current": False}, "effect": {"decision": "deny", "reason": "literature_screening_overdue", "required_action": "perform_literature_screening"}},
	{"name": "cross_tenant_denied", "condition": {"operation_type": "write", "cross_tenant": True}, "effect": {"decision": "deny", "reason": "cross_tenant_operation_denied", "required_action": "use_correct_tenant_context"}},
	{"name": "case_status_supported", "condition": {"operation": "update_case_status", "case_status_supported": False}, "effect": {"decision": "deny", "reason": "case_status_not_supported", "required_action": "select_supported_case_status"}},
	{"name": "seriousness_criteria_supported", "condition": {"operation": "assess_seriousness", "seriousness_criteria_supported": False}, "effect": {"decision": "deny", "reason": "seriousness_criteria_not_supported", "required_action": "select_supported_seriousness_criteria"}},
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
			"api_prefix": "/pharma-pvi/api/v1",
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
