"""Executable capability contract for APG Clinical Analytics."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "healthcare_ana"
CAPABILITY_NAME = "Clinical Analytics"
CAPABILITY_VERSION = "1.0.0"
ANA_EVENT_STREAM = "apg.healthcare.ana.lifecycle"

SUPPORTED_ANALYSIS_TYPES = [
	"population_health", "outcomes_measurement", "readmission_prediction",
	"quality_indicators", "cost_analysis", "utilization_review",
	"clinical_pathway", "comorbidity_analysis", "preventive_care",
]
SUPPORTED_METRIC_TYPES = [
	"mortality_rate", "readmission_rate", "length_of_stay", "complication_rate",
	"patient_satisfaction", "medication_adherence", "care_gap", "cost_per_episode",
	"preventable_admission", "falls_rate", "infection_rate",
]
SUPPORTED_POPULATION_SEGMENTS = [
	"chronic_disease", "high_risk", "post_acute", "pediatric", "geriatric",
	"maternal", "mental_health", "substance_use", "oncology", "cardiac",
]
SUPPORTED_PREDICTION_MODELS = [
	"logistic_regression", "random_forest", "gradient_boosting", "neural_network",
	"cox_proportional_hazards", "lasso_regression", "ensemble",
]
SUPPORTED_REPORT_FORMATS = ["pdf", "excel", "csv", "json", "hl7_fhir", "cda"]
SUPPORTED_AGGREGATION_PERIODS = ["daily", "weekly", "monthly", "quarterly", "annual", "rolling_30d", "rolling_90d"]
SUPPORTED_BENCHMARK_TYPES = ["national", "regional", "peer_group", "internal", "cms_star", "joint_commission"]
SUPPORTED_ALERT_SEVERITIES = ["informational", "warning", "critical"]
SUPPORTED_DATA_SOURCES = ["emr", "lab", "pharmacy", "claims", "registry", "device", "patient_reported"]
SUPPORTED_COHORT_STATUSES = ["draft", "active", "archived"]
SUPPORTED_DASHBOARD_TYPES = ["executive", "clinical", "operational", "quality", "population"]
SUPPORTED_EXPORT_PERMISSIONS = ["analyst", "clinician", "administrator", "executive"]
SUPPORTED_AGENT_ROLES = ["analytics_steward", "model_reviewer", "report_reviewer", "cohort_manager"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"analysis": {
		"supported_analysis_types": SUPPORTED_ANALYSIS_TYPES,
		"supported_metric_types": SUPPORTED_METRIC_TYPES,
		"evidence_required": True,
	},
	"population": {
		"supported_segments": SUPPORTED_POPULATION_SEGMENTS,
		"cohort_size_min": 1,
		"cohort_size_max": 1_000_000,
	},
	"prediction": {
		"supported_models": SUPPORTED_PREDICTION_MODELS,
		"min_auc": 0.70,
		"retraining_days": 90,
	},
	"reporting": {
		"supported_formats": SUPPORTED_REPORT_FORMATS,
		"supported_periods": SUPPORTED_AGGREGATION_PERIODS,
		"supported_benchmarks": SUPPORTED_BENCHMARK_TYPES,
	},
	"governance": {
		"require_tenant_context": True,
		"policy_attached_for_writes": True,
		"audit_events": True,
		"phi_de_identification_required": True,
		"cross_tenant_data_denied": True,
		"model_deployment_requires_approval": True,
		"benchmark_source_required": True,
	},
	"observability": {
		"event_stream": ANA_EVENT_STREAM,
		"stream_processor": "bytewax",
	},
	"adapters": {
		"auth": "auth",
		"audit": "audl",
		"notifications": "ntfy",
		"nlp": "nlpc",
		"monitoring": "moni",
		"event_stream": "bytewax",
		"scheduler": "schd",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_cohorts": True,
		"enable_metrics": True,
		"enable_predictions": True,
		"enable_reports": True,
		"enable_benchmarks": True,
	},
	"theme": {
		"default_theme": "healthcare_ana_clinical",
		"allow_tenant_overrides": True,
	},
	"agents": {
		"enabled": True,
		"supported_runtimes": SUPPORTED_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_AGENT_ROLES,
		"human_approval_required_for_privileged_actions": True,
	},
}

PROVIDES = [
	"population_health_analytics",
	"clinical_outcomes_measurement",
	"readmission_prediction",
	"quality_indicator_tracking",
	"cohort_management",
	"clinical_benchmarking",
	"analytics_report_generation",
	"care_gap_identification",
	"predictive_model_management",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "nlpc", "moni", "mqeb", "schd"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/healthcare-ana/dashboard", "component": "AnaDashboard", "permission": "healthcare_ana:view", "nav_group": "Overview"},
	{"name": "population", "path": "/healthcare-ana/population", "component": "AnaPopulationHealth", "permission": "healthcare_ana:population", "nav_group": "Analysis"},
	{"name": "cohorts", "path": "/healthcare-ana/cohorts", "component": "AnaCohortManager", "permission": "healthcare_ana:cohorts", "nav_group": "Analysis"},
	{"name": "cohort_detail", "path": "/healthcare-ana/cohorts/<id>", "component": "AnaCohortDetail", "permission": "healthcare_ana:cohorts", "nav_group": "Analysis"},
	{"name": "metrics", "path": "/healthcare-ana/metrics", "component": "AnaMetricLedger", "permission": "healthcare_ana:metrics", "nav_group": "Quality"},
	{"name": "predictions", "path": "/healthcare-ana/predictions", "component": "AnaPredictionWorkbench", "permission": "healthcare_ana:predictions", "nav_group": "Predictive"},
	{"name": "benchmarks", "path": "/healthcare-ana/benchmarks", "component": "AnaBenchmarkConsole", "permission": "healthcare_ana:benchmarks", "nav_group": "Quality"},
	{"name": "care_gaps", "path": "/healthcare-ana/care-gaps", "component": "AnaCareGapConsole", "permission": "healthcare_ana:care_gaps", "nav_group": "Quality"},
	{"name": "reports", "path": "/healthcare-ana/reports", "component": "AnaReportBuilder", "permission": "healthcare_ana:reports", "nav_group": "Reporting"},
	{"name": "report_detail", "path": "/healthcare-ana/reports/<id>", "component": "AnaReportDetail", "permission": "healthcare_ana:reports", "nav_group": "Reporting"},
	{"name": "quality_indicators", "path": "/healthcare-ana/quality", "component": "AnaQualityIndicators", "permission": "healthcare_ana:quality", "nav_group": "Quality"},
	{"name": "agents", "path": "/healthcare-ana/agents", "component": "AnaAgentWorkbench", "permission": "healthcare_ana:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/healthcare-ana/settings", "component": "AnaSettings", "permission": "healthcare_ana:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "healthcare_ana_clinical",
	"tokens": {
		"color.primary": "#0369A1",
		"color.accent": "#0891B2",
		"color.success": "#166534",
		"color.warning": "#A16207",
		"color.danger": "#B91C1C",
		"surface.canvas": "#F0F9FF",
		"surface.panel": "#FFFFFF",
		"text.primary": "#0C4A6E",
		"text.secondary": "#075985",
		"border.radius": "6px",
		"density": "comfortable",
	},
	"components": {
		"cohorts": {"icon": "users", "status_indicator": "cohort-status-chip"},
		"metrics": {"icon": "bar-chart-2", "status_indicator": "metric-trend-chip"},
		"predictions": {"icon": "trending-up", "status_indicator": "model-status-chip"},
		"benchmarks": {"icon": "award", "status_indicator": "benchmark-chip"},
		"reports": {"icon": "file-text", "status_indicator": "report-status-chip"},
		"quality_indicators": {"icon": "check-circle", "status_indicator": "quality-chip"},
		"care_gaps": {"icon": "alert-circle", "status_indicator": "gap-severity-chip"},
		"agents": {"icon": "cpu", "status_indicator": "agent-runtime-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": ANA_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"cohort_created", "cohort_updated", "metric_recorded",
		"prediction_generated", "benchmark_updated", "care_gap_identified",
		"report_generated", "quality_indicator_updated", "model_deployed",
	],
	"guardrails": [
		"phi_de_identification_required_for_export",
		"cross_tenant_data_access_denied",
		"model_deployment_requires_approval",
		"benchmark_source_must_be_validated",
		"privileged_agent_action_requires_human_approval",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "write_policy_required", "required_action": "attach_write_policy"}},
	{"name": "analysis_type_supported", "condition": {"operation": "create_analysis", "analysis_type_supported": False}, "effect": {"decision": "deny", "reason": "analysis_type_not_supported", "required_action": "select_supported_analysis_type"}},
	{"name": "cohort_requires_segment", "condition": {"operation": "create_cohort", "segment_present": False}, "effect": {"decision": "deny", "reason": "cohort_segment_required", "required_action": "specify_population_segment"}},
	{"name": "metric_type_supported", "condition": {"operation": "record_metric", "metric_type_supported": False}, "effect": {"decision": "deny", "reason": "metric_type_not_supported", "required_action": "select_supported_metric_type"}},
	{"name": "prediction_model_supported", "condition": {"operation": "deploy_model", "model_type_supported": False}, "effect": {"decision": "deny", "reason": "prediction_model_not_supported", "required_action": "select_supported_model"}},
	{"name": "model_deployment_requires_approval", "condition": {"operation": "deploy_model", "approval_present": False}, "effect": {"decision": "deny", "reason": "model_deployment_approval_required", "required_action": "obtain_deployment_approval"}},
	{"name": "phi_export_requires_deidentification", "condition": {"operation": "export_data", "phi_deidentified": False}, "effect": {"decision": "deny", "reason": "phi_deidentification_required", "required_action": "apply_deidentification"}},
	{"name": "benchmark_source_required", "condition": {"operation": "add_benchmark", "source_present": False}, "effect": {"decision": "deny", "reason": "benchmark_source_required", "required_action": "specify_benchmark_source"}},
	{"name": "benchmark_type_supported", "condition": {"operation": "add_benchmark", "benchmark_type_supported": False}, "effect": {"decision": "deny", "reason": "benchmark_type_not_supported", "required_action": "select_supported_benchmark_type"}},
	{"name": "cross_tenant_data_denied", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_data_access_prohibited", "required_action": "use_tenant_scoped_query"}},
	{"name": "report_format_supported", "condition": {"operation": "generate_report", "format_supported": False}, "effect": {"decision": "deny", "reason": "report_format_not_supported", "required_action": "select_supported_format"}},
	{"name": "cohort_min_size", "condition": {"operation": "create_cohort", "cohort_size_valid": False}, "effect": {"decision": "deny", "reason": "cohort_size_invalid", "required_action": "adjust_cohort_criteria"}},
	{"name": "aggregation_period_supported", "condition": {"operation": "aggregate_metrics", "period_supported": False}, "effect": {"decision": "deny", "reason": "aggregation_period_not_supported", "required_action": "select_supported_period"}},
	{"name": "care_gap_evidence_required", "condition": {"operation": "identify_care_gap", "evidence_present": False}, "effect": {"decision": "deny", "reason": "care_gap_evidence_required", "required_action": "attach_clinical_evidence"}},
	{"name": "quality_indicator_source_required", "condition": {"operation": "record_quality_indicator", "source_present": False}, "effect": {"decision": "deny", "reason": "quality_indicator_source_required", "required_action": "specify_data_source"}},
	{"name": "data_source_supported", "condition": {"operation": "configure_data_source", "data_source_supported": False}, "effect": {"decision": "deny", "reason": "data_source_not_supported", "required_action": "select_supported_data_source"}},
	{"name": "agent_privileged_action_requires_approval", "condition": {"agent_action": True, "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "privileged_agent_action_requires_human_approval", "required_action": "record_human_approval"}},
	{"name": "scheduled_report_requires_scheduler", "condition": {"operation": "schedule_report", "scheduler_configured": False}, "effect": {"decision": "deny", "reason": "scheduler_not_configured", "required_action": "configure_scheduler_adapter"}},
	{"name": "prediction_auc_threshold", "condition": {"operation": "deploy_model", "auc_above_threshold": False}, "effect": {"decision": "deny", "reason": "model_auc_below_minimum", "required_action": "improve_model_performance"}},
	{"name": "model_retraining_overdue", "condition": {"operation": "generate_prediction", "model_retraining_overdue": True}, "effect": {"decision": "warn", "reason": "model_retraining_overdue", "required_action": "schedule_model_retraining"}},
	{"name": "cohort_delete_requires_no_active_analyses", "condition": {"operation": "delete_cohort", "active_analyses_exist": True}, "effect": {"decision": "deny", "reason": "cohort_has_active_analyses", "required_action": "complete_or_archive_analyses"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	"""Return the full capability contract for the given tenant."""
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
				"analysis": {"type": "object"},
				"population": {"type": "object"},
				"prediction": {"type": "object"},
			},
		},
		"rule_engine": {
			"type": "deterministic",
			"default_decision": "allow",
			"rules": RULES,
		},
		"ui": {
			"shell": "apg_python",
			"requires_theme": True,
			"template_roots": ["healthcare/ana/templates"],
			"routes": UI_ROUTES,
		},
		"theme": THEME,
		"streaming": STREAMING,
		"provides": PROVIDES,
		"requires": REQUIRES,
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate deterministic rules against the provided context dict."""
	for rule in RULES:
		cond = rule["condition"]
		if all(context.get(k) == v for k, v in cond.items()):
			effect = rule["effect"]
			return {
				"rule": rule["name"],
				"decision": effect["decision"],
				"reason": effect["reason"],
				"required_action": effect.get("required_action"),
			}
	return {"rule": None, "decision": "allow", "reason": "no_rule_matched", "required_action": None}
