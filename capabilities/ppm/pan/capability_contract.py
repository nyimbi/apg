"""Executable capability contract for APG Portfolio Analytics (pan)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "ppm_pan"
CAPABILITY_NAME = "Portfolio Analytics"
CAPABILITY_VERSION = "1.0.0"
PAN_EVENT_STREAM = "apg.ppm.pan.lifecycle"

# ── Supported enum values ────────────────────────────────────────────────────
SUPPORTED_PORTFOLIO_STATUSES = ["active", "archived", "under_review", "proposed", "approved", "closed"]
SUPPORTED_ALIGNMENT_DIMENSIONS = ["strategic_fit", "risk_appetite", "resource_capacity", "financial_return", "innovation_index", "sustainability_score"]
SUPPORTED_RISK_CATEGORIES = ["market", "technology", "operational", "financial", "regulatory", "reputational", "strategic"]
SUPPORTED_RETURN_METRICS = ["npv", "irr", "roi", "payback_period", "benefit_cost_ratio", "ev_ebitda"]
SUPPORTED_DASHBOARD_TYPES = ["executive_summary", "strategic_alignment", "risk_return_matrix", "capacity_heat_map", "performance_scoreboard", "pipeline_funnel", "investment_map"]
SUPPORTED_SCORING_METHODS = ["weighted_criteria", "ahp", "topsis", "multi_criteria_analysis", "balanced_scorecard"]
SUPPORTED_HEAT_MAP_DIMENSIONS = ["resource_type", "skill_category", "department", "geography", "project_phase", "time_horizon"]
SUPPORTED_PERFORMANCE_PERIODS = ["current_quarter", "ytd", "rolling_12m", "project_lifetime", "custom"]
SUPPORTED_BENCHMARK_TYPES = ["industry_average", "peer_group", "historical", "target", "best_in_class"]
SUPPORTED_REPORT_FORMATS = ["dashboard", "pdf_export", "excel_export", "api_json", "scheduled_email"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["portfolio_analyst", "alignment_scorer", "risk_assessor", "performance_reporter", "capacity_planner"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_CLASSIFICATION_LEVELS = ["internal", "confidential", "restricted", "public"]

PROVIDES = [
	"portfolio_performance_dashboard",
	"strategic_alignment_scoring",
	"risk_return_analysis",
	"capacity_heat_map",
	"portfolio_investment_analysis",
	"project_pipeline_reporting",
	"benchmark_comparison",
	"portfolio_optimisation_recommendations",
	"executive_portfolio_briefings",
	"scenario_analysis",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "nlpc", "moni", "mqeb"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/ppm-pan/dashboard", "component": "PanDashboard", "permission": "ppm_pan:view", "nav_group": "Overview"},
	{"name": "portfolios", "path": "/ppm-pan/portfolios", "component": "PortfolioList", "permission": "ppm_pan:portfolios", "nav_group": "Portfolios"},
	{"name": "portfolio_detail", "path": "/ppm-pan/portfolios/<id>", "component": "PortfolioDetail", "permission": "ppm_pan:portfolios", "nav_group": "Portfolios"},
	{"name": "strategic_alignment", "path": "/ppm-pan/alignment", "component": "StrategicAlignmentScorecard", "permission": "ppm_pan:alignment", "nav_group": "Strategy"},
	{"name": "risk_return", "path": "/ppm-pan/risk-return", "component": "RiskReturnMatrix", "permission": "ppm_pan:risk", "nav_group": "Risk & Return"},
	{"name": "capacity_heat_map", "path": "/ppm-pan/capacity", "component": "CapacityHeatMap", "permission": "ppm_pan:capacity", "nav_group": "Capacity"},
	{"name": "performance", "path": "/ppm-pan/performance", "component": "PerformanceScoreboard", "permission": "ppm_pan:performance", "nav_group": "Performance"},
	{"name": "pipeline", "path": "/ppm-pan/pipeline", "component": "ProjectPipelineFunnel", "permission": "ppm_pan:pipeline", "nav_group": "Pipeline"},
	{"name": "benchmarks", "path": "/ppm-pan/benchmarks", "component": "BenchmarkComparison", "permission": "ppm_pan:benchmarks", "nav_group": "Analysis"},
	{"name": "scenarios", "path": "/ppm-pan/scenarios", "component": "ScenarioAnalysis", "permission": "ppm_pan:scenarios", "nav_group": "Analysis"},
	{"name": "reports", "path": "/ppm-pan/reports", "component": "PortfolioReportBuilder", "permission": "ppm_pan:reports", "nav_group": "Reports"},
	{"name": "investments", "path": "/ppm-pan/investments", "component": "InvestmentMapView", "permission": "ppm_pan:investments", "nav_group": "Finance"},
	{"name": "agents", "path": "/ppm-pan/agents", "component": "PanAgentWorkbench", "permission": "ppm_pan:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/ppm-pan/settings", "component": "PanSettings", "permission": "ppm_pan:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "ppm_pan_control",
	"tokens": {
		"color.primary": "#7C3AED",
		"color.accent": "#0891B2",
		"color.success": "#15803D",
		"color.warning": "#B45309",
		"color.danger": "#B91C1C",
		"surface.canvas": "#F8FAFC",
		"surface.panel": "#FFFFFF",
		"text.primary": "#111827",
		"text.secondary": "#4B5563",
		"border.radius": "8px",
		"density": "comfortable",
	},
	"components": {
		"portfolio": {"icon": "folder-open", "status_indicator": "portfolio-status-chip"},
		"alignment_score": {"icon": "compass", "status_indicator": "alignment-score-chip"},
		"risk_return": {"icon": "activity", "status_indicator": "risk-level-chip"},
		"capacity_heat_map": {"icon": "grid", "status_indicator": "utilisation-chip"},
		"performance": {"icon": "trending-up", "status_indicator": "kpi-chip"},
		"pipeline": {"icon": "git-merge", "status_indicator": "phase-chip"},
		"investment": {"icon": "dollar-sign", "status_indicator": "return-chip"},
		"benchmark": {"icon": "bar-chart", "status_indicator": "benchmark-chip"},
		"agent": {"icon": "bot", "status_indicator": "agent-runtime-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": PAN_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"portfolio_created",
		"portfolio_updated",
		"alignment_score_calculated",
		"risk_return_analysed",
		"capacity_heat_map_generated",
		"performance_snapshot_taken",
		"benchmark_comparison_run",
		"scenario_analysed",
		"report_generated",
		"agent_registered",
	],
	"guardrails": [
		"analytics_batch_requires_bytewax",
		"portfolio_write_requires_approval",
		"cross_tenant_portfolio_access_denied",
		"classification_downgrade_denied",
		"privileged_agent_action_requires_human_approval",
		"scenario_override_requires_analyst",
	],
}

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"portfolios": {
		"supported_statuses": SUPPORTED_PORTFOLIO_STATUSES,
		"supported_classification_levels": SUPPORTED_CLASSIFICATION_LEVELS,
		"owner_required": True,
		"evidence_required": True,
	},
	"alignment": {
		"supported_dimensions": SUPPORTED_ALIGNMENT_DIMENSIONS,
		"supported_scoring_methods": SUPPORTED_SCORING_METHODS,
		"portfolio_required": True,
		"evidence_required": True,
	},
	"risk_return": {
		"supported_risk_categories": SUPPORTED_RISK_CATEGORIES,
		"supported_return_metrics": SUPPORTED_RETURN_METRICS,
		"portfolio_required": True,
		"evidence_required": True,
	},
	"heat_map": {
		"supported_dimensions": SUPPORTED_HEAT_MAP_DIMENSIONS,
		"portfolio_required": True,
	},
	"performance": {
		"supported_periods": SUPPORTED_PERFORMANCE_PERIODS,
		"supported_benchmark_types": SUPPORTED_BENCHMARK_TYPES,
		"portfolio_required": True,
	},
	"reports": {
		"supported_dashboard_types": SUPPORTED_DASHBOARD_TYPES,
		"supported_formats": SUPPORTED_REPORT_FORMATS,
	},
	"agents": {
		"enabled": True,
		"supported_runtimes": SUPPORTED_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_AGENT_ROLES,
		"name_required": True,
		"scope_required": True,
		"human_approval_required_for_privileged_actions": True,
	},
	"governance": {
		"require_tenant_context": True,
		"policy_attached_for_writes": True,
		"audit_events": True,
		"cross_tenant_portfolio_access_denied": True,
		"classification_downgrade_denied": True,
		"portfolio_write_requires_approval": True,
		"scenario_override_requires_analyst": True,
	},
	"observability": {"event_stream": PAN_EVENT_STREAM, "stream_processor": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_portfolios": True, "enable_alignment": True, "enable_risk_return": True, "enable_capacity": True, "enable_performance": True, "enable_pipeline": True, "enable_benchmarks": True, "enable_scenarios": True, "enable_agents": True},
	"theme": {"default_theme": "ppm_pan_control", "allow_tenant_overrides": True},
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "analytics_policy_required", "required_action": "attach_analytics_policy"}},
	{"name": "portfolio_status_supported", "condition": {"operation": "create_portfolio", "status_supported": False}, "effect": {"decision": "deny", "reason": "portfolio_status_not_supported", "required_action": "select_supported_status"}},
	{"name": "portfolio_owner_required", "condition": {"operation": "create_portfolio", "owner_present": False}, "effect": {"decision": "deny", "reason": "portfolio_owner_required", "required_action": "assign_portfolio_owner"}},
	{"name": "portfolio_classification_supported", "condition": {"operation": "create_portfolio", "classification_supported": False}, "effect": {"decision": "deny", "reason": "classification_not_supported", "required_action": "select_supported_classification"}},
	{"name": "portfolio_evidence_required", "condition": {"operation": "create_portfolio", "evidence_present": False}, "effect": {"decision": "deny", "reason": "portfolio_evidence_required", "required_action": "attach_portfolio_evidence"}},
	{"name": "portfolio_write_requires_approval", "condition": {"operation": "create_portfolio", "approval_present": False}, "effect": {"decision": "deny", "reason": "portfolio_write_requires_approval", "required_action": "obtain_portfolio_approval"}},
	{"name": "alignment_dimension_supported", "condition": {"operation": "score_alignment", "dimension_supported": False}, "effect": {"decision": "deny", "reason": "alignment_dimension_not_supported", "required_action": "select_supported_dimension"}},
	{"name": "alignment_scoring_method_supported", "condition": {"operation": "score_alignment", "scoring_method_supported": False}, "effect": {"decision": "deny", "reason": "scoring_method_not_supported", "required_action": "select_supported_scoring_method"}},
	{"name": "alignment_portfolio_required", "condition": {"operation": "score_alignment", "portfolio_present": False}, "effect": {"decision": "deny", "reason": "portfolio_required", "required_action": "select_portfolio"}},
	{"name": "alignment_evidence_required", "condition": {"operation": "score_alignment", "evidence_present": False}, "effect": {"decision": "deny", "reason": "alignment_evidence_required", "required_action": "attach_alignment_evidence"}},
	{"name": "risk_category_supported", "condition": {"operation": "analyse_risk_return", "risk_category_supported": False}, "effect": {"decision": "deny", "reason": "risk_category_not_supported", "required_action": "select_supported_risk_category"}},
	{"name": "return_metric_supported", "condition": {"operation": "analyse_risk_return", "return_metric_supported": False}, "effect": {"decision": "deny", "reason": "return_metric_not_supported", "required_action": "select_supported_return_metric"}},
	{"name": "risk_return_portfolio_required", "condition": {"operation": "analyse_risk_return", "portfolio_present": False}, "effect": {"decision": "deny", "reason": "portfolio_required", "required_action": "select_portfolio"}},
	{"name": "risk_return_evidence_required", "condition": {"operation": "analyse_risk_return", "evidence_present": False}, "effect": {"decision": "deny", "reason": "risk_return_evidence_required", "required_action": "attach_risk_return_evidence"}},
	{"name": "heat_map_dimension_supported", "condition": {"operation": "generate_heat_map", "dimension_supported": False}, "effect": {"decision": "deny", "reason": "heat_map_dimension_not_supported", "required_action": "select_supported_dimension"}},
	{"name": "heat_map_portfolio_required", "condition": {"operation": "generate_heat_map", "portfolio_present": False}, "effect": {"decision": "deny", "reason": "portfolio_required", "required_action": "select_portfolio"}},
	{"name": "performance_period_supported", "condition": {"operation": "snapshot_performance", "period_supported": False}, "effect": {"decision": "deny", "reason": "performance_period_not_supported", "required_action": "select_supported_period"}},
	{"name": "benchmark_type_supported", "condition": {"operation": "compare_benchmark", "benchmark_type_supported": False}, "effect": {"decision": "deny", "reason": "benchmark_type_not_supported", "required_action": "select_supported_benchmark_type"}},
	{"name": "scenario_analyst_required", "condition": {"operation": "run_scenario", "analyst_present": False}, "effect": {"decision": "deny", "reason": "scenario_override_requires_analyst", "required_action": "assign_scenario_analyst"}},
	{"name": "classification_downgrade_denied", "condition": {"operation": "update_portfolio", "classification_downgrade": True}, "effect": {"decision": "deny", "reason": "classification_downgrade_denied", "required_action": "maintain_classification_level"}},
	{"name": "cross_tenant_access_denied", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_portfolio_access_denied", "required_action": "use_own_tenant_context"}},
	{"name": "analytics_batch_requires_bytewax", "condition": {"operation": "analytics_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_analytics_batch_to_bytewax"}},
	{"name": "agent_runtime_supported", "condition": {"operation": "register_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "agent_role_supported", "condition": {"operation": "register_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "agent_name_required", "condition": {"operation": "register_agent", "agent_name_present": False}, "effect": {"decision": "deny", "reason": "agent_name_required", "required_action": "name_agent"}},
	{"name": "agent_scope_required", "condition": {"operation": "register_agent", "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "agent_scope_required", "required_action": "bound_agent_scope"}},
	{"name": "privileged_agent_action_requires_human_approval", "condition": {"operation": "agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	return {
		"capability": CAPABILITY_ID,
		"name": CAPABILITY_NAME,
		"display_name": CAPABILITY_NAME,
		"version": CAPABILITY_VERSION,
		"provides": list(PROVIDES),
		"requires": list(REQUIRES),
		"configuration": configuration,
		"configuration_schema": {
			"type": "object",
			"required": ["tenant_id", "ui", "theme"],
			"properties": {key: {"type": "object"} for key in configuration if key != "tenant_id"} | {"tenant_id": {"type": "string", "minLength": 1}},
		},
		"rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"api_prefix": "/ppm-pan/api/v1",
			"requires_theme": True,
			"view_module": "views.py",
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
