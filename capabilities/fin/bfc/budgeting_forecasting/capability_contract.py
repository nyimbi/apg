"""Executable capability contract for APG budgeting and forecasting."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


SUPPORTED_BFC_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_BFC_AGENT_ROLES = [
	"budget_planning_reviewer",
	"forecast_reviewer",
	"variance_reviewer",
	"scenario_reviewer",
	"approval_reviewer",
	"cash_flow_reviewer",
]
BFC_EVENT_STREAM = "apg.fin.bfc.lifecycle"
SUPPORTED_FORECAST_METHODS = ["trend", "driver", "statistical", "manual"]
SUPPORTED_LINE_TYPES = ["revenue", "expense", "capital", "headcount"]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"budgets": {
		"owner_required": True,
		"fiscal_year_required": True,
		"currency_required": True,
		"period_dates_required": True,
		"line_required_for_submission": True,
		"high_value_review_threshold": 1000000,
	},
	"budget_lines": {
		"account_required": True,
		"line_type_required": True,
		"positive_amount_required": True,
		"supported_line_types": SUPPORTED_LINE_TYPES,
	},
	"approvals": {
		"submission_required": True,
		"approval_record_required": True,
		"separation_of_duties_required": True,
	},
	"forecasts": {
		"supported_methods": SUPPORTED_FORECAST_METHODS,
		"max_horizon_months": 60,
		"confidence_bounds": [0, 100],
		"base_budget_optional": True,
	},
	"scenarios": {
		"probability_bounds": [0, 100],
		"driver_required": True,
		"minimum_driver_count": 1,
	},
	"variances": {
		"budget_required": True,
		"actual_required": True,
		"review_threshold_percent": 10,
		"review_required_above_threshold": True,
	},
	"collaboration": {
		"enabled": True,
		"participant_required": True,
		"session_audit_required": True,
	},
	"bfc_agents": {
		"enabled": True,
		"supported_runtimes": SUPPORTED_BFC_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_BFC_AGENT_ROLES,
		"human_approval_required": True,
		"max_autonomous_scope": "recommend_validate_and_prepare",
	},
	"governance": {
		"require_tenant_context": True,
		"audit_state_changes": True,
		"policy_attached_for_writes": True,
		"segregation_of_duties": True,
	},
	"observability": {
		"event_stream": BFC_EVENT_STREAM,
		"stream_processor": "bytewax",
		"emit_budget_events": True,
		"emit_forecast_events": True,
		"emit_scenario_events": True,
		"emit_variance_events": True,
		"emit_collaboration_events": True,
		"emit_agent_events": True,
	},
	"adapters": {
		"authorization": "adapter",
		"audit": "adapter",
		"event_stream": "bytewax",
		"notification": "adapter",
		"general_ledger": "adapter",
		"accounts_payable": "adapter",
		"accounts_receivable": "adapter",
		"cash_management": "adapter",
		"business_intelligence": "adapter",
		"theme": "adapter",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_budgets": True,
		"enable_budget_lines": True,
		"enable_forecasts": True,
		"enable_scenarios": True,
		"enable_variances": True,
		"enable_approvals": True,
		"enable_collaboration": True,
		"enable_agents": True,
		"enable_settings": True,
	},
	"theme": {"default_theme": "bfc_budgeting_forecasting_control", "allow_tenant_overrides": True},
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"budgets",
		"budget_lines",
		"approvals",
		"forecasts",
		"scenarios",
		"variances",
		"collaboration",
		"bfc_agents",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {
		"tenant_id": {"type": "string", "minLength": 1},
		"budgets": {"type": "object"},
		"budget_lines": {"type": "object"},
		"approvals": {"type": "object"},
		"forecasts": {"type": "object"},
		"scenarios": {"type": "object"},
		"variances": {"type": "object"},
		"collaboration": {"type": "object"},
		"bfc_agents": {"type": "object"},
		"governance": {"type": "object"},
		"observability": {"type": "object"},
		"adapters": {"type": "object"},
		"ui": {"type": "object"},
		"theme": {"type": "object"},
	},
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "BFC operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "bfc_write_requires_policy", "description": "BFC writes require policy attachment.", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "operation_policy_required", "required_action": "attach_operation_policy"}},
	{"name": "budget_requires_owner", "description": "Budgets require an accountable owner.", "condition": {"operation": "create_budget", "budget_owner_assigned": False}, "effect": {"decision": "deny", "reason": "budget_owner_required", "required_action": "assign_budget_owner"}},
	{"name": "budget_requires_fiscal_year", "description": "Budgets require a fiscal year.", "condition": {"operation": "create_budget", "fiscal_year_present": False}, "effect": {"decision": "deny", "reason": "budget_fiscal_year_required", "required_action": "set_fiscal_year"}},
	{"name": "budget_requires_currency", "description": "Budgets require currency.", "condition": {"operation": "create_budget", "currency_present": False}, "effect": {"decision": "deny", "reason": "budget_currency_required", "required_action": "set_currency"}},
	{"name": "budget_requires_period_dates", "description": "Budgets require start and end dates.", "condition": {"operation": "create_budget", "period_dates_present": False}, "effect": {"decision": "deny", "reason": "budget_period_dates_required", "required_action": "set_period_dates"}},
	{"name": "budget_period_end_after_start", "description": "Budget period end must be after start.", "condition": {"operation": "create_budget", "period_range_valid": False}, "effect": {"decision": "deny", "reason": "budget_period_range_invalid", "required_action": "set_valid_period_range"}},
	{"name": "budget_line_requires_budget", "description": "Budget lines require a budget.", "condition": {"operation": "add_budget_line", "budget_present": False}, "effect": {"decision": "deny", "reason": "budget_line_budget_required", "required_action": "attach_budget"}},
	{"name": "budget_line_requires_account", "description": "Budget lines require an account.", "condition": {"operation": "add_budget_line", "account_present": False}, "effect": {"decision": "deny", "reason": "budget_line_account_required", "required_action": "attach_account"}},
	{"name": "budget_line_type_supported", "description": "Budget line type must be supported.", "condition": {"operation": "add_budget_line", "line_type_supported": False}, "effect": {"decision": "deny", "reason": "budget_line_type_not_supported", "required_action": "select_supported_line_type"}},
	{"name": "budget_line_amount_positive", "description": "Budget line amount must be positive.", "condition": {"operation": "add_budget_line", "line_amount_lte": 0}, "effect": {"decision": "deny", "reason": "budget_line_amount_must_be_positive", "required_action": "set_positive_line_amount"}},
	{"name": "budget_submission_requires_lines", "description": "Budget submission requires at least one line.", "condition": {"operation": "submit_budget", "line_count_lte": 0}, "effect": {"decision": "deny", "reason": "budget_lines_required_for_submission", "required_action": "add_budget_lines"}},
	{"name": "budget_approval_requires_submitted_budget", "description": "Approvals require submitted budget state.", "condition": {"operation": "approve_budget", "budget_submitted": False}, "effect": {"decision": "deny", "reason": "budget_submission_required", "required_action": "submit_budget"}},
	{"name": "budget_approval_requires_record", "description": "Budget approval requires approval evidence.", "condition": {"operation": "approve_budget", "approval_recorded": False}, "effect": {"decision": "deny", "reason": "budget_approval_record_required", "required_action": "record_approval"}},
	{"name": "budget_approval_requires_separation", "description": "Budget submitter cannot self-approve.", "condition": {"operation": "approve_budget", "separation_of_duties_passed": False}, "effect": {"decision": "deny", "reason": "separation_of_duties_required", "required_action": "select_independent_approver"}},
	{"name": "high_value_budget_requires_review", "description": "High value budgets require review.", "condition": {"operation": "approve_budget", "budget_total_gt": 1000000, "high_value_review_recorded": False}, "effect": {"decision": "require_review", "reason": "high_value_budget_review_required", "required_action": "record_high_value_review"}},
	{"name": "forecast_method_supported", "description": "Forecast method must be supported.", "condition": {"operation": "create_forecast", "forecast_method_supported": False}, "effect": {"decision": "deny", "reason": "forecast_method_not_supported", "required_action": "select_supported_forecast_method"}},
	{"name": "forecast_horizon_positive", "description": "Forecast horizon must be positive.", "condition": {"operation": "create_forecast", "horizon_months_lte": 0}, "effect": {"decision": "deny", "reason": "forecast_horizon_must_be_positive", "required_action": "set_positive_horizon"}},
	{"name": "forecast_horizon_within_limit", "description": "Forecast horizon must stay within configured limit.", "condition": {"operation": "create_forecast", "horizon_months_gt": 60}, "effect": {"decision": "deny", "reason": "forecast_horizon_exceeds_limit", "required_action": "reduce_forecast_horizon"}},
	{"name": "forecast_confidence_within_bounds", "description": "Forecast confidence must be between 0 and 100.", "condition": {"operation": "create_forecast", "confidence_out_of_bounds": True}, "effect": {"decision": "deny", "reason": "forecast_confidence_out_of_bounds", "required_action": "set_valid_confidence"}},
	{"name": "forecast_point_requires_forecast", "description": "Forecast points require a forecast.", "condition": {"operation": "record_forecast_point", "forecast_present": False}, "effect": {"decision": "deny", "reason": "forecast_point_forecast_required", "required_action": "attach_forecast"}},
	{"name": "forecast_point_requires_period", "description": "Forecast points require a period.", "condition": {"operation": "record_forecast_point", "period_present": False}, "effect": {"decision": "deny", "reason": "forecast_point_period_required", "required_action": "set_period"}},
	{"name": "scenario_requires_name", "description": "Scenarios require a name.", "condition": {"operation": "create_scenario", "scenario_name_present": False}, "effect": {"decision": "deny", "reason": "scenario_name_required", "required_action": "name_scenario"}},
	{"name": "scenario_probability_within_bounds", "description": "Scenario probability must be between 0 and 100.", "condition": {"operation": "create_scenario", "probability_out_of_bounds": True}, "effect": {"decision": "deny", "reason": "scenario_probability_out_of_bounds", "required_action": "set_valid_probability"}},
	{"name": "scenario_requires_driver", "description": "Scenarios require planning drivers.", "condition": {"operation": "create_scenario", "driver_count_lte": 0}, "effect": {"decision": "deny", "reason": "scenario_driver_required", "required_action": "attach_driver_assumptions"}},
	{"name": "variance_requires_budget", "description": "Variance analysis requires a budget.", "condition": {"operation": "record_variance", "budget_present": False}, "effect": {"decision": "deny", "reason": "variance_budget_required", "required_action": "attach_budget"}},
	{"name": "variance_requires_actual", "description": "Variance analysis requires actual amount.", "condition": {"operation": "record_variance", "actual_amount_present": False}, "effect": {"decision": "deny", "reason": "variance_actual_amount_required", "required_action": "record_actual_amount"}},
	{"name": "variance_above_threshold_requires_review", "description": "Material variance requires review.", "condition": {"operation": "record_variance", "variance_percent_abs_gt": 10, "variance_review_recorded": False}, "effect": {"decision": "require_review", "reason": "variance_review_required", "required_action": "record_variance_review"}},
	{"name": "collaboration_requires_budget", "description": "Collaboration sessions require a budget.", "condition": {"operation": "start_collaboration_session", "budget_present": False}, "effect": {"decision": "deny", "reason": "collaboration_budget_required", "required_action": "attach_budget"}},
	{"name": "collaboration_requires_participants", "description": "Collaboration sessions require participants.", "condition": {"operation": "start_collaboration_session", "participant_count_lte": 0}, "effect": {"decision": "deny", "reason": "collaboration_participant_required", "required_action": "add_participants"}},
	{"name": "bfc_batch_requires_bytewax", "description": "BFC batches require Bytewax coordination.", "condition": {"operation": "bfc_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_bfc_batch_to_bytewax"}},
	{"name": "bfc_event_requires_bytewax", "description": "BFC lifecycle events require Bytewax.", "condition": {"operation": "bfc_event", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_bfc_event_to_bytewax"}},
	{"name": "bfc_agent_runtime_supported", "description": "BFC agents must use an approved runtime.", "condition": {"operation": "register_bfc_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "bfc_agent_runtime_not_supported", "required_action": "select_supported_agent_runtime"}},
	{"name": "bfc_agent_role_supported", "description": "BFC agents must use an approved role.", "condition": {"operation": "register_bfc_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "bfc_agent_role_not_supported", "required_action": "select_supported_agent_role"}},
	{"name": "privileged_agent_bfc_action_requires_human_approval", "description": "Privileged BFC actions proposed by agents require human approval.", "condition": {"operation": "agent_bfc_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/bfc-budgeting-forecasting/dashboard", "component": "BFCDashboard", "permission": "bfc_budgeting_forecasting:view", "nav_group": "Overview"},
	{"name": "budgets", "path": "/bfc-budgeting-forecasting/budgets", "component": "BudgetWorkbench", "permission": "bfc_budgeting_forecasting:manage_budgets", "nav_group": "Budgets"},
	{"name": "budget_lines", "path": "/bfc-budgeting-forecasting/budget-lines", "component": "BudgetLineGrid", "permission": "bfc_budgeting_forecasting:manage_budgets", "nav_group": "Budgets"},
	{"name": "forecasts", "path": "/bfc-budgeting-forecasting/forecasts", "component": "ForecastWorkbench", "permission": "bfc_budgeting_forecasting:forecast", "nav_group": "Forecasts"},
	{"name": "scenarios", "path": "/bfc-budgeting-forecasting/scenarios", "component": "ScenarioWorkbench", "permission": "bfc_budgeting_forecasting:scenario", "nav_group": "Planning"},
	{"name": "variances", "path": "/bfc-budgeting-forecasting/variances", "component": "VarianceConsole", "permission": "bfc_budgeting_forecasting:analyze", "nav_group": "Analysis"},
	{"name": "approvals", "path": "/bfc-budgeting-forecasting/approvals", "component": "BudgetApprovalQueue", "permission": "bfc_budgeting_forecasting:approve", "nav_group": "Approvals"},
	{"name": "collaboration", "path": "/bfc-budgeting-forecasting/collaboration", "component": "PlanningCollaboration", "permission": "bfc_budgeting_forecasting:collaborate", "nav_group": "Planning"},
	{"name": "agents", "path": "/bfc-budgeting-forecasting/agents", "component": "BFCAgentWorkbench", "permission": "bfc_budgeting_forecasting:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/bfc-budgeting-forecasting/settings", "component": "BFCSettings", "permission": "bfc_budgeting_forecasting:admin", "nav_group": "Administration"},
]

THEME: dict[str, Any] = {
	"name": "bfc_budgeting_forecasting_control",
	"tokens": {"color.primary": "#28536B", "color.accent": "#C44536", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F7F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"},
	"components": {
		"budgets": {"icon": "calculator", "status_indicator": "budget-pill", "risk_style": "budget-band"},
		"budget_lines": {"visual": "planning-grid", "status_style": "line-chip"},
		"forecasts": {"visual": "forecast-curve", "status_style": "confidence-chip"},
		"scenarios": {"visual": "scenario-map", "status_style": "probability-chip"},
		"variances": {"visual": "variance-waterfall", "status_style": "variance-chip"},
		"approvals": {"visual": "approval-queue", "status_style": "approval-chip"},
		"collaboration": {"visual": "planning-room", "status_style": "presence-chip"},
		"agent_workbench": {"visual": "review-lane", "status_style": "agent-chip"},
	},
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "bfc_budgeting_forecasting",
		"display_name": "Budgeting and Forecasting",
		"provides": [
			"budget_planning_lifecycle",
			"budget_line_management",
			"budget_approval_workflow",
			"forecast_lifecycle",
			"scenario_planning",
			"variance_analysis_lifecycle",
			"planning_collaboration",
			"bfc_agents",
		],
		"requires": ["auth", "audl", "ntfy", "composition_events", "composition_config", "general_ledger", "accounts_payable", "accounts_receivable", "cash_management", "business_intelligence"],
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {"shell": "apg_python", "view_module": "views.py", "api_prefix": "/bfc-budgeting-forecasting/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True},
		"theme": deepcopy(THEME),
		"streaming": streaming_manifest(),
	}


def streaming_manifest() -> dict[str, Any]:
	return {
		"processor": "bytewax",
		"stream": BFC_EVENT_STREAM,
		"key": "tenant_id",
		"events": [
			"budget_created",
			"budget_line_added",
			"budget_submitted",
			"budget_approved",
			"forecast_created",
			"forecast_point_recorded",
			"scenario_created",
			"variance_recorded",
			"collaboration_session_started",
			"bfc_agent_registered",
		],
		"states": ["draft", "planning", "submitted", "approved", "active", "locked", "forecasted", "reviewed", "closed", "blocked"],
		"guardrails": [
			"bfc_batch_requires_bytewax",
			"bfc_event_requires_bytewax",
			"privileged_agent_bfc_action_requires_human_approval",
		],
	}


def event_stream_name() -> str:
	return BFC_EVENT_STREAM


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	matched: list[str] = []
	actions: list[dict[str, Any]] = []
	decision = "allow"
	for rule in RULES:
		if _matches(rule["condition"], context):
			matched.append(rule["name"])
			effect = rule["effect"]
			actions.append(effect)
			if effect["decision"] == "deny":
				decision = "deny"
			elif effect["decision"] == "require_review" and decision != "deny":
				decision = "require_review"
	return {"decision": decision, "matched_rules": matched, "actions": actions, "context": context}


def _matches(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if key.endswith("_lte"):
			if not context.get(key[:-4], 0) <= expected:
				return False
		elif key.endswith("_lt"):
			if not context.get(key[:-3], 0) < expected:
				return False
		elif key.endswith("_gte"):
			if not context.get(key[:-4], 0) >= expected:
				return False
		elif key.endswith("_gt"):
			if not context.get(key[:-3], 0) > expected:
				return False
		elif key.endswith("_ne"):
			if context.get(key[:-3]) == expected:
				return False
		elif context.get(key) != expected:
			return False
	return True


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
	for key, value in source.items():
		if isinstance(value, dict) and isinstance(target.get(key), dict):
			_deep_merge(target[key], value)
		else:
			target[key] = value
