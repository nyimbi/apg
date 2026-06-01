"""Executable capability contract for APG Robo Advisory."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "fintech_robo"
CAPABILITY_NAME = "Robo Advisory"
CAPABILITY_VERSION = "1.1.0"
ROBO_EVENT_STREAM = "apg.fintech.robo.lifecycle"

SUPPORTED_RISK_PROFILES = ["conservative", "balanced", "growth", "aggressive"]
SUPPORTED_GOAL_TYPES = ["retirement", "education", "home", "wealth_growth", "income", "emergency"]
SUPPORTED_CURRENCIES = ["USD", "KES", "EUR", "GBP", "NGN", "GHS", "ZAR"]
SUPPORTED_CADENCES = ["one_time", "weekly", "monthly", "quarterly"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["robo_suitability_reviewer", "model_portfolio_reviewer", "recommendation_reviewer", "drift_reviewer", "tax_loss_reviewer", "robo_compliance_reviewer"]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"profiles": {"kyc_required": True, "suitability_required": True, "supported_risk_profiles": SUPPORTED_RISK_PROFILES},
	"goals": {"supported_goal_types": SUPPORTED_GOAL_TYPES, "supported_currencies": SUPPORTED_CURRENCIES, "positive_target_required": True, "horizon_required": True},
	"models": {"supported_risk_profiles": SUPPORTED_RISK_PROFILES, "allocation_total_percent": 100, "policy_required": True},
	"recommendations": {"analysis_required": True, "profile_goal_model_required": True},
	"automation": {"supported_cadences": SUPPORTED_CADENCES, "funding_source_required": True, "approved_recommendation_required": True},
	"drift": {"analysis_required": True, "threshold_bps": 500},
	"tax_loss": {"tax_lot_required": True, "positive_loss_required": True},
	"reviews": {"supported_statuses": SUPPORTED_REVIEW_STATUSES, "reviewer_required": True, "evidence_required": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True},
	"observability": {"event_stream": ROBO_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "nlp": "nlpc", "keys": "keym", "wealth": "fintech_wealth", "kyc": "fintech_kyc", "aml": "fintech_aml", "fraud": "fintech_fraud", "analytics": "bia", "reporting": "fin_rpt", "market_data": "market_data", "event_stream": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_profiles": True, "enable_goals": True, "enable_models": True, "enable_recommendations": True, "enable_automation": True, "enable_drift": True, "enable_tax_loss": True, "enable_reviews": True, "enable_agents": True},
	"theme": {"default_theme": "robo_advisory_control", "allow_tenant_overrides": True},
}

PROVIDES = ["robo_investor_profile_workflow", "robo_goal_plan_workflow", "robo_model_portfolio_workflow", "robo_recommendation_workflow", "robo_automation_workflow", "robo_drift_workflow", "robo_tax_loss_workflow", "robo_review_workflow", "robo_agent_workflow"]
REQUIRES = ["auth", "audl", "ntfy", "nlpc", "keym", "fintech_wealth", "fintech_kyc", "fintech_aml", "fintech_fraud", "bia", "fin_rpt"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/fintech-robo/dashboard", "component": "RoboAdvisoryDashboard", "permission": "fintech_robo:view", "nav_group": "Overview"},
	{"name": "profiles", "path": "/fintech-robo/profiles", "component": "InvestorProfileConsole", "permission": "fintech_robo:profiles", "nav_group": "Investors"},
	{"name": "goals", "path": "/fintech-robo/goals", "component": "GoalPlanConsole", "permission": "fintech_robo:goals", "nav_group": "Investors"},
	{"name": "models", "path": "/fintech-robo/models", "component": "ModelPortfolioConsole", "permission": "fintech_robo:models", "nav_group": "Models"},
	{"name": "recommendations", "path": "/fintech-robo/recommendations", "component": "RecommendationWorkbench", "permission": "fintech_robo:recommendations", "nav_group": "Advice"},
	{"name": "automation", "path": "/fintech-robo/automation", "component": "AutomationPlanConsole", "permission": "fintech_robo:automation", "nav_group": "Advice"},
	{"name": "drift", "path": "/fintech-robo/drift", "component": "DriftMonitor", "permission": "fintech_robo:drift", "nav_group": "Operations"},
	{"name": "tax_loss", "path": "/fintech-robo/tax-loss", "component": "TaxLossWorkbench", "permission": "fintech_robo:tax_loss", "nav_group": "Operations"},
	{"name": "reviews", "path": "/fintech-robo/reviews", "component": "RoboReviewConsole", "permission": "fintech_robo:reviews", "nav_group": "Governance"},
	{"name": "agents", "path": "/fintech-robo/agents", "component": "RoboAgentWorkbench", "permission": "fintech_robo:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/fintech-robo/settings", "component": "RoboSettings", "permission": "fintech_robo:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "robo_advisory_control",
	"tokens": {"color.primary": "#155E75", "color.accent": "#7C3AED", "color.success": "#15803D", "color.warning": "#A16207", "color.danger": "#B91C1C", "surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF", "text.primary": "#111827", "text.secondary": "#4B5563", "border.radius": "8px", "density": "compact"},
	"components": {"profiles": {"icon": "user-round-check", "status_indicator": "profile-chip"}, "goals": {"icon": "target", "status_indicator": "goal-chip"}, "models": {"icon": "pie-chart", "status_indicator": "model-chip"}, "recommendations": {"icon": "sparkles", "status_indicator": "recommendation-chip"}, "automation": {"icon": "repeat", "status_indicator": "automation-chip"}, "drift": {"icon": "activity", "status_indicator": "drift-chip"}, "tax_loss": {"icon": "receipt", "status_indicator": "tax-chip"}, "reviews": {"icon": "clipboard-check", "status_indicator": "review-chip"}, "agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"}},
}

STREAMING = {
	"processor": "bytewax",
	"stream": ROBO_EVENT_STREAM,
	"key": "tenant_id",
	"events": ["investor_profile_created", "goal_plan_defined", "model_portfolio_published", "recommendation_generated", "recommendation_approved", "automation_plan_configured", "drift_recorded", "tax_loss_candidate_recorded", "robo_review_recorded", "robo_agent_registered"],
	"guardrails": ["robo_batch_requires_bytewax", "privileged_robo_agent_action_requires_human_approval"],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "robo_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "policy_evidence_required", "required_action": "attach_policy_evidence"}},
	{"name": "profile_client_required", "condition": {"operation": "create_investor_profile", "client_present": False}, "effect": {"decision": "deny", "reason": "investor_client_required", "required_action": "select_client"}},
	{"name": "profile_kyc_required", "condition": {"operation": "create_investor_profile", "kyc_present": False}, "effect": {"decision": "deny", "reason": "investor_kyc_required", "required_action": "attach_kyc"}},
	{"name": "profile_suitability_required", "condition": {"operation": "create_investor_profile", "suitability_present": False}, "effect": {"decision": "deny", "reason": "investor_suitability_required", "required_action": "attach_suitability"}},
	{"name": "profile_risk_supported", "condition": {"operation": "create_investor_profile", "risk_profile_supported": False}, "effect": {"decision": "deny", "reason": "risk_profile_not_supported", "required_action": "select_supported_risk_profile"}},
	{"name": "goal_profile_required", "condition": {"operation": "define_goal_plan", "profile_present": False}, "effect": {"decision": "deny", "reason": "goal_profile_required", "required_action": "select_investor_profile"}},
	{"name": "goal_type_supported", "condition": {"operation": "define_goal_plan", "goal_type_supported": False}, "effect": {"decision": "deny", "reason": "goal_type_not_supported", "required_action": "select_supported_goal"}},
	{"name": "goal_positive_target", "condition": {"operation": "define_goal_plan", "positive_target": False}, "effect": {"decision": "deny", "reason": "positive_goal_target_required", "required_action": "set_positive_target"}},
	{"name": "goal_currency_supported", "condition": {"operation": "define_goal_plan", "currency_supported": False}, "effect": {"decision": "deny", "reason": "goal_currency_not_supported", "required_action": "select_supported_currency"}},
	{"name": "goal_horizon_required", "condition": {"operation": "define_goal_plan", "horizon_present": False}, "effect": {"decision": "deny", "reason": "goal_horizon_required", "required_action": "attach_horizon"}},
	{"name": "model_risk_supported", "condition": {"operation": "publish_model_portfolio", "risk_profile_supported": False}, "effect": {"decision": "deny", "reason": "model_risk_profile_not_supported", "required_action": "select_supported_risk_profile"}},
	{"name": "model_allocation_total", "condition": {"operation": "publish_model_portfolio", "allocation_totals_100": False}, "effect": {"decision": "deny", "reason": "model_allocation_total_must_equal_100", "required_action": "rebalance_model_allocation"}},
	{"name": "model_policy_required", "condition": {"operation": "publish_model_portfolio", "policy_present": False}, "effect": {"decision": "deny", "reason": "model_policy_required", "required_action": "attach_policy"}},
	{"name": "recommendation_profile_required", "condition": {"operation": "generate_recommendation", "profile_present": False}, "effect": {"decision": "deny", "reason": "recommendation_profile_required", "required_action": "select_profile"}},
	{"name": "recommendation_goal_required", "condition": {"operation": "generate_recommendation", "goal_present": False}, "effect": {"decision": "deny", "reason": "recommendation_goal_required", "required_action": "select_goal"}},
	{"name": "recommendation_model_required", "condition": {"operation": "generate_recommendation", "model_present": False}, "effect": {"decision": "deny", "reason": "recommendation_model_required", "required_action": "select_model"}},
	{"name": "recommendation_analysis_required", "condition": {"operation": "generate_recommendation", "analysis_present": False}, "effect": {"decision": "deny", "reason": "recommendation_analysis_required", "required_action": "attach_analysis"}},
	{"name": "recommendation_approval_required", "condition": {"operation": "approve_recommendation", "recommendation_present": False}, "effect": {"decision": "deny", "reason": "recommendation_required", "required_action": "select_recommendation"}},
	{"name": "recommendation_reviewer_required", "condition": {"operation": "approve_recommendation", "reviewer_present": False}, "effect": {"decision": "deny", "reason": "recommendation_reviewer_required", "required_action": "assign_reviewer"}},
	{"name": "automation_recommendation_required", "condition": {"operation": "configure_automation_plan", "approved_recommendation_present": False}, "effect": {"decision": "deny", "reason": "approved_recommendation_required", "required_action": "approve_recommendation"}},
	{"name": "automation_cadence_supported", "condition": {"operation": "configure_automation_plan", "cadence_supported": False}, "effect": {"decision": "deny", "reason": "automation_cadence_not_supported", "required_action": "select_supported_cadence"}},
	{"name": "automation_funding_source_required", "condition": {"operation": "configure_automation_plan", "funding_source_present": False}, "effect": {"decision": "deny", "reason": "funding_source_required", "required_action": "attach_funding_source"}},
	{"name": "drift_profile_required", "condition": {"operation": "record_drift", "profile_present": False}, "effect": {"decision": "deny", "reason": "drift_profile_required", "required_action": "select_profile"}},
	{"name": "drift_analysis_required", "condition": {"operation": "record_drift", "analysis_present": False}, "effect": {"decision": "deny", "reason": "drift_analysis_required", "required_action": "attach_analysis"}},
	{"name": "tax_profile_required", "condition": {"operation": "record_tax_loss_candidate", "profile_present": False}, "effect": {"decision": "deny", "reason": "tax_loss_profile_required", "required_action": "select_profile"}},
	{"name": "tax_lot_required", "condition": {"operation": "record_tax_loss_candidate", "tax_lot_present": False}, "effect": {"decision": "deny", "reason": "tax_lot_required", "required_action": "attach_tax_lot"}},
	{"name": "tax_positive_loss", "condition": {"operation": "record_tax_loss_candidate", "positive_loss": False}, "effect": {"decision": "deny", "reason": "positive_tax_loss_required", "required_action": "set_positive_loss"}},
	{"name": "review_status_supported", "condition": {"operation": "record_review", "status_supported": False}, "effect": {"decision": "deny", "reason": "review_status_not_supported", "required_action": "select_supported_status"}},
	{"name": "review_evidence_required", "condition": {"operation": "record_review", "evidence_present": False}, "effect": {"decision": "deny", "reason": "review_evidence_required", "required_action": "attach_review_evidence"}},
	{"name": "robo_batch_requires_bytewax", "condition": {"operation": "robo_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_robo_batch_to_bytewax"}},
	{"name": "robo_agent_runtime_supported", "condition": {"operation": "register_robo_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "robo_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "robo_agent_role_supported", "condition": {"operation": "register_robo_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "robo_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "privileged_robo_agent_action_requires_human_approval", "condition": {"operation": "robo_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
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
		"configuration_schema": {"type": "object", "required": list(configuration), "properties": {key: {"type": "object"} for key in configuration if key != "tenant_id"} | {"tenant_id": {"type": "string", "minLength": 1}}},
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {"shell": "apg_python", "requires_theme": True, "api_prefix": "/fintech-robo/api/v1", "template_roots": ["templates/", "static/"], "view_module": "views.py", "routes": deepcopy(UI_ROUTES)},
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
