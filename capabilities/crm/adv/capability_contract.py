"""Executable capability contract for APG advanced CRM analytics."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


SUPPORTED_CRM_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_CRM_AGENT_ROLES = [
	"pipeline_analyst",
	"lead_quality_reviewer",
	"account_strategist",
	"forecast_reviewer",
	"campaign_reviewer",
	"privacy_reviewer",
]
CRM_EVENT_STREAM = "apg.crm.adv.lifecycle"


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"accounts": {"owner_required": True, "segment_required": True, "territory_supported": True},
	"contacts": {"consent_required_for_outreach": True, "relationship_mapping_supported": True},
	"leads": {"source_required": True, "score_required_for_assignment": True, "assignment_policy_required": True},
	"opportunities": {"account_required": True, "stage_required": True, "amount_required": True, "close_date_required": True},
	"activities": {"owner_required": True, "next_step_required_for_open_pipeline": True},
	"analytics": {"forecast_evidence_required": True, "confidence_required": True, "pipeline_health_supported": True},
	"campaigns": {"audience_required": True, "consent_required": True, "budget_review_required": True},
	"crm_agents": {
		"enabled": True,
		"supported_runtimes": SUPPORTED_CRM_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_CRM_AGENT_ROLES,
		"human_approval_required": True,
		"max_autonomous_scope": "recommend_validate_and_prepare",
	},
	"governance": {
		"require_tenant_context": True,
		"audit_state_changes": True,
		"policy_attached_for_writes": True,
		"privacy_review_for_bulk_outreach": True,
	},
	"observability": {
		"event_stream": CRM_EVENT_STREAM,
		"stream_processor": "bytewax",
		"emit_account_events": True,
		"emit_lead_events": True,
		"emit_opportunity_events": True,
		"emit_agent_events": True,
	},
	"adapters": {
		"authorization": "adapter",
		"audit": "adapter",
		"event_stream": "bytewax",
		"notification": "adapter",
		"customer_data": "adapter",
		"theme": "adapter",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_accounts": True,
		"enable_contacts": True,
		"enable_leads": True,
		"enable_pipeline": True,
		"enable_activities": True,
		"enable_campaigns": True,
		"enable_forecasts": True,
		"enable_agents": True,
		"enable_settings": True,
	},
	"theme": {"default_theme": "crm_adv_control", "allow_tenant_overrides": True},
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"accounts",
		"contacts",
		"leads",
		"opportunities",
		"activities",
		"analytics",
		"campaigns",
		"crm_agents",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {
		"tenant_id": {"type": "string", "minLength": 1},
		"accounts": {"type": "object"},
		"contacts": {"type": "object"},
		"leads": {"type": "object"},
		"opportunities": {"type": "object"},
		"activities": {"type": "object"},
		"analytics": {"type": "object"},
		"campaigns": {"type": "object"},
		"crm_agents": {"type": "object"},
		"governance": {"type": "object"},
		"observability": {"type": "object"},
		"adapters": {"type": "object"},
		"ui": {"type": "object"},
		"theme": {"type": "object"},
	},
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "Advanced CRM operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "crm_write_requires_policy", "description": "Advanced CRM write operations require policy attachment.", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "operation_policy_required", "required_action": "attach_operation_policy"}},
	{"name": "account_requires_owner", "description": "Accounts require an owner.", "condition": {"operation": "create_account", "account_owner_assigned": False}, "effect": {"decision": "deny", "reason": "account_owner_required", "required_action": "assign_account_owner"}},
	{"name": "account_requires_segment", "description": "Accounts require a segment.", "condition": {"operation": "create_account", "account_segment_present": False}, "effect": {"decision": "deny", "reason": "account_segment_required", "required_action": "set_account_segment"}},
	{"name": "contact_outreach_requires_consent", "description": "Outreach contacts require consent evidence.", "condition": {"operation": "create_contact", "outreach_enabled": True, "consent_recorded": False}, "effect": {"decision": "deny", "reason": "contact_consent_required", "required_action": "record_contact_consent"}},
	{"name": "lead_requires_source", "description": "Leads require a source.", "condition": {"operation": "create_lead", "lead_source_present": False}, "effect": {"decision": "deny", "reason": "lead_source_required", "required_action": "set_lead_source"}},
	{"name": "lead_assignment_requires_score", "description": "Lead assignment requires a score.", "condition": {"operation": "assign_lead", "lead_score_present": False}, "effect": {"decision": "deny", "reason": "lead_score_required", "required_action": "score_lead"}},
	{"name": "lead_assignment_requires_policy", "description": "Lead assignment requires an assignment policy.", "condition": {"operation": "assign_lead", "assignment_policy_present": False}, "effect": {"decision": "deny", "reason": "lead_assignment_policy_required", "required_action": "attach_assignment_policy"}},
	{"name": "opportunity_requires_account", "description": "Opportunities require an account.", "condition": {"operation": "create_opportunity", "account_present": False}, "effect": {"decision": "deny", "reason": "opportunity_account_required", "required_action": "attach_account"}},
	{"name": "opportunity_requires_stage", "description": "Opportunities require a sales stage.", "condition": {"operation": "create_opportunity", "stage_present": False}, "effect": {"decision": "deny", "reason": "opportunity_stage_required", "required_action": "set_sales_stage"}},
	{"name": "opportunity_requires_amount", "description": "Opportunities require an amount.", "condition": {"operation": "create_opportunity", "amount_present": False}, "effect": {"decision": "deny", "reason": "opportunity_amount_required", "required_action": "set_opportunity_amount"}},
	{"name": "opportunity_amount_must_be_positive", "description": "Opportunities require positive amount.", "condition": {"operation": "create_opportunity", "amount_lte": 0}, "effect": {"decision": "deny", "reason": "opportunity_amount_positive_required", "required_action": "set_positive_opportunity_amount"}},
	{"name": "opportunity_requires_close_date", "description": "Opportunities require a close date.", "condition": {"operation": "create_opportunity", "close_date_present": False}, "effect": {"decision": "deny", "reason": "opportunity_close_date_required", "required_action": "set_close_date"}},
	{"name": "activity_requires_owner", "description": "CRM activities require an owner.", "condition": {"operation": "record_activity", "activity_owner_assigned": False}, "effect": {"decision": "deny", "reason": "activity_owner_required", "required_action": "assign_activity_owner"}},
	{"name": "open_pipeline_requires_next_step", "description": "Open pipeline opportunities require a next step.", "condition": {"operation": "record_activity", "open_pipeline": True, "next_step_present": False}, "effect": {"decision": "require_review", "reason": "next_step_required", "required_action": "record_next_step"}},
	{"name": "forecast_requires_evidence", "description": "Forecasts require evidence.", "condition": {"operation": "record_forecast", "forecast_evidence_present": False}, "effect": {"decision": "deny", "reason": "forecast_evidence_required", "required_action": "attach_forecast_evidence"}},
	{"name": "forecast_requires_confidence", "description": "Forecasts require confidence level.", "condition": {"operation": "record_forecast", "confidence_present": False}, "effect": {"decision": "deny", "reason": "forecast_confidence_required", "required_action": "set_forecast_confidence"}},
	{"name": "forecast_confidence_minimum", "description": "Forecast confidence cannot be below zero.", "condition": {"operation": "record_forecast", "confidence_lt": 0}, "effect": {"decision": "deny", "reason": "forecast_confidence_out_of_range", "required_action": "set_forecast_confidence_between_zero_and_one"}},
	{"name": "forecast_confidence_maximum", "description": "Forecast confidence cannot exceed one.", "condition": {"operation": "record_forecast", "confidence_gt": 1}, "effect": {"decision": "deny", "reason": "forecast_confidence_out_of_range", "required_action": "set_forecast_confidence_between_zero_and_one"}},
	{"name": "campaign_requires_audience", "description": "Campaigns require an audience.", "condition": {"operation": "launch_campaign", "audience_present": False}, "effect": {"decision": "deny", "reason": "campaign_audience_required", "required_action": "define_campaign_audience"}},
	{"name": "campaign_requires_consent", "description": "Campaigns require consent evidence.", "condition": {"operation": "launch_campaign", "consent_evidence_present": False}, "effect": {"decision": "deny", "reason": "campaign_consent_required", "required_action": "attach_consent_evidence"}},
	{"name": "bulk_outreach_requires_privacy_review", "description": "Bulk outreach requires privacy review.", "condition": {"operation": "launch_campaign", "bulk_outreach": True, "privacy_review_recorded": False}, "effect": {"decision": "require_review", "reason": "privacy_review_required", "required_action": "record_privacy_review"}},
	{"name": "large_campaign_requires_budget_review", "description": "Large campaigns require budget review.", "condition": {"operation": "launch_campaign", "budget_gt": 50000, "budget_review_recorded": False}, "effect": {"decision": "require_review", "reason": "campaign_budget_review_required", "required_action": "record_budget_review"}},
	{"name": "crm_batch_import_requires_bytewax", "description": "CRM batch imports require Bytewax coordination.", "condition": {"operation": "crm_batch_import", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_crm_import_to_bytewax"}},
	{"name": "crm_event_requires_bytewax", "description": "CRM lifecycle events require Bytewax.", "condition": {"operation": "crm_event", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_crm_event_to_bytewax"}},
	{"name": "crm_agent_runtime_supported", "description": "CRM agents must use an approved runtime.", "condition": {"operation": "register_crm_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "crm_agent_runtime_not_supported", "required_action": "select_supported_agent_runtime"}},
	{"name": "crm_agent_role_supported", "description": "CRM agents must use an approved role.", "condition": {"operation": "register_crm_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "crm_agent_role_not_supported", "required_action": "select_supported_agent_role"}},
	{"name": "privileged_agent_crm_action_requires_human_approval", "description": "Privileged CRM actions proposed by agents require human approval.", "condition": {"operation": "agent_crm_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/crm-adv/dashboard", "component": "AdvancedCRMDashboard", "permission": "crm_adv:view", "nav_group": "Overview"},
	{"name": "accounts", "path": "/crm-adv/accounts", "component": "AccountCommandCenter", "permission": "crm_adv:manage_accounts", "nav_group": "Accounts"},
	{"name": "contacts", "path": "/crm-adv/contacts", "component": "ContactRelationshipMap", "permission": "crm_adv:manage_contacts", "nav_group": "Contacts"},
	{"name": "leads", "path": "/crm-adv/leads", "component": "LeadScoringConsole", "permission": "crm_adv:manage_leads", "nav_group": "Pipeline"},
	{"name": "pipeline", "path": "/crm-adv/pipeline", "component": "SalesPipelineConsole", "permission": "crm_adv:manage_pipeline", "nav_group": "Pipeline"},
	{"name": "activities", "path": "/crm-adv/activities", "component": "ActivityTimeline", "permission": "crm_adv:manage_activities", "nav_group": "Engagement"},
	{"name": "campaigns", "path": "/crm-adv/campaigns", "component": "CampaignConsole", "permission": "crm_adv:manage_campaigns", "nav_group": "Engagement"},
	{"name": "forecasts", "path": "/crm-adv/forecasts", "component": "ForecastWorkbench", "permission": "crm_adv:forecast", "nav_group": "Analytics"},
	{"name": "agents", "path": "/crm-adv/agents", "component": "CRMAgentWorkbench", "permission": "crm_adv:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/crm-adv/settings", "component": "CRMSettings", "permission": "crm_adv:admin", "nav_group": "Administration"},
]

THEME: dict[str, Any] = {
	"name": "crm_adv_control",
	"tokens": {"color.primary": "#28536B", "color.accent": "#C44536", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F7F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"},
	"components": {
		"accounts": {"icon": "building-2", "status_indicator": "account-pill", "risk_style": "segment-band"},
		"contacts": {"visual": "relationship-map", "status_style": "consent-chip"},
		"leads": {"visual": "score-lanes", "status_style": "quality-chip"},
		"pipeline": {"visual": "stage-board", "status_style": "stage-chip"},
		"activities": {"visual": "timeline", "status_style": "next-step-chip"},
		"campaigns": {"visual": "audience-list", "status_style": "privacy-chip"},
		"forecasts": {"visual": "forecast-grid", "status_style": "confidence-chip"},
		"agent_workbench": {"visual": "review-lane", "status_style": "approval-chip"},
	},
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "crm_adv",
		"display_name": "Advanced CRM Analytics",
		"provides": [
			"account_lifecycle",
			"contact_relationship_management",
			"lead_scoring_and_assignment",
			"sales_pipeline_management",
			"activity_timeline",
			"campaign_governance",
			"forecast_analytics",
			"crm_agents",
		],
		"requires": ["auth", "audl", "ntfy", "composition_events", "composition_config", "mdm"],
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {"shell": "apg_python", "view_module": "views.py", "api_prefix": "/crm-adv/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True},
		"theme": deepcopy(THEME),
		"streaming": streaming_manifest(),
	}


def streaming_manifest() -> dict[str, Any]:
	return {
		"processor": "bytewax",
		"stream": CRM_EVENT_STREAM,
		"key": "tenant_id",
		"events": [
			"account_created",
			"contact_created",
			"lead_created",
			"lead_assigned",
			"opportunity_created",
			"activity_recorded",
			"campaign_launched",
			"forecast_recorded",
			"crm_agent_registered",
		],
		"states": ["draft", "active", "qualified", "assigned", "open", "won", "lost", "archived"],
		"guardrails": [
			"crm_batch_import_requires_bytewax",
			"crm_event_requires_bytewax",
			"privileged_agent_crm_action_requires_human_approval",
		],
	}


def event_stream_name() -> str:
	return CRM_EVENT_STREAM


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
