"""Executable capability contract for APG Renewable Energy."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "energy_ren"
CAPABILITY_NAME = "Renewable Energy"
CAPABILITY_VERSION = "1.0.0"
REN_EVENT_STREAM = "apg.energy.ren.lifecycle"

SUPPORTED_RENEWABLE_TYPES = ["solar_pv_utility", "solar_pv_rooftop", "solar_csp", "wind_onshore", "wind_offshore", "small_hydro", "large_hydro", "run_of_river", "pumped_hydro", "biomass_power", "biogas", "geothermal", "wave_tidal", "waste_to_energy"]
SUPPORTED_ASSET_STATUSES = ["operating", "under_construction", "commissioning", "curtailed", "maintenance", "decommissioned", "mothballed", "planning"]
SUPPORTED_CURTAILMENT_REASONS = ["grid_congestion", "frequency_regulation", "voltage_control", "market_oversupply", "technical_limit", "weather_variability", "operator_instruction", "force_majeure"]
SUPPORTED_REC_TYPES = ["renewable_energy_certificate", "guarantees_of_origin", "international_rec", "solar_renewable_energy_certificate", "wind_renewable_energy_certificate", "hydro_renewable_energy_certificate", "bioenergy_renewable_energy_certificate"]
SUPPORTED_REC_STATUSES = ["issued", "transferred", "retired", "cancelled", "expired"]
SUPPORTED_CARBON_CREDIT_TYPES = ["voluntary_carbon_unit", "verified_emission_reduction", "gold_standard", "clean_development_mechanism", "emission_reduction_unit", "removal_unit"]
SUPPORTED_FEED_IN_TARIFF_TYPES = ["fixed_fit", "premium_fit", "net_metering_fit", "net_billing_fit", "time_differentiated_fit"]
SUPPORTED_FORECAST_TYPES = ["solar_irradiance", "wind_speed", "hydro_inflow", "generation_output", "capacity_availability"]
SUPPORTED_FORECAST_HORIZONS = ["1h", "4h", "24h", "48h", "7d", "30d"]
SUPPORTED_PERFORMANCE_METRICS = ["capacity_factor", "performance_ratio", "specific_yield", "availability", "pr_ratio", "clipping_loss", "soiling_loss", "shading_loss"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["curtailment_optimizer", "rec_manager", "carbon_credit_analyst", "forecast_analyst", "performance_analyst"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"ui": {"enable_dashboard": True, "enable_assets": True, "enable_curtailment": True, "enable_recs": True, "enable_carbon_credits": True, "enable_feed_in_tariffs": True, "enable_forecasting": True, "enable_performance": True},
	"theme": {"default_theme": "energy_ren_ops", "allow_tenant_overrides": True},
	"assets": {"supported_types": SUPPORTED_RENEWABLE_TYPES, "supported_statuses": SUPPORTED_ASSET_STATUSES, "capacity_mw_required": True, "commissioning_date_required": True},
	"curtailment": {"supported_reasons": SUPPORTED_CURTAILMENT_REASONS, "mwh_tracking": True, "revenue_loss_tracking": True, "approval_required": True},
	"recs": {"supported_types": SUPPORTED_REC_TYPES, "supported_statuses": SUPPORTED_REC_STATUSES, "registry_required": True, "vintage_year_required": True},
	"carbon_credits": {"supported_types": SUPPORTED_CARBON_CREDIT_TYPES, "standard_required": True, "vintage_required": True, "verification_required": True},
	"feed_in_tariffs": {"supported_types": SUPPORTED_FEED_IN_TARIFF_TYPES, "approval_required": True, "effective_date_required": True},
	"forecasting": {"supported_types": SUPPORTED_FORECAST_TYPES, "supported_horizons": SUPPORTED_FORECAST_HORIZONS, "model_versioning": True},
	"performance": {"supported_metrics": SUPPORTED_PERFORMANCE_METRICS, "auto_calculate": True, "benchmark_comparison": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_curtailment": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "cross_tenant_denied": True, "unapproved_curtailment_denied": True},
	"observability": {"event_stream": REN_EVENT_STREAM, "stream_processor": "bytewax"},
}

PROVIDES = [
	"renewable_asset_registry",
	"curtailment_tracking",
	"rec_certificate_management",
	"carbon_credit_management",
	"feed_in_tariff_management",
	"generation_forecasting",
	"renewable_performance_analytics",
	"green_energy_reporting",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "moni", "comp", "mqeb", "schd"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/energy-ren/dashboard", "component": "RenDashboard", "permission": "energy_ren:view", "nav_group": "Overview"},
	{"name": "assets", "path": "/energy-ren/assets", "component": "RenewableAssetRegistry", "permission": "energy_ren:assets", "nav_group": "Assets"},
	{"name": "asset_detail", "path": "/energy-ren/assets/<id>", "component": "RenewableAssetDetail", "permission": "energy_ren:assets", "nav_group": "Assets"},
	{"name": "curtailment", "path": "/energy-ren/curtailment", "component": "CurtailmentTracker", "permission": "energy_ren:curtailment", "nav_group": "Operations"},
	{"name": "recs", "path": "/energy-ren/recs", "component": "RecCertificateManager", "permission": "energy_ren:recs", "nav_group": "Certificates"},
	{"name": "carbon_credits", "path": "/energy-ren/carbon-credits", "component": "CarbonCreditManager", "permission": "energy_ren:carbon_credits", "nav_group": "Certificates"},
	{"name": "feed_in_tariffs", "path": "/energy-ren/feed-in-tariffs", "component": "FeedInTariffManager", "permission": "energy_ren:feed_in_tariffs", "nav_group": "Finance"},
	{"name": "forecasting", "path": "/energy-ren/forecasting", "component": "GenerationForecaster", "permission": "energy_ren:forecasting", "nav_group": "Analytics"},
	{"name": "performance", "path": "/energy-ren/performance", "component": "RenewablePerformance", "permission": "energy_ren:performance", "nav_group": "Analytics"},
	{"name": "reports", "path": "/energy-ren/reports", "component": "RenewableReports", "permission": "energy_ren:reports", "nav_group": "Reporting"},
	{"name": "agents", "path": "/energy-ren/agents", "component": "RenAgentWorkbench", "permission": "energy_ren:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/energy-ren/settings", "component": "RenSettings", "permission": "energy_ren:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "energy_ren_ops",
	"tokens": {
		"color.primary": "#16A34A",
		"color.accent": "#F59E0B",
		"color.success": "#15803D",
		"color.warning": "#D97706",
		"color.danger": "#DC2626",
		"surface.canvas": "#F0FDF4",
		"surface.panel": "#FFFFFF",
		"text.primary": "#14532D",
		"text.secondary": "#166534",
		"border.radius": "8px",
		"density": "comfortable",
	},
	"components": {
		"assets": {"icon": "sun", "status_indicator": "asset-status-chip"},
		"curtailment": {"icon": "cloud-off", "status_indicator": "curtailment-reason-chip"},
		"recs": {"icon": "award", "status_indicator": "rec-status-chip"},
		"carbon_credits": {"icon": "leaf", "status_indicator": "carbon-credit-type-chip"},
		"feed_in_tariffs": {"icon": "dollar-sign", "status_indicator": "fit-type-chip"},
		"forecasting": {"icon": "wind", "status_indicator": "forecast-horizon-chip"},
		"performance": {"icon": "bar-chart", "status_indicator": "perf-metric-chip"},
		"agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": REN_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"renewable_asset_registered", "asset_status_changed", "curtailment_event_created",
		"curtailment_event_approved", "rec_issued", "rec_transferred", "rec_retired",
		"carbon_credit_issued", "carbon_credit_retired", "feed_in_tariff_activated",
		"generation_forecast_published", "performance_metric_calculated",
	],
	"guardrails": [
		"unapproved_curtailment_denied",
		"cross_tenant_renewable_data_denied",
		"privileged_ren_agent_requires_human_approval",
		"rec_double_issuance_denied",
		"carbon_credit_verification_required",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "write_policy_required", "required_action": "attach_write_policy"}},
	{"name": "renewable_type_supported", "condition": {"operation": "register_asset", "renewable_type_supported": False}, "effect": {"decision": "deny", "reason": "renewable_type_not_supported", "required_action": "select_supported_renewable_type"}},
	{"name": "asset_capacity_positive", "condition": {"operation": "register_asset", "capacity_positive": False}, "effect": {"decision": "deny", "reason": "capacity_mw_must_be_positive", "required_action": "set_positive_capacity_mw"}},
	{"name": "asset_commissioning_date_required", "condition": {"operation": "register_asset", "commissioning_date_present": False}, "effect": {"decision": "deny", "reason": "commissioning_date_required", "required_action": "set_commissioning_date"}},
	{"name": "asset_location_required", "condition": {"operation": "register_asset", "location_present": False}, "effect": {"decision": "deny", "reason": "asset_location_required", "required_action": "provide_asset_location"}},
	{"name": "curtailment_reason_supported", "condition": {"operation": "record_curtailment", "curtailment_reason_supported": False}, "effect": {"decision": "deny", "reason": "curtailment_reason_not_supported", "required_action": "select_supported_curtailment_reason"}},
	{"name": "curtailment_mwh_positive", "condition": {"operation": "record_curtailment", "mwh_positive": False}, "effect": {"decision": "deny", "reason": "curtailed_mwh_must_be_positive", "required_action": "set_positive_mwh"}},
	{"name": "curtailment_approval_required", "condition": {"operation": "approve_curtailment", "approval_present": False}, "effect": {"decision": "deny", "reason": "curtailment_approval_required", "required_action": "obtain_curtailment_approval"}},
	{"name": "rec_type_supported", "condition": {"operation": "issue_rec", "rec_type_supported": False}, "effect": {"decision": "deny", "reason": "rec_type_not_supported", "required_action": "select_supported_rec_type"}},
	{"name": "rec_registry_required", "condition": {"operation": "issue_rec", "registry_present": False}, "effect": {"decision": "deny", "reason": "rec_registry_required", "required_action": "specify_rec_registry"}},
	{"name": "rec_vintage_year_required", "condition": {"operation": "issue_rec", "vintage_year_present": False}, "effect": {"decision": "deny", "reason": "rec_vintage_year_required", "required_action": "set_vintage_year"}},
	{"name": "rec_double_issuance_denied", "condition": {"operation": "issue_rec", "rec_already_issued": True}, "effect": {"decision": "deny", "reason": "rec_already_issued_for_period", "required_action": "verify_no_prior_issuance"}},
	{"name": "carbon_credit_type_supported", "condition": {"operation": "issue_carbon_credit", "credit_type_supported": False}, "effect": {"decision": "deny", "reason": "carbon_credit_type_not_supported", "required_action": "select_supported_credit_type"}},
	{"name": "carbon_credit_verification_required", "condition": {"operation": "issue_carbon_credit", "verification_present": False}, "effect": {"decision": "deny", "reason": "carbon_credit_verification_required", "required_action": "attach_verification_report"}},
	{"name": "feed_in_tariff_type_supported", "condition": {"operation": "create_fit", "fit_type_supported": False}, "effect": {"decision": "deny", "reason": "feed_in_tariff_type_not_supported", "required_action": "select_supported_fit_type"}},
	{"name": "feed_in_tariff_approval_required", "condition": {"operation": "activate_fit", "approval_present": False}, "effect": {"decision": "deny", "reason": "fit_activation_approval_required", "required_action": "obtain_fit_approval"}},
	{"name": "forecast_type_supported", "condition": {"operation": "publish_forecast", "forecast_type_supported": False}, "effect": {"decision": "deny", "reason": "forecast_type_not_supported", "required_action": "select_supported_forecast_type"}},
	{"name": "forecast_horizon_supported", "condition": {"operation": "publish_forecast", "forecast_horizon_supported": False}, "effect": {"decision": "deny", "reason": "forecast_horizon_not_supported", "required_action": "select_supported_horizon"}},
	{"name": "cross_tenant_denied", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_access_denied", "required_action": "use_correct_tenant_context"}},
	{"name": "ren_agent_runtime_supported", "condition": {"operation": "register_ren_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "ren_agent_role_supported", "condition": {"operation": "register_ren_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "privileged_ren_agent_requires_human_approval", "condition": {"operation": "ren_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required_for_curtailment_command", "required_action": "record_human_approval"}},
	{"name": "rec_retirement_irreversible", "condition": {"operation": "cancel_rec", "rec_status": "retired"}, "effect": {"decision": "deny", "reason": "retired_rec_cannot_be_cancelled", "required_action": "verify_rec_status_before_cancel"}},
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
			"properties": {key: {"type": "object"} for key in configuration if key != "tenant_id"} | {"tenant_id": {"type": "string", "minLength": 1}},
		},
		"rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"api_prefix": "/energy-ren/api/v1",
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
