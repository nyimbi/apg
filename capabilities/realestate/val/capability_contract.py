"""Executable capability contract for APG Property Valuation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "realestate_val"
CAPABILITY_NAME = "Property Valuation"
CAPABILITY_VERSION = "1.0.0"
VAL_EVENT_STREAM = "apg.realestate.val.lifecycle"

SUPPORTED_VALUATION_METHODS = ["dcf", "comparable_sales", "investment_method", "residual_method", "cost_method", "profits_method", "mass_appraisal", "desk_review", "drive_by", "full_inspection"]
SUPPORTED_VALUATION_PURPOSES = ["mortgage_security", "insurance_reinstatement", "purchase", "sale", "financial_reporting", "ifrs16_commencement", "rating_appeal", "compulsory_purchase", "inheritance_tax", "rental_review"]
SUPPORTED_VALUATION_STATUSES = ["instructed", "in_progress", "draft_issued", "under_review", "approved", "signed_off", "published", "superseded", "challenged"]
SUPPORTED_COMPARABLE_TYPES = ["sale", "lease", "letting", "auction", "off_market", "distressed"]
SUPPORTED_REVALUATION_TRIGGERS = ["periodic_cycle", "material_change", "refinancing", "acquisition", "disposal", "ifrs_reporting_date", "insurance_renewal", "legal_dispute", "management_request"]
SUPPORTED_ADJUSTMENT_FACTORS = ["location", "size", "age", "condition", "specification", "tenure", "planning", "environmental", "market_timing", "occupancy"]
SUPPORTED_YIELD_TYPES = ["net_initial_yield", "equivalent_yield", "reversionary_yield", "running_yield", "true_equivalent_yield", "net_income_yield"]
SUPPORTED_DCF_PARAMETERS = ["discount_rate", "holding_period", "exit_yield", "rental_growth", "void_period", "capex_allowance", "purchasers_costs"]
SUPPORTED_MASS_APPRAISAL_MODELS = ["regression", "spatial_interpolation", "hedonic_pricing", "ai_avms", "comparable_grid"]
SUPPORTED_VALUER_GRADES = ["rics_registered", "rics_fellow", "api_registered", "internal_valuer", "external_valuer", "independent_valuer"]
SUPPORTED_REPORT_TYPES = ["desktop_valuation", "restricted_report", "full_red_book", "market_appraisal", "reinstatement_cost_assessment", "schedule_of_condition", "mass_appraisal_report"]
SUPPORTED_CURRENCIES = ["KES", "USD", "EUR", "GBP", "ZAR"]
SUPPORTED_AREA_UNITS = ["sqm", "sqft", "acres", "hectares"]
SUPPORTED_REVALUATION_FREQUENCIES = ["monthly", "quarterly", "semi_annual", "annual", "biennial", "triennial", "as_needed"]
SUPPORTED_APPROVAL_LEVELS = ["senior_valuer", "chief_valuer", "audit_committee", "board"]

PROVIDES = [
	"comparable_sales_analysis",
	"dcf_valuation_engine",
	"mass_appraisal_engine",
	"valuation_roll_management",
	"revaluation_cycle_management",
	"valuation_report_generation",
	"yield_analysis",
	"valuer_panel_management",
	"valuation_challenge_workflow",
	"valuation_benchmarking",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "nlpc", "comp", "mqeb", "schd"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/realestate/val/dashboard", "component": "ValDashboard", "permission": "realestate_val:view", "nav_group": "Overview"},
	{"name": "valuations", "path": "/realestate/val/valuations", "component": "ValuationRegistry", "permission": "realestate_val:valuations", "nav_group": "Valuations"},
	{"name": "valuation-detail", "path": "/realestate/val/valuations/<id>", "component": "ValuationDetail", "permission": "realestate_val:valuations", "nav_group": "Valuations"},
	{"name": "comparables", "path": "/realestate/val/comparables", "component": "ComparablesSalesDatabase", "permission": "realestate_val:comparables", "nav_group": "Analysis"},
	{"name": "dcf-builder", "path": "/realestate/val/dcf", "component": "DcfModelBuilder", "permission": "realestate_val:dcf", "nav_group": "Models"},
	{"name": "mass-appraisal", "path": "/realestate/val/mass-appraisal", "component": "MassAppraisalConsole", "permission": "realestate_val:mass_appraisal", "nav_group": "Models"},
	{"name": "valuation-roll", "path": "/realestate/val/roll", "component": "ValuationRollConsole", "permission": "realestate_val:roll", "nav_group": "Roll"},
	{"name": "revaluation-cycles", "path": "/realestate/val/cycles", "component": "RevaluationCyclePlanner", "permission": "realestate_val:cycles", "nav_group": "Planning"},
	{"name": "yield-analysis", "path": "/realestate/val/yields", "component": "YieldAnalysisConsole", "permission": "realestate_val:yields", "nav_group": "Analysis"},
	{"name": "valuers", "path": "/realestate/val/valuers", "component": "ValuerPanelRegistry", "permission": "realestate_val:valuers", "nav_group": "Registry"},
	{"name": "challenges", "path": "/realestate/val/challenges", "component": "ValuationChallengeWorkflow", "permission": "realestate_val:challenges", "nav_group": "Governance"},
	{"name": "reports", "path": "/realestate/val/reports", "component": "ValuationReportBuilder", "permission": "realestate_val:reports", "nav_group": "Reporting"},
	{"name": "benchmarking", "path": "/realestate/val/benchmarking", "component": "ValuationBenchmarkDashboard", "permission": "realestate_val:benchmarking", "nav_group": "Analytics"},
	{"name": "settings", "path": "/realestate/val/settings", "component": "ValSettings", "permission": "realestate_val:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "realestate_val_appraisal",
	"tokens": {
		"color.primary": "#312E81",
		"color.accent": "#B45309",
		"color.success": "#166534",
		"color.warning": "#92400E",
		"color.danger": "#991B1B",
		"surface.canvas": "#EEF2FF",
		"surface.panel": "#FFFFFF",
		"text.primary": "#1E1B4B",
		"text.secondary": "#4B5563",
		"border.radius": "8px",
		"density": "comfortable",
	},
	"components": {
		"valuations": {"icon": "trending-up", "status_indicator": "valuation-status-chip"},
		"comparables": {"icon": "bar-chart", "status_indicator": "comparable-type-chip"},
		"dcf": {"icon": "calculator", "status_indicator": "dcf-parameter-chip"},
		"mass_appraisal": {"icon": "cpu", "status_indicator": "model-type-chip"},
		"valuation_roll": {"icon": "table", "status_indicator": "revaluation-frequency-chip"},
		"yields": {"icon": "percent", "status_indicator": "yield-type-chip"},
		"valuers": {"icon": "user-check", "status_indicator": "valuer-grade-chip"},
		"challenges": {"icon": "shield-alert", "status_indicator": "challenge-status-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": VAL_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"valuation_instructed", "valuation_completed", "valuation_approved", "valuation_published",
		"comparable_added", "comparable_verified",
		"dcf_model_run", "mass_appraisal_run_completed",
		"revaluation_cycle_triggered", "valuation_roll_updated",
		"yield_analysis_calculated", "valuation_challenged", "challenge_resolved",
		"valuer_registered", "valuation_benchmark_generated",
	],
	"guardrails": [
		"valuation_sign_off_requires_qualified_valuer",
		"red_book_valuation_requires_independent_valuer",
		"mass_appraisal_model_requires_calibration",
		"challenge_requires_counter_evidence",
		"dcf_discount_rate_range_enforced",
	],
}

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"valuations": {"supported_methods": SUPPORTED_VALUATION_METHODS, "supported_purposes": SUPPORTED_VALUATION_PURPOSES, "supported_statuses": SUPPORTED_VALUATION_STATUSES},
	"comparables": {"supported_types": SUPPORTED_COMPARABLE_TYPES, "supported_area_units": SUPPORTED_AREA_UNITS},
	"dcf": {"supported_parameters": SUPPORTED_DCF_PARAMETERS, "min_discount_rate": 0.03, "max_discount_rate": 0.30},
	"mass_appraisal": {"supported_models": SUPPORTED_MASS_APPRAISAL_MODELS, "calibration_required": True},
	"yields": {"supported_types": SUPPORTED_YIELD_TYPES},
	"adjustments": {"supported_factors": SUPPORTED_ADJUSTMENT_FACTORS},
	"revaluation": {"supported_triggers": SUPPORTED_REVALUATION_TRIGGERS, "supported_frequencies": SUPPORTED_REVALUATION_FREQUENCIES},
	"valuers": {"supported_grades": SUPPORTED_VALUER_GRADES, "independence_required_for_red_book": True},
	"reports": {"supported_types": SUPPORTED_REPORT_TYPES},
	"approvals": {"supported_levels": SUPPORTED_APPROVAL_LEVELS},
	"currencies": {"supported": SUPPORTED_CURRENCIES, "default": "KES"},
	"ui": {"enable_dashboard": True, "enable_valuations": True, "enable_dcf": True, "enable_mass_appraisal": True},
	"theme": {"default_theme": "realestate_val_appraisal", "allow_tenant_overrides": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "rics_compliance_required": True},
	"observability": {"event_stream": VAL_EVENT_STREAM, "stream_processor": "bytewax"},
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "valuation_policy_required", "required_action": "attach_valuation_policy"}},
	{"name": "valuation_method_supported", "condition": {"operation": "instruct_valuation", "method_supported": False}, "effect": {"decision": "deny", "reason": "valuation_method_not_supported", "required_action": "select_supported_valuation_method"}},
	{"name": "valuation_purpose_supported", "condition": {"operation": "instruct_valuation", "purpose_supported": False}, "effect": {"decision": "deny", "reason": "valuation_purpose_not_supported", "required_action": "select_supported_valuation_purpose"}},
	{"name": "valuation_requires_property", "condition": {"operation": "instruct_valuation", "property_present": False}, "effect": {"decision": "deny", "reason": "property_required_for_valuation", "required_action": "link_property"}},
	{"name": "valuation_requires_qualified_valuer", "condition": {"operation": "instruct_valuation", "qualified_valuer_assigned": False}, "effect": {"decision": "deny", "reason": "qualified_valuer_required_for_valuation", "required_action": "assign_qualified_valuer"}},
	{"name": "red_book_requires_independent_valuer", "condition": {"operation": "publish_valuation", "report_type": "full_red_book", "valuer_independent": False}, "effect": {"decision": "deny", "reason": "red_book_valuation_requires_independent_valuer", "required_action": "assign_independent_valuer"}},
	{"name": "sign_off_requires_approved_valuer_grade", "condition": {"operation": "sign_off_valuation", "valuer_grade_approved": False}, "effect": {"decision": "deny", "reason": "sign_off_requires_rics_registered_or_above", "required_action": "assign_qualified_senior_valuer"}},
	{"name": "comparable_type_supported", "condition": {"operation": "add_comparable", "comparable_type_supported": False}, "effect": {"decision": "deny", "reason": "comparable_type_not_supported", "required_action": "select_supported_comparable_type"}},
	{"name": "dcf_discount_rate_in_range", "condition": {"operation": "run_dcf", "discount_rate_in_range": False}, "effect": {"decision": "deny", "reason": "dcf_discount_rate_outside_permitted_range", "required_action": "set_discount_rate_within_range"}},
	{"name": "dcf_requires_all_parameters", "condition": {"operation": "run_dcf", "all_dcf_parameters_present": False}, "effect": {"decision": "deny", "reason": "all_dcf_parameters_required", "required_action": "complete_dcf_parameters"}},
	{"name": "mass_appraisal_requires_calibrated_model", "condition": {"operation": "run_mass_appraisal", "model_calibrated": False}, "effect": {"decision": "deny", "reason": "mass_appraisal_model_must_be_calibrated", "required_action": "calibrate_model"}},
	{"name": "yield_type_supported", "condition": {"operation": "calculate_yield", "yield_type_supported": False}, "effect": {"decision": "deny", "reason": "yield_type_not_supported", "required_action": "select_supported_yield_type"}},
	{"name": "revaluation_trigger_supported", "condition": {"operation": "trigger_revaluation", "trigger_supported": False}, "effect": {"decision": "deny", "reason": "revaluation_trigger_not_supported", "required_action": "select_supported_revaluation_trigger"}},
	{"name": "challenge_requires_counter_evidence", "condition": {"operation": "raise_challenge", "counter_evidence_present": False}, "effect": {"decision": "deny", "reason": "counter_evidence_required_to_challenge_valuation", "required_action": "attach_counter_evidence"}},
	{"name": "challenge_requires_active_valuation", "condition": {"operation": "raise_challenge", "valuation_status_challengeable": False}, "effect": {"decision": "deny", "reason": "valuation_must_be_in_challengeable_status", "required_action": "check_valuation_status"}},
	{"name": "adjustment_factor_supported", "condition": {"operation": "apply_adjustment", "adjustment_factor_supported": False}, "effect": {"decision": "deny", "reason": "adjustment_factor_not_supported", "required_action": "select_supported_adjustment_factor"}},
	{"name": "valuer_grade_supported", "condition": {"operation": "register_valuer", "valuer_grade_supported": False}, "effect": {"decision": "deny", "reason": "valuer_grade_not_supported", "required_action": "select_supported_valuer_grade"}},
	{"name": "published_valuation_immutable", "condition": {"operation_type": "write", "valuation_status": "published"}, "effect": {"decision": "deny", "reason": "published_valuation_cannot_be_modified", "required_action": "instruct_new_valuation_or_raise_challenge"}},
	{"name": "cross_tenant_valuation_denied", "condition": {"operation_type": "write", "cross_tenant": True}, "effect": {"decision": "deny", "reason": "cross_tenant_valuation_not_allowed", "required_action": "use_correct_tenant_context"}},
	{"name": "currency_supported", "condition": {"operation_type": "write", "currency_supported": False}, "effect": {"decision": "deny", "reason": "currency_not_supported", "required_action": "select_supported_currency"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	"""Return the full capability contract for the given tenant."""
	cfg = deepcopy(DEFAULT_CONFIGURATION)
	cfg["tenant_id"] = tenant_id
	return {
		"capability": CAPABILITY_ID,
		"display_name": CAPABILITY_NAME,
		"version": CAPABILITY_VERSION,
		"configuration": cfg,
		"configuration_schema": {
			"required": ["tenant_id", "ui", "theme"],
			"properties": {"tenant_id": {"type": "string"}, "ui": {"type": "object"}, "theme": {"type": "object"}},
		},
		"rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": RULES},
		"ui": {"shell": "apg_python", "requires_theme": True, "template_roots": ["realestate/val/templates"], "routes": UI_ROUTES},
		"theme": THEME,
		"streaming": STREAMING,
		"provides": PROVIDES,
		"requires": REQUIRES,
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate all rules against context. Returns first denial or allow."""
	for rule in RULES:
		cond = rule["condition"]
		if all(context.get(k) == v for k, v in cond.items()):
			effect = rule["effect"]
			if effect["decision"] == "deny":
				return {"decision": "deny", "rule": rule["name"], "reason": effect["reason"], "required_action": effect.get("required_action")}
	return {"decision": "allow", "rule": None, "reason": None}
