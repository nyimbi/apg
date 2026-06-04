"""Executable capability contract for APG Property Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "realestate_prm"
CAPABILITY_NAME = "Property Management"
CAPABILITY_VERSION = "1.0.0"
PRM_EVENT_STREAM = "apg.realestate.prm.lifecycle"

SUPPORTED_PROPERTY_TYPES = ["office", "retail", "industrial", "residential", "mixed_use", "hotel", "student_accommodation", "data_centre", "healthcare_facility", "land", "special_purpose"]
SUPPORTED_PROPERTY_STATUSES = ["development", "pre_completion", "active", "partially_let", "vacant", "under_refurbishment", "for_sale", "sold", "demolished"]
SUPPORTED_UNIT_TYPES = ["office_suite", "retail_unit", "industrial_unit", "apartment", "penthouse", "studio", "car_park", "storage", "roof_terrace", "amenity"]
SUPPORTED_UNIT_STATUSES = ["available", "under_offer", "let", "owner_occupied", "under_refurbishment", "held_back", "not_available"]
SUPPORTED_OWNERSHIP_STRUCTURES = ["freehold", "leasehold", "commonhold", "joint_venture", "spv", "reit", "unit_trust", "managed_fund"]
SUPPORTED_PORTFOLIO_TIERS = ["core", "core_plus", "value_add", "opportunistic", "development"]
SUPPORTED_MANAGEMENT_MODELS = ["full_service", "facilities_only", "lease_management_only", "financial_only", "owner_managed"]
SUPPORTED_REPORTING_PERIODS = ["monthly", "quarterly", "semi_annual", "annual"]
SUPPORTED_PERFORMANCE_KPIS = ["occupancy_rate", "void_rate", "wault", "net_initial_yield", "equivalent_yield", "irr", "net_rental_income", "total_return", "capex_ratio"]
SUPPORTED_OWNER_TYPES = ["institutional", "private_individual", "corporate", "pension_fund", "sovereign_wealth", "family_office", "reit", "government"]
SUPPORTED_HANDOVER_TYPES = ["landlord_to_tenant", "developer_to_landlord", "contractor_to_developer", "management_handover"]
SUPPORTED_AREA_UNITS = ["sqm", "sqft", "acres", "hectares"]
SUPPORTED_GRADE_TYPES = ["grade_a", "grade_b", "grade_c", "heritage", "listed_building"]
SUPPORTED_CURRENCIES = ["KES", "USD", "EUR", "GBP", "ZAR"]
SUPPORTED_APPROVAL_LEVELS = ["property_manager", "asset_manager", "portfolio_director", "investment_committee", "board"]

PROVIDES = [
	"property_portfolio_management",
	"unit_management",
	"owner_portal_service",
	"property_performance_reporting",
	"portfolio_analytics",
	"handover_management",
	"owner_distribution_management",
	"property_data_room",
	"performance_kpi_engine",
	"property_benchmarking",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "moni", "mqeb", "srch"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/realestate/prm/dashboard", "component": "PrmDashboard", "permission": "realestate_prm:view", "nav_group": "Overview"},
	{"name": "portfolio", "path": "/realestate/prm/portfolio", "component": "PortfolioOverview", "permission": "realestate_prm:portfolio", "nav_group": "Portfolio"},
	{"name": "properties", "path": "/realestate/prm/properties", "component": "PropertyRegistry", "permission": "realestate_prm:properties", "nav_group": "Properties"},
	{"name": "property-detail", "path": "/realestate/prm/properties/<id>", "component": "PropertyDetail", "permission": "realestate_prm:properties", "nav_group": "Properties"},
	{"name": "units", "path": "/realestate/prm/units", "component": "UnitManagementConsole", "permission": "realestate_prm:units", "nav_group": "Units"},
	{"name": "owners", "path": "/realestate/prm/owners", "component": "OwnerRegistry", "permission": "realestate_prm:owners", "nav_group": "Owners"},
	{"name": "owner-portal", "path": "/realestate/prm/owner-portal", "component": "OwnerPortal", "permission": "realestate_prm:owner_portal", "nav_group": "Owners"},
	{"name": "performance", "path": "/realestate/prm/performance", "component": "PropertyPerformanceDashboard", "permission": "realestate_prm:performance", "nav_group": "Analytics"},
	{"name": "kpi-builder", "path": "/realestate/prm/kpis", "component": "KpiBuilder", "permission": "realestate_prm:kpis", "nav_group": "Analytics"},
	{"name": "handovers", "path": "/realestate/prm/handovers", "component": "HandoverConsole", "permission": "realestate_prm:handovers", "nav_group": "Transactions"},
	{"name": "distributions", "path": "/realestate/prm/distributions", "component": "OwnerDistributionConsole", "permission": "realestate_prm:distributions", "nav_group": "Financial"},
	{"name": "data-room", "path": "/realestate/prm/data-room", "component": "PropertyDataRoom", "permission": "realestate_prm:data_room", "nav_group": "Documents"},
	{"name": "reports", "path": "/realestate/prm/reports", "component": "PropertyReportBuilder", "permission": "realestate_prm:reports", "nav_group": "Reporting"},
	{"name": "settings", "path": "/realestate/prm/settings", "component": "PrmSettings", "permission": "realestate_prm:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "realestate_prm_portfolio",
	"tokens": {
		"color.primary": "#1E3A5F",
		"color.accent": "#D4A017",
		"color.success": "#166534",
		"color.warning": "#A16207",
		"color.danger": "#B91C1C",
		"surface.canvas": "#F8F9FA",
		"surface.panel": "#FFFFFF",
		"text.primary": "#1E3A5F",
		"text.secondary": "#6B7280",
		"border.radius": "8px",
		"density": "comfortable",
	},
	"components": {
		"properties": {"icon": "building-2", "status_indicator": "property-status-chip"},
		"units": {"icon": "grid", "status_indicator": "unit-status-chip"},
		"owners": {"icon": "users", "status_indicator": "owner-type-chip"},
		"portfolio": {"icon": "briefcase", "status_indicator": "portfolio-tier-chip"},
		"performance": {"icon": "bar-chart-2", "status_indicator": "kpi-chip"},
		"handovers": {"icon": "arrow-left-right", "status_indicator": "handover-type-chip"},
		"distributions": {"icon": "split", "status_indicator": "distribution-status-chip"},
		"data_room": {"icon": "folder-lock", "status_indicator": "access-level-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": PRM_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"property_registered", "property_status_changed", "property_sold",
		"unit_status_changed", "unit_let", "unit_vacated",
		"owner_registered", "owner_distribution_paid",
		"performance_kpi_calculated", "occupancy_threshold_breached",
		"handover_completed", "data_room_accessed",
		"portfolio_benchmark_generated",
	],
	"guardrails": [
		"property_deletion_requires_board_approval",
		"owner_distribution_requires_dual_control",
		"data_room_access_logged_always",
		"kpi_benchmark_requires_verified_data",
		"unit_status_change_triggers_rent_roll_update",
	],
}

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"properties": {"supported_types": SUPPORTED_PROPERTY_TYPES, "supported_statuses": SUPPORTED_PROPERTY_STATUSES, "supported_grade_types": SUPPORTED_GRADE_TYPES, "supported_area_units": SUPPORTED_AREA_UNITS},
	"units": {"supported_types": SUPPORTED_UNIT_TYPES, "supported_statuses": SUPPORTED_UNIT_STATUSES},
	"ownership": {"supported_structures": SUPPORTED_OWNERSHIP_STRUCTURES, "supported_owner_types": SUPPORTED_OWNER_TYPES},
	"portfolio": {"supported_tiers": SUPPORTED_PORTFOLIO_TIERS, "supported_management_models": SUPPORTED_MANAGEMENT_MODELS},
	"performance": {"supported_kpis": SUPPORTED_PERFORMANCE_KPIS, "reporting_periods": SUPPORTED_REPORTING_PERIODS},
	"handovers": {"supported_types": SUPPORTED_HANDOVER_TYPES},
	"approvals": {"supported_levels": SUPPORTED_APPROVAL_LEVELS},
	"currencies": {"supported": SUPPORTED_CURRENCIES, "default": "KES"},
	"ui": {"enable_dashboard": True, "enable_portfolio": True, "enable_owner_portal": True, "enable_performance": True},
	"theme": {"default_theme": "realestate_prm_portfolio", "allow_tenant_overrides": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "data_room_access_always_logged": True},
	"observability": {"event_stream": PRM_EVENT_STREAM, "stream_processor": "bytewax"},
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "property_management_policy_required", "required_action": "attach_property_policy"}},
	{"name": "property_type_supported", "condition": {"operation": "register_property", "property_type_supported": False}, "effect": {"decision": "deny", "reason": "property_type_not_supported", "required_action": "select_supported_property_type"}},
	{"name": "property_requires_owner", "condition": {"operation": "register_property", "owner_present": False}, "effect": {"decision": "deny", "reason": "property_owner_required", "required_action": "link_property_owner"}},
	{"name": "property_requires_address", "condition": {"operation": "register_property", "address_present": False}, "effect": {"decision": "deny", "reason": "property_address_required", "required_action": "set_property_address"}},
	{"name": "property_deletion_requires_board_approval", "condition": {"operation": "delete_property", "board_approved": False}, "effect": {"decision": "deny", "reason": "board_approval_required_to_delete_property", "required_action": "submit_to_board"}},
	{"name": "unit_type_supported", "condition": {"operation": "create_unit", "unit_type_supported": False}, "effect": {"decision": "deny", "reason": "unit_type_not_supported", "required_action": "select_supported_unit_type"}},
	{"name": "unit_requires_property", "condition": {"operation": "create_unit", "property_present": False}, "effect": {"decision": "deny", "reason": "property_required_for_unit", "required_action": "link_property"}},
	{"name": "owner_type_supported", "condition": {"operation": "register_owner", "owner_type_supported": False}, "effect": {"decision": "deny", "reason": "owner_type_not_supported", "required_action": "select_supported_owner_type"}},
	{"name": "ownership_structure_supported", "condition": {"operation": "register_property", "ownership_structure_supported": False}, "effect": {"decision": "deny", "reason": "ownership_structure_not_supported", "required_action": "select_supported_ownership_structure"}},
	{"name": "portfolio_tier_supported", "condition": {"operation": "assign_portfolio_tier", "tier_supported": False}, "effect": {"decision": "deny", "reason": "portfolio_tier_not_supported", "required_action": "select_supported_portfolio_tier"}},
	{"name": "distribution_requires_dual_control", "condition": {"operation": "process_distribution", "dual_control_satisfied": False}, "effect": {"decision": "deny", "reason": "owner_distribution_requires_dual_control", "required_action": "obtain_second_approver"}},
	{"name": "handover_type_supported", "condition": {"operation": "create_handover", "handover_type_supported": False}, "effect": {"decision": "deny", "reason": "handover_type_not_supported", "required_action": "select_supported_handover_type"}},
	{"name": "data_room_access_always_logged", "condition": {"operation": "access_data_room", "access_logged": False}, "effect": {"decision": "deny", "reason": "data_room_access_must_always_be_logged", "required_action": "enable_access_logging"}},
	{"name": "kpi_requires_verified_data", "condition": {"operation": "calculate_kpi", "data_verified": False}, "effect": {"decision": "deny", "reason": "kpi_calculation_requires_verified_data", "required_action": "verify_source_data"}},
	{"name": "property_status_transition_valid", "condition": {"operation": "update_property_status", "status_transition_valid": False}, "effect": {"decision": "deny", "reason": "invalid_property_status_transition", "required_action": "follow_valid_status_transition"}},
	{"name": "sold_property_modification_denied", "condition": {"operation_type": "write", "property_status": "sold"}, "effect": {"decision": "deny", "reason": "sold_property_cannot_be_modified", "required_action": "create_new_property_record"}},
	{"name": "management_model_supported", "condition": {"operation": "set_management_model", "model_supported": False}, "effect": {"decision": "deny", "reason": "management_model_not_supported", "required_action": "select_supported_management_model"}},
	{"name": "cross_tenant_property_denied", "condition": {"operation_type": "write", "cross_tenant": True}, "effect": {"decision": "deny", "reason": "cross_tenant_property_management_not_allowed", "required_action": "use_correct_tenant_context"}},
	{"name": "currency_supported", "condition": {"operation_type": "write", "currency_supported": False}, "effect": {"decision": "deny", "reason": "currency_not_supported", "required_action": "select_supported_currency"}},
	{"name": "report_period_required", "condition": {"operation": "generate_report", "period_present": False}, "effect": {"decision": "deny", "reason": "reporting_period_required_for_report", "required_action": "specify_reporting_period"}},
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
		"ui": {"shell": "apg_python", "requires_theme": True, "template_roots": ["realestate/prm/templates"], "routes": UI_ROUTES},
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
