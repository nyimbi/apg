"""Executable capability contract for APG Lease Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "realestate_lea"
CAPABILITY_NAME = "Lease Management"
CAPABILITY_VERSION = "1.0.0"
LEA_EVENT_STREAM = "apg.realestate.lea.lifecycle"

SUPPORTED_LEASE_TYPES = ["commercial", "retail", "industrial", "residential", "ground_lease", "sublease", "licence_to_occupy", "peppercorn", "assured_shorthold", "regulated"]
SUPPORTED_LEASE_STATUSES = ["heads_of_terms", "negotiating", "signed", "active", "holding_over", "notice_served", "expired", "surrendered", "forfeited", "assigned"]
SUPPORTED_ESCALATION_TYPES = ["fixed_percentage", "cpi_linked", "open_market_review", "ratchet", "turnover_linked", "base_plus_variable", "stepped"]
SUPPORTED_OPTION_TYPES = ["break_option_tenant", "break_option_landlord", "renewal_option", "purchase_option", "expansion_option", "contraction_option", "extension_option"]
SUPPORTED_RENT_REVIEW_TYPES = ["upward_only", "upward_downward", "fixed", "open_market", "indexed"]
SUPPORTED_IFRS16_CATEGORIES = ["finance_lease", "operating_lease", "short_term_exemption", "low_value_exemption"]
SUPPORTED_ASC842_CATEGORIES = ["finance_lease", "operating_lease", "practical_expedient_short_term", "practical_expedient_low_value"]
SUPPORTED_HOLDING_OVER_TYPES = ["statutory", "contractual", "periodic_tenancy", "licence"]
SUPPORTED_ASSIGNMENT_TYPES = ["absolute_assignment", "assignment_with_guarantee", "sub_letting", "licence_assignment"]
SUPPORTED_DILAPIDATION_TYPES = {"schedule_of_condition": "pre_lease", "interim_schedule": "mid_lease", "terminal_schedule": "end_lease"}
SUPPORTED_CURRENCIES = ["KES", "USD", "EUR", "GBP", "ZAR"]
SUPPORTED_AREA_UNITS = ["sqm", "sqft", "acres", "hectares"]
SUPPORTED_NOTICE_TYPES = ["break_notice", "renewal_notice", "quit_notice", "rent_review_notice", "forfeiture_notice"]
SUPPORTED_APPROVAL_LEVELS = ["property_manager", "asset_manager", "investment_committee", "board"]
SUPPORTED_ABSTRACTION_STATUSES = ["pending", "in_progress", "complete", "verified", "exception"]

PROVIDES = [
	"lease_abstraction_engine",
	"rent_escalation_scheduler",
	"lease_option_tracker",
	"ifrs16_asc842_compliance",
	"lease_expiry_pipeline",
	"rent_review_workflow",
	"lease_assignment_management",
	"dilapidation_management",
	"lease_renewal_workflow",
	"lease_performance_reporting",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "nlpc", "comp", "mqeb", "schd"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/realestate/lea/dashboard", "component": "LeaDashboard", "permission": "realestate_lea:view", "nav_group": "Overview"},
	{"name": "leases", "path": "/realestate/lea/leases", "component": "LeaseRegistry", "permission": "realestate_lea:leases", "nav_group": "Leases"},
	{"name": "lease-detail", "path": "/realestate/lea/leases/<id>", "component": "LeaseDetail", "permission": "realestate_lea:leases", "nav_group": "Leases"},
	{"name": "abstraction", "path": "/realestate/lea/abstraction", "component": "LeaseAbstractionWorkbench", "permission": "realestate_lea:abstraction", "nav_group": "Abstraction"},
	{"name": "escalations", "path": "/realestate/lea/escalations", "component": "RentEscalationScheduler", "permission": "realestate_lea:escalations", "nav_group": "Rent"},
	{"name": "rent-reviews", "path": "/realestate/lea/rent-reviews", "component": "RentReviewWorkflow", "permission": "realestate_lea:rent_reviews", "nav_group": "Rent"},
	{"name": "options", "path": "/realestate/lea/options", "component": "LeaseOptionTracker", "permission": "realestate_lea:options", "nav_group": "Options"},
	{"name": "ifrs16", "path": "/realestate/lea/ifrs16", "component": "Ifrs16ComplianceConsole", "permission": "realestate_lea:ifrs16", "nav_group": "Compliance"},
	{"name": "expiry-pipeline", "path": "/realestate/lea/expiry", "component": "LeaseExpiryPipeline", "permission": "realestate_lea:view", "nav_group": "Planning"},
	{"name": "assignments", "path": "/realestate/lea/assignments", "component": "LeaseAssignmentConsole", "permission": "realestate_lea:assignments", "nav_group": "Transactions"},
	{"name": "dilapidations", "path": "/realestate/lea/dilapidations", "component": "DilapidationConsole", "permission": "realestate_lea:dilapidations", "nav_group": "Transactions"},
	{"name": "renewals", "path": "/realestate/lea/renewals", "component": "LeaseRenewalPipeline", "permission": "realestate_lea:renewals", "nav_group": "Planning"},
	{"name": "reports", "path": "/realestate/lea/reports", "component": "LeaseReportBuilder", "permission": "realestate_lea:reports", "nav_group": "Reporting"},
	{"name": "settings", "path": "/realestate/lea/settings", "component": "LeaSettings", "permission": "realestate_lea:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "realestate_lea_portfolio",
	"tokens": {
		"color.primary": "#4338CA",
		"color.accent": "#0D9488",
		"color.success": "#166534",
		"color.warning": "#A16207",
		"color.danger": "#B91C1C",
		"surface.canvas": "#F5F3FF",
		"surface.panel": "#FFFFFF",
		"text.primary": "#1E1B4B",
		"text.secondary": "#4B5563",
		"border.radius": "8px",
		"density": "comfortable",
	},
	"components": {
		"leases": {"icon": "file-contract", "status_indicator": "lease-status-chip"},
		"escalations": {"icon": "trending-up", "status_indicator": "escalation-type-chip"},
		"options": {"icon": "toggle-right", "status_indicator": "option-type-chip"},
		"rent_reviews": {"icon": "refresh-cw", "status_indicator": "review-type-chip"},
		"ifrs16": {"icon": "layers", "status_indicator": "ifrs16-category-chip"},
		"assignments": {"icon": "arrow-right-circle", "status_indicator": "assignment-type-chip"},
		"dilapidations": {"icon": "tool", "status_indicator": "dilapidation-type-chip"},
		"renewals": {"icon": "repeat", "status_indicator": "renewal-status-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": LEA_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"lease_created", "lease_signed", "lease_activated", "lease_expired", "lease_surrendered",
		"rent_escalation_applied", "rent_review_commenced", "rent_review_agreed",
		"option_exercised", "option_lapsed", "option_expiring_soon",
		"ifrs16_schedule_generated", "lease_expiry_alert_sent",
		"assignment_completed", "subletting_approved",
		"dilapidation_schedule_issued", "lease_renewal_completed",
	],
	"guardrails": [
		"rent_review_cannot_backdate_beyond_review_date",
		"ifrs16_reclassification_requires_auditor",
		"option_exercise_requires_notice_period",
		"assignment_requires_landlord_consent",
		"forfeiture_requires_legal_process",
	],
}

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"leases": {"supported_types": SUPPORTED_LEASE_TYPES, "supported_statuses": SUPPORTED_LEASE_STATUSES, "supported_currencies": SUPPORTED_CURRENCIES, "supported_area_units": SUPPORTED_AREA_UNITS},
	"escalations": {"supported_types": SUPPORTED_ESCALATION_TYPES, "auto_apply": False},
	"options": {"supported_types": SUPPORTED_OPTION_TYPES, "early_warning_days": 180},
	"rent_reviews": {"supported_types": SUPPORTED_RENT_REVIEW_TYPES, "notice_required_days": 30},
	"ifrs16": {"supported_categories": SUPPORTED_IFRS16_CATEGORIES, "asc842_categories": SUPPORTED_ASC842_CATEGORIES},
	"assignments": {"supported_types": SUPPORTED_ASSIGNMENT_TYPES, "landlord_consent_required": True},
	"notices": {"supported_types": SUPPORTED_NOTICE_TYPES},
	"abstractions": {"supported_statuses": SUPPORTED_ABSTRACTION_STATUSES, "ai_assisted": True},
	"approvals": {"supported_levels": SUPPORTED_APPROVAL_LEVELS},
	"ui": {"enable_dashboard": True, "enable_leases": True, "enable_ifrs16": True, "enable_options": True},
	"theme": {"default_theme": "realestate_lea_portfolio", "allow_tenant_overrides": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True},
	"observability": {"event_stream": LEA_EVENT_STREAM, "stream_processor": "bytewax"},
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "lease_policy_required", "required_action": "attach_lease_policy"}},
	{"name": "lease_type_supported", "condition": {"operation": "create_lease", "lease_type_supported": False}, "effect": {"decision": "deny", "reason": "lease_type_not_supported", "required_action": "select_supported_lease_type"}},
	{"name": "lease_requires_property", "condition": {"operation": "create_lease", "property_present": False}, "effect": {"decision": "deny", "reason": "property_required_for_lease", "required_action": "link_property"}},
	{"name": "lease_requires_tenant", "condition": {"operation": "create_lease", "tenant_present": False}, "effect": {"decision": "deny", "reason": "tenant_required_for_lease", "required_action": "link_tenant"}},
	{"name": "lease_requires_commencement_date", "condition": {"operation": "activate_lease", "commencement_date_present": False}, "effect": {"decision": "deny", "reason": "commencement_date_required_for_activation", "required_action": "set_commencement_date"}},
	{"name": "lease_requires_expiry_date", "condition": {"operation": "activate_lease", "expiry_date_present": False}, "effect": {"decision": "deny", "reason": "expiry_date_required_for_activation", "required_action": "set_expiry_date"}},
	{"name": "escalation_type_supported", "condition": {"operation": "create_escalation", "escalation_type_supported": False}, "effect": {"decision": "deny", "reason": "escalation_type_not_supported", "required_action": "select_supported_escalation_type"}},
	{"name": "rent_review_cannot_backdate", "condition": {"operation": "apply_rent_review", "review_date_in_past": True, "backdating_authorised": False}, "effect": {"decision": "deny", "reason": "rent_review_backdating_requires_authorisation", "required_action": "obtain_backdating_authorisation"}},
	{"name": "option_type_supported", "condition": {"operation": "create_option", "option_type_supported": False}, "effect": {"decision": "deny", "reason": "option_type_not_supported", "required_action": "select_supported_option_type"}},
	{"name": "option_exercise_requires_notice", "condition": {"operation": "exercise_option", "notice_served": False}, "effect": {"decision": "deny", "reason": "notice_required_before_option_exercise", "required_action": "serve_option_notice"}},
	{"name": "option_exercise_window_required", "condition": {"operation": "exercise_option", "within_exercise_window": False}, "effect": {"decision": "deny", "reason": "option_exercise_outside_permitted_window", "required_action": "check_option_exercise_dates"}},
	{"name": "ifrs16_requires_discount_rate", "condition": {"operation": "generate_ifrs16_schedule", "discount_rate_present": False}, "effect": {"decision": "deny", "reason": "discount_rate_required_for_ifrs16", "required_action": "set_discount_rate"}},
	{"name": "ifrs16_reclassification_requires_auditor", "condition": {"operation": "reclassify_ifrs16", "auditor_approved": False}, "effect": {"decision": "deny", "reason": "auditor_approval_required_for_ifrs16_reclassification", "required_action": "obtain_auditor_approval"}},
	{"name": "assignment_requires_landlord_consent", "condition": {"operation": "assign_lease", "landlord_consent_obtained": False}, "effect": {"decision": "deny", "reason": "landlord_consent_required_for_assignment", "required_action": "obtain_landlord_consent"}},
	{"name": "assignment_type_supported", "condition": {"operation": "assign_lease", "assignment_type_supported": False}, "effect": {"decision": "deny", "reason": "assignment_type_not_supported", "required_action": "select_supported_assignment_type"}},
	{"name": "surrender_requires_active_lease", "condition": {"operation": "surrender_lease", "lease_status": "active", "lease_active": False}, "effect": {"decision": "deny", "reason": "lease_must_be_active_to_surrender", "required_action": "check_lease_status"}},
	{"name": "forfeiture_requires_legal_process", "condition": {"operation": "forfeit_lease", "legal_process_complete": False}, "effect": {"decision": "deny", "reason": "legal_process_required_for_forfeiture", "required_action": "complete_legal_forfeiture_process"}},
	{"name": "abstraction_verification_required", "condition": {"operation": "activate_lease", "abstraction_verified": False}, "effect": {"decision": "deny", "reason": "lease_abstraction_must_be_verified_before_activation", "required_action": "verify_lease_abstraction"}},
	{"name": "renewal_requires_investment_committee", "condition": {"operation": "renew_lease", "lease_value_above_threshold": True, "investment_committee_approved": False}, "effect": {"decision": "deny", "reason": "investment_committee_approval_required_for_major_renewal", "required_action": "submit_to_investment_committee"}},
	{"name": "cross_tenant_lease_denied", "condition": {"operation_type": "write", "cross_tenant": True}, "effect": {"decision": "deny", "reason": "cross_tenant_lease_not_allowed", "required_action": "use_correct_tenant_context"}},
	{"name": "rent_review_type_supported", "condition": {"operation": "commence_rent_review", "review_type_supported": False}, "effect": {"decision": "deny", "reason": "rent_review_type_not_supported", "required_action": "select_supported_review_type"}},
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
		"ui": {"shell": "apg_python", "requires_theme": True, "template_roots": ["realestate/lea/templates"], "routes": UI_ROUTES},
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
