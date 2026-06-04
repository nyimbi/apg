"""Executable capability contract for APG Rental Operations."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "realestate_ren"
CAPABILITY_NAME = "Rental Operations"
CAPABILITY_VERSION = "1.0.0"
REN_EVENT_STREAM = "apg.realestate.ren.lifecycle"

SUPPORTED_TENANCY_TYPES = ["assured_shorthold", "fixed_term", "periodic", "licence", "commercial", "regulated", "student", "social_housing", "serviced_office"]
SUPPORTED_TENANCY_STATUSES = ["application", "referencing", "approved", "notice_signed", "active", "notice_served", "holding_over", "vacating", "vacated", "dispute"]
SUPPORTED_RENT_FREQUENCIES = ["weekly", "fortnightly", "monthly", "quarterly", "semi_annual", "annual", "in_advance"]
SUPPORTED_PAYMENT_METHODS = ["bank_transfer", "direct_debit", "standing_order", "cheque", "cash", "mpesa", "credit_card", "debit_card"]
SUPPORTED_ARREARS_STATUSES = ["current", "1_30_days", "31_60_days", "61_90_days", "90_plus_days", "legal_action", "write_off"]
SUPPORTED_DEPOSIT_TYPES = ["cash_deposit", "deposit_replacement_insurance", "guarantor_deposit", "deed_of_guarantee", "zero_deposit"]
SUPPORTED_DEPOSIT_STATUSES = ["held", "registered", "released", "disputed", "deducted", "refunded"]
SUPPORTED_RENEWAL_TYPES = ["fixed_term_renewal", "periodic_continuation", "new_lease", "holdover_regularisation"]
SUPPORTED_NOTICE_TYPES = ["section_21", "section_8", "notice_to_quit", "break_notice", "forfeiture_notice", "rent_increase_notice"]
SUPPORTED_LEGAL_ACTIONS = ["letter_before_action", "county_court_claim", "possession_order", "bailiff_eviction", "charging_order", "attachment_of_earnings"]
SUPPORTED_REFERENCING_TYPES = ["credit_check", "employment_check", "landlord_reference", "right_to_rent", "bank_statement", "guarantor_check"]
SUPPORTED_DEPOSIT_SCHEMES = ["TDS", "DPS", "mydeposits", "client_account", "insured_scheme"]
SUPPORTED_ARREARS_ACTIONS = ["reminder_email", "sms_reminder", "phone_call", "formal_letter", "agent_visit", "legal_referral"]
SUPPORTED_CURRENCIES = ["KES", "USD", "EUR", "GBP", "ZAR"]
SUPPORTED_APPROVAL_LEVELS = ["property_manager", "senior_property_manager", "portfolio_director"]

PROVIDES = [
	"tenancy_lifecycle_management",
	"rent_collection_engine",
	"arrears_management_workflow",
	"deposit_accounting",
	"tenancy_renewal_pipeline",
	"referencing_workflow",
	"notice_management",
	"legal_action_tracking",
	"rent_roll_management",
	"tenancy_performance_reporting",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "comp", "mqeb", "schd"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/realestate/ren/dashboard", "component": "RenDashboard", "permission": "realestate_ren:view", "nav_group": "Overview"},
	{"name": "tenancies", "path": "/realestate/ren/tenancies", "component": "TenancyRegistry", "permission": "realestate_ren:tenancies", "nav_group": "Tenancies"},
	{"name": "tenancy-detail", "path": "/realestate/ren/tenancies/<id>", "component": "TenancyDetail", "permission": "realestate_ren:tenancies", "nav_group": "Tenancies"},
	{"name": "referencing", "path": "/realestate/ren/referencing", "component": "ReferencingWorkflow", "permission": "realestate_ren:referencing", "nav_group": "Onboarding"},
	{"name": "rent-collection", "path": "/realestate/ren/rent-collection", "component": "RentCollectionConsole", "permission": "realestate_ren:rent_collection", "nav_group": "Collections"},
	{"name": "arrears", "path": "/realestate/ren/arrears", "component": "ArrearsManagementConsole", "permission": "realestate_ren:arrears", "nav_group": "Collections"},
	{"name": "deposits", "path": "/realestate/ren/deposits", "component": "DepositAccountingConsole", "permission": "realestate_ren:deposits", "nav_group": "Financial"},
	{"name": "renewals", "path": "/realestate/ren/renewals", "component": "TenancyRenewalPipeline", "permission": "realestate_ren:renewals", "nav_group": "Planning"},
	{"name": "notices", "path": "/realestate/ren/notices", "component": "NoticeManagementConsole", "permission": "realestate_ren:notices", "nav_group": "Legal"},
	{"name": "legal-actions", "path": "/realestate/ren/legal", "component": "LegalActionTracker", "permission": "realestate_ren:legal", "nav_group": "Legal"},
	{"name": "rent-roll", "path": "/realestate/ren/rent-roll", "component": "RentRollView", "permission": "realestate_ren:rent_roll", "nav_group": "Reporting"},
	{"name": "reports", "path": "/realestate/ren/reports", "component": "RentalReportBuilder", "permission": "realestate_ren:reports", "nav_group": "Reporting"},
	{"name": "settings", "path": "/realestate/ren/settings", "component": "RenSettings", "permission": "realestate_ren:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "realestate_ren_operations",
	"tokens": {
		"color.primary": "#1C3D5A",
		"color.accent": "#F59E0B",
		"color.success": "#15803D",
		"color.warning": "#B45309",
		"color.danger": "#DC2626",
		"surface.canvas": "#F9FAFB",
		"surface.panel": "#FFFFFF",
		"text.primary": "#111827",
		"text.secondary": "#6B7280",
		"border.radius": "6px",
		"density": "comfortable",
	},
	"components": {
		"tenancies": {"icon": "home", "status_indicator": "tenancy-status-chip"},
		"rent_collection": {"icon": "credit-card", "status_indicator": "payment-method-chip"},
		"arrears": {"icon": "alert-circle", "status_indicator": "arrears-status-chip"},
		"deposits": {"icon": "lock", "status_indicator": "deposit-status-chip"},
		"renewals": {"icon": "repeat", "status_indicator": "renewal-type-chip"},
		"notices": {"icon": "mail", "status_indicator": "notice-type-chip"},
		"legal_actions": {"icon": "gavel", "status_indicator": "legal-action-chip"},
		"rent_roll": {"icon": "table", "status_indicator": "rent-frequency-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": REN_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"tenancy_created", "tenancy_activated", "tenancy_vacated",
		"rent_received", "rent_overdue", "arrears_escalated",
		"deposit_registered", "deposit_released", "deposit_disputed",
		"renewal_initiated", "renewal_completed",
		"notice_served", "legal_action_commenced",
		"right_to_rent_expiry_alert", "referencing_completed",
	],
	"guardrails": [
		"deposit_must_be_registered_before_tenancy_activation",
		"right_to_rent_check_required_for_residential",
		"legal_action_requires_arrears_threshold",
		"deposit_deduction_requires_evidence",
		"notice_served_triggers_void_pipeline",
	],
}

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"tenancies": {"supported_types": SUPPORTED_TENANCY_TYPES, "supported_statuses": SUPPORTED_TENANCY_STATUSES},
	"rent": {"supported_frequencies": SUPPORTED_RENT_FREQUENCIES, "supported_payment_methods": SUPPORTED_PAYMENT_METHODS, "supported_currencies": SUPPORTED_CURRENCIES},
	"arrears": {"supported_statuses": SUPPORTED_ARREARS_STATUSES, "supported_actions": SUPPORTED_ARREARS_ACTIONS, "legal_threshold_days": 90},
	"deposits": {"supported_types": SUPPORTED_DEPOSIT_TYPES, "supported_statuses": SUPPORTED_DEPOSIT_STATUSES, "supported_schemes": SUPPORTED_DEPOSIT_SCHEMES, "registration_required": True},
	"renewals": {"supported_types": SUPPORTED_RENEWAL_TYPES, "early_warning_days": 90},
	"notices": {"supported_types": SUPPORTED_NOTICE_TYPES},
	"legal": {"supported_actions": SUPPORTED_LEGAL_ACTIONS},
	"referencing": {"supported_types": SUPPORTED_REFERENCING_TYPES},
	"approvals": {"supported_levels": SUPPORTED_APPROVAL_LEVELS},
	"ui": {"enable_dashboard": True, "enable_tenancies": True, "enable_arrears": True, "enable_deposits": True},
	"theme": {"default_theme": "realestate_ren_operations", "allow_tenant_overrides": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True},
	"observability": {"event_stream": REN_EVENT_STREAM, "stream_processor": "bytewax"},
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "rental_policy_required", "required_action": "attach_rental_policy"}},
	{"name": "tenancy_type_supported", "condition": {"operation": "create_tenancy", "tenancy_type_supported": False}, "effect": {"decision": "deny", "reason": "tenancy_type_not_supported", "required_action": "select_supported_tenancy_type"}},
	{"name": "tenancy_requires_unit", "condition": {"operation": "create_tenancy", "unit_present": False}, "effect": {"decision": "deny", "reason": "unit_required_for_tenancy", "required_action": "link_unit"}},
	{"name": "tenancy_requires_tenant", "condition": {"operation": "create_tenancy", "tenant_present": False}, "effect": {"decision": "deny", "reason": "tenant_required_for_tenancy", "required_action": "link_tenant"}},
	{"name": "activation_requires_deposit_registered", "condition": {"operation": "activate_tenancy", "deposit_registered": False}, "effect": {"decision": "deny", "reason": "deposit_must_be_registered_before_activation", "required_action": "register_deposit"}},
	{"name": "activation_requires_referencing_complete", "condition": {"operation": "activate_tenancy", "referencing_complete": False}, "effect": {"decision": "deny", "reason": "referencing_must_be_complete_before_activation", "required_action": "complete_referencing"}},
	{"name": "right_to_rent_required_for_residential", "condition": {"operation": "activate_tenancy", "tenancy_type": "assured_shorthold", "right_to_rent_checked": False}, "effect": {"decision": "deny", "reason": "right_to_rent_check_required_for_residential_tenancy", "required_action": "complete_right_to_rent_check"}},
	{"name": "rent_frequency_supported", "condition": {"operation": "create_tenancy", "rent_frequency_supported": False}, "effect": {"decision": "deny", "reason": "rent_frequency_not_supported", "required_action": "select_supported_rent_frequency"}},
	{"name": "payment_method_supported", "condition": {"operation": "record_payment", "payment_method_supported": False}, "effect": {"decision": "deny", "reason": "payment_method_not_supported", "required_action": "select_supported_payment_method"}},
	{"name": "deposit_type_supported", "condition": {"operation": "register_deposit", "deposit_type_supported": False}, "effect": {"decision": "deny", "reason": "deposit_type_not_supported", "required_action": "select_supported_deposit_type"}},
	{"name": "deposit_deduction_requires_evidence", "condition": {"operation": "deduct_from_deposit", "evidence_present": False}, "effect": {"decision": "deny", "reason": "evidence_required_for_deposit_deduction", "required_action": "attach_evidence"}},
	{"name": "deposit_deduction_cannot_exceed_held_amount", "condition": {"operation": "deduct_from_deposit", "deduction_exceeds_held": True}, "effect": {"decision": "deny", "reason": "deduction_cannot_exceed_held_deposit_amount", "required_action": "adjust_deduction_amount"}},
	{"name": "legal_action_requires_arrears_threshold", "condition": {"operation": "commence_legal_action", "arrears_above_threshold": False}, "effect": {"decision": "deny", "reason": "arrears_threshold_not_met_for_legal_action", "required_action": "follow_arrears_escalation_process"}},
	{"name": "notice_type_supported", "condition": {"operation": "serve_notice", "notice_type_supported": False}, "effect": {"decision": "deny", "reason": "notice_type_not_supported", "required_action": "select_supported_notice_type"}},
	{"name": "renewal_type_supported", "condition": {"operation": "initiate_renewal", "renewal_type_supported": False}, "effect": {"decision": "deny", "reason": "renewal_type_not_supported", "required_action": "select_supported_renewal_type"}},
	{"name": "referencing_type_supported", "condition": {"operation": "run_referencing", "referencing_type_supported": False}, "effect": {"decision": "deny", "reason": "referencing_type_not_supported", "required_action": "select_supported_referencing_type"}},
	{"name": "arrears_action_type_supported", "condition": {"operation": "take_arrears_action", "action_type_supported": False}, "effect": {"decision": "deny", "reason": "arrears_action_type_not_supported", "required_action": "select_supported_arrears_action"}},
	{"name": "vacated_tenancy_modification_restricted", "condition": {"operation_type": "write", "tenancy_status": "vacated"}, "effect": {"decision": "deny", "reason": "vacated_tenancy_cannot_be_modified", "required_action": "create_new_tenancy_record"}},
	{"name": "cross_tenant_rental_denied", "condition": {"operation_type": "write", "cross_tenant": True}, "effect": {"decision": "deny", "reason": "cross_tenant_rental_operations_not_allowed", "required_action": "use_correct_tenant_context"}},
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
		"ui": {"shell": "apg_python", "requires_theme": True, "template_roots": ["realestate/ren/templates"], "routes": UI_ROUTES},
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
