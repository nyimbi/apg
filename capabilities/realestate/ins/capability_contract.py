"""Executable capability contract for APG Property Insurance."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "realestate_ins"
CAPABILITY_NAME = "Property Insurance"
CAPABILITY_VERSION = "1.0.0"
INS_EVENT_STREAM = "apg.realestate.ins.lifecycle"

SUPPORTED_POLICY_TYPES = ["property_all_risk", "fire_perils", "public_liability", "employers_liability", "professional_indemnity", "contractors_all_risk", "fidelity_guarantee", "loss_of_rent", "terrorism", "flood", "earthquake"]
SUPPORTED_CLAIM_STATUSES = ["lodged", "under_investigation", "awaiting_assessment", "approved", "partially_approved", "rejected", "appealed", "settled", "closed"]
SUPPORTED_PERIL_TYPES = ["fire", "flood", "theft", "malicious_damage", "storm", "subsidence", "impact", "explosion", "escape_of_water", "terrorism", "earthquake"]
SUPPORTED_ASSET_TYPES = ["building", "plant_equipment", "fit_out", "fixtures", "stock", "electronic_equipment", "vehicles", "art_valuables"]
SUPPORTED_VALUATION_BASES = ["reinstatement_cost", "market_value", "agreed_value", "indemnity_value", "replacement_cost"]
SUPPORTED_PREMIUM_ALLOCATION_METHODS = ["floor_area", "insured_value", "pro_rata", "risk_weighted", "metered"]
SUPPORTED_CLAIM_TYPES = ["partial_loss", "total_loss", "business_interruption", "liability", "third_party", "ad_hoc"]
SUPPORTED_INSURER_GRADES = ["preferred", "approved", "conditional", "suspended"]
SUPPORTED_BROKER_ROLES = ["lead_broker", "co_broker", "reinsurance_broker", "claims_handler"]
SUPPORTED_COVERAGE_STATUSES = ["active", "lapsed", "expiring_soon", "expired", "cancelled", "endorsed"]
SUPPORTED_ENDORSEMENT_TYPES = ["addition_of_property", "deletion_of_property", "sum_insured_change", "premium_adjustment", "clause_amendment", "extension", "reinstatement"]
SUPPORTED_DEDUCTIBLE_TYPES = ["fixed", "percentage", "franchise", "excess"]
SUPPORTED_RENEWAL_STATUSES = ["pending", "in_negotiation", "quoted", "accepted", "bound", "lapsed"]
SUPPORTED_GAP_SEVERITIES = ["critical", "high", "medium", "low"]
SUPPORTED_CURRENCIES = ["KES", "USD", "EUR", "GBP", "ZAR"]

PROVIDES = [
	"policy_lifecycle_management",
	"asset_schedule_management",
	"claims_processing_workflow",
	"premium_allocation_engine",
	"coverage_gap_analysis",
	"endorsement_management",
	"insurer_broker_registry",
	"renewal_pipeline_tracking",
	"insurance_reporting",
	"compliance_certificate_management",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "comp", "mqeb", "schd"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/realestate/ins/dashboard", "component": "InsDashboard", "permission": "realestate_ins:view", "nav_group": "Overview"},
	{"name": "policies", "path": "/realestate/ins/policies", "component": "PolicyRegistry", "permission": "realestate_ins:policies", "nav_group": "Policies"},
	{"name": "asset-schedule", "path": "/realestate/ins/assets", "component": "AssetScheduleConsole", "permission": "realestate_ins:assets", "nav_group": "Assets"},
	{"name": "claims", "path": "/realestate/ins/claims", "component": "ClaimsProcessingQueue", "permission": "realestate_ins:claims", "nav_group": "Claims"},
	{"name": "premium-allocation", "path": "/realestate/ins/premiums", "component": "PremiumAllocationConsole", "permission": "realestate_ins:premiums", "nav_group": "Financial"},
	{"name": "coverage-gaps", "path": "/realestate/ins/gaps", "component": "CoverageGapAnalyser", "permission": "realestate_ins:gaps", "nav_group": "Analysis"},
	{"name": "endorsements", "path": "/realestate/ins/endorsements", "component": "EndorsementConsole", "permission": "realestate_ins:endorsements", "nav_group": "Policies"},
	{"name": "insurers", "path": "/realestate/ins/insurers", "component": "InsurerRegistry", "permission": "realestate_ins:insurers", "nav_group": "Registry"},
	{"name": "brokers", "path": "/realestate/ins/brokers", "component": "BrokerRegistry", "permission": "realestate_ins:brokers", "nav_group": "Registry"},
	{"name": "renewals", "path": "/realestate/ins/renewals", "component": "RenewalPipeline", "permission": "realestate_ins:renewals", "nav_group": "Planning"},
	{"name": "certificates", "path": "/realestate/ins/certificates", "component": "CertificateConsole", "permission": "realestate_ins:certificates", "nav_group": "Compliance"},
	{"name": "reports", "path": "/realestate/ins/reports", "component": "InsuranceReportBuilder", "permission": "realestate_ins:reports", "nav_group": "Reporting"},
	{"name": "settings", "path": "/realestate/ins/settings", "component": "InsSettings", "permission": "realestate_ins:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "realestate_ins_cover",
	"tokens": {
		"color.primary": "#0C4A6E",
		"color.accent": "#0369A1",
		"color.success": "#14532D",
		"color.warning": "#78350F",
		"color.danger": "#7F1D1D",
		"surface.canvas": "#F0F9FF",
		"surface.panel": "#FFFFFF",
		"text.primary": "#0C0A09",
		"text.secondary": "#44403C",
		"border.radius": "6px",
		"density": "comfortable",
	},
	"components": {
		"policies": {"icon": "shield", "status_indicator": "coverage-status-chip"},
		"assets": {"icon": "building", "status_indicator": "asset-type-chip"},
		"claims": {"icon": "alert-octagon", "status_indicator": "claim-status-chip"},
		"premiums": {"icon": "dollar-sign", "status_indicator": "allocation-method-chip"},
		"coverage_gaps": {"icon": "alert-circle", "status_indicator": "gap-severity-chip"},
		"endorsements": {"icon": "edit-3", "status_indicator": "endorsement-type-chip"},
		"renewals": {"icon": "refresh-cw", "status_indicator": "renewal-status-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": INS_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"policy_created", "policy_bound", "policy_lapsed", "policy_expired", "policy_cancelled",
		"asset_added_to_schedule", "asset_removed_from_schedule",
		"claim_lodged", "claim_assessed", "claim_approved", "claim_rejected", "claim_settled",
		"premium_allocated", "endorsement_issued", "renewal_due", "coverage_gap_detected",
		"certificate_issued",
	],
	"guardrails": [
		"claim_against_lapsed_policy_denied",
		"settlement_above_threshold_requires_approval",
		"endorsed_sum_cannot_exceed_market_value",
		"coverage_gap_critical_triggers_alert",
		"suspended_insurer_policy_cannot_be_bound",
	],
}

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"policies": {"supported_types": SUPPORTED_POLICY_TYPES, "supported_statuses": SUPPORTED_COVERAGE_STATUSES, "supported_valuation_bases": SUPPORTED_VALUATION_BASES},
	"claims": {"supported_statuses": SUPPORTED_CLAIM_STATUSES, "supported_types": SUPPORTED_CLAIM_TYPES, "large_claim_threshold": 1000000},
	"assets": {"supported_types": SUPPORTED_ASSET_TYPES, "supported_perils": SUPPORTED_PERIL_TYPES},
	"premiums": {"supported_allocation_methods": SUPPORTED_PREMIUM_ALLOCATION_METHODS},
	"endorsements": {"supported_types": SUPPORTED_ENDORSEMENT_TYPES},
	"deductibles": {"supported_types": SUPPORTED_DEDUCTIBLE_TYPES},
	"insurers": {"supported_grades": SUPPORTED_INSURER_GRADES},
	"brokers": {"supported_roles": SUPPORTED_BROKER_ROLES},
	"renewals": {"supported_statuses": SUPPORTED_RENEWAL_STATUSES, "early_warning_days": 90},
	"gaps": {"supported_severities": SUPPORTED_GAP_SEVERITIES, "auto_alert_on_critical": True},
	"currencies": {"supported": SUPPORTED_CURRENCIES, "default": "KES"},
	"ui": {"enable_dashboard": True, "enable_policies": True, "enable_claims": True, "enable_gap_analysis": True},
	"theme": {"default_theme": "realestate_ins_cover", "allow_tenant_overrides": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True},
	"observability": {"event_stream": INS_EVENT_STREAM, "stream_processor": "bytewax"},
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "insurance_policy_required", "required_action": "attach_insurance_policy"}},
	{"name": "policy_type_supported", "condition": {"operation": "create_policy", "policy_type_supported": False}, "effect": {"decision": "deny", "reason": "policy_type_not_supported", "required_action": "select_supported_policy_type"}},
	{"name": "policy_requires_insurer", "condition": {"operation": "create_policy", "insurer_present": False}, "effect": {"decision": "deny", "reason": "insurer_required_for_policy", "required_action": "link_insurer"}},
	{"name": "suspended_insurer_cannot_bind", "condition": {"operation": "bind_policy", "insurer_grade": "suspended"}, "effect": {"decision": "deny", "reason": "suspended_insurer_cannot_bind_new_policy", "required_action": "select_approved_insurer"}},
	{"name": "policy_requires_asset_schedule", "condition": {"operation": "bind_policy", "asset_schedule_present": False}, "effect": {"decision": "deny", "reason": "asset_schedule_required_before_binding", "required_action": "complete_asset_schedule"}},
	{"name": "claim_requires_active_policy", "condition": {"operation": "lodge_claim", "policy_active": False}, "effect": {"decision": "deny", "reason": "policy_must_be_active_to_lodge_claim", "required_action": "reinstate_or_renew_policy"}},
	{"name": "claim_peril_must_be_covered", "condition": {"operation": "lodge_claim", "peril_covered": False}, "effect": {"decision": "deny", "reason": "claimed_peril_not_covered_by_policy", "required_action": "verify_policy_coverage"}},
	{"name": "claim_type_supported", "condition": {"operation": "lodge_claim", "claim_type_supported": False}, "effect": {"decision": "deny", "reason": "claim_type_not_supported", "required_action": "select_supported_claim_type"}},
	{"name": "large_claim_requires_approval", "condition": {"operation": "approve_claim", "amount_above_threshold": True, "senior_approved": False}, "effect": {"decision": "deny", "reason": "large_claim_requires_senior_approval", "required_action": "escalate_to_senior_approver"}},
	{"name": "settlement_cannot_exceed_sum_insured", "condition": {"operation": "settle_claim", "settlement_exceeds_sum_insured": True}, "effect": {"decision": "deny", "reason": "settlement_cannot_exceed_sum_insured", "required_action": "adjust_settlement_amount"}},
	{"name": "premium_allocation_method_supported", "condition": {"operation": "allocate_premium", "method_supported": False}, "effect": {"decision": "deny", "reason": "allocation_method_not_supported", "required_action": "select_supported_allocation_method"}},
	{"name": "endorsement_type_supported", "condition": {"operation": "issue_endorsement", "endorsement_type_supported": False}, "effect": {"decision": "deny", "reason": "endorsement_type_not_supported", "required_action": "select_supported_endorsement_type"}},
	{"name": "endorsed_sum_cannot_exceed_market_value", "condition": {"operation": "issue_endorsement", "endorsed_sum_exceeds_market_value": True}, "effect": {"decision": "deny", "reason": "endorsed_sum_insured_exceeds_market_value", "required_action": "obtain_fresh_valuation"}},
	{"name": "valuation_basis_supported", "condition": {"operation": "create_policy", "valuation_basis_supported": False}, "effect": {"decision": "deny", "reason": "valuation_basis_not_supported", "required_action": "select_supported_valuation_basis"}},
	{"name": "asset_type_supported", "condition": {"operation": "add_asset_to_schedule", "asset_type_supported": False}, "effect": {"decision": "deny", "reason": "asset_type_not_supported", "required_action": "select_supported_asset_type"}},
	{"name": "renewal_requires_broker_confirmation", "condition": {"operation": "bind_renewal", "broker_confirmed": False}, "effect": {"decision": "deny", "reason": "broker_confirmation_required_for_renewal", "required_action": "obtain_broker_confirmation"}},
	{"name": "critical_gap_triggers_mandatory_alert", "condition": {"operation": "analyse_gaps", "critical_gap_detected": True, "alert_sent": False}, "effect": {"decision": "deny", "reason": "critical_coverage_gap_alert_mandatory", "required_action": "send_coverage_gap_alert"}},
	{"name": "certificate_requires_active_policy", "condition": {"operation": "issue_certificate", "policy_active": False}, "effect": {"decision": "deny", "reason": "certificate_requires_active_policy", "required_action": "activate_policy_before_issuing_certificate"}},
	{"name": "cross_tenant_insurance_denied", "condition": {"operation_type": "write", "cross_tenant": True}, "effect": {"decision": "deny", "reason": "cross_tenant_insurance_not_allowed", "required_action": "use_correct_tenant_context"}},
	{"name": "deductible_type_supported", "condition": {"operation": "set_deductible", "deductible_type_supported": False}, "effect": {"decision": "deny", "reason": "deductible_type_not_supported", "required_action": "select_supported_deductible_type"}},
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
		"ui": {"shell": "apg_python", "requires_theme": True, "template_roots": ["realestate/ins/templates"], "routes": UI_ROUTES},
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
