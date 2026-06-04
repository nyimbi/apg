"""Executable capability contract for APG Property Contracts."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "realestate_con"
CAPABILITY_NAME = "Property Contracts"
CAPABILITY_VERSION = "1.0.0"
CON_EVENT_STREAM = "apg.realestate.con.lifecycle"

SUPPORTED_CONTRACT_TYPES = ["sale_purchase", "management_contract", "construction_contract", "service_agreement", "joint_venture", "development_agreement", "agency_agreement", "facility_management"]
SUPPORTED_CONTRACT_STATUSES = ["draft", "negotiating", "pending_signature", "active", "suspended", "expired", "terminated", "disputed"]
SUPPORTED_PARTY_ROLES = ["buyer", "seller", "landlord", "tenant", "developer", "contractor", "subcontractor", "agent", "managing_agent", "guarantor"]
SUPPORTED_MILESTONE_TYPES = ["payment", "handover", "inspection", "approval", "completion", "possession", "registration", "defect_liability"]
SUPPORTED_CLAUSE_TYPES = ["payment_terms", "penalty", "force_majeure", "termination", "dispute_resolution", "warranties", "insurance", "confidentiality", "indemnity", "variation"]
SUPPORTED_VARIATION_TYPES = ["price_adjustment", "scope_change", "timeline_extension", "party_substitution", "clause_amendment", "schedule_update"]
SUPPORTED_DISPUTE_TYPES = ["payment_dispute", "quality_dispute", "delay_dispute", "scope_dispute", "termination_dispute", "title_dispute"]
SUPPORTED_TERMINATION_REASONS = ["mutual_agreement", "breach", "insolvency", "force_majeure", "non_performance", "expiry", "regulatory"]
SUPPORTED_SIGNATURE_METHODS = ["wet_ink", "digital", "electronic", "notarised", "witnessed"]
SUPPORTED_APPROVAL_LEVELS = ["legal_review", "management", "board", "regulatory", "notary"]
SUPPORTED_DOCUMENT_TYPES = ["contract_draft", "signed_contract", "amendment", "addendum", "schedule", "variation_order", "correspondence", "evidence"]
SUPPORTED_CONTRACTOR_GRADES = ["preferred", "approved", "conditional", "suspended", "blacklisted"]
SUPPORTED_RETENTION_METHODS = ["percentage", "fixed_amount", "milestone_linked", "performance_bond"]
SUPPORTED_CURRENCIES = ["KES", "USD", "EUR", "GBP", "ZAR"]
SUPPORTED_GOVERNING_LAWS = ["Kenya", "Uganda", "Tanzania", "Nigeria", "South_Africa", "England_Wales"]

PROVIDES = [
	"contract_lifecycle_management",
	"contractor_registry_management",
	"milestone_tracking_workflow",
	"variation_order_management",
	"dispute_resolution_workflow",
	"contract_clause_library",
	"retention_management",
	"contract_expiry_alerts",
	"digital_signature_workflow",
	"contract_performance_reporting",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "nlpc", "comp", "mqeb"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/realestate/con/dashboard", "component": "ConDashboard", "permission": "realestate_con:view", "nav_group": "Overview"},
	{"name": "contracts", "path": "/realestate/con/contracts", "component": "ContractRegistry", "permission": "realestate_con:contracts", "nav_group": "Contracts"},
	{"name": "contract-detail", "path": "/realestate/con/contracts/<id>", "component": "ContractDetail", "permission": "realestate_con:contracts", "nav_group": "Contracts"},
	{"name": "contractors", "path": "/realestate/con/contractors", "component": "ContractorRegistry", "permission": "realestate_con:contractors", "nav_group": "Contractors"},
	{"name": "milestones", "path": "/realestate/con/milestones", "component": "MilestoneTracker", "permission": "realestate_con:milestones", "nav_group": "Execution"},
	{"name": "variations", "path": "/realestate/con/variations", "component": "VariationOrderConsole", "permission": "realestate_con:variations", "nav_group": "Execution"},
	{"name": "disputes", "path": "/realestate/con/disputes", "component": "DisputeResolutionConsole", "permission": "realestate_con:disputes", "nav_group": "Disputes"},
	{"name": "clauses", "path": "/realestate/con/clauses", "component": "ClauseLibrary", "permission": "realestate_con:clauses", "nav_group": "Library"},
	{"name": "retention", "path": "/realestate/con/retention", "component": "RetentionConsole", "permission": "realestate_con:retention", "nav_group": "Financial"},
	{"name": "approvals", "path": "/realestate/con/approvals", "component": "ContractApprovalQueue", "permission": "realestate_con:approvals", "nav_group": "Governance"},
	{"name": "expiry-pipeline", "path": "/realestate/con/expiry", "component": "ContractExpiryPipeline", "permission": "realestate_con:view", "nav_group": "Planning"},
	{"name": "reports", "path": "/realestate/con/reports", "component": "ContractReportBuilder", "permission": "realestate_con:reports", "nav_group": "Reporting"},
	{"name": "settings", "path": "/realestate/con/settings", "component": "ConSettings", "permission": "realestate_con:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "realestate_con_registry",
	"tokens": {
		"color.primary": "#1E40AF",
		"color.accent": "#7C3AED",
		"color.success": "#065F46",
		"color.warning": "#92400E",
		"color.danger": "#991B1B",
		"surface.canvas": "#F8FAFC",
		"surface.panel": "#FFFFFF",
		"text.primary": "#0F172A",
		"text.secondary": "#374151",
		"border.radius": "8px",
		"density": "comfortable",
	},
	"components": {
		"contracts": {"icon": "file-signature", "status_indicator": "contract-status-chip"},
		"contractors": {"icon": "hard-hat", "status_indicator": "contractor-grade-chip"},
		"milestones": {"icon": "flag", "status_indicator": "milestone-type-chip"},
		"variations": {"icon": "git-branch", "status_indicator": "variation-status-chip"},
		"disputes": {"icon": "alert-triangle", "status_indicator": "dispute-status-chip"},
		"retention": {"icon": "lock", "status_indicator": "retention-method-chip"},
		"clauses": {"icon": "book", "status_indicator": "clause-type-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": CON_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"contract_created", "contract_executed", "contract_suspended", "contract_terminated",
		"contractor_registered", "contractor_graded",
		"milestone_reached", "milestone_overdue",
		"variation_raised", "variation_approved", "variation_rejected",
		"dispute_raised", "dispute_resolved",
		"retention_released", "contract_expiring_soon",
	],
	"guardrails": [
		"contract_execution_requires_all_signatures",
		"variation_above_threshold_requires_board_approval",
		"dispute_resolution_requires_legal_review",
		"blacklisted_contractor_engagement_denied",
		"retention_release_requires_defect_clearance",
	],
}

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"contracts": {"supported_types": SUPPORTED_CONTRACT_TYPES, "supported_statuses": SUPPORTED_CONTRACT_STATUSES, "supported_signature_methods": SUPPORTED_SIGNATURE_METHODS},
	"parties": {"supported_roles": SUPPORTED_PARTY_ROLES, "supported_governing_laws": SUPPORTED_GOVERNING_LAWS},
	"milestones": {"supported_types": SUPPORTED_MILESTONE_TYPES, "overdue_alert_days": 7},
	"variations": {"supported_types": SUPPORTED_VARIATION_TYPES, "board_approval_threshold": 500000},
	"disputes": {"supported_types": SUPPORTED_DISPUTE_TYPES, "legal_review_required": True},
	"clauses": {"supported_types": SUPPORTED_CLAUSE_TYPES},
	"retention": {"supported_methods": SUPPORTED_RETENTION_METHODS, "default_percentage": 5.0},
	"contractors": {"supported_grades": SUPPORTED_CONTRACTOR_GRADES, "grading_review_months": 12},
	"approvals": {"supported_levels": SUPPORTED_APPROVAL_LEVELS},
	"currencies": {"supported": SUPPORTED_CURRENCIES, "default": "KES"},
	"termination": {"supported_reasons": SUPPORTED_TERMINATION_REASONS},
	"ui": {"enable_dashboard": True, "enable_contracts": True, "enable_contractors": True, "enable_milestones": True},
	"theme": {"default_theme": "realestate_con_registry", "allow_tenant_overrides": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True},
	"observability": {"event_stream": CON_EVENT_STREAM, "stream_processor": "bytewax"},
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "contract_policy_required", "required_action": "attach_contract_policy"}},
	{"name": "contract_type_supported", "condition": {"operation": "create_contract", "contract_type_supported": False}, "effect": {"decision": "deny", "reason": "contract_type_not_supported", "required_action": "select_supported_contract_type"}},
	{"name": "contract_requires_parties", "condition": {"operation": "create_contract", "parties_present": False}, "effect": {"decision": "deny", "reason": "contracting_parties_required", "required_action": "add_contract_parties"}},
	{"name": "contract_requires_governing_law", "condition": {"operation": "create_contract", "governing_law_present": False}, "effect": {"decision": "deny", "reason": "governing_law_required", "required_action": "specify_governing_law"}},
	{"name": "execution_requires_all_signatures", "condition": {"operation": "execute_contract", "all_signatures_present": False}, "effect": {"decision": "deny", "reason": "all_party_signatures_required", "required_action": "collect_all_signatures"}},
	{"name": "execution_requires_legal_review", "condition": {"operation": "execute_contract", "legal_review_complete": False}, "effect": {"decision": "deny", "reason": "legal_review_required_before_execution", "required_action": "complete_legal_review"}},
	{"name": "contractor_blacklisted_engagement_denied", "condition": {"operation": "create_contract", "contractor_grade": "blacklisted"}, "effect": {"decision": "deny", "reason": "blacklisted_contractor_cannot_be_engaged", "required_action": "select_approved_contractor"}},
	{"name": "milestone_requires_contract", "condition": {"operation": "create_milestone", "contract_present": False}, "effect": {"decision": "deny", "reason": "contract_required_for_milestone", "required_action": "link_contract"}},
	{"name": "milestone_date_required", "condition": {"operation": "create_milestone", "due_date_present": False}, "effect": {"decision": "deny", "reason": "milestone_due_date_required", "required_action": "set_milestone_due_date"}},
	{"name": "variation_requires_active_contract", "condition": {"operation": "raise_variation", "contract_status": "active", "contract_active": False}, "effect": {"decision": "deny", "reason": "contract_must_be_active_for_variation", "required_action": "activate_contract_first"}},
	{"name": "variation_above_threshold_requires_board", "condition": {"operation": "raise_variation", "amount_above_threshold": True, "board_approved": False}, "effect": {"decision": "deny", "reason": "board_approval_required_for_large_variation", "required_action": "submit_to_board"}},
	{"name": "dispute_requires_contract", "condition": {"operation": "raise_dispute", "contract_present": False}, "effect": {"decision": "deny", "reason": "contract_required_for_dispute", "required_action": "link_contract_to_dispute"}},
	{"name": "dispute_type_supported", "condition": {"operation": "raise_dispute", "dispute_type_supported": False}, "effect": {"decision": "deny", "reason": "dispute_type_not_supported", "required_action": "select_supported_dispute_type"}},
	{"name": "retention_release_requires_defect_clearance", "condition": {"operation": "release_retention", "defect_liability_cleared": False}, "effect": {"decision": "deny", "reason": "defect_liability_period_not_cleared", "required_action": "obtain_defect_clearance_certificate"}},
	{"name": "retention_release_requires_approval", "condition": {"operation": "release_retention", "approved": False}, "effect": {"decision": "deny", "reason": "retention_release_requires_approval", "required_action": "submit_retention_release_for_approval"}},
	{"name": "termination_requires_reason", "condition": {"operation": "terminate_contract", "reason_present": False}, "effect": {"decision": "deny", "reason": "termination_reason_required", "required_action": "specify_termination_reason"}},
	{"name": "termination_requires_notice_period", "condition": {"operation": "terminate_contract", "notice_period_satisfied": False}, "effect": {"decision": "deny", "reason": "contractual_notice_period_not_satisfied", "required_action": "serve_notice_period"}},
	{"name": "contractor_grade_supported", "condition": {"operation": "grade_contractor", "grade_supported": False}, "effect": {"decision": "deny", "reason": "contractor_grade_not_supported", "required_action": "select_supported_grade"}},
	{"name": "variation_type_supported", "condition": {"operation": "raise_variation", "variation_type_supported": False}, "effect": {"decision": "deny", "reason": "variation_type_not_supported", "required_action": "select_supported_variation_type"}},
	{"name": "document_type_supported", "condition": {"operation": "attach_document", "document_type_supported": False}, "effect": {"decision": "deny", "reason": "document_type_not_supported", "required_action": "select_supported_document_type"}},
	{"name": "cross_tenant_contract_denied", "condition": {"operation_type": "write", "cross_tenant": True}, "effect": {"decision": "deny", "reason": "cross_tenant_contract_not_allowed", "required_action": "use_correct_tenant_context"}},
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
			"properties": {
				"tenant_id": {"type": "string"},
				"ui": {"type": "object"},
				"theme": {"type": "object"},
			},
		},
		"rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": RULES},
		"ui": {"shell": "apg_python", "requires_theme": True, "template_roots": ["realestate/con/templates"], "routes": UI_ROUTES},
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
