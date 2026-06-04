"""Executable capability contract for APG Tax Administration."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "government_tax"
CAPABILITY_NAME = "Tax Administration"
CAPABILITY_VERSION = "1.0.0"
TAX_EVENT_STREAM = "apg.government.tax.lifecycle"

SUPPORTED_TAX_TYPES = ["income_tax", "vat", "corporate_tax", "withholding_tax", "capital_gains_tax", "excise_duty", "customs_duty", "stamp_duty", "rental_income_tax", "turnover_tax", "digital_services_tax"]
SUPPORTED_REGISTRATION_STATUSES = ["pending", "active", "suspended", "deregistered", "under_investigation"]
SUPPORTED_RETURN_TYPES = ["monthly_vat", "annual_income", "quarterly_advance", "withholding_tax_return", "corporate_annual", "customs_entry"]
SUPPORTED_RETURN_STATUSES = ["draft", "filed", "amended", "under_review", "assessed", "disputed", "finalised"]
SUPPORTED_ASSESSMENT_TYPES = ["self_assessment", "amended_assessment", "best_judgement", "audit_assessment", "estimated_assessment"]
SUPPORTED_OBJECTION_STATUSES = ["submitted", "under_review", "upheld", "partially_upheld", "dismissed", "appealed"]
SUPPORTED_DEBT_COLLECTION_METHODS = ["payment_plan", "garnishment", "asset_seizure", "third_party_demand", "legal_proceedings", "write_off"]
SUPPORTED_AUDIT_TYPES = ["desk_audit", "field_audit", "it_audit", "transfer_pricing", "vat_refund_audit", "forensic_audit"]
SUPPORTED_AUDIT_STATUSES = ["planned", "in_progress", "completed", "report_issued", "objection_filed", "finalised"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["registration_officer", "return_processor", "assessment_reviewer", "debt_collector", "audit_analyst"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"registrations": {
		"supported_tax_types": SUPPORTED_TAX_TYPES,
		"supported_statuses": SUPPORTED_REGISTRATION_STATUSES,
		"pin_required": True,
		"national_id_required": True,
		"business_registration_required": False,
		"evidence_required": True,
	},
	"returns": {
		"supported_return_types": SUPPORTED_RETURN_TYPES,
		"supported_statuses": SUPPORTED_RETURN_STATUSES,
		"taxpayer_pin_required": True,
		"period_required": True,
		"evidence_required": True,
	},
	"assessments": {
		"supported_assessment_types": SUPPORTED_ASSESSMENT_TYPES,
		"return_required": True,
		"assessor_required": True,
		"evidence_required": True,
	},
	"objections": {
		"supported_statuses": SUPPORTED_OBJECTION_STATUSES,
		"assessment_required": True,
		"grounds_required": True,
		"evidence_required": True,
		"deadline_enforced": True,
	},
	"debt_collection": {
		"supported_methods": SUPPORTED_DEBT_COLLECTION_METHODS,
		"assessed_liability_required": True,
		"demand_notice_required": True,
		"approval_required": True,
		"evidence_required": True,
	},
	"audits": {
		"supported_audit_types": SUPPORTED_AUDIT_TYPES,
		"supported_statuses": SUPPORTED_AUDIT_STATUSES,
		"taxpayer_pin_required": True,
		"auditor_required": True,
		"evidence_required": True,
	},
	"reviews": {
		"supported_statuses": SUPPORTED_REVIEW_STATUSES,
		"reviewer_required": True,
		"evidence_required": True,
	},
	"agents": {
		"enabled": True,
		"supported_runtimes": SUPPORTED_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_AGENT_ROLES,
		"name_required": True,
		"scope_required": True,
		"human_approval_required_for_privileged_actions": True,
	},
	"governance": {
		"require_tenant_context": True,
		"policy_attached_for_writes": True,
		"audit_events": True,
		"duplicate_pin_denied": True,
		"late_filing_penalty_automated": True,
		"objection_outside_deadline_denied": True,
		"debt_collection_without_demand_notice_denied": True,
		"tax_evasion_flagged_for_audit": True,
		"taxpayer_data_confidentiality_enforced": True,
		"evidence_fabrication_denied": True,
	},
	"observability": {"event_stream": TAX_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {
		"auth": "auth",
		"audit": "audl",
		"notifications": "ntfy",
		"workflow": "wflo",
		"compliance": "comp",
		"monitoring": "moni",
		"scheduling": "schd",
		"event_stream": "bytewax",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_registrations": True,
		"enable_returns": True,
		"enable_assessments": True,
		"enable_objections": True,
		"enable_debt_collection": True,
		"enable_audits": True,
		"enable_reviews": True,
		"enable_agents": True,
	},
	"theme": {"default_theme": "government_tax_control", "allow_tenant_overrides": True},
}

PROVIDES = [
	"taxpayer_registration_workflow",
	"return_filing_workflow",
	"tax_assessment_workflow",
	"objection_management_workflow",
	"debt_collection_workflow",
	"audit_case_management_workflow",
	"tax_review_workflow",
	"tax_agent_workflow",
	"tax_refund_workflow",
	"compliance_risk_scoring_workflow",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "comp", "schd", "moni", "mqeb"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/government-tax/dashboard", "component": "TaxDashboard", "permission": "government_tax:view", "nav_group": "Overview"},
	{"name": "registrations", "path": "/government-tax/registrations", "component": "TaxpayerRegistrationConsole", "permission": "government_tax:register", "nav_group": "Registration"},
	{"name": "returns", "path": "/government-tax/returns", "component": "TaxReturnFilingConsole", "permission": "government_tax:returns", "nav_group": "Returns"},
	{"name": "assessments", "path": "/government-tax/assessments", "component": "TaxAssessmentConsole", "permission": "government_tax:assess", "nav_group": "Assessment"},
	{"name": "objections", "path": "/government-tax/objections", "component": "ObjectionManagementConsole", "permission": "government_tax:object", "nav_group": "Disputes"},
	{"name": "debt_collection", "path": "/government-tax/debt-collection", "component": "DebtCollectionConsole", "permission": "government_tax:collect", "nav_group": "Collections"},
	{"name": "audits", "path": "/government-tax/audits", "component": "AuditCaseConsole", "permission": "government_tax:audit", "nav_group": "Audits"},
	{"name": "refunds", "path": "/government-tax/refunds", "component": "TaxRefundConsole", "permission": "government_tax:refunds", "nav_group": "Refunds"},
	{"name": "compliance", "path": "/government-tax/compliance", "component": "TaxComplianceDashboard", "permission": "government_tax:compliance", "nav_group": "Compliance"},
	{"name": "reviews", "path": "/government-tax/reviews", "component": "TaxReviewConsole", "permission": "government_tax:review", "nav_group": "Governance"},
	{"name": "agents", "path": "/government-tax/agents", "component": "TaxAgentWorkbench", "permission": "government_tax:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/government-tax/settings", "component": "TaxSettings", "permission": "government_tax:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "government_tax_control",
	"tokens": {
		"color.primary": "#1E40AF",
		"color.accent": "#D97706",
		"color.success": "#166534",
		"color.warning": "#B45309",
		"color.danger": "#991B1B",
		"surface.canvas": "#EFF6FF",
		"surface.panel": "#FFFFFF",
		"text.primary": "#1E3A5F",
		"text.secondary": "#374151",
		"border.radius": "6px",
		"density": "comfortable",
	},
	"components": {
		"registrations": {"icon": "user-plus", "status_indicator": "registration-status-chip"},
		"returns": {"icon": "file-text", "status_indicator": "return-status-chip"},
		"assessments": {"icon": "calculator", "status_indicator": "assessment-type-chip"},
		"objections": {"icon": "message-square", "status_indicator": "objection-status-chip"},
		"debt_collection": {"icon": "trending-down", "status_indicator": "collection-method-chip"},
		"audits": {"icon": "search", "status_indicator": "audit-status-chip"},
		"reviews": {"icon": "clipboard-check", "status_indicator": "review-status-chip"},
		"agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": TAX_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"taxpayer_registered",
		"tax_return_filed",
		"tax_assessed",
		"objection_filed",
		"objection_determined",
		"debt_collection_initiated",
		"payment_received",
		"audit_case_opened",
		"audit_completed",
		"tax_agent_registered",
	],
	"guardrails": [
		"tax_batch_requires_bytewax",
		"duplicate_pin_denied",
		"objection_outside_deadline_denied",
		"debt_collection_without_demand_notice_denied",
		"taxpayer_data_confidentiality_enforced",
		"evidence_fabrication_denied",
		"privileged_tax_agent_action_requires_human_approval",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "tax_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "tax_policy_required", "required_action": "attach_tax_policy"}},
	{"name": "tax_type_supported", "condition": {"operation": "register_taxpayer", "tax_type_supported": False}, "effect": {"decision": "deny", "reason": "tax_type_not_supported", "required_action": "select_supported_tax_type"}},
	{"name": "registration_pin_required", "condition": {"operation": "register_taxpayer", "pin_present": False}, "effect": {"decision": "deny", "reason": "tax_pin_required", "required_action": "generate_tax_pin"}},
	{"name": "registration_national_id_required", "condition": {"operation": "register_taxpayer", "national_id_present": False}, "effect": {"decision": "deny", "reason": "national_id_required", "required_action": "provide_national_id"}},
	{"name": "registration_evidence_required", "condition": {"operation": "register_taxpayer", "evidence_present": False}, "effect": {"decision": "deny", "reason": "registration_evidence_required", "required_action": "upload_registration_documents"}},
	{"name": "duplicate_pin_denied", "condition": {"operation": "register_taxpayer", "duplicate_pin": True}, "effect": {"decision": "deny", "reason": "duplicate_pin_denied", "required_action": "resolve_duplicate_registration"}},
	{"name": "return_type_supported", "condition": {"operation": "file_return", "return_type_supported": False}, "effect": {"decision": "deny", "reason": "return_type_not_supported", "required_action": "select_supported_return_type"}},
	{"name": "return_taxpayer_pin_required", "condition": {"operation": "file_return", "taxpayer_pin_present": False}, "effect": {"decision": "deny", "reason": "taxpayer_pin_required", "required_action": "provide_taxpayer_pin"}},
	{"name": "return_period_required", "condition": {"operation": "file_return", "period_present": False}, "effect": {"decision": "deny", "reason": "return_period_required", "required_action": "specify_return_period"}},
	{"name": "return_evidence_required", "condition": {"operation": "file_return", "evidence_present": False}, "effect": {"decision": "deny", "reason": "return_evidence_required", "required_action": "attach_return_evidence"}},
	{"name": "assessment_type_supported", "condition": {"operation": "raise_assessment", "assessment_type_supported": False}, "effect": {"decision": "deny", "reason": "assessment_type_not_supported", "required_action": "select_supported_assessment_type"}},
	{"name": "assessment_return_required", "condition": {"operation": "raise_assessment", "return_present": False}, "effect": {"decision": "deny", "reason": "return_required", "required_action": "select_tax_return"}},
	{"name": "assessment_assessor_required", "condition": {"operation": "raise_assessment", "assessor_present": False}, "effect": {"decision": "deny", "reason": "assessor_required", "required_action": "assign_assessor"}},
	{"name": "objection_assessment_required", "condition": {"operation": "file_objection", "assessment_present": False}, "effect": {"decision": "deny", "reason": "assessment_required", "required_action": "select_assessment"}},
	{"name": "objection_grounds_required", "condition": {"operation": "file_objection", "grounds_present": False}, "effect": {"decision": "deny", "reason": "objection_grounds_required", "required_action": "state_grounds_of_objection"}},
	{"name": "objection_deadline_enforced", "condition": {"operation": "file_objection", "within_deadline": False}, "effect": {"decision": "deny", "reason": "objection_deadline_passed", "required_action": "apply_for_extension_or_appeal"}},
	{"name": "debt_collection_liability_required", "condition": {"operation": "initiate_collection", "assessed_liability_present": False}, "effect": {"decision": "deny", "reason": "assessed_liability_required", "required_action": "raise_assessment_first"}},
	{"name": "debt_collection_demand_required", "condition": {"operation": "initiate_collection", "demand_notice_issued": False}, "effect": {"decision": "deny", "reason": "demand_notice_required", "required_action": "issue_demand_notice"}},
	{"name": "debt_collection_method_supported", "condition": {"operation": "initiate_collection", "collection_method_supported": False}, "effect": {"decision": "deny", "reason": "collection_method_not_supported", "required_action": "select_supported_collection_method"}},
	{"name": "audit_type_supported", "condition": {"operation": "open_audit", "audit_type_supported": False}, "effect": {"decision": "deny", "reason": "audit_type_not_supported", "required_action": "select_supported_audit_type"}},
	{"name": "audit_taxpayer_required", "condition": {"operation": "open_audit", "taxpayer_pin_present": False}, "effect": {"decision": "deny", "reason": "taxpayer_pin_required", "required_action": "provide_taxpayer_pin"}},
	{"name": "audit_auditor_required", "condition": {"operation": "open_audit", "auditor_present": False}, "effect": {"decision": "deny", "reason": "auditor_required", "required_action": "assign_auditor"}},
	{"name": "tax_batch_requires_bytewax", "condition": {"operation": "tax_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_tax_batch_to_bytewax"}},
	{"name": "tax_agent_runtime_supported", "condition": {"operation": "register_tax_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "tax_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "tax_agent_role_supported", "condition": {"operation": "register_tax_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "tax_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "tax_agent_name_required", "condition": {"operation": "register_tax_agent", "agent_name_present": False}, "effect": {"decision": "deny", "reason": "tax_agent_name_required", "required_action": "name_tax_agent"}},
	{"name": "tax_agent_scope_required", "condition": {"operation": "register_tax_agent", "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "tax_agent_scope_required", "required_action": "bound_tax_agent_scope"}},
	{"name": "privileged_tax_agent_action_requires_human_approval", "condition": {"operation": "tax_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
	{"name": "evidence_fabrication_denied", "condition": {"operation": "tax_agent_action", "evidence_fabrication_scope": True}, "effect": {"decision": "deny", "reason": "evidence_fabrication_denied", "required_action": "remove_evidence_fabrication_scope"}},
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
		"configuration_schema": {
			"type": "object",
			"required": list(configuration),
			"properties": {key: {"type": "object"} for key in configuration if key != "tenant_id"} | {"tenant_id": {"type": "string", "minLength": 1}},
		},
		"rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"api_prefix": "/government-tax/api/v1",
			"requires_theme": True,
			"view_module": "views.py",
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
