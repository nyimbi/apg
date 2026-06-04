"""Executable capability contract for APG Multi-Country Operations."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "loc_mco"
CAPABILITY_NAME = "Multi-Country Operations"
CAPABILITY_VERSION = "1.0.0"
MCO_EVENT_STREAM = "apg.loc.mco.lifecycle"

# --- Supported enum constants ---
SUPPORTED_COUNTRY_STATUSES = ["active", "inactive", "suspended", "under_review"]
SUPPORTED_ENTITY_TYPES = ["subsidiary", "branch", "representative_office", "joint_venture", "holding_company", "partnership", "sole_proprietorship"]
SUPPORTED_REGULATORY_FRAMEWORKS = ["ifrs", "gaap", "local_gaap", "sme_ifrs", "cash_basis", "tax_basis"]
SUPPORTED_COMPLIANCE_DOMAINS = ["tax", "statutory_reporting", "labour", "environmental", "data_protection", "anti_money_laundering", "sanctions", "import_export", "corporate_governance"]
SUPPORTED_COMPLIANCE_STATUSES = ["compliant", "non_compliant", "under_review", "exempted", "pending_assessment"]
SUPPORTED_INTERCOMPANY_TYPES = ["loan", "dividend", "management_fee", "royalty", "cost_allocation", "goods_transfer", "services_transfer", "capital_contribution", "guarantee"]
SUPPORTED_INTERCOMPANY_STATUSES = ["draft", "pending_approval", "approved", "settled", "voided", "disputed"]
SUPPORTED_STATUTORY_REPORT_TYPES = ["annual_return", "financial_statements", "tax_return", "vat_return", "payroll_return", "regulatory_filing", "beneficial_ownership", "transfer_pricing"]
SUPPORTED_STATUTORY_STATUSES = ["draft", "under_review", "approved", "filed", "accepted", "rejected", "overdue"]
SUPPORTED_APPROVAL_STATUSES = ["pending", "approved", "rejected", "escalated", "withdrawn"]
SUPPORTED_CURRENCIES = ["KES", "USD", "EUR", "GBP", "ZAR", "NGN", "GHS", "TZS", "UGX", "RWF", "ETB", "XOF", "XAF", "MWK", "ZMW"]
SUPPORTED_JURISDICTIONS = ["ke", "ug", "tz", "rw", "et", "gh", "ng", "za", "us", "gb", "de", "fr", "ae", "cn", "in"]
SUPPORTED_TRANSFER_PRICING_METHODS = ["comparable_uncontrolled_price", "resale_price", "cost_plus", "transactional_net_margin", "profit_split"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["compliance_monitor", "intercompany_reviewer", "statutory_filer", "regulatory_mapper", "entity_steward"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"countries": {
		"supported_statuses": SUPPORTED_COUNTRY_STATUSES,
		"supported_jurisdictions": SUPPORTED_JURISDICTIONS,
		"supported_currencies": SUPPORTED_CURRENCIES,
		"regulatory_framework_required": True,
		"functional_currency_required": True,
	},
	"entities": {
		"supported_entity_types": SUPPORTED_ENTITY_TYPES,
		"registration_number_required": True,
		"country_required": True,
		"functional_currency_required": True,
	},
	"compliance": {
		"supported_domains": SUPPORTED_COMPLIANCE_DOMAINS,
		"supported_statuses": SUPPORTED_COMPLIANCE_STATUSES,
		"supported_frameworks": SUPPORTED_REGULATORY_FRAMEWORKS,
		"owner_required": True,
		"evidence_required": True,
		"next_review_required": True,
	},
	"intercompany": {
		"supported_types": SUPPORTED_INTERCOMPANY_TYPES,
		"supported_statuses": SUPPORTED_INTERCOMPANY_STATUSES,
		"transfer_pricing_required": True,
		"approval_required": True,
		"arms_length_validation": True,
	},
	"statutory_reports": {
		"supported_types": SUPPORTED_STATUTORY_REPORT_TYPES,
		"supported_statuses": SUPPORTED_STATUTORY_STATUSES,
		"entity_required": True,
		"period_required": True,
		"filer_required": True,
	},
	"governance": {
		"require_tenant_context": True,
		"policy_attached_for_writes": True,
		"audit_events": True,
		"cross_tenant_entity_denied": True,
		"unapproved_intercompany_denied": True,
		"unsupported_jurisdiction_denied": True,
		"unfiled_overdue_report_blocked": True,
		"arms_length_bypass_denied": True,
	},
	"observability": {"event_stream": MCO_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {
		"auth": "auth",
		"audit": "audl",
		"notifications": "ntfy",
		"workflow": "wflo",
		"compliance": "comp",
		"monitoring": "moni",
		"event_stream": "bytewax",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_countries": True,
		"enable_entities": True,
		"enable_compliance": True,
		"enable_intercompany": True,
		"enable_statutory_reports": True,
		"enable_transfer_pricing": True,
		"enable_agents": True,
	},
	"theme": {"default_theme": "loc_mco_global", "allow_tenant_overrides": True},
}

PROVIDES = [
	"country_entity_management",
	"regulatory_compliance_mapping",
	"intercompany_transaction_workflow",
	"statutory_reporting_workflow",
	"transfer_pricing_validation",
	"cross_border_governance",
	"multi_entity_consolidation_data",
	"jurisdiction_registry",
	"compliance_monitoring",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "comp", "moni", "mqeb"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/loc-mco/dashboard", "component": "McoDashboard", "permission": "loc_mco:view", "nav_group": "Overview"},
	{"name": "countries", "path": "/loc-mco/countries", "component": "McoCountryRegistry", "permission": "loc_mco:countries", "nav_group": "Setup"},
	{"name": "countries_create", "path": "/loc-mco/countries/create", "component": "McoCountryCreate", "permission": "loc_mco:countries_write", "nav_group": "Setup"},
	{"name": "entities", "path": "/loc-mco/entities", "component": "McoEntityList", "permission": "loc_mco:entities", "nav_group": "Setup"},
	{"name": "entities_create", "path": "/loc-mco/entities/create", "component": "McoEntityCreate", "permission": "loc_mco:entities_write", "nav_group": "Setup"},
	{"name": "compliance", "path": "/loc-mco/compliance", "component": "McoComplianceMatrix", "permission": "loc_mco:compliance", "nav_group": "Compliance"},
	{"name": "compliance_create", "path": "/loc-mco/compliance/create", "component": "McoComplianceCreate", "permission": "loc_mco:compliance_write", "nav_group": "Compliance"},
	{"name": "intercompany", "path": "/loc-mco/intercompany", "component": "McoIntercompanyLedger", "permission": "loc_mco:intercompany", "nav_group": "Transactions"},
	{"name": "intercompany_create", "path": "/loc-mco/intercompany/create", "component": "McoIntercompanyCreate", "permission": "loc_mco:intercompany_write", "nav_group": "Transactions"},
	{"name": "transfer_pricing", "path": "/loc-mco/transfer-pricing", "component": "McoTransferPricingConsole", "permission": "loc_mco:transfer_pricing", "nav_group": "Transactions"},
	{"name": "statutory_reports", "path": "/loc-mco/statutory-reports", "component": "McoStatutoryReportList", "permission": "loc_mco:statutory_reports", "nav_group": "Reporting"},
	{"name": "statutory_reports_create", "path": "/loc-mco/statutory-reports/create", "component": "McoStatutoryReportCreate", "permission": "loc_mco:statutory_reports_write", "nav_group": "Reporting"},
	{"name": "agents", "path": "/loc-mco/agents", "component": "McoAgentWorkbench", "permission": "loc_mco:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/loc-mco/settings", "component": "McoSettings", "permission": "loc_mco:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "loc_mco_global",
	"tokens": {
		"color.primary": "#1D4ED8",
		"color.accent": "#0891B2",
		"color.success": "#15803D",
		"color.warning": "#B45309",
		"color.danger": "#B91C1C",
		"surface.canvas": "#F8FAFC",
		"surface.panel": "#FFFFFF",
		"text.primary": "#111827",
		"text.secondary": "#4B5563",
		"border.radius": "6px",
		"density": "comfortable",
	},
	"components": {
		"countries": {"icon": "globe", "status_indicator": "country-status-chip"},
		"entities": {"icon": "building-2", "status_indicator": "entity-type-chip"},
		"compliance": {"icon": "shield-check", "status_indicator": "compliance-status-chip"},
		"intercompany": {"icon": "arrows-left-right", "status_indicator": "intercompany-status-chip"},
		"transfer_pricing": {"icon": "scale", "status_indicator": "tp-method-chip"},
		"statutory_reports": {"icon": "file-text", "status_indicator": "statutory-status-chip"},
		"agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": MCO_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"country_registered",
		"country_updated",
		"entity_registered",
		"entity_updated",
		"compliance_mapping_recorded",
		"compliance_status_updated",
		"intercompany_transaction_created",
		"intercompany_transaction_approved",
		"intercompany_transaction_settled",
		"statutory_report_created",
		"statutory_report_filed",
		"statutory_report_accepted",
		"transfer_pricing_validated",
		"agent_registered",
	],
	"guardrails": [
		"cross_tenant_entity_denied",
		"unapproved_intercompany_denied",
		"unsupported_jurisdiction_denied",
		"arms_length_bypass_denied",
		"unfiled_overdue_report_blocked",
		"privileged_agent_action_requires_human_approval",
	],
}

RULES: list[dict[str, Any]] = [
	# Tenant governance
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "policy_required_for_writes", "required_action": "attach_policy"}},
	{"name": "cross_tenant_entity_denied", "condition": {"cross_tenant_operation": True}, "effect": {"decision": "deny", "reason": "cross_tenant_entity_access_denied", "required_action": "use_tenant_scoped_operation"}},
	# Country rules
	{"name": "country_jurisdiction_supported", "condition": {"operation": "register_country", "jurisdiction_supported": False}, "effect": {"decision": "deny", "reason": "jurisdiction_not_supported", "required_action": "select_supported_jurisdiction"}},
	{"name": "country_currency_supported", "condition": {"operation": "register_country", "currency_supported": False}, "effect": {"decision": "deny", "reason": "functional_currency_not_supported", "required_action": "select_supported_currency"}},
	{"name": "country_regulatory_framework_required", "condition": {"operation": "register_country", "regulatory_framework_present": False}, "effect": {"decision": "deny", "reason": "regulatory_framework_required", "required_action": "select_regulatory_framework"}},
	{"name": "country_name_required", "condition": {"operation": "register_country", "country_name_present": False}, "effect": {"decision": "deny", "reason": "country_name_required", "required_action": "provide_country_name"}},
	# Entity rules
	{"name": "entity_type_supported", "condition": {"operation": "register_entity", "entity_type_supported": False}, "effect": {"decision": "deny", "reason": "entity_type_not_supported", "required_action": "select_supported_entity_type"}},
	{"name": "entity_country_required", "condition": {"operation": "register_entity", "country_present": False}, "effect": {"decision": "deny", "reason": "country_required", "required_action": "assign_country"}},
	{"name": "entity_registration_number_required", "condition": {"operation": "register_entity", "registration_number_present": False}, "effect": {"decision": "deny", "reason": "registration_number_required", "required_action": "provide_registration_number"}},
	{"name": "entity_functional_currency_required", "condition": {"operation": "register_entity", "functional_currency_present": False}, "effect": {"decision": "deny", "reason": "functional_currency_required", "required_action": "assign_functional_currency"}},
	# Compliance rules
	{"name": "compliance_domain_supported", "condition": {"operation": "record_compliance", "domain_supported": False}, "effect": {"decision": "deny", "reason": "compliance_domain_not_supported", "required_action": "select_supported_domain"}},
	{"name": "compliance_framework_supported", "condition": {"operation": "record_compliance", "framework_supported": False}, "effect": {"decision": "deny", "reason": "regulatory_framework_not_supported", "required_action": "select_supported_framework"}},
	{"name": "compliance_owner_required", "condition": {"operation": "record_compliance", "owner_present": False}, "effect": {"decision": "deny", "reason": "compliance_owner_required", "required_action": "assign_compliance_owner"}},
	{"name": "compliance_evidence_required", "condition": {"operation": "record_compliance", "evidence_present": False}, "effect": {"decision": "deny", "reason": "compliance_evidence_required", "required_action": "attach_compliance_evidence"}},
	{"name": "compliance_review_date_required", "condition": {"operation": "record_compliance", "review_date_present": False}, "effect": {"decision": "deny", "reason": "next_review_date_required", "required_action": "set_review_date"}},
	# Intercompany rules
	{"name": "intercompany_type_supported", "condition": {"operation": "create_intercompany", "transaction_type_supported": False}, "effect": {"decision": "deny", "reason": "intercompany_type_not_supported", "required_action": "select_supported_type"}},
	{"name": "intercompany_originator_required", "condition": {"operation": "create_intercompany", "originator_present": False}, "effect": {"decision": "deny", "reason": "originator_entity_required", "required_action": "assign_originator"}},
	{"name": "intercompany_counterparty_required", "condition": {"operation": "create_intercompany", "counterparty_present": False}, "effect": {"decision": "deny", "reason": "counterparty_entity_required", "required_action": "assign_counterparty"}},
	{"name": "intercompany_approval_required", "condition": {"operation": "approve_intercompany", "approver_present": False}, "effect": {"decision": "deny", "reason": "approver_required", "required_action": "assign_approver"}},
	{"name": "arms_length_bypass_denied", "condition": {"operation": "create_intercompany", "arms_length_bypass": True}, "effect": {"decision": "deny", "reason": "arms_length_bypass_denied", "required_action": "apply_transfer_pricing_method"}},
	{"name": "intercompany_currency_supported", "condition": {"operation": "create_intercompany", "currency_supported": False}, "effect": {"decision": "deny", "reason": "transaction_currency_not_supported", "required_action": "select_supported_currency"}},
	# Transfer pricing rules
	{"name": "transfer_pricing_method_supported", "condition": {"operation": "validate_transfer_pricing", "tp_method_supported": False}, "effect": {"decision": "deny", "reason": "transfer_pricing_method_not_supported", "required_action": "select_supported_tp_method"}},
	{"name": "transfer_pricing_documentation_required", "condition": {"operation": "validate_transfer_pricing", "documentation_present": False}, "effect": {"decision": "deny", "reason": "tp_documentation_required", "required_action": "attach_tp_documentation"}},
	# Statutory report rules
	{"name": "statutory_report_type_supported", "condition": {"operation": "create_statutory_report", "report_type_supported": False}, "effect": {"decision": "deny", "reason": "statutory_report_type_not_supported", "required_action": "select_supported_report_type"}},
	{"name": "statutory_report_entity_required", "condition": {"operation": "create_statutory_report", "entity_present": False}, "effect": {"decision": "deny", "reason": "entity_required", "required_action": "assign_entity"}},
	{"name": "statutory_report_period_required", "condition": {"operation": "create_statutory_report", "period_present": False}, "effect": {"decision": "deny", "reason": "reporting_period_required", "required_action": "set_reporting_period"}},
	{"name": "statutory_report_filer_required", "condition": {"operation": "file_statutory_report", "filer_present": False}, "effect": {"decision": "deny", "reason": "filer_identity_required", "required_action": "assign_filer"}},
	{"name": "overdue_report_filing_blocked", "condition": {"operation": "create_statutory_report", "existing_overdue_unfiled": True}, "effect": {"decision": "deny", "reason": "overdue_statutory_report_must_be_filed_first", "required_action": "file_overdue_report"}},
	# Agent rules
	{"name": "agent_runtime_supported", "condition": {"operation": "register_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "agent_role_supported", "condition": {"operation": "register_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "privileged_agent_action_requires_human_approval", "condition": {"operation": "agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required_for_privileged_action", "required_action": "record_human_approval"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	schema_props = {k: {"type": "object"} for k in configuration if k != "tenant_id"}
	schema_props["tenant_id"] = {"type": "string", "minLength": 1}
	schema_props["ui"] = {"type": "object"}
	schema_props["theme"] = {"type": "object"}
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
			"required": ["tenant_id", "ui", "theme"],
			"properties": schema_props,
		},
		"rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"api_prefix": "/loc-mco/api/v1",
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
