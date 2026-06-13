"""Executable capability contract for APG Tax Calculation Engine."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "common_tax_calc"
CAPABILITY_NAME = "Tax Calculation Engine"
CAPABILITY_VERSION = "1.0.0"
TAX_EVENT_STREAM = "apg.common.tax.calculations"

# Supported tax types across African jurisdictions
SUPPORTED_TAX_TYPES = [
	"vat",            # Value Added Tax / GST
	"wht",            # Withholding Tax
	"excise",         # Excise duty
	"paye",           # Pay As You Earn (employment income tax)
	"corporate",      # Corporate income tax
	"customs",        # Customs/import duty
	"capital_gains",  # Capital Gains Tax
	"stamp_duty",     # Stamp duty
	"turnover",       # Turnover tax (SME simplified regimes)
	"digital_services", # Digital services tax (Kenya DST, Nigeria etc.)
]

# ISO 3166-1 alpha-2 country codes with active APG tax packs
SUPPORTED_COUNTRY_CODES = [
	"KE",  # Kenya - KRA iTax
	"NG",  # Nigeria - FIRS
	"GH",  # Ghana - GRA
	"UG",  # Uganda - URA
	"TZ",  # Tanzania - TRA
	"ZA",  # South Africa - SARS
	"RW",  # Rwanda - RRA
	"ET",  # Ethiopia - ERCA
	"EG",  # Egypt - ETA
	"MA",  # Morocco - DGI
	"TN",  # Tunisia - DGI
	"CI",  # Côte d'Ivoire - DGI
	"SN",  # Senegal - DGID
	"CM",  # Cameroon - DGI
	"ZM",  # Zambia - ZRA
	"ZW",  # Zimbabwe - ZIMRA
	"BW",  # Botswana - BURS
	"MZ",  # Mozambique - AT
	"AO",  # Angola - AGT
	"NA",  # Namibia - NamRA
]

SUPPORTED_PRODUCT_CATEGORIES = [
	"standard",           # Standard rated
	"zero_rated",         # Zero-rated (exports, basic foods)
	"exempt",             # Fully exempt
	"reduced_rate",       # Reduced rate (some countries)
	"financial_services", # Typically exempt
	"health",             # Medical supplies / services
	"education",          # Educational services
	"agriculture",        # Agricultural inputs
	"fuel",               # Petroleum products (often excise-heavy)
	"digital",            # Digital goods / services
	"insurance",          # Insurance premiums
	"real_estate",        # Property transactions
]

SUPPORTED_ENTITY_TYPES = [
	"individual",
	"company",
	"partnership",
	"trust",
	"ngo",
	"government",
	"cooperative",
]

SUPPORTED_TREATY_STATUSES = ["domestic", "treaty_reduced", "treaty_exempt", "non_resident"]

SUPPORTED_CALCULATION_STATUSES = ["pending", "calculated", "verified", "filed", "paid", "amended"]

SUPPORTED_AUDIT_ACTIONS = [
	"rate_lookup",
	"calculation_performed",
	"rule_evaluated",
	"override_applied",
	"amendment_recorded",
	"filing_triggered",
]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"tax_types": {
		"supported": SUPPORTED_TAX_TYPES,
		"require_country_code": True,
		"require_product_category": True,
		"require_entity_type": False,
	},
	"countries": {
		"supported": SUPPORTED_COUNTRY_CODES,
		"default_country": "KE",
		"multi_country_enabled": True,
	},
	"rates": {
		"cache_ttl_seconds": 3600,
		"allow_manual_override": True,
		"require_override_justification": True,
	},
	"audit": {
		"enabled": True,
		"every_calculation": True,
		"retain_days": 2555,  # 7 years — typical statutory minimum
	},
	"governance": {
		"require_tenant_context": True,
		"policy_attached_for_writes": True,
		"cross_tenant_calc_denied": True,
		"unapproved_rate_override_denied": True,
		"negative_tax_denied": True,
		"future_period_filing_denied": True,
	},
	"observability": {
		"event_stream": TAX_EVENT_STREAM,
		"stream_processor": "bytewax",
	},
	"adapters": {
		"audit": "audl",
		"auth": "auth",
		"vat_rules": "tax_vat",
		"wht_rules": "tax_wht",
	},
}

PROVIDES = [
	"tax_calculation_workflow",
	"tax_rate_lookup",
	"tax_period_management",
	"tax_audit_trail",
	"tax_cross_capability_api",
]

REQUIRES = ["audl", "auth", "tax_vat", "tax_wht"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/tax/dashboard", "component": "TaxDashboard", "permission": "tax:view", "nav_group": "Overview"},
	{"name": "calculations", "path": "/tax/calculations", "component": "TaxCalculationLedger", "permission": "tax:calculations", "nav_group": "Operations"},
	{"name": "rates", "path": "/tax/rates", "component": "TaxRateWorkbench", "permission": "tax:rates", "nav_group": "Configuration"},
	{"name": "periods", "path": "/tax/periods", "component": "TaxPeriodConsole", "permission": "tax:periods", "nav_group": "Compliance"},
	{"name": "audit", "path": "/tax/audit", "component": "TaxAuditLedger", "permission": "tax:audit", "nav_group": "Governance"},
	{"name": "settings", "path": "/tax/settings", "component": "TaxSettings", "permission": "tax:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "tax_engine_control",
	"tokens": {
		"color.primary": "#1E40AF",
		"color.accent": "#059669",
		"color.success": "#166534",
		"color.warning": "#B45309",
		"color.danger": "#991B1B",
		"surface.canvas": "#F8FAFC",
		"surface.panel": "#FFFFFF",
		"text.primary": "#111827",
		"text.secondary": "#4B5563",
		"border.radius": "8px",
		"density": "compact",
	},
	"components": {
		"calculations": {"icon": "calculator", "status_indicator": "calc-status-chip"},
		"rates": {"icon": "percent", "status_indicator": "rate-chip"},
		"periods": {"icon": "calendar", "status_indicator": "period-chip"},
		"audit": {"icon": "shield-check", "status_indicator": "audit-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": TAX_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"tax_calculation_performed",
		"tax_rate_cached",
		"tax_period_opened",
		"tax_period_closed",
		"tax_audit_recorded",
		"tax_rate_override_applied",
	],
	"guardrails": [
		"cross_tenant_calc_denied",
		"negative_tax_denied",
		"unapproved_rate_override_denied",
		"future_period_filing_denied",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "tax_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "tax_policy_required", "required_action": "attach_tax_policy"}},
	{"name": "tax_type_supported", "condition": {"operation": "calculate_tax", "tax_type_supported": False}, "effect": {"decision": "deny", "reason": "tax_type_not_supported", "required_action": "select_supported_tax_type"}},
	{"name": "country_code_supported", "condition": {"operation": "calculate_tax", "country_code_supported": False}, "effect": {"decision": "deny", "reason": "country_not_supported", "required_action": "select_supported_country"}},
	{"name": "amount_positive_required", "condition": {"operation": "calculate_tax", "amount_positive": False}, "effect": {"decision": "deny", "reason": "amount_must_be_positive", "required_action": "provide_positive_amount"}},
	{"name": "product_category_required", "condition": {"operation": "calculate_tax", "product_category_present": False}, "effect": {"decision": "deny", "reason": "product_category_required", "required_action": "attach_product_category"}},
	{"name": "cross_tenant_calc_denied", "condition": {"operation": "calculate_tax", "cross_tenant": True}, "effect": {"decision": "deny", "reason": "cross_tenant_calc_denied", "required_action": "use_own_tenant_id"}},
	{"name": "negative_tax_denied", "condition": {"operation": "calculate_tax", "result_negative": True}, "effect": {"decision": "deny", "reason": "negative_tax_result_denied", "required_action": "review_rate_configuration"}},
	{"name": "rate_override_requires_justification", "condition": {"operation": "override_rate", "justification_present": False}, "effect": {"decision": "deny", "reason": "override_justification_required", "required_action": "provide_override_justification"}},
	{"name": "rate_override_requires_approval", "condition": {"operation": "override_rate", "approved_by_present": False}, "effect": {"decision": "deny", "reason": "rate_override_approval_required", "required_action": "obtain_override_approval"}},
	{"name": "period_dates_required", "condition": {"operation": "open_period", "period_dates_present": False}, "effect": {"decision": "deny", "reason": "period_dates_required", "required_action": "provide_period_dates"}},
	{"name": "future_period_filing_denied", "condition": {"operation": "file_period", "period_in_future": True}, "effect": {"decision": "deny", "reason": "cannot_file_future_period", "required_action": "wait_for_period_end"}},
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
		"ui": {"shell": "apg_python", "api_prefix": "/tax/api/v1", "requires_theme": True, "view_module": "views.py", "template_roots": ["templates/", "static/"], "routes": deepcopy(UI_ROUTES)},
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
