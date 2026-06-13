"""Executable capability contract for APG VAT/GST Country Rule Packs."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "common_tax_vat"
CAPABILITY_NAME = "VAT/GST Country Rule Packs"
CAPABILITY_VERSION = "1.0.0"
VAT_EVENT_STREAM = "apg.common.tax.vat"

SUPPORTED_COUNTRY_CODES = [
	"KE", "NG", "GH", "UG", "TZ", "ZA", "RW", "ET", "EG", "MA",
	"TN", "CI", "SN", "CM", "ZM", "ZW", "BW", "MZ", "AO", "NA",
]

SUPPORTED_VAT_CATEGORIES = [
	"standard", "zero_rated", "exempt", "reduced_rate",
	"financial_services", "health", "education", "agriculture",
	"fuel", "digital", "insurance", "real_estate",
]

SUPPORTED_RETURN_STATUSES = ["draft", "submitted", "accepted", "rejected", "amended"]

SUPPORTED_EXEMPTION_TYPES = [
	"basic_foodstuffs", "medical_supplies", "educational_materials",
	"financial_services", "agricultural_inputs", "exports",
	"diplomatic", "government", "nonprofit",
]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"countries": {"supported": SUPPORTED_COUNTRY_CODES, "multi_country_enabled": True},
	"vat_categories": {"supported": SUPPORTED_VAT_CATEGORIES},
	"returns": {"supported_statuses": SUPPORTED_RETURN_STATUSES, "auto_compute_enabled": True},
	"exemptions": {"supported_types": SUPPORTED_EXEMPTION_TYPES, "evidence_required": True},
	"governance": {
		"require_tenant_context": True,
		"policy_attached_for_writes": True,
		"rate_freeze_denial": True,
		"backdated_return_denial": True,
	},
	"observability": {"event_stream": VAT_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {"audit": "audl", "auth": "auth", "tax_calc": "tax_calc"},
}

PROVIDES = [
	"vat_rate_lookup",
	"vat_return_workflow",
	"vat_exemption_registry",
	"vat_country_config",
]

REQUIRES = ["audl", "auth", "tax_calc"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/tax/vat/dashboard", "component": "VatDashboard", "permission": "tax_vat:view", "nav_group": "Overview"},
	{"name": "rates", "path": "/tax/vat/rates", "component": "VatRateWorkbench", "permission": "tax_vat:rates", "nav_group": "Configuration"},
	{"name": "returns", "path": "/tax/vat/returns", "component": "VatReturnConsole", "permission": "tax_vat:returns", "nav_group": "Compliance"},
	{"name": "exemptions", "path": "/tax/vat/exemptions", "component": "VatExemptionRegistry", "permission": "tax_vat:exemptions", "nav_group": "Compliance"},
	{"name": "settings", "path": "/tax/vat/settings", "component": "VatSettings", "permission": "tax_vat:admin", "nav_group": "Administration"},
]

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "vat_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "vat_policy_required", "required_action": "attach_vat_policy"}},
	{"name": "country_code_supported", "condition": {"operation": "get_vat_rate", "country_code_supported": False}, "effect": {"decision": "deny", "reason": "country_not_in_vat_pack", "required_action": "select_supported_country"}},
	{"name": "vat_category_supported", "condition": {"operation": "get_vat_rate", "vat_category_supported": False}, "effect": {"decision": "deny", "reason": "vat_category_not_supported", "required_action": "select_supported_category"}},
	{"name": "return_status_supported", "condition": {"operation": "submit_return", "status_supported": False}, "effect": {"decision": "deny", "reason": "return_status_not_supported", "required_action": "select_supported_status"}},
	{"name": "exemption_evidence_required", "condition": {"operation": "register_exemption", "evidence_present": False}, "effect": {"decision": "deny", "reason": "exemption_evidence_required", "required_action": "attach_exemption_evidence"}},
	{"name": "backdated_return_denial", "condition": {"operation": "submit_return", "period_in_future": True}, "effect": {"decision": "deny", "reason": "cannot_submit_future_return", "required_action": "wait_for_period_end"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	return {
		"capability": CAPABILITY_ID,
		"name": CAPABILITY_NAME,
		"version": CAPABILITY_VERSION,
		"provides": list(PROVIDES),
		"requires": list(REQUIRES),
		"configuration": configuration,
		"rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)},
		"ui": {"shell": "apg_python", "api_prefix": "/tax/vat/api/v1", "routes": deepcopy(UI_ROUTES)},
		"streaming": {"processor": "bytewax", "stream": VAT_EVENT_STREAM, "key": "tenant_id"},
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
