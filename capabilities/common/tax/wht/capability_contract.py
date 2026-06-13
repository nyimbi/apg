"""Executable capability contract for APG Withholding Tax (WHT)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "common_tax_wht"
CAPABILITY_NAME = "Withholding Tax Engine"
CAPABILITY_VERSION = "1.0.0"
WHT_EVENT_STREAM = "apg.common.tax.wht"

SUPPORTED_COUNTRY_CODES = [
	"KE", "NG", "GH", "UG", "TZ", "ZA", "RW", "ET", "EG", "MA",
	"TN", "CI", "SN", "CM", "ZM", "ZW", "BW", "MZ", "AO", "NA",
]

SUPPORTED_PAYMENT_TYPES = [
	"dividends",
	"interest",
	"royalties",
	"professional_fees",
	"management_fees",
	"rent",
	"technical_services",
	"construction",
	"supply_of_goods",
	"commissions",
	"insurance_premiums",
	"winnings",
	"pension",
	"employment_income",
]

SUPPORTED_TREATY_STATUSES = ["domestic", "treaty_reduced", "treaty_exempt", "non_resident"]

SUPPORTED_ENTITY_TYPES = ["individual", "company", "partnership", "trust", "ngo", "government"]

SUPPORTED_CERTIFICATE_STATUSES = ["issued", "verified", "expired", "cancelled"]

SUPPORTED_RETURN_STATUSES = ["draft", "submitted", "accepted", "rejected", "amended"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"countries": {"supported": SUPPORTED_COUNTRY_CODES},
	"payment_types": {"supported": SUPPORTED_PAYMENT_TYPES},
	"treaty_statuses": {"supported": SUPPORTED_TREATY_STATUSES},
	"certificates": {
		"auto_generate": True,
		"require_payment_proof": True,
		"supported_statuses": SUPPORTED_CERTIFICATE_STATUSES,
	},
	"returns": {
		"frequency": "quarterly",
		"supported_statuses": SUPPORTED_RETURN_STATUSES,
	},
	"governance": {
		"require_tenant_context": True,
		"policy_attached_for_writes": True,
		"treaty_without_evidence_denied": True,
		"zero_rate_without_treaty_denied": False,
	},
	"observability": {"event_stream": WHT_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {"audit": "audl", "auth": "auth", "tax_calc": "tax_calc"},
}

PROVIDES = [
	"wht_rate_lookup",
	"wht_certificate_workflow",
	"wht_return_workflow",
	"wht_payment_record",
]

REQUIRES = ["audl", "auth", "tax_calc"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/tax/wht/dashboard", "component": "WhtDashboard", "permission": "tax_wht:view", "nav_group": "Overview"},
	{"name": "rates", "path": "/tax/wht/rates", "component": "WhtRateWorkbench", "permission": "tax_wht:rates", "nav_group": "Configuration"},
	{"name": "certificates", "path": "/tax/wht/certificates", "component": "WhtCertificateLedger", "permission": "tax_wht:certificates", "nav_group": "Compliance"},
	{"name": "returns", "path": "/tax/wht/returns", "component": "WhtReturnConsole", "permission": "tax_wht:returns", "nav_group": "Compliance"},
	{"name": "payments", "path": "/tax/wht/payments", "component": "WhtPaymentLedger", "permission": "tax_wht:payments", "nav_group": "Operations"},
	{"name": "settings", "path": "/tax/wht/settings", "component": "WhtSettings", "permission": "tax_wht:admin", "nav_group": "Administration"},
]

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "wht_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "wht_policy_required", "required_action": "attach_wht_policy"}},
	{"name": "country_code_supported", "condition": {"operation": "get_wht_rate", "country_code_supported": False}, "effect": {"decision": "deny", "reason": "country_not_in_wht_pack", "required_action": "select_supported_country"}},
	{"name": "payment_type_supported", "condition": {"operation": "get_wht_rate", "payment_type_supported": False}, "effect": {"decision": "deny", "reason": "payment_type_not_supported", "required_action": "select_supported_payment_type"}},
	{"name": "treaty_evidence_required", "condition": {"operation": "get_wht_rate", "treaty_status": "treaty_reduced", "treaty_evidence_present": False}, "effect": {"decision": "deny", "reason": "treaty_evidence_required", "required_action": "attach_treaty_evidence"}},
	{"name": "certificate_payment_proof_required", "condition": {"operation": "issue_certificate", "payment_proof_present": False}, "effect": {"decision": "deny", "reason": "payment_proof_required", "required_action": "attach_payment_proof"}},
	{"name": "return_period_not_future", "condition": {"operation": "submit_return", "period_in_future": True}, "effect": {"decision": "deny", "reason": "cannot_submit_future_wht_return", "required_action": "wait_for_period_end"}},
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
		"ui": {"shell": "apg_python", "api_prefix": "/tax/wht/api/v1", "routes": deepcopy(UI_ROUTES)},
		"streaming": {"processor": "bytewax", "stream": WHT_EVENT_STREAM, "key": "tenant_id"},
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
