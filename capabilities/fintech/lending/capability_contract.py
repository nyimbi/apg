"""Executable capability contract for APG Digital Lending."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "fintech_lending"
CAPABILITY_NAME = "Digital Lending"
CAPABILITY_VERSION = "1.1.0"
LENDING_EVENT_STREAM = "apg.fintech.lending.lifecycle"

SUPPORTED_CURRENCIES = ["USD", "EUR", "GBP", "KES", "ZAR", "NGN", "GHS", "UGX", "TZS"]
SUPPORTED_COUNTRIES = ["KE", "UG", "TZ", "RW", "GH", "NG", "ZA", "GB", "US", "AE"]
SUPPORTED_PRODUCT_TYPES = ["term_loan", "revolving_credit", "invoice_finance", "asset_finance", "salary_advance", "merchant_cash_advance"]
SUPPORTED_REPAYMENT_FREQUENCIES = ["weekly", "biweekly", "monthly", "quarterly"]
SUPPORTED_APPLICATION_PURPOSES = ["working_capital", "inventory", "asset_purchase", "education", "medical", "home_improvement", "emergency", "refinance"]
SUPPORTED_UNDERWRITING_DECISIONS = ["approve", "decline", "refer", "counteroffer"]
SUPPORTED_OFFER_STATUSES = ["issued", "accepted", "expired", "withdrawn"]
SUPPORTED_DISBURSEMENT_RAILS = ["payment_account", "wallet", "card", "bank_transfer"]
SUPPORTED_COLLECTION_REASONS = ["missed_payment", "promise_broken", "hardship", "fraud_review", "restructure_request"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["lending_ops_reviewer", "underwriting_reviewer", "credit_risk_reviewer", "collections_reviewer", "compliance_reviewer"]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"products": {"owner_required": True, "supported_currencies": SUPPORTED_CURRENCIES, "supported_types": SUPPORTED_PRODUCT_TYPES, "min_term_days": 7, "max_term_days": 3650, "max_nominal_rate": 0.75, "max_amount": 1000000, "supported_repayment_frequencies": SUPPORTED_REPAYMENT_FREQUENCIES},
	"borrowers": {"kyc_required": True, "income_evidence_required": True, "consent_required": True, "supported_countries": SUPPORTED_COUNTRIES},
	"applications": {"supported_purposes": SUPPORTED_APPLICATION_PURPOSES, "affordability_required": True, "bank_statement_required": True, "aml_required": True, "fraud_required": True, "remittance_evidence_supported": True, "card_evidence_supported": True, "high_amount_threshold": 100000},
	"underwriting": {"supported_decisions": SUPPORTED_UNDERWRITING_DECISIONS, "min_score": 0, "max_score": 1000, "human_approval_required_for_final_decision": True, "adverse_reason_required_for_declines": True},
	"offers": {"supported_statuses": SUPPORTED_OFFER_STATUSES, "apr_required": True, "expiry_required": True, "borrower_acceptance_required": True},
	"disbursements": {"supported_rails": SUPPORTED_DISBURSEMENT_RAILS, "funding_account_required": True, "human_approval_required": True},
	"repayments": {"positive_due_amount_required": True, "due_date_required": True, "supported_frequencies": SUPPORTED_REPAYMENT_FREQUENCIES},
	"collections": {"supported_reasons": SUPPORTED_COLLECTION_REASONS, "reviewer_required": True, "contact_policy_required": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_lending_events": True, "borrower_consent_required": True},
	"observability": {"event_stream": LENDING_EVENT_STREAM, "stream_processor": "bytewax", "emit_product_events": True, "emit_borrower_events": True, "emit_application_events": True, "emit_underwriting_events": True, "emit_offer_events": True, "emit_disbursement_events": True, "emit_repayment_events": True, "emit_collection_events": True, "emit_agent_events": True},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "nlp": "nlpc", "keys": "keym", "payments": "fintech_payments", "wallets": "fintech_wallets", "cards": "fintech_cards", "kyc": "fintech_kyc", "aml": "fintech_aml", "fraud": "fintech_fraud", "remittance": "fintech_remittance", "event_stream": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_products": True, "enable_borrowers": True, "enable_applications": True, "enable_underwriting": True, "enable_offers": True, "enable_disbursements": True, "enable_repayments": True, "enable_collections": True, "enable_agents": True},
	"theme": {"default_theme": "fintech_lending_control", "allow_tenant_overrides": True},
}

PROVIDES = ["loan_product_governance", "borrower_lifecycle", "credit_application_workflow", "underwriting_decisioning", "loan_offer_workflow", "disbursement_control", "repayment_schedule_workflow", "collections_workflow", "lending_agent_workflow"]
REQUIRES = ["auth", "audl", "ntfy", "nlpc", "keym", "fintech_payments", "fintech_wallets", "fintech_cards", "fintech_kyc", "fintech_aml", "fintech_fraud", "fintech_remittance"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/fintech-lending/dashboard", "component": "LendingDashboard", "permission": "fintech_lending:view", "nav_group": "Overview"},
	{"name": "products", "path": "/fintech-lending/products", "component": "LoanProductConsole", "permission": "fintech_lending:manage_products", "nav_group": "Products"},
	{"name": "borrowers", "path": "/fintech-lending/borrowers", "component": "BorrowerWorkbench", "permission": "fintech_lending:manage_borrowers", "nav_group": "Borrowers"},
	{"name": "applications", "path": "/fintech-lending/applications", "component": "ApplicationWorkbench", "permission": "fintech_lending:submit", "nav_group": "Applications"},
	{"name": "underwriting", "path": "/fintech-lending/underwriting", "component": "UnderwritingConsole", "permission": "fintech_lending:underwrite", "nav_group": "Risk"},
	{"name": "offers", "path": "/fintech-lending/offers", "component": "LoanOfferWorkbench", "permission": "fintech_lending:offer", "nav_group": "Offers"},
	{"name": "disbursements", "path": "/fintech-lending/disbursements", "component": "DisbursementConsole", "permission": "fintech_lending:disburse", "nav_group": "Funding"},
	{"name": "repayments", "path": "/fintech-lending/repayments", "component": "RepaymentScheduleWorkbench", "permission": "fintech_lending:repayments", "nav_group": "Servicing"},
	{"name": "collections", "path": "/fintech-lending/collections", "component": "CollectionsWorkbench", "permission": "fintech_lending:collections", "nav_group": "Servicing"},
	{"name": "agents", "path": "/fintech-lending/agents", "component": "LendingAgentWorkbench", "permission": "fintech_lending:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/fintech-lending/settings", "component": "LendingSettings", "permission": "fintech_lending:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "fintech_lending_control",
	"tokens": {"color.primary": "#0F766E", "color.accent": "#7C3AED", "color.success": "#15803D", "color.warning": "#B45309", "color.danger": "#B91C1C", "surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF", "text.primary": "#111827", "text.secondary": "#4B5563", "border.radius": "8px", "density": "compact"},
	"components": {"products": {"icon": "landmark", "status_indicator": "product-chip"}, "applications": {"icon": "file-check-2", "status_indicator": "application-chip"}, "underwriting": {"icon": "scale", "status_indicator": "decision-chip"}, "offers": {"icon": "badge-dollar-sign", "status_indicator": "offer-chip"}, "repayments": {"icon": "calendar-clock", "status_indicator": "repayment-chip"}, "collections": {"icon": "phone-call", "status_indicator": "collection-chip"}, "agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"}},
}

STREAMING = {
	"processor": "bytewax",
	"stream": LENDING_EVENT_STREAM,
	"key": "tenant_id",
	"events": ["loan_product_registered", "borrower_onboarded", "loan_application_submitted", "underwriting_recorded", "loan_offer_issued", "loan_disbursement_recorded", "repayment_schedule_created", "collection_case_opened", "lending_agent_registered"],
	"guardrails": ["lending_batch_requires_bytewax", "privileged_lending_agent_action_requires_human_approval"],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "Lending operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "lending_write_requires_policy", "description": "Lending writes require policy evidence.", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "lending_policy_required", "required_action": "attach_lending_policy"}},
	{"name": "loan_product_owner_required", "description": "Loan products require owner.", "condition": {"operation": "register_product", "owner_present": False}, "effect": {"decision": "deny", "reason": "loan_product_owner_required", "required_action": "assign_product_owner"}},
	{"name": "loan_product_currency_supported", "description": "Loan product currency must be supported.", "condition": {"operation": "register_product", "currency_supported": False}, "effect": {"decision": "deny", "reason": "lending_currency_not_supported", "required_action": "select_supported_currency"}},
	{"name": "loan_product_type_supported", "description": "Loan product type must be supported.", "condition": {"operation": "register_product", "product_type_supported": False}, "effect": {"decision": "deny", "reason": "loan_product_type_not_supported", "required_action": "select_supported_product_type"}},
	{"name": "loan_product_term_valid", "description": "Loan product term must be valid.", "condition": {"operation": "register_product", "term_valid": False}, "effect": {"decision": "deny", "reason": "loan_product_term_invalid", "required_action": "set_supported_term"}},
	{"name": "loan_product_rate_valid", "description": "Loan product rate must be within policy.", "condition": {"operation": "register_product", "rate_valid": False}, "effect": {"decision": "deny", "reason": "loan_product_rate_invalid", "required_action": "set_policy_rate"}},
	{"name": "loan_product_amount_limits_valid", "description": "Loan product amount limits must be valid.", "condition": {"operation": "register_product", "amount_limits_valid": False}, "effect": {"decision": "deny", "reason": "loan_product_amount_limits_invalid", "required_action": "set_amount_limits"}},
	{"name": "loan_product_repayment_frequency_supported", "description": "Repayment frequency must be supported.", "condition": {"operation": "register_product", "repayment_frequency_supported": False}, "effect": {"decision": "deny", "reason": "repayment_frequency_not_supported", "required_action": "select_supported_frequency"}},
	{"name": "borrower_customer_required", "description": "Borrowers require customer reference.", "condition": {"operation": "onboard_borrower", "customer_present": False}, "effect": {"decision": "deny", "reason": "borrower_customer_required", "required_action": "attach_customer_reference"}},
	{"name": "borrower_kyc_required", "description": "Borrowers require KYC evidence.", "condition": {"operation": "onboard_borrower", "kyc_present": False}, "effect": {"decision": "deny", "reason": "borrower_kyc_required", "required_action": "attach_kyc_profile"}},
	{"name": "borrower_country_supported", "description": "Borrower country must be supported.", "condition": {"operation": "onboard_borrower", "country_supported": False}, "effect": {"decision": "deny", "reason": "borrower_country_not_supported", "required_action": "select_supported_country"}},
	{"name": "borrower_income_evidence_required", "description": "Borrowers require income evidence.", "condition": {"operation": "onboard_borrower", "income_evidence_present": False}, "effect": {"decision": "deny", "reason": "income_evidence_required", "required_action": "attach_income_evidence"}},
	{"name": "borrower_consent_required", "description": "Borrowers require lending consent evidence.", "condition": {"operation": "onboard_borrower", "consent_present": False}, "effect": {"decision": "deny", "reason": "borrower_consent_required", "required_action": "attach_borrower_consent"}},
	{"name": "application_borrower_required", "description": "Applications require borrower.", "condition": {"operation": "submit_application", "borrower_present": False}, "effect": {"decision": "deny", "reason": "borrower_required", "required_action": "select_borrower"}},
	{"name": "application_product_required", "description": "Applications require loan product.", "condition": {"operation": "submit_application", "product_present": False}, "effect": {"decision": "deny", "reason": "loan_product_required", "required_action": "select_loan_product"}},
	{"name": "application_amount_positive", "description": "Application amount must be positive.", "condition": {"operation": "submit_application", "positive_amount": False}, "effect": {"decision": "deny", "reason": "positive_amount_required", "required_action": "set_positive_amount"}},
	{"name": "application_amount_within_limits", "description": "Application amount must fit product limits.", "condition": {"operation": "submit_application", "amount_within_limits": False}, "effect": {"decision": "deny", "reason": "application_amount_outside_limits", "required_action": "adjust_requested_amount"}},
	{"name": "application_purpose_supported", "description": "Application purpose must be supported.", "condition": {"operation": "submit_application", "purpose_supported": False}, "effect": {"decision": "deny", "reason": "application_purpose_not_supported", "required_action": "select_supported_purpose"}},
	{"name": "application_affordability_required", "description": "Applications require affordability evidence.", "condition": {"operation": "submit_application", "affordability_present": False}, "effect": {"decision": "deny", "reason": "affordability_evidence_required", "required_action": "attach_affordability_evidence"}},
	{"name": "application_bank_statement_required", "description": "Applications require bank-statement evidence.", "condition": {"operation": "submit_application", "bank_statement_present": False}, "effect": {"decision": "deny", "reason": "bank_statement_evidence_required", "required_action": "attach_bank_statement_evidence"}},
	{"name": "application_aml_required", "description": "Applications require AML evidence.", "condition": {"operation": "submit_application", "aml_present": False}, "effect": {"decision": "deny", "reason": "aml_evidence_required", "required_action": "attach_aml_evidence"}},
	{"name": "application_fraud_required", "description": "Applications require fraud evidence.", "condition": {"operation": "submit_application", "fraud_present": False}, "effect": {"decision": "deny", "reason": "fraud_evidence_required", "required_action": "attach_fraud_evidence"}},
	{"name": "application_remittance_or_card_evidence_required", "description": "Credit files require remittance or card behavior evidence.", "condition": {"operation": "submit_application", "remittance_or_card_evidence_present": False}, "effect": {"decision": "require_review", "reason": "behavior_evidence_review_required", "required_action": "record_behavior_evidence_review"}},
	{"name": "high_amount_application_requires_review", "description": "High amount applications require review.", "condition": {"operation": "submit_application", "high_amount": True, "human_review_recorded": False}, "effect": {"decision": "require_review", "reason": "high_amount_application_review_required", "required_action": "record_credit_committee_review"}},
	{"name": "underwriting_application_required", "description": "Underwriting requires application.", "condition": {"operation": "record_underwriting", "application_present": False}, "effect": {"decision": "deny", "reason": "application_required", "required_action": "select_application"}},
	{"name": "underwriting_score_in_range", "description": "Underwriting score must be in range.", "condition": {"operation": "record_underwriting", "score_in_range": False}, "effect": {"decision": "deny", "reason": "underwriting_score_out_of_range", "required_action": "set_valid_score"}},
	{"name": "underwriting_decision_supported", "description": "Underwriting decision must be supported.", "condition": {"operation": "record_underwriting", "decision_supported": False}, "effect": {"decision": "deny", "reason": "underwriting_decision_not_supported", "required_action": "select_supported_decision"}},
	{"name": "underwriting_decision_evidence_required", "description": "Underwriting decisions require evidence.", "condition": {"operation": "record_underwriting", "decision_evidence_present": False}, "effect": {"decision": "deny", "reason": "underwriting_evidence_required", "required_action": "attach_underwriting_evidence"}},
	{"name": "underwriting_adverse_reason_required", "description": "Declines require adverse-action reason.", "condition": {"operation": "record_underwriting", "adverse_decision": True, "adverse_reason_present": False}, "effect": {"decision": "deny", "reason": "adverse_reason_required", "required_action": "record_adverse_action_reason"}},
	{"name": "underwriting_final_decision_requires_approval", "description": "Approve or decline decisions require human approval.", "condition": {"operation": "record_underwriting", "final_decision": True, "human_approval_recorded": False}, "effect": {"decision": "require_review", "reason": "underwriting_approval_required", "required_action": "record_underwriting_approval"}},
	{"name": "offer_application_required", "description": "Offers require application.", "condition": {"operation": "issue_offer", "application_present": False}, "effect": {"decision": "deny", "reason": "application_required", "required_action": "select_application"}},
	{"name": "offer_underwriting_required", "description": "Offers require underwriting decision.", "condition": {"operation": "issue_offer", "underwriting_present": False}, "effect": {"decision": "deny", "reason": "underwriting_decision_required", "required_action": "attach_underwriting_decision"}},
	{"name": "offer_apr_valid", "description": "Offer APR must be valid.", "condition": {"operation": "issue_offer", "apr_valid": False}, "effect": {"decision": "deny", "reason": "offer_apr_invalid", "required_action": "set_valid_apr"}},
	{"name": "offer_term_valid", "description": "Offer term must be valid.", "condition": {"operation": "issue_offer", "term_valid": False}, "effect": {"decision": "deny", "reason": "offer_term_invalid", "required_action": "set_valid_term"}},
	{"name": "offer_status_supported", "description": "Offer status must be supported.", "condition": {"operation": "issue_offer", "offer_status_supported": False}, "effect": {"decision": "deny", "reason": "offer_status_not_supported", "required_action": "select_supported_offer_status"}},
	{"name": "offer_expiry_required", "description": "Offers require expiry.", "condition": {"operation": "issue_offer", "expiry_present": False}, "effect": {"decision": "deny", "reason": "offer_expiry_required", "required_action": "set_offer_expiry"}},
	{"name": "offer_acceptance_required", "description": "Accepted offers require borrower acceptance evidence.", "condition": {"operation": "issue_offer", "accepted_offer": True, "borrower_acceptance_present": False}, "effect": {"decision": "deny", "reason": "borrower_acceptance_required", "required_action": "attach_borrower_acceptance"}},
	{"name": "disbursement_offer_required", "description": "Disbursements require accepted offer.", "condition": {"operation": "record_disbursement", "offer_present": False}, "effect": {"decision": "deny", "reason": "accepted_offer_required", "required_action": "select_accepted_offer"}},
	{"name": "disbursement_funding_account_required", "description": "Disbursements require funding account.", "condition": {"operation": "record_disbursement", "funding_account_present": False}, "effect": {"decision": "deny", "reason": "funding_account_required", "required_action": "attach_funding_account"}},
	{"name": "disbursement_rail_supported", "description": "Disbursement rail must be supported.", "condition": {"operation": "record_disbursement", "rail_supported": False}, "effect": {"decision": "deny", "reason": "disbursement_rail_not_supported", "required_action": "select_supported_disbursement_rail"}},
	{"name": "disbursement_destination_required", "description": "Disbursements require payment, wallet, card, or bank destination.", "condition": {"operation": "record_disbursement", "destination_present": False}, "effect": {"decision": "deny", "reason": "disbursement_destination_required", "required_action": "attach_disbursement_destination"}},
	{"name": "disbursement_requires_human_approval", "description": "Disbursement requires human approval.", "condition": {"operation": "record_disbursement", "human_approval_recorded": False}, "effect": {"decision": "require_review", "reason": "disbursement_approval_required", "required_action": "record_disbursement_approval"}},
	{"name": "repayment_offer_required", "description": "Repayment schedules require offer.", "condition": {"operation": "schedule_repayment", "offer_present": False}, "effect": {"decision": "deny", "reason": "loan_offer_required", "required_action": "select_loan_offer"}},
	{"name": "repayment_due_amount_positive", "description": "Repayment due amount must be positive.", "condition": {"operation": "schedule_repayment", "positive_due_amount": False}, "effect": {"decision": "deny", "reason": "positive_due_amount_required", "required_action": "set_positive_due_amount"}},
	{"name": "repayment_due_date_required", "description": "Repayment schedules require due date.", "condition": {"operation": "schedule_repayment", "due_date_present": False}, "effect": {"decision": "deny", "reason": "due_date_required", "required_action": "set_due_date"}},
	{"name": "repayment_frequency_supported", "description": "Repayment frequency must be supported.", "condition": {"operation": "schedule_repayment", "repayment_frequency_supported": False}, "effect": {"decision": "deny", "reason": "repayment_frequency_not_supported", "required_action": "select_supported_frequency"}},
	{"name": "collection_overdue_account_required", "description": "Collection cases require overdue account.", "condition": {"operation": "open_collection_case", "overdue_account_present": False}, "effect": {"decision": "deny", "reason": "overdue_account_required", "required_action": "attach_overdue_account"}},
	{"name": "collection_reason_supported", "description": "Collection reason must be supported.", "condition": {"operation": "open_collection_case", "collection_reason_supported": False}, "effect": {"decision": "deny", "reason": "collection_reason_not_supported", "required_action": "select_supported_collection_reason"}},
	{"name": "collection_reviewer_required", "description": "Collection cases require reviewer.", "condition": {"operation": "open_collection_case", "reviewer_present": False}, "effect": {"decision": "deny", "reason": "collection_reviewer_required", "required_action": "assign_collection_reviewer"}},
	{"name": "collection_contact_policy_required", "description": "Collection cases require contact policy.", "condition": {"operation": "open_collection_case", "contact_policy_present": False}, "effect": {"decision": "deny", "reason": "contact_policy_required", "required_action": "attach_contact_policy"}},
	{"name": "lending_batch_requires_bytewax", "description": "Lending batches require Bytewax.", "condition": {"operation": "lending_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_lending_batch_to_bytewax"}},
	{"name": "lending_agent_runtime_supported", "description": "Lending agents must use a supported runtime.", "condition": {"operation": "register_lending_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "lending_agent_runtime_not_supported", "required_action": "select_supported_agent_runtime"}},
	{"name": "lending_agent_role_supported", "description": "Lending agents must use a supported role.", "condition": {"operation": "register_lending_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "lending_agent_role_not_supported", "required_action": "select_supported_agent_role"}},
	{"name": "privileged_lending_agent_action_requires_human_approval", "description": "Privileged lending-agent actions require human approval.", "condition": {"operation": "lending_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
]


def _configuration_schema() -> dict[str, Any]:
	return {"type": "object", "required": list(DEFAULT_CONFIGURATION), "properties": {key: {"type": "object"} for key in DEFAULT_CONFIGURATION if key != "tenant_id"} | {"tenant_id": {"type": "string", "minLength": 1}}}


def _matches_condition(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if key.endswith("_ne"):
			if context.get(key[:-3]) == expected:
				return False
			continue
		if context.get(key) != expected:
			return False
	return True


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	if overrides:
		for key, value in overrides.items():
			if isinstance(value, dict) and isinstance(configuration.get(key), dict):
				configuration[key].update(value)
			else:
				configuration[key] = value
	return {"capability": CAPABILITY_ID, "display_name": CAPABILITY_NAME, "name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "configuration": configuration, "configuration_schema": _configuration_schema(), "provides": PROVIDES, "requires": REQUIRES, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "api_prefix": "/fintech-lending/api/v1", "routes": deepcopy(UI_ROUTES), "requires_theme": True, "view_module": "views.py", "template_roots": ["templates/", "static/"]}, "theme": deepcopy(THEME), "streaming": deepcopy(STREAMING)}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	contract = get_capability_contract(str(context.get("tenant_id") or "default"))
	matched = [rule for rule in contract["rule_engine"]["rules"] if _matches_condition(rule["condition"], context)]
	decision = "allow"
	for rule in matched:
		effect = rule["effect"]["decision"]
		if effect == "deny":
			decision = "deny"
			break
		if effect == "require_review" and decision == "allow":
			decision = "require_review"
	return {"decision": decision, "matched_rules": [rule["name"] for rule in matched], "actions": [rule["effect"] for rule in matched], "effects": [rule["effect"] for rule in matched]}
