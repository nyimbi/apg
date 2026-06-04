"""Executable capability contract for APG Digital Neobanking."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "fintech_neobanking"
CAPABILITY_NAME = "Digital Neobanking"
CAPABILITY_VERSION = "1.1.0"
NEOBANKING_EVENT_STREAM = "apg.fintech.neobanking.lifecycle"

SUPPORTED_CURRENCIES = ["USD", "EUR", "GBP", "KES", "ZAR", "NGN", "GHS", "UGX", "TZS"]
SUPPORTED_COUNTRIES = ["KE", "UG", "TZ", "RW", "GH", "NG", "ZA", "GB", "US", "AE"]
SUPPORTED_ACCOUNT_TYPES = ["current", "savings", "joint", "business", "youth", "merchant"]
SUPPORTED_PAYMENT_RAILS = ["bank_transfer", "card", "wallet", "mobile_money", "internal_transfer"]
SUPPORTED_TRANSACTION_TYPES = ["deposit", "withdrawal", "transfer_in", "transfer_out", "card_purchase", "fee", "refund", "interest"]
SUPPORTED_CASE_REASONS = ["account_access", "card_issue", "payment_dispute", "kyc_review", "fraud_review", "fee_query", "statement_query"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["neobank_ops_reviewer", "account_risk_reviewer", "payments_reviewer", "customer_service_reviewer", "compliance_reviewer"]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"programs": {"owner_required": True, "settlement_account_required": True, "supported_countries": SUPPORTED_COUNTRIES, "supported_currencies": SUPPORTED_CURRENCIES},
	"customers": {"kyc_required": True, "aml_required": True, "fraud_required": True, "consent_required": True, "supported_countries": SUPPORTED_COUNTRIES},
	"accounts": {"supported_types": SUPPORTED_ACCOUNT_TYPES, "supported_currencies": SUPPORTED_CURRENCIES, "program_required": True, "customer_required": True},
	"rails": {"supported_rails": SUPPORTED_PAYMENT_RAILS, "provider_reference_required": True, "wallet_or_card_reference_supported": True},
	"transactions": {"supported_types": SUPPORTED_TRANSACTION_TYPES, "positive_amount_required": True, "risk_reference_required": True, "high_value_threshold": 100000, "human_approval_required_for_high_impact": True},
	"savings": {"target_amount_required": True, "source_account_required": True},
	"statements": {"period_required": True, "audit_statement_events": True},
	"service_cases": {"supported_reasons": SUPPORTED_CASE_REASONS, "reviewer_required": True, "evidence_required": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "customer_consent_required": True, "audit_neobanking_events": True},
	"observability": {"event_stream": NEOBANKING_EVENT_STREAM, "stream_processor": "bytewax", "emit_program_events": True, "emit_customer_events": True, "emit_account_events": True, "emit_rail_events": True, "emit_transaction_events": True, "emit_savings_events": True, "emit_statement_events": True, "emit_case_events": True, "emit_agent_events": True},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "nlp": "nlpc", "keys": "keym", "payments": "fintech_payments", "wallets": "fintech_wallets", "cards": "fintech_cards", "kyc": "fintech_kyc", "aml": "fintech_aml", "fraud": "fintech_fraud", "lending": "fintech_lending", "remittance": "fintech_remittance", "event_stream": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_programs": True, "enable_customers": True, "enable_accounts": True, "enable_rails": True, "enable_transactions": True, "enable_savings": True, "enable_statements": True, "enable_cases": True, "enable_agents": True},
	"theme": {"default_theme": "fintech_neobanking_control", "allow_tenant_overrides": True},
}

PROVIDES = ["neobank_program_governance", "digital_customer_onboarding", "deposit_account_lifecycle", "payment_rail_linking", "account_transaction_posting", "savings_pot_workflow", "statement_workflow", "customer_service_case_workflow", "neobanking_agent_workflow"]
REQUIRES = ["auth", "audl", "ntfy", "nlpc", "keym", "fintech_payments", "fintech_wallets", "fintech_cards", "fintech_kyc", "fintech_aml", "fintech_fraud", "fintech_lending", "fintech_remittance"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/fintech-neobanking/dashboard", "component": "NeobankingDashboard", "permission": "fintech_neobanking:view", "nav_group": "Overview"},
	{"name": "programs", "path": "/fintech-neobanking/programs", "component": "BankProgramConsole", "permission": "fintech_neobanking:manage_programs", "nav_group": "Programs"},
	{"name": "customers", "path": "/fintech-neobanking/customers", "component": "DigitalCustomerWorkbench", "permission": "fintech_neobanking:manage_customers", "nav_group": "Customers"},
	{"name": "accounts", "path": "/fintech-neobanking/accounts", "component": "DepositAccountWorkbench", "permission": "fintech_neobanking:manage_accounts", "nav_group": "Accounts"},
	{"name": "rails", "path": "/fintech-neobanking/rails", "component": "PaymentRailConsole", "permission": "fintech_neobanking:manage_rails", "nav_group": "Payments"},
	{"name": "transactions", "path": "/fintech-neobanking/transactions", "component": "TransactionConsole", "permission": "fintech_neobanking:post_transactions", "nav_group": "Payments"},
	{"name": "savings", "path": "/fintech-neobanking/savings", "component": "SavingsPotWorkbench", "permission": "fintech_neobanking:savings", "nav_group": "Accounts"},
	{"name": "statements", "path": "/fintech-neobanking/statements", "component": "StatementWorkbench", "permission": "fintech_neobanking:statements", "nav_group": "Servicing"},
	{"name": "cases", "path": "/fintech-neobanking/cases", "component": "ServiceCaseWorkbench", "permission": "fintech_neobanking:cases", "nav_group": "Servicing"},
	{"name": "agents", "path": "/fintech-neobanking/agents", "component": "NeobankingAgentWorkbench", "permission": "fintech_neobanking:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/fintech-neobanking/settings", "component": "NeobankingSettings", "permission": "fintech_neobanking:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "fintech_neobanking_control",
	"tokens": {"color.primary": "#2563EB", "color.accent": "#059669", "color.success": "#15803D", "color.warning": "#B45309", "color.danger": "#B91C1C", "surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF", "text.primary": "#111827", "text.secondary": "#4B5563", "border.radius": "8px", "density": "compact"},
	"components": {"programs": {"icon": "building-2", "status_indicator": "program-chip"}, "customers": {"icon": "user-check", "status_indicator": "customer-chip"}, "accounts": {"icon": "landmark", "status_indicator": "account-chip"}, "rails": {"icon": "waypoints", "status_indicator": "rail-chip"}, "transactions": {"icon": "receipt", "status_indicator": "transaction-chip"}, "savings": {"icon": "piggy-bank", "status_indicator": "savings-chip"}, "cases": {"icon": "life-buoy", "status_indicator": "case-chip"}, "agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"}},
}

STREAMING = {
	"processor": "bytewax",
	"stream": NEOBANKING_EVENT_STREAM,
	"key": "tenant_id",
	"events": ["bank_program_registered", "digital_customer_onboarded", "deposit_account_opened", "payment_rail_linked", "account_transaction_posted", "savings_pot_created", "statement_issued", "service_case_opened", "neobanking_agent_registered"],
	"guardrails": ["neobanking_batch_requires_bytewax", "privileged_neobanking_agent_action_requires_human_approval"],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "Neobanking operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "neobanking_write_requires_policy", "description": "Neobanking writes require policy evidence.", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "neobanking_policy_required", "required_action": "attach_neobanking_policy"}},
	{"name": "program_owner_required", "description": "Bank programs require owner.", "condition": {"operation": "register_program", "owner_present": False}, "effect": {"decision": "deny", "reason": "program_owner_required", "required_action": "assign_program_owner"}},
	{"name": "program_country_supported", "description": "Bank program country must be supported.", "condition": {"operation": "register_program", "country_supported": False}, "effect": {"decision": "deny", "reason": "program_country_not_supported", "required_action": "select_supported_country"}},
	{"name": "program_currency_supported", "description": "Bank program currency must be supported.", "condition": {"operation": "register_program", "currency_supported": False}, "effect": {"decision": "deny", "reason": "program_currency_not_supported", "required_action": "select_supported_currency"}},
	{"name": "program_settlement_required", "description": "Bank programs require settlement account.", "condition": {"operation": "register_program", "settlement_account_present": False}, "effect": {"decision": "deny", "reason": "settlement_account_required", "required_action": "attach_settlement_account"}},
	{"name": "customer_reference_required", "description": "Digital customers require customer reference.", "condition": {"operation": "onboard_customer", "customer_present": False}, "effect": {"decision": "deny", "reason": "customer_reference_required", "required_action": "attach_customer_reference"}},
	{"name": "customer_kyc_required", "description": "Digital customers require KYC evidence.", "condition": {"operation": "onboard_customer", "kyc_present": False}, "effect": {"decision": "deny", "reason": "customer_kyc_required", "required_action": "attach_kyc_profile"}},
	{"name": "customer_aml_required", "description": "Digital customers require AML evidence.", "condition": {"operation": "onboard_customer", "aml_present": False}, "effect": {"decision": "deny", "reason": "customer_aml_required", "required_action": "attach_aml_evidence"}},
	{"name": "customer_fraud_required", "description": "Digital customers require fraud evidence.", "condition": {"operation": "onboard_customer", "fraud_present": False}, "effect": {"decision": "deny", "reason": "customer_fraud_required", "required_action": "attach_fraud_evidence"}},
	{"name": "customer_country_supported", "description": "Customer country must be supported.", "condition": {"operation": "onboard_customer", "country_supported": False}, "effect": {"decision": "deny", "reason": "customer_country_not_supported", "required_action": "select_supported_country"}},
	{"name": "customer_consent_required", "description": "Digital customers require consent.", "condition": {"operation": "onboard_customer", "consent_present": False}, "effect": {"decision": "deny", "reason": "customer_consent_required", "required_action": "attach_customer_consent"}},
	{"name": "account_program_required", "description": "Account opening requires program.", "condition": {"operation": "open_account", "program_present": False}, "effect": {"decision": "deny", "reason": "program_required", "required_action": "select_bank_program"}},
	{"name": "account_customer_required", "description": "Account opening requires customer.", "condition": {"operation": "open_account", "customer_present": False}, "effect": {"decision": "deny", "reason": "customer_required", "required_action": "select_customer"}},
	{"name": "account_type_supported", "description": "Account type must be supported.", "condition": {"operation": "open_account", "account_type_supported": False}, "effect": {"decision": "deny", "reason": "account_type_not_supported", "required_action": "select_supported_account_type"}},
	{"name": "account_currency_supported", "description": "Account currency must be supported.", "condition": {"operation": "open_account", "currency_supported": False}, "effect": {"decision": "deny", "reason": "account_currency_not_supported", "required_action": "select_supported_currency"}},
	{"name": "account_initial_balance_non_negative", "description": "Initial balance cannot be negative.", "condition": {"operation": "open_account", "initial_balance_non_negative": False}, "effect": {"decision": "deny", "reason": "initial_balance_negative", "required_action": "set_non_negative_initial_balance"}},
	{"name": "rail_account_required", "description": "Rail links require account.", "condition": {"operation": "link_payment_rail", "account_present": False}, "effect": {"decision": "deny", "reason": "account_required", "required_action": "select_account"}},
	{"name": "rail_supported", "description": "Payment rail must be supported.", "condition": {"operation": "link_payment_rail", "rail_supported": False}, "effect": {"decision": "deny", "reason": "payment_rail_not_supported", "required_action": "select_supported_payment_rail"}},
	{"name": "rail_provider_reference_required", "description": "Payment rail links require provider reference.", "condition": {"operation": "link_payment_rail", "provider_reference_present": False}, "effect": {"decision": "deny", "reason": "provider_reference_required", "required_action": "attach_provider_reference"}},
	{"name": "transaction_account_required", "description": "Transactions require account.", "condition": {"operation": "post_transaction", "account_present": False}, "effect": {"decision": "deny", "reason": "account_required", "required_action": "select_account"}},
	{"name": "transaction_type_supported", "description": "Transaction type must be supported.", "condition": {"operation": "post_transaction", "transaction_type_supported": False}, "effect": {"decision": "deny", "reason": "transaction_type_not_supported", "required_action": "select_supported_transaction_type"}},
	{"name": "transaction_amount_positive", "description": "Transaction amount must be positive.", "condition": {"operation": "post_transaction", "positive_amount": False}, "effect": {"decision": "deny", "reason": "positive_amount_required", "required_action": "set_positive_amount"}},
	{"name": "transaction_currency_matches_account", "description": "Transaction currency must match account.", "condition": {"operation": "post_transaction", "currency_matches_account": False}, "effect": {"decision": "deny", "reason": "transaction_currency_mismatch", "required_action": "use_account_currency"}},
	{"name": "transaction_risk_reference_required", "description": "Transactions require AML/Fraud risk reference.", "condition": {"operation": "post_transaction", "risk_reference_present": False}, "effect": {"decision": "deny", "reason": "risk_reference_required", "required_action": "attach_risk_reference"}},
	{"name": "high_impact_transaction_requires_approval", "description": "High-impact transactions require human approval.", "condition": {"operation": "post_transaction", "high_impact": True, "human_approval_recorded": False}, "effect": {"decision": "require_review", "reason": "transaction_approval_required", "required_action": "record_transaction_approval"}},
	{"name": "savings_account_required", "description": "Savings pots require account.", "condition": {"operation": "create_savings_pot", "account_present": False}, "effect": {"decision": "deny", "reason": "account_required", "required_action": "select_account"}},
	{"name": "savings_name_required", "description": "Savings pots require name.", "condition": {"operation": "create_savings_pot", "name_present": False}, "effect": {"decision": "deny", "reason": "savings_name_required", "required_action": "set_savings_name"}},
	{"name": "savings_target_positive", "description": "Savings target must be positive.", "condition": {"operation": "create_savings_pot", "positive_target": False}, "effect": {"decision": "deny", "reason": "positive_target_required", "required_action": "set_positive_target"}},
	{"name": "statement_account_required", "description": "Statements require account.", "condition": {"operation": "issue_statement", "account_present": False}, "effect": {"decision": "deny", "reason": "account_required", "required_action": "select_account"}},
	{"name": "statement_period_required", "description": "Statements require period.", "condition": {"operation": "issue_statement", "period_present": False}, "effect": {"decision": "deny", "reason": "statement_period_required", "required_action": "set_statement_period"}},
	{"name": "case_customer_required", "description": "Service cases require customer.", "condition": {"operation": "open_service_case", "customer_present": False}, "effect": {"decision": "deny", "reason": "customer_required", "required_action": "select_customer"}},
	{"name": "case_account_required", "description": "Service cases require account.", "condition": {"operation": "open_service_case", "account_present": False}, "effect": {"decision": "deny", "reason": "account_required", "required_action": "select_account"}},
	{"name": "case_reason_supported", "description": "Case reason must be supported.", "condition": {"operation": "open_service_case", "case_reason_supported": False}, "effect": {"decision": "deny", "reason": "case_reason_not_supported", "required_action": "select_supported_case_reason"}},
	{"name": "case_reviewer_required", "description": "Service cases require reviewer.", "condition": {"operation": "open_service_case", "reviewer_present": False}, "effect": {"decision": "deny", "reason": "case_reviewer_required", "required_action": "assign_case_reviewer"}},
	{"name": "case_evidence_required", "description": "Service cases require evidence.", "condition": {"operation": "open_service_case", "evidence_present": False}, "effect": {"decision": "deny", "reason": "case_evidence_required", "required_action": "attach_case_evidence"}},
	{"name": "neobanking_batch_requires_bytewax", "description": "Neobanking batches require Bytewax.", "condition": {"operation": "neobanking_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_neobanking_batch_to_bytewax"}},
	{"name": "neobanking_agent_runtime_supported", "description": "Neobanking agents must use a supported runtime.", "condition": {"operation": "register_neobanking_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "neobanking_agent_runtime_not_supported", "required_action": "select_supported_agent_runtime"}},
	{"name": "neobanking_agent_role_supported", "description": "Neobanking agents must use a supported role.", "condition": {"operation": "register_neobanking_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "neobanking_agent_role_not_supported", "required_action": "select_supported_agent_role"}},
	{"name": "privileged_neobanking_agent_action_requires_human_approval", "description": "Privileged neobanking-agent actions require human approval.", "condition": {"operation": "neobanking_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},

	# Cross-tenant and privilege escalation guards
	{"name": "cross_tenant_neobanking_access_denied", "description": "Neobanking resources cannot be accessed across tenant boundaries.", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_access_denied", "required_action": "use_tenant_scoped_credentials"}},
	{"name": "privilege_escalation_denied", "description": "Neobanking privilege escalation without approval is denied.", "condition": {"privilege_escalation_attempt": True, "approval_recorded": False}, "effect": {"decision": "deny", "reason": "privilege_escalation_denied", "required_action": "obtain_escalation_approval"}},

	# Africa-specific neobanking rules
	{"name": "ke_cbk_banking_licence_required", "description": "Kenya CBK banking licence (or fintech partnership) required for neobank operations.", "condition": {"operation": "launch_neobank", "country": "KE", "cbk_licence_present": False}, "effect": {"decision": "deny", "reason": "ke_cbk_banking_licence_required", "required_action": "obtain_cbk_banking_licence_or_partnership"}},
	{"name": "mpesa_neobank_integration_required", "description": "Kenya neobanks must integrate M-Pesa for mobile money.", "condition": {"operation": "launch_neobank", "country": "KE", "mpesa_integrated": False}, "effect": {"decision": "deny", "reason": "mpesa_integration_required", "required_action": "integrate_mpesa_daraja"}},
	{"name": "mobile_money_neobank_kyc_tier", "description": "Neobank mobile money customers require CBK tiered KYC.", "condition": {"operation": "open_account", "account_type": "mobile_money", "kyc_tier_assigned": False}, "effect": {"decision": "deny", "reason": "cbk_kyc_tier_required", "required_action": "assign_cbk_kyc_tier"}},
	{"name": "ke_deposit_insurance_required", "description": "Kenya neobanks holding deposits require Kenya Deposit Insurance Corporation membership.", "condition": {"operation": "hold_customer_deposits", "country": "KE", "kdic_member": False}, "effect": {"decision": "deny", "reason": "ke_kdic_membership_required", "required_action": "obtain_kdic_membership"}},
	{"name": "ng_cbn_neobank_licence_required", "description": "Nigeria CBN payment service bank or microfinance licence required for neobanking.", "condition": {"operation": "launch_neobank", "country": "NG", "cbn_licence_present": False}, "effect": {"decision": "deny", "reason": "ng_cbn_neobank_licence_required", "required_action": "obtain_cbn_neobank_licence"}},
	{"name": "neobank_aml_transaction_monitoring", "description": "Neobank transactions require real-time AML transaction monitoring.", "condition": {"operation": "process_transaction", "aml_monitoring_active": False}, "effect": {"decision": "deny", "reason": "aml_transaction_monitoring_required", "required_action": "enable_aml_monitoring"}},
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
	return {"capability": CAPABILITY_ID, "display_name": CAPABILITY_NAME, "name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "configuration": configuration, "configuration_schema": _configuration_schema(), "provides": PROVIDES, "requires": REQUIRES, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "api_prefix": "/fintech-neobanking/api/v1", "routes": deepcopy(UI_ROUTES), "requires_theme": True, "view_module": "views.py", "template_roots": ["templates/", "static/"]}, "theme": deepcopy(THEME), "streaming": deepcopy(STREAMING)}


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
