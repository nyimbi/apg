"""Executable capability contract for APG Mobile Banking."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "fintech_mobile"
CAPABILITY_NAME = "Mobile Banking"
CAPABILITY_VERSION = "1.1.0"
MOBILE_EVENT_STREAM = "apg.fintech.mobile.lifecycle"

SUPPORTED_CURRENCIES = ["USD", "EUR", "GBP", "KES", "ZAR", "NGN", "GHS", "UGX", "TZS"]
SUPPORTED_COUNTRIES = ["KE", "UG", "TZ", "RW", "GH", "NG", "ZA", "GB", "US", "AE"]
SUPPORTED_PLATFORMS = ["ios", "android", "web", "ussd", "sms"]
SUPPORTED_AUTH_FACTORS = ["passcode", "biometric", "device_binding", "otp", "hardware_key"]
SUPPORTED_ACCOUNT_LINK_TYPES = ["deposit", "wallet", "card", "loan", "savings", "bnpl", "agency_float"]
SUPPORTED_PAYMENT_TYPES = ["peer_transfer", "merchant_payment", "bill_payment", "airtime", "loan_repayment", "savings_transfer", "card_payment", "wallet_cash_out"]
SUPPORTED_SERVICE_REASONS = ["account_access", "device_change", "payment_dispute", "card_issue", "loan_question", "bnpl_question", "fraud_report", "profile_update"]
SUPPORTED_NOTIFICATION_CHANNELS = ["push", "sms", "email", "in_app", "ussd"]
SUPPORTED_FRAUD_SEVERITIES = ["low", "medium", "high", "critical"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["mobile_ops_reviewer", "device_risk_reviewer", "mobile_payments_reviewer", "mobile_service_reviewer", "mobile_fraud_reviewer", "mobile_compliance_reviewer"]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"programs": {"owner_required": True, "supported_countries": SUPPORTED_COUNTRIES, "supported_currencies": SUPPORTED_CURRENCIES, "supported_platforms": SUPPORTED_PLATFORMS},
	"customers": {"kyc_required": True, "aml_required": True, "fraud_required": True, "consent_required": True, "supported_countries": SUPPORTED_COUNTRIES},
	"devices": {"attestation_required": True, "supported_platforms": SUPPORTED_PLATFORMS, "risk_tier_required": True},
	"auth_factors": {"supported_types": SUPPORTED_AUTH_FACTORS, "strength_reference_required": True, "device_required": True},
	"account_links": {"supported_types": SUPPORTED_ACCOUNT_LINK_TYPES, "supported_currencies": SUPPORTED_CURRENCIES, "provider_reference_required": True},
	"payments": {"supported_types": SUPPORTED_PAYMENT_TYPES, "supported_currencies": SUPPORTED_CURRENCIES, "positive_amount_required": True, "risk_reference_required": True, "high_value_threshold": 100000, "human_approval_required_for_high_value": True},
	"bills": {"biller_reference_required": True, "payment_reference_required": True},
	"airtime": {"operator_reference_required": True, "phone_reference_required": True},
	"service_requests": {"supported_reasons": SUPPORTED_SERVICE_REASONS, "evidence_required": True, "reviewer_required": True},
	"notifications": {"supported_channels": SUPPORTED_NOTIFICATION_CHANNELS, "consent_required": True},
	"fraud_events": {"supported_severities": SUPPORTED_FRAUD_SEVERITIES, "evidence_required": True, "high_severity_requires_approval": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_mobile_events": True},
	"observability": {"event_stream": MOBILE_EVENT_STREAM, "stream_processor": "bytewax", "emit_program_events": True, "emit_customer_events": True, "emit_device_events": True, "emit_auth_events": True, "emit_link_events": True, "emit_payment_events": True, "emit_bill_events": True, "emit_airtime_events": True, "emit_service_events": True, "emit_notification_events": True, "emit_fraud_events": True, "emit_agent_events": True},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "nlp": "nlpc", "keys": "keym", "payments": "fintech_payments", "wallets": "fintech_wallets", "cards": "fintech_cards", "kyc": "fintech_kyc", "aml": "fintech_aml", "fraud": "fintech_fraud", "neobanking": "fintech_neobanking", "lending": "fintech_lending", "bnpl": "fintech_bnpl", "agency": "fintech_agency", "event_stream": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_programs": True, "enable_customers": True, "enable_devices": True, "enable_auth_factors": True, "enable_account_links": True, "enable_payments": True, "enable_bills": True, "enable_airtime": True, "enable_service_requests": True, "enable_notifications": True, "enable_fraud_events": True, "enable_agents": True},
	"theme": {"default_theme": "fintech_mobile_control", "allow_tenant_overrides": True},
}

PROVIDES = ["mobile_banking_program_governance", "mobile_customer_enrollment", "trusted_device_lifecycle", "mobile_authentication_factor_workflow", "mobile_account_linking", "mobile_payment_workflow", "mobile_bill_payment_workflow", "mobile_airtime_workflow", "mobile_service_request_workflow", "mobile_notification_workflow", "mobile_fraud_event_workflow", "mobile_banking_agent_workflow"]
REQUIRES = ["auth", "audl", "ntfy", "nlpc", "keym", "fintech_payments", "fintech_wallets", "fintech_cards", "fintech_kyc", "fintech_aml", "fintech_fraud", "fintech_neobanking", "fintech_lending", "fintech_bnpl", "fintech_agency"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/fintech-mobile/dashboard", "component": "MobileBankingDashboard", "permission": "fintech_mobile:view", "nav_group": "Overview"},
	{"name": "programs", "path": "/fintech-mobile/programs", "component": "MobileProgramConsole", "permission": "fintech_mobile:manage_programs", "nav_group": "Programs"},
	{"name": "customers", "path": "/fintech-mobile/customers", "component": "MobileCustomerWorkbench", "permission": "fintech_mobile:customers", "nav_group": "Customers"},
	{"name": "devices", "path": "/fintech-mobile/devices", "component": "TrustedDeviceConsole", "permission": "fintech_mobile:devices", "nav_group": "Security"},
	{"name": "auth_factors", "path": "/fintech-mobile/auth-factors", "component": "AuthFactorConsole", "permission": "fintech_mobile:auth", "nav_group": "Security"},
	{"name": "account_links", "path": "/fintech-mobile/account-links", "component": "AccountLinkConsole", "permission": "fintech_mobile:accounts", "nav_group": "Accounts"},
	{"name": "payments", "path": "/fintech-mobile/payments", "component": "MobilePaymentConsole", "permission": "fintech_mobile:payments", "nav_group": "Payments"},
	{"name": "bills", "path": "/fintech-mobile/bills", "component": "BillPaymentWorkbench", "permission": "fintech_mobile:bills", "nav_group": "Payments"},
	{"name": "airtime", "path": "/fintech-mobile/airtime", "component": "AirtimeWorkbench", "permission": "fintech_mobile:airtime", "nav_group": "Payments"},
	{"name": "service_requests", "path": "/fintech-mobile/service-requests", "component": "MobileServiceWorkbench", "permission": "fintech_mobile:service", "nav_group": "Servicing"},
	{"name": "notifications", "path": "/fintech-mobile/notifications", "component": "NotificationPreferenceConsole", "permission": "fintech_mobile:notifications", "nav_group": "Engagement"},
	{"name": "fraud_events", "path": "/fintech-mobile/fraud-events", "component": "MobileFraudEventConsole", "permission": "fintech_mobile:fraud", "nav_group": "Risk"},
	{"name": "agents", "path": "/fintech-mobile/agents", "component": "MobileBankingAgentWorkbench", "permission": "fintech_mobile:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/fintech-mobile/settings", "component": "MobileBankingSettings", "permission": "fintech_mobile:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "fintech_mobile_control",
	"tokens": {"color.primary": "#0284C7", "color.accent": "#16A34A", "color.success": "#15803D", "color.warning": "#B45309", "color.danger": "#B91C1C", "surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF", "text.primary": "#111827", "text.secondary": "#4B5563", "border.radius": "8px", "density": "compact"},
	"components": {"programs": {"icon": "smartphone", "status_indicator": "program-chip"}, "customers": {"icon": "user-check", "status_indicator": "customer-chip"}, "devices": {"icon": "shield-check", "status_indicator": "device-chip"}, "auth_factors": {"icon": "key-round", "status_indicator": "auth-chip"}, "account_links": {"icon": "link", "status_indicator": "link-chip"}, "payments": {"icon": "send", "status_indicator": "payment-chip"}, "bills": {"icon": "receipt", "status_indicator": "bill-chip"}, "airtime": {"icon": "radio-tower", "status_indicator": "airtime-chip"}, "service_requests": {"icon": "life-buoy", "status_indicator": "service-chip"}, "notifications": {"icon": "bell", "status_indicator": "notification-chip"}, "fraud_events": {"icon": "shield-alert", "status_indicator": "fraud-chip"}, "agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"}},
}

STREAMING = {
	"processor": "bytewax",
	"stream": MOBILE_EVENT_STREAM,
	"key": "tenant_id",
	"events": ["mobile_program_registered", "mobile_customer_enrolled", "trusted_device_bound", "auth_factor_registered", "account_linked", "mobile_payment_initiated", "bill_payment_recorded", "airtime_purchased", "service_request_opened", "notification_preference_set", "fraud_event_recorded", "mobile_agent_registered"],
	"guardrails": ["mobile_batch_requires_bytewax", "privileged_mobile_agent_action_requires_human_approval"],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "Mobile banking operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "mobile_write_requires_policy", "description": "Mobile banking writes require policy evidence.", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "policy_evidence_required", "required_action": "attach_policy_evidence"}},
	{"name": "program_owner_required", "description": "Programs require owner.", "condition": {"operation": "register_program", "owner_present": False}, "effect": {"decision": "deny", "reason": "program_owner_required", "required_action": "assign_program_owner"}},
	{"name": "program_country_supported", "description": "Program country must be supported.", "condition": {"operation": "register_program", "country_supported": False}, "effect": {"decision": "deny", "reason": "program_country_not_supported", "required_action": "select_supported_country"}},
	{"name": "program_currency_supported", "description": "Program currency must be supported.", "condition": {"operation": "register_program", "currency_supported": False}, "effect": {"decision": "deny", "reason": "program_currency_not_supported", "required_action": "select_supported_currency"}},
	{"name": "program_platforms_valid", "description": "Program platforms must be supported.", "condition": {"operation": "register_program", "platforms_valid": False}, "effect": {"decision": "deny", "reason": "program_platforms_invalid", "required_action": "select_supported_platforms"}},
	{"name": "customer_reference_required", "description": "Customers require reference.", "condition": {"operation": "enroll_customer", "customer_present": False}, "effect": {"decision": "deny", "reason": "customer_reference_required", "required_action": "attach_customer_reference"}},
	{"name": "customer_country_supported", "description": "Customer country must be supported.", "condition": {"operation": "enroll_customer", "country_supported": False}, "effect": {"decision": "deny", "reason": "customer_country_not_supported", "required_action": "select_supported_country"}},
	{"name": "customer_kyc_required", "description": "Customers require KYC evidence.", "condition": {"operation": "enroll_customer", "kyc_present": False}, "effect": {"decision": "deny", "reason": "customer_kyc_required", "required_action": "attach_kyc_profile"}},
	{"name": "customer_consent_required", "description": "Customers require consent.", "condition": {"operation": "enroll_customer", "consent_present": False}, "effect": {"decision": "deny", "reason": "customer_consent_required", "required_action": "capture_consent"}},
	{"name": "customer_aml_required", "description": "Customers require AML evidence.", "condition": {"operation": "enroll_customer", "aml_present": False}, "effect": {"decision": "deny", "reason": "customer_aml_required", "required_action": "attach_aml_evidence"}},
	{"name": "customer_fraud_required", "description": "Customers require fraud evidence.", "condition": {"operation": "enroll_customer", "fraud_present": False}, "effect": {"decision": "deny", "reason": "customer_fraud_required", "required_action": "attach_fraud_evidence"}},
	{"name": "device_customer_required", "description": "Devices require customer.", "condition": {"operation": "bind_device", "customer_present": False}, "effect": {"decision": "deny", "reason": "device_customer_required", "required_action": "select_customer"}},
	{"name": "device_platform_supported", "description": "Device platform must be supported.", "condition": {"operation": "bind_device", "platform_supported": False}, "effect": {"decision": "deny", "reason": "device_platform_not_supported", "required_action": "select_supported_platform"}},
	{"name": "device_fingerprint_required", "description": "Devices require fingerprint.", "condition": {"operation": "bind_device", "fingerprint_present": False}, "effect": {"decision": "deny", "reason": "device_fingerprint_required", "required_action": "attach_device_fingerprint"}},
	{"name": "device_attestation_required", "description": "Devices require attestation.", "condition": {"operation": "bind_device", "attestation_present": False}, "effect": {"decision": "deny", "reason": "device_attestation_required", "required_action": "attach_attestation"}},
	{"name": "device_risk_tier_required", "description": "Devices require risk tier.", "condition": {"operation": "bind_device", "risk_tier_present": False}, "effect": {"decision": "deny", "reason": "device_risk_tier_required", "required_action": "assign_risk_tier"}},
	{"name": "auth_customer_required", "description": "Auth factors require customer.", "condition": {"operation": "register_auth_factor", "customer_present": False}, "effect": {"decision": "deny", "reason": "auth_customer_required", "required_action": "select_customer"}},
	{"name": "auth_device_required", "description": "Auth factors require device.", "condition": {"operation": "register_auth_factor", "device_present": False}, "effect": {"decision": "deny", "reason": "auth_device_required", "required_action": "select_device"}},
	{"name": "auth_factor_type_supported", "description": "Auth factor type must be supported.", "condition": {"operation": "register_auth_factor", "factor_type_supported": False}, "effect": {"decision": "deny", "reason": "auth_factor_type_not_supported", "required_action": "select_supported_factor"}},
	{"name": "auth_strength_required", "description": "Auth factors require strength reference.", "condition": {"operation": "register_auth_factor", "strength_reference_present": False}, "effect": {"decision": "deny", "reason": "auth_strength_reference_required", "required_action": "attach_strength_reference"}},
	{"name": "link_customer_required", "description": "Account links require customer.", "condition": {"operation": "link_account", "customer_present": False}, "effect": {"decision": "deny", "reason": "link_customer_required", "required_action": "select_customer"}},
	{"name": "link_type_supported", "description": "Account link type must be supported.", "condition": {"operation": "link_account", "link_type_supported": False}, "effect": {"decision": "deny", "reason": "account_link_type_not_supported", "required_action": "select_supported_link_type"}},
	{"name": "link_reference_required", "description": "Account links require account reference.", "condition": {"operation": "link_account", "account_reference_present": False}, "effect": {"decision": "deny", "reason": "account_reference_required", "required_action": "attach_account_reference"}},
	{"name": "link_currency_supported", "description": "Account link currency must be supported.", "condition": {"operation": "link_account", "currency_supported": False}, "effect": {"decision": "deny", "reason": "account_link_currency_not_supported", "required_action": "select_supported_currency"}},
	{"name": "link_provider_reference_required", "description": "Account links require provider reference.", "condition": {"operation": "link_account", "provider_reference_present": False}, "effect": {"decision": "deny", "reason": "provider_reference_required", "required_action": "attach_provider_reference"}},
	{"name": "payment_customer_required", "description": "Payments require customer.", "condition": {"operation": "initiate_payment", "customer_present": False}, "effect": {"decision": "deny", "reason": "payment_customer_required", "required_action": "select_customer"}},
	{"name": "payment_device_required", "description": "Payments require trusted device.", "condition": {"operation": "initiate_payment", "device_present": False}, "effect": {"decision": "deny", "reason": "payment_device_required", "required_action": "select_device"}},
	{"name": "payment_link_required", "description": "Payments require account link.", "condition": {"operation": "initiate_payment", "account_link_present": False}, "effect": {"decision": "deny", "reason": "payment_account_link_required", "required_action": "select_account_link"}},
	{"name": "payment_type_supported", "description": "Payment type must be supported.", "condition": {"operation": "initiate_payment", "payment_type_supported": False}, "effect": {"decision": "deny", "reason": "payment_type_not_supported", "required_action": "select_supported_payment_type"}},
	{"name": "payment_amount_positive", "description": "Payment amount must be positive.", "condition": {"operation": "initiate_payment", "positive_amount": False}, "effect": {"decision": "deny", "reason": "positive_payment_amount_required", "required_action": "set_positive_amount"}},
	{"name": "payment_currency_supported", "description": "Payment currency must be supported.", "condition": {"operation": "initiate_payment", "currency_supported": False}, "effect": {"decision": "deny", "reason": "payment_currency_not_supported", "required_action": "select_supported_currency"}},
	{"name": "payment_currency_matches_link", "description": "Payment currency must match account link.", "condition": {"operation": "initiate_payment", "currency_matches_link": False}, "effect": {"decision": "deny", "reason": "payment_link_currency_mismatch", "required_action": "use_link_currency"}},
	{"name": "payment_recipient_required", "description": "Payments require recipient.", "condition": {"operation": "initiate_payment", "recipient_present": False}, "effect": {"decision": "deny", "reason": "payment_recipient_required", "required_action": "attach_recipient"}},
	{"name": "payment_risk_reference_required", "description": "Payments require risk reference.", "condition": {"operation": "initiate_payment", "risk_reference_present": False}, "effect": {"decision": "deny", "reason": "payment_risk_reference_required", "required_action": "attach_risk_reference"}},
	{"name": "high_value_payment_requires_approval", "description": "High-value payments require approval.", "condition": {"operation": "initiate_payment", "high_value": True, "human_approval_recorded": False}, "effect": {"decision": "require_review", "reason": "payment_approval_required", "required_action": "record_payment_approval"}},
	{"name": "bill_payment_reference_required", "description": "Bill payments require biller reference.", "condition": {"operation": "record_bill_payment", "biller_reference_present": False}, "effect": {"decision": "deny", "reason": "biller_reference_required", "required_action": "attach_biller_reference"}},
	{"name": "bill_payment_payment_required", "description": "Bill payments require payment.", "condition": {"operation": "record_bill_payment", "payment_present": False}, "effect": {"decision": "deny", "reason": "bill_payment_required", "required_action": "select_payment"}},
	{"name": "bill_payment_type_required", "description": "Bill payment records require a bill-payment transaction.", "condition": {"operation": "record_bill_payment", "payment_type_matches": False}, "effect": {"decision": "deny", "reason": "bill_payment_type_required", "required_action": "use_bill_payment_transaction"}},
	{"name": "airtime_operator_required", "description": "Airtime purchases require operator.", "condition": {"operation": "purchase_airtime", "operator_reference_present": False}, "effect": {"decision": "deny", "reason": "airtime_operator_required", "required_action": "attach_operator_reference"}},
	{"name": "airtime_phone_required", "description": "Airtime purchases require phone reference.", "condition": {"operation": "purchase_airtime", "phone_reference_present": False}, "effect": {"decision": "deny", "reason": "airtime_phone_required", "required_action": "attach_phone_reference"}},
	{"name": "airtime_payment_required", "description": "Airtime purchases require payment.", "condition": {"operation": "purchase_airtime", "payment_present": False}, "effect": {"decision": "deny", "reason": "airtime_payment_required", "required_action": "select_payment"}},
	{"name": "airtime_payment_type_required", "description": "Airtime records require an airtime transaction.", "condition": {"operation": "purchase_airtime", "payment_type_matches": False}, "effect": {"decision": "deny", "reason": "airtime_payment_type_required", "required_action": "use_airtime_transaction"}},
	{"name": "service_customer_required", "description": "Service requests require customer.", "condition": {"operation": "open_service_request", "customer_present": False}, "effect": {"decision": "deny", "reason": "service_customer_required", "required_action": "select_customer"}},
	{"name": "service_reason_supported", "description": "Service reason must be supported.", "condition": {"operation": "open_service_request", "service_reason_supported": False}, "effect": {"decision": "deny", "reason": "service_reason_not_supported", "required_action": "select_supported_reason"}},
	{"name": "service_evidence_required", "description": "Service requests require evidence.", "condition": {"operation": "open_service_request", "evidence_present": False}, "effect": {"decision": "deny", "reason": "service_evidence_required", "required_action": "attach_service_evidence"}},
	{"name": "service_reviewer_required", "description": "Service requests require reviewer.", "condition": {"operation": "open_service_request", "reviewer_present": False}, "effect": {"decision": "deny", "reason": "service_reviewer_required", "required_action": "assign_reviewer"}},
	{"name": "notification_customer_required", "description": "Notification preferences require customer.", "condition": {"operation": "set_notification_preference", "customer_present": False}, "effect": {"decision": "deny", "reason": "notification_customer_required", "required_action": "select_customer"}},
	{"name": "notification_channel_supported", "description": "Notification channel must be supported.", "condition": {"operation": "set_notification_preference", "notification_channel_supported": False}, "effect": {"decision": "deny", "reason": "notification_channel_not_supported", "required_action": "select_supported_channel"}},
	{"name": "notification_consent_required", "description": "Notification preference requires consent.", "condition": {"operation": "set_notification_preference", "consent_present": False}, "effect": {"decision": "deny", "reason": "notification_consent_required", "required_action": "capture_notification_consent"}},
	{"name": "fraud_customer_required", "description": "Fraud events require customer.", "condition": {"operation": "record_fraud_event", "customer_present": False}, "effect": {"decision": "deny", "reason": "fraud_customer_required", "required_action": "select_customer"}},
	{"name": "fraud_severity_supported", "description": "Fraud severity must be supported.", "condition": {"operation": "record_fraud_event", "severity_supported": False}, "effect": {"decision": "deny", "reason": "fraud_severity_not_supported", "required_action": "select_supported_severity"}},
	{"name": "fraud_evidence_required", "description": "Fraud events require evidence.", "condition": {"operation": "record_fraud_event", "evidence_present": False}, "effect": {"decision": "deny", "reason": "fraud_evidence_required", "required_action": "attach_fraud_evidence"}},
	{"name": "high_severity_fraud_requires_approval", "description": "High-severity fraud events require approval.", "condition": {"operation": "record_fraud_event", "high_severity": True, "human_approval_recorded": False}, "effect": {"decision": "require_review", "reason": "fraud_approval_required", "required_action": "record_fraud_approval"}},
	{"name": "mobile_batch_requires_bytewax", "description": "Mobile batches require Bytewax.", "condition": {"operation": "mobile_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_mobile_batch_to_bytewax"}},
	{"name": "mobile_agent_runtime_supported", "description": "Mobile agents must use a supported runtime.", "condition": {"operation": "register_mobile_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "mobile_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "mobile_agent_role_supported", "description": "Mobile agents must use a supported role.", "condition": {"operation": "register_mobile_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "mobile_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "privileged_mobile_agent_action_requires_human_approval", "description": "Privileged mobile-agent actions require human approval.", "condition": {"operation": "mobile_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
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
	return {"capability": CAPABILITY_ID, "display_name": CAPABILITY_NAME, "name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "configuration": configuration, "configuration_schema": _configuration_schema(), "provides": PROVIDES, "requires": REQUIRES, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "api_prefix": "/fintech-mobile/api/v1", "routes": deepcopy(UI_ROUTES), "requires_theme": True, "view_module": "views.py", "template_roots": ["templates/", "static/"]}, "theme": deepcopy(THEME), "streaming": deepcopy(STREAMING)}


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
