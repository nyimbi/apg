"""Executable capability contract for the Fintech Gateway capability."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "fintech_gateway"
CAPABILITY_NAME = "Fintech Gateway"
CAPABILITY_VERSION = "2.1.0"
GATEWAY_EVENT_STREAM = "apg.fintech.gateway.lifecycle"

SUPPORTED_PROVIDER_TYPES = ["card", "bank", "mobile_money", "wallet", "settlement", "fraud"]
SUPPORTED_PROVIDERS = ["stripe", "adyen", "mpesa", "dpo", "flutterwave", "pesapal", "paypal", "manual"]
SUPPORTED_CURRENCIES = ["USD", "EUR", "GBP", "KES", "ZAR", "NGN", "GHS", "UGX", "TZS"]
SUPPORTED_PAYMENT_METHODS = ["card", "bank_transfer", "mobile_money", "wallet", "cash_voucher"]
SUPPORTED_RISK_LEVELS = ["low", "medium", "high", "blocked"]
SUPPORTED_DISPUTE_REASONS = ["fraud", "duplicate", "product_not_received", "service_not_provided", "authorization", "processing_error", "other"]
SUPPORTED_GATEWAY_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_GATEWAY_AGENT_ROLES = [
	"merchant_underwriter",
	"routing_reviewer",
	"fraud_reviewer",
	"settlement_reviewer",
	"dispute_reviewer",
	"provider_operations_reviewer",
]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"merchants": {
		"merchant_code_required": True,
		"legal_name_required": True,
		"country_required": True,
		"kyc_required": True,
		"manual_review_required_for_high_risk": True,
	},
	"provider_connections": {
		"provider_required": True,
		"provider_type_required": True,
		"supported_providers": SUPPORTED_PROVIDERS,
		"supported_provider_types": SUPPORTED_PROVIDER_TYPES,
		"credential_reference_required": True,
	},
	"payment_methods": {
		"customer_reference_required": True,
		"method_type_required": True,
		"supported_methods": SUPPORTED_PAYMENT_METHODS,
		"token_reference_required": True,
	},
	"payment_intents": {
		"merchant_required": True,
		"amount_positive_required": True,
		"currency_supported_required": True,
		"payment_method_required": True,
		"risk_review_required_for_high_risk": True,
	},
	"authorization": {
		"provider_required": True,
		"routing_decision_required": True,
		"blocked_risk_denied": True,
		"approval_required_for_high_value": True,
	},
	"capture": {
		"authorized_payment_required": True,
		"capture_amount_positive_required": True,
		"overcapture_blocked": True,
	},
	"refunds": {
		"captured_payment_required": True,
		"refund_amount_positive_required": True,
		"overrefund_blocked": True,
		"review_required_for_large_refund": True,
	},
	"webhooks": {
		"provider_required": True,
		"event_id_required": True,
		"signature_required": True,
		"idempotency_required": True,
	},
	"settlements": {
		"provider_required": True,
		"settlement_reference_required": True,
		"amount_nonnegative_required": True,
		"reconciliation_review_required_for_variance": True,
	},
	"disputes": {
		"payment_required": True,
		"reason_required": True,
		"supported_reasons": SUPPORTED_DISPUTE_REASONS,
		"owner_required": True,
		"resolution_review_required": True,
	},
	"gateway_agents": {
		"enabled": True,
		"supported_runtimes": SUPPORTED_GATEWAY_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_GATEWAY_AGENT_ROLES,
		"max_autonomous_scope": "recommend_validate_and_prepare",
		"human_approval_required": True,
	},
	"governance": {
		"require_tenant_context": True,
		"policy_attached_for_writes": True,
		"audit_state_changes": True,
		"segregation_of_duties": True,
	},
	"observability": {
		"event_stream": GATEWAY_EVENT_STREAM,
		"stream_processor": "bytewax",
		"emit_merchant_events": True,
		"emit_provider_events": True,
		"emit_payment_events": True,
		"emit_risk_events": True,
		"emit_webhook_events": True,
		"emit_settlement_events": True,
		"emit_dispute_events": True,
		"emit_agent_events": True,
	},
	"adapters": {
		"authorization": "adapter",
		"audit": "adapter",
		"notification": "adapter",
		"vault": "adapter",
		"ledger": "adapter",
		"cash_management": "adapter",
		"accounts_receivable": "adapter",
		"customer_relationship_management": "adapter",
		"business_intelligence": "adapter",
		"provider_network": "adapter",
		"event_stream": "bytewax",
		"theme": "adapter",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_merchants": True,
		"enable_providers": True,
		"enable_payment_methods": True,
		"enable_payments": True,
		"enable_routing": True,
		"enable_risk": True,
		"enable_webhooks": True,
		"enable_settlements": True,
		"enable_disputes": True,
		"enable_agents": True,
		"enable_settings": True,
	},
	"theme": {
		"default_theme": "fintech_gateway_control",
		"allow_tenant_overrides": True,
	},
}


PROVIDES = [
	"merchant_onboarding_lifecycle",
	"provider_connection_lifecycle",
	"payment_method_tokenization_workflow",
	"payment_intent_lifecycle",
	"payment_routing_workflow",
	"fraud_risk_review_workflow",
	"authorization_capture_workflow",
	"refund_lifecycle",
	"webhook_ingestion_workflow",
	"settlement_reconciliation_workflow",
	"payment_dispute_workflow",
	"gateway_agents",
]

REQUIRES = [
	"auth",
	"audl",
	"ntfy",
	"composition_events",
	"composition_config",
	"keym",
	"encr",
	"cbm_cash_management",
	"arc_accounts_receivable",
	"crm_adv",
	"bia_anl",
]


UI_ROUTES = [
	{"name": "dashboard", "path": "/fintech-gateway/dashboard", "component": "GatewayDashboard", "permission": "fintech_gateway:view", "nav_group": "Overview"},
	{"name": "merchants", "path": "/fintech-gateway/merchants", "component": "MerchantWorkbench", "permission": "fintech_gateway:manage_merchants", "nav_group": "Merchants"},
	{"name": "providers", "path": "/fintech-gateway/providers", "component": "ProviderConnectionWorkbench", "permission": "fintech_gateway:manage_providers", "nav_group": "Providers"},
	{"name": "payment_methods", "path": "/fintech-gateway/payment-methods", "component": "PaymentMethodVault", "permission": "fintech_gateway:manage_payment_methods", "nav_group": "Payments"},
	{"name": "payments", "path": "/fintech-gateway/payments", "component": "PaymentOperationsConsole", "permission": "fintech_gateway:process", "nav_group": "Payments"},
	{"name": "routing", "path": "/fintech-gateway/routing", "component": "RoutingWorkbench", "permission": "fintech_gateway:route", "nav_group": "Operations"},
	{"name": "risk", "path": "/fintech-gateway/risk", "component": "FraudRiskConsole", "permission": "fintech_gateway:risk", "nav_group": "Risk"},
	{"name": "webhooks", "path": "/fintech-gateway/webhooks", "component": "WebhookInbox", "permission": "fintech_gateway:webhooks", "nav_group": "Operations"},
	{"name": "settlements", "path": "/fintech-gateway/settlements", "component": "SettlementConsole", "permission": "fintech_gateway:settle", "nav_group": "Finance"},
	{"name": "disputes", "path": "/fintech-gateway/disputes", "component": "PaymentDisputeWorkbench", "permission": "fintech_gateway:disputes", "nav_group": "Risk"},
	{"name": "agents", "path": "/fintech-gateway/agents", "component": "GatewayAgentWorkbench", "permission": "fintech_gateway:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/fintech-gateway/settings", "component": "GatewaySettings", "permission": "fintech_gateway:admin", "nav_group": "Administration"},
]


THEME = {
	"name": "fintech_gateway_control",
	"tokens": {
		"color.primary": "#1E5B5A",
		"color.accent": "#D97706",
		"color.success": "#237A57",
		"color.warning": "#B7791F",
		"color.danger": "#B42318",
		"surface.canvas": "#F6F8F8",
		"surface.panel": "#FFFFFF",
		"text.primary": "#172033",
		"text.secondary": "#52606D",
		"border.radius": "8px",
		"density": "compact",
	},
	"components": {
		"merchants": {"icon": "store", "status_indicator": "merchant-pill", "risk_style": "underwriting-band"},
		"providers": {"visual": "provider-grid", "status_style": "connectivity-chip"},
		"payments": {"visual": "payment-timeline", "status_style": "payment-chip"},
		"routing": {"visual": "route-lane", "status_style": "decision-chip"},
		"risk": {"visual": "risk-queue", "status_style": "risk-chip"},
		"webhooks": {"visual": "event-inbox", "status_style": "event-chip"},
		"settlements": {"visual": "settlement-grid", "status_style": "variance-chip"},
		"disputes": {"visual": "case-board", "status_style": "dispute-chip"},
		"agents": {"visual": "review-lane", "status_style": "agent-chip"},
	},
}


STREAMING = {
	"processor": "bytewax",
	"stream": GATEWAY_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"merchant_onboarded",
		"provider_connected",
		"payment_method_tokenized",
		"payment_intent_created",
		"payment_risk_assessed",
		"payment_authorized",
		"payment_captured",
		"payment_refunded",
		"webhook_ingested",
		"settlement_recorded",
		"payment_dispute_opened",
		"payment_dispute_resolved",
		"gateway_agent_registered",
	],
	"states": ["draft", "active", "review", "authorized", "captured", "refunded", "settled", "disputed", "resolved", "blocked"],
	"guardrails": [
		"gateway_batch_requires_bytewax",
		"gateway_event_requires_bytewax",
		"privileged_gateway_agent_action_requires_human_approval",
	],
}


RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "Gateway operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "gateway_write_requires_policy", "description": "Gateway writes require policy attachment.", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "operation_policy_required", "required_action": "attach_operation_policy"}},
	{"name": "merchant_requires_code", "description": "Merchants require merchant code.", "condition": {"operation": "onboard_merchant", "merchant_code_present": False}, "effect": {"decision": "deny", "reason": "merchant_code_required", "required_action": "set_merchant_code"}},
	{"name": "merchant_requires_legal_name", "description": "Merchants require legal name.", "condition": {"operation": "onboard_merchant", "legal_name_present": False}, "effect": {"decision": "deny", "reason": "merchant_legal_name_required", "required_action": "set_merchant_legal_name"}},
	{"name": "merchant_requires_country", "description": "Merchants require country.", "condition": {"operation": "onboard_merchant", "country_present": False}, "effect": {"decision": "deny", "reason": "merchant_country_required", "required_action": "set_country"}},
	{"name": "high_risk_merchant_requires_review", "description": "High-risk merchant onboarding requires review.", "condition": {"operation": "onboard_merchant", "risk_level": "high", "review_recorded": False}, "effect": {"decision": "require_review", "reason": "merchant_review_required", "required_action": "record_merchant_review"}},
	{"name": "provider_name_supported", "description": "Provider must be supported.", "condition": {"operation": "connect_provider", "provider_supported": False}, "effect": {"decision": "deny", "reason": "provider_not_supported", "required_action": "select_supported_provider"}},
	{"name": "provider_type_supported", "description": "Provider type must be supported.", "condition": {"operation": "connect_provider", "provider_type_supported": False}, "effect": {"decision": "deny", "reason": "provider_type_not_supported", "required_action": "select_supported_provider_type"}},
	{"name": "provider_requires_credentials", "description": "Provider connections require credential reference.", "condition": {"operation": "connect_provider", "credential_reference_present": False}, "effect": {"decision": "deny", "reason": "provider_credential_reference_required", "required_action": "attach_credential_reference"}},
	{"name": "payment_method_requires_merchant", "description": "Payment methods require merchant.", "condition": {"operation": "tokenize_payment_method", "merchant_present": False}, "effect": {"decision": "deny", "reason": "payment_method_merchant_required", "required_action": "select_merchant"}},
	{"name": "payment_method_requires_customer", "description": "Payment methods require customer reference.", "condition": {"operation": "tokenize_payment_method", "customer_reference_present": False}, "effect": {"decision": "deny", "reason": "customer_reference_required", "required_action": "attach_customer_reference"}},
	{"name": "payment_method_type_supported", "description": "Payment method type must be supported.", "condition": {"operation": "tokenize_payment_method", "payment_method_type_supported": False}, "effect": {"decision": "deny", "reason": "payment_method_type_not_supported", "required_action": "select_supported_payment_method_type"}},
	{"name": "payment_method_requires_token", "description": "Payment methods require token reference.", "condition": {"operation": "tokenize_payment_method", "token_reference_present": False}, "effect": {"decision": "deny", "reason": "token_reference_required", "required_action": "attach_token_reference"}},
	{"name": "payment_intent_requires_merchant", "description": "Payment intents require merchant.", "condition": {"operation": "create_payment_intent", "merchant_present": False}, "effect": {"decision": "deny", "reason": "merchant_required", "required_action": "select_merchant"}},
	{"name": "payment_intent_amount_positive", "description": "Payment intent amount must be positive.", "condition": {"operation": "create_payment_intent", "amount_lte": 0}, "effect": {"decision": "deny", "reason": "payment_amount_positive_required", "required_action": "set_positive_amount"}},
	{"name": "payment_intent_currency_supported", "description": "Payment intent currency must be supported.", "condition": {"operation": "create_payment_intent", "currency_supported": False}, "effect": {"decision": "deny", "reason": "currency_not_supported", "required_action": "select_supported_currency"}},
	{"name": "payment_intent_requires_method", "description": "Payment intents require payment method.", "condition": {"operation": "create_payment_intent", "payment_method_present": False}, "effect": {"decision": "deny", "reason": "payment_method_required", "required_action": "select_payment_method"}},
	{"name": "risk_requires_payment", "description": "Risk assessment requires payment intent.", "condition": {"operation": "assess_payment_risk", "payment_present": False}, "effect": {"decision": "deny", "reason": "payment_required", "required_action": "select_payment"}},
	{"name": "high_risk_payment_requires_review", "description": "High-risk payments require review.", "condition": {"operation": "assess_payment_risk", "risk_level": "high", "review_recorded": False}, "effect": {"decision": "require_review", "reason": "payment_risk_review_required", "required_action": "record_risk_review"}},
	{"name": "blocked_risk_denies_authorization", "description": "Blocked risk denies authorization.", "condition": {"operation": "authorize_payment", "risk_level": "blocked"}, "effect": {"decision": "deny", "reason": "payment_risk_blocked", "required_action": "resolve_risk_block"}},
	{"name": "authorization_requires_payment_intent", "description": "Authorization requires payment intent.", "condition": {"operation": "authorize_payment", "payment_present": False}, "effect": {"decision": "deny", "reason": "payment_intent_required", "required_action": "select_payment_intent"}},
	{"name": "authorization_requires_provider", "description": "Payment authorization requires provider.", "condition": {"operation": "authorize_payment", "provider_present": False}, "effect": {"decision": "deny", "reason": "provider_required", "required_action": "select_provider"}},
	{"name": "high_value_authorization_requires_approval", "description": "High-value authorization requires approval.", "condition": {"operation": "authorize_payment", "high_value": True, "approval_recorded": False}, "effect": {"decision": "require_review", "reason": "high_value_payment_approval_required", "required_action": "record_payment_approval"}},
	{"name": "capture_requires_authorized_payment", "description": "Capture requires authorized payment.", "condition": {"operation": "capture_payment", "authorized_payment_present": False}, "effect": {"decision": "deny", "reason": "authorized_payment_required", "required_action": "authorize_payment"}},
	{"name": "capture_amount_positive", "description": "Capture amount must be positive.", "condition": {"operation": "capture_payment", "capture_amount_lte": 0}, "effect": {"decision": "deny", "reason": "capture_amount_positive_required", "required_action": "set_positive_capture_amount"}},
	{"name": "capture_blocks_overcapture", "description": "Capture cannot exceed authorized amount.", "condition": {"operation": "capture_payment", "overcapture": True}, "effect": {"decision": "deny", "reason": "overcapture_blocked", "required_action": "reduce_capture_amount"}},
	{"name": "refund_requires_captured_payment", "description": "Refund requires captured payment.", "condition": {"operation": "refund_payment", "captured_payment_present": False}, "effect": {"decision": "deny", "reason": "captured_payment_required", "required_action": "capture_payment"}},
	{"name": "refund_amount_positive", "description": "Refund amount must be positive.", "condition": {"operation": "refund_payment", "refund_amount_lte": 0}, "effect": {"decision": "deny", "reason": "refund_amount_positive_required", "required_action": "set_positive_refund_amount"}},
	{"name": "refund_blocks_overrefund", "description": "Refund cannot exceed captured balance.", "condition": {"operation": "refund_payment", "overrefund": True}, "effect": {"decision": "deny", "reason": "overrefund_blocked", "required_action": "reduce_refund_amount"}},
	{"name": "large_refund_requires_review", "description": "Large refunds require review.", "condition": {"operation": "refund_payment", "large_refund": True, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "refund_review_required", "required_action": "record_refund_review"}},
	{"name": "webhook_requires_provider", "description": "Webhook ingestion requires provider.", "condition": {"operation": "ingest_webhook", "provider_present": False}, "effect": {"decision": "deny", "reason": "webhook_provider_required", "required_action": "attach_provider"}},
	{"name": "webhook_requires_event_id", "description": "Webhook ingestion requires event ID.", "condition": {"operation": "ingest_webhook", "event_id_present": False}, "effect": {"decision": "deny", "reason": "webhook_event_id_required", "required_action": "attach_event_id"}},
	{"name": "webhook_requires_signature", "description": "Webhook ingestion requires signature.", "condition": {"operation": "ingest_webhook", "signature_present": False}, "effect": {"decision": "deny", "reason": "webhook_signature_required", "required_action": "attach_signature"}},
	{"name": "webhook_requires_idempotency", "description": "Webhook ingestion requires idempotency key.", "condition": {"operation": "ingest_webhook", "idempotency_key_present": False}, "effect": {"decision": "deny", "reason": "webhook_idempotency_required", "required_action": "attach_idempotency_key"}},
	{"name": "settlement_requires_provider", "description": "Settlement requires provider.", "condition": {"operation": "record_settlement", "provider_present": False}, "effect": {"decision": "deny", "reason": "settlement_provider_required", "required_action": "attach_provider"}},
	{"name": "settlement_requires_reference", "description": "Settlement requires reference.", "condition": {"operation": "record_settlement", "settlement_reference_present": False}, "effect": {"decision": "deny", "reason": "settlement_reference_required", "required_action": "attach_settlement_reference"}},
	{"name": "settlement_amount_nonnegative", "description": "Settlement amount cannot be negative.", "condition": {"operation": "record_settlement", "settlement_amount_lt": 0}, "effect": {"decision": "deny", "reason": "settlement_amount_nonnegative_required", "required_action": "set_nonnegative_settlement_amount"}},
	{"name": "settlement_variance_requires_review", "description": "Settlement variance requires review.", "condition": {"operation": "record_settlement", "variance_detected": True, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "settlement_variance_review_required", "required_action": "record_settlement_review"}},
	{"name": "dispute_requires_payment", "description": "Disputes require payment.", "condition": {"operation": "open_dispute", "payment_present": False}, "effect": {"decision": "deny", "reason": "payment_required", "required_action": "select_payment"}},
	{"name": "dispute_reason_supported", "description": "Dispute reason must be supported.", "condition": {"operation": "open_dispute", "dispute_reason_supported": False}, "effect": {"decision": "deny", "reason": "dispute_reason_not_supported", "required_action": "select_supported_reason"}},
	{"name": "dispute_requires_owner", "description": "Disputes require owner.", "condition": {"operation": "open_dispute", "owner_present": False}, "effect": {"decision": "deny", "reason": "dispute_owner_required", "required_action": "assign_owner"}},
	{"name": "dispute_resolution_requires_review", "description": "Dispute resolution requires review.", "condition": {"operation": "resolve_dispute", "resolution_review_recorded": False}, "effect": {"decision": "deny", "reason": "dispute_resolution_review_required", "required_action": "record_resolution_review"}},
	{"name": "gateway_batch_requires_bytewax", "description": "Gateway batches require Bytewax coordination.", "condition": {"operation": "gateway_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_gateway_batch_to_bytewax"}},
	{"name": "gateway_event_requires_bytewax", "description": "Gateway events require Bytewax.", "condition": {"operation": "gateway_event", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_gateway_event_to_bytewax"}},
	{"name": "gateway_agent_runtime_supported", "description": "Gateway agents must use an approved runtime.", "condition": {"operation": "register_gateway_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "gateway_agent_runtime_not_supported", "required_action": "select_supported_agent_runtime"}},
	{"name": "gateway_agent_role_supported", "description": "Gateway agents must use an approved role.", "condition": {"operation": "register_gateway_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "gateway_agent_role_not_supported", "required_action": "select_supported_agent_role"}},
	{"name": "privileged_gateway_agent_action_requires_human_approval", "description": "Privileged gateway actions proposed by agents require human approval.", "condition": {"operation": "gateway_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},

	# Cross-tenant and privilege escalation guards
	{"name": "cross_tenant_gateway_access_denied", "description": "Gateway resources cannot be accessed across tenant boundaries.", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_access_denied", "required_action": "use_tenant_scoped_credentials"}},
	{"name": "privilege_escalation_denied", "description": "Gateway privilege escalation without approval is denied.", "condition": {"privilege_escalation_attempt": True, "approval_recorded": False}, "effect": {"decision": "deny", "reason": "privilege_escalation_denied", "required_action": "obtain_escalation_approval"}},

	# Africa-specific payment gateway rules
	{"name": "ke_cbk_psp_licence_required", "description": "Kenya CBK Payment Service Provider licence required for gateway operations.", "condition": {"operation": "register_gateway_provider", "country": "KE", "cbk_psp_licence_present": False}, "effect": {"decision": "deny", "reason": "ke_cbk_psp_licence_required", "required_action": "obtain_cbk_psp_licence"}},
	{"name": "mpesa_daraja_gateway_credentials_required", "description": "M-Pesa Daraja gateway integration requires registered consumer key and secret.", "condition": {"operation": "register_mpesa_gateway", "mpesa_consumer_key_present": False}, "effect": {"decision": "deny", "reason": "mpesa_daraja_credentials_required", "required_action": "register_mpesa_daraja_credentials"}},
	{"name": "mpesa_stk_push_gateway_shortcode_required", "description": "M-Pesa STK push gateway requires a registered Safaricom shortcode.", "condition": {"operation": "mpesa_stk_push", "mpesa_shortcode_present": False}, "effect": {"decision": "deny", "reason": "mpesa_shortcode_required", "required_action": "register_mpesa_shortcode"}},
	{"name": "mobile_money_gateway_kyc_required", "description": "Mobile money gateway transactions require sender KYC.", "condition": {"operation": "process_mobile_money", "kyc_present": False}, "effect": {"decision": "deny", "reason": "mobile_money_gateway_kyc_required", "required_action": "verify_sender_kyc"}},
	{"name": "ke_gateway_pci_dss_required", "description": "Kenya payment gateway must comply with PCI DSS.", "condition": {"operation": "register_gateway_provider", "country": "KE", "pci_dss_compliant": False}, "effect": {"decision": "deny", "reason": "ke_pci_dss_compliance_required", "required_action": "achieve_pci_dss_compliance"}},
	{"name": "pesalink_gateway_registered_required", "description": "PesaLink gateway requires registered bank account.", "condition": {"operation": "process_pesalink", "bank_account_registered": False}, "effect": {"decision": "deny", "reason": "pesalink_account_registration_required", "required_action": "register_pesalink_bank_account"}},
	{"name": "ng_cbn_payment_gateway_licence", "description": "Nigeria CBN payment solution services licence required for gateway operations.", "condition": {"operation": "register_gateway_provider", "country": "NG", "cbn_pss_licence_present": False}, "effect": {"decision": "deny", "reason": "ng_cbn_pss_licence_required", "required_action": "obtain_cbn_payment_solution_licence"}},
	{"name": "gateway_cbk_large_value_reporting", "description": "Kenya CBK large value payment gateway transactions require regulatory reporting.", "condition": {"operation": "process_payment", "country": "KE", "exceeds_cbk_reporting_threshold": True, "cbk_report_filed": False}, "effect": {"decision": "require_review", "reason": "ke_cbk_large_value_reporting_required", "required_action": "file_cbk_large_value_report"}},
]



def _configuration_schema() -> dict[str, Any]:
	return {
		"type": "object",
		"required": list(DEFAULT_CONFIGURATION),
		"properties": {
			key: {"type": "object"} for key in DEFAULT_CONFIGURATION if key != "tenant_id"
		} | {"tenant_id": {"type": "string", "minLength": 1}},
	}


def _matches_condition(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if key.endswith("_lte"):
			if context.get(key[:-4]) is None or context[key[:-4]] > expected:
				return False
			continue
		if key.endswith("_lt"):
			if context.get(key[:-3]) is None or context[key[:-3]] >= expected:
				return False
			continue
		if key.endswith("_gte"):
			if context.get(key[:-4]) is None or context[key[:-4]] < expected:
				return False
			continue
		if key.endswith("_gt"):
			if context.get(key[:-3]) is None or context[key[:-3]] <= expected:
				return False
			continue
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

	return {
		"capability": CAPABILITY_ID,
		"display_name": CAPABILITY_NAME,
		"name": CAPABILITY_NAME,
		"version": CAPABILITY_VERSION,
		"configuration": configuration,
		"configuration_schema": _configuration_schema(),
		"provides": PROVIDES,
		"requires": REQUIRES,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"api_prefix": "/fintech-gateway/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"requires_theme": True,
			"view_module": "views.py",
			"template_roots": ["templates/", "static/"],
		},
		"theme": deepcopy(THEME),
		"streaming": deepcopy(STREAMING),
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	contract = get_capability_contract(context.get("tenant_id", "default"))
	matched = [
		rule for rule in contract["rule_engine"]["rules"]
		if _matches_condition(rule["condition"], context)
	]
	decision = "allow"
	for rule in matched:
		rule_decision = rule["effect"]["decision"]
		if rule_decision == "deny":
			decision = "deny"
			break
		if rule_decision == "require_review" and decision == "allow":
			decision = "require_review"
	return {
		"decision": decision,
		"matched_rules": [rule["name"] for rule in matched],
		"effects": [rule["effect"] for rule in matched],
	}
