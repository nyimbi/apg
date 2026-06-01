"""Executable capability contract for APG Buy Now Pay Later."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "fintech_bnpl"
CAPABILITY_NAME = "Buy Now Pay Later"
CAPABILITY_VERSION = "1.1.0"
BNPL_EVENT_STREAM = "apg.fintech.bnpl.lifecycle"

SUPPORTED_CURRENCIES = ["USD", "EUR", "GBP", "KES", "ZAR", "NGN", "GHS", "UGX", "TZS"]
SUPPORTED_COUNTRIES = ["KE", "UG", "TZ", "RW", "GH", "NG", "ZA", "GB", "US", "AE"]
SUPPORTED_MERCHANT_CATEGORIES = ["retail", "electronics", "grocery", "travel", "education", "medical", "services", "marketplace"]
SUPPORTED_CHECKOUT_CHANNELS = ["web", "mobile", "pos", "marketplace", "api"]
SUPPORTED_PLAN_TYPES = ["pay_in_3", "pay_in_4", "monthly_installments", "invoice_split"]
SUPPORTED_INSTALLMENT_STATUSES = ["scheduled", "due", "paid", "missed", "waived"]
SUPPORTED_SETTLEMENT_STATUSES = ["pending", "released", "held", "reconciled"]
SUPPORTED_DISPUTE_REASONS = ["goods_not_received", "duplicate", "refund_not_processed", "fraud", "quality_issue", "merchant_error"]
SUPPORTED_AFFORDABILITY_DECISIONS = ["approve", "decline", "refer"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["bnpl_ops_reviewer", "affordability_reviewer", "merchant_risk_reviewer", "settlement_reviewer", "dispute_reviewer", "compliance_reviewer"]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"programs": {"owner_required": True, "supported_countries": SUPPORTED_COUNTRIES, "supported_currencies": SUPPORTED_CURRENCIES, "fee_disclosure_required": True, "max_installment_count": 24},
	"consumers": {"kyc_required": True, "aml_required": True, "fraud_required": True, "consent_required": True, "supported_countries": SUPPORTED_COUNTRIES},
	"merchants": {"program_required": True, "legal_entity_required": True, "supported_categories": SUPPORTED_MERCHANT_CATEGORIES, "settlement_account_required": True},
	"checkout": {"supported_channels": SUPPORTED_CHECKOUT_CHANNELS, "supported_categories": SUPPORTED_MERCHANT_CATEGORIES, "supported_currencies": SUPPORTED_CURRENCIES, "high_value_threshold": 100000, "payment_reference_required": True},
	"affordability": {"supported_decisions": SUPPORTED_AFFORDABILITY_DECISIONS, "score_min": 0, "score_max": 1000, "human_approval_required_for_final_decisions": True, "adverse_reason_required_for_decline": True},
	"plans": {"supported_plan_types": SUPPORTED_PLAN_TYPES, "min_term_days": 1, "max_term_days": 730, "fee_disclosure_required": True, "customer_acceptance_required": True},
	"installments": {"supported_statuses": SUPPORTED_INSTALLMENT_STATUSES, "positive_due_amount_required": True},
	"settlements": {"supported_statuses": SUPPORTED_SETTLEMENT_STATUSES, "reconciliation_required": True, "high_value_threshold": 100000, "human_approval_required_for_holds_and_high_value_releases": True},
	"disputes": {"supported_reasons": SUPPORTED_DISPUTE_REASONS, "evidence_required": True, "reviewer_required": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "customer_consent_required": True, "audit_bnpl_events": True},
	"observability": {"event_stream": BNPL_EVENT_STREAM, "stream_processor": "bytewax", "emit_program_events": True, "emit_consumer_events": True, "emit_merchant_events": True, "emit_checkout_events": True, "emit_affordability_events": True, "emit_plan_events": True, "emit_installment_events": True, "emit_settlement_events": True, "emit_dispute_events": True, "emit_agent_events": True},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "nlp": "nlpc", "keys": "keym", "payments": "fintech_payments", "wallets": "fintech_wallets", "cards": "fintech_cards", "kyc": "fintech_kyc", "aml": "fintech_aml", "fraud": "fintech_fraud", "lending": "fintech_lending", "neobanking": "fintech_neobanking", "event_stream": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_programs": True, "enable_consumers": True, "enable_merchants": True, "enable_checkouts": True, "enable_affordability": True, "enable_plans": True, "enable_installments": True, "enable_settlements": True, "enable_disputes": True, "enable_agents": True},
	"theme": {"default_theme": "fintech_bnpl_control", "allow_tenant_overrides": True},
}

PROVIDES = ["bnpl_merchant_program_governance", "consumer_bnpl_lifecycle", "merchant_checkout_workflow", "affordability_decisioning", "bnpl_plan_workflow", "installment_schedule_workflow", "merchant_settlement_workflow", "bnpl_dispute_workflow", "bnpl_agent_workflow"]
REQUIRES = ["auth", "audl", "ntfy", "nlpc", "keym", "fintech_payments", "fintech_wallets", "fintech_cards", "fintech_kyc", "fintech_aml", "fintech_fraud", "fintech_lending", "fintech_neobanking"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/fintech-bnpl/dashboard", "component": "BNPLDashboard", "permission": "fintech_bnpl:view", "nav_group": "Overview"},
	{"name": "programs", "path": "/fintech-bnpl/programs", "component": "BNPLProgramConsole", "permission": "fintech_bnpl:manage_programs", "nav_group": "Programs"},
	{"name": "consumers", "path": "/fintech-bnpl/consumers", "component": "BNPLConsumerWorkbench", "permission": "fintech_bnpl:manage_consumers", "nav_group": "Consumers"},
	{"name": "merchants", "path": "/fintech-bnpl/merchants", "component": "MerchantRiskConsole", "permission": "fintech_bnpl:manage_merchants", "nav_group": "Merchants"},
	{"name": "checkouts", "path": "/fintech-bnpl/checkouts", "component": "CheckoutSessionConsole", "permission": "fintech_bnpl:manage_checkouts", "nav_group": "Checkout"},
	{"name": "affordability", "path": "/fintech-bnpl/affordability", "component": "AffordabilityDecisionWorkbench", "permission": "fintech_bnpl:decisioning", "nav_group": "Risk"},
	{"name": "plans", "path": "/fintech-bnpl/plans", "component": "BNPLPlanWorkbench", "permission": "fintech_bnpl:plans", "nav_group": "Plans"},
	{"name": "installments", "path": "/fintech-bnpl/installments", "component": "InstallmentScheduleConsole", "permission": "fintech_bnpl:installments", "nav_group": "Plans"},
	{"name": "settlements", "path": "/fintech-bnpl/settlements", "component": "MerchantSettlementWorkbench", "permission": "fintech_bnpl:settlements", "nav_group": "Settlement"},
	{"name": "disputes", "path": "/fintech-bnpl/disputes", "component": "BNPLDisputeWorkbench", "permission": "fintech_bnpl:disputes", "nav_group": "Servicing"},
	{"name": "agents", "path": "/fintech-bnpl/agents", "component": "BNPLAgentWorkbench", "permission": "fintech_bnpl:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/fintech-bnpl/settings", "component": "BNPLSettings", "permission": "fintech_bnpl:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "fintech_bnpl_control",
	"tokens": {"color.primary": "#0F766E", "color.accent": "#7C3AED", "color.success": "#15803D", "color.warning": "#B45309", "color.danger": "#B91C1C", "surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF", "text.primary": "#111827", "text.secondary": "#4B5563", "border.radius": "8px", "density": "compact"},
	"components": {"programs": {"icon": "badge-percent", "status_indicator": "program-chip"}, "consumers": {"icon": "user-check", "status_indicator": "consumer-chip"}, "merchants": {"icon": "store", "status_indicator": "merchant-chip"}, "checkouts": {"icon": "shopping-cart", "status_indicator": "checkout-chip"}, "affordability": {"icon": "shield-check", "status_indicator": "decision-chip"}, "plans": {"icon": "calendar-clock", "status_indicator": "plan-chip"}, "settlements": {"icon": "landmark", "status_indicator": "settlement-chip"}, "disputes": {"icon": "circle-alert", "status_indicator": "dispute-chip"}, "agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"}},
}

STREAMING = {
	"processor": "bytewax",
	"stream": BNPL_EVENT_STREAM,
	"key": "tenant_id",
	"events": ["bnpl_program_registered", "bnpl_consumer_onboarded", "bnpl_merchant_registered", "checkout_session_created", "affordability_decision_recorded", "bnpl_plan_created", "installment_scheduled", "merchant_settlement_recorded", "bnpl_dispute_opened", "bnpl_agent_registered"],
	"guardrails": ["bnpl_batch_requires_bytewax", "privileged_bnpl_agent_action_requires_human_approval"],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "BNPL operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "bnpl_write_requires_policy", "description": "BNPL writes require policy evidence.", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "policy_evidence_required", "required_action": "attach_policy_evidence"}},
	{"name": "program_owner_required", "description": "BNPL programs require an owner.", "condition": {"operation": "register_merchant_program", "owner_present": False}, "effect": {"decision": "deny", "reason": "program_owner_required", "required_action": "assign_program_owner"}},
	{"name": "program_country_supported", "description": "BNPL program country must be supported.", "condition": {"operation": "register_merchant_program", "country_supported": False}, "effect": {"decision": "deny", "reason": "program_country_not_supported", "required_action": "select_supported_country"}},
	{"name": "program_currency_supported", "description": "BNPL program currency must be supported.", "condition": {"operation": "register_merchant_program", "currency_supported": False}, "effect": {"decision": "deny", "reason": "program_currency_not_supported", "required_action": "select_supported_currency"}},
	{"name": "program_settlement_policy_required", "description": "BNPL programs require settlement policy evidence.", "condition": {"operation": "register_merchant_program", "settlement_policy_present": False}, "effect": {"decision": "deny", "reason": "settlement_policy_required", "required_action": "attach_settlement_policy"}},
	{"name": "program_fee_disclosure_required", "description": "BNPL programs require fee disclosure.", "condition": {"operation": "register_merchant_program", "fee_disclosure_present": False}, "effect": {"decision": "deny", "reason": "fee_disclosure_required", "required_action": "attach_fee_disclosure"}},
	{"name": "program_installment_count_valid", "description": "BNPL program installment count must be in range.", "condition": {"operation": "register_merchant_program", "installment_count_valid": False}, "effect": {"decision": "deny", "reason": "installment_count_invalid", "required_action": "set_valid_installment_count"}},
	{"name": "consumer_customer_required", "description": "Consumers require customer reference.", "condition": {"operation": "onboard_consumer", "customer_present": False}, "effect": {"decision": "deny", "reason": "customer_reference_required", "required_action": "attach_customer_reference"}},
	{"name": "consumer_kyc_required", "description": "Consumers require KYC evidence.", "condition": {"operation": "onboard_consumer", "kyc_present": False}, "effect": {"decision": "deny", "reason": "consumer_kyc_required", "required_action": "attach_kyc_profile"}},
	{"name": "consumer_country_supported", "description": "Consumer country must be supported.", "condition": {"operation": "onboard_consumer", "country_supported": False}, "effect": {"decision": "deny", "reason": "consumer_country_not_supported", "required_action": "select_supported_country"}},
	{"name": "consumer_consent_required", "description": "Consumers require consent.", "condition": {"operation": "onboard_consumer", "consent_present": False}, "effect": {"decision": "deny", "reason": "consumer_consent_required", "required_action": "capture_consumer_consent"}},
	{"name": "consumer_aml_required", "description": "Consumers require AML evidence.", "condition": {"operation": "onboard_consumer", "aml_present": False}, "effect": {"decision": "deny", "reason": "consumer_aml_required", "required_action": "attach_aml_evidence"}},
	{"name": "consumer_fraud_required", "description": "Consumers require fraud evidence.", "condition": {"operation": "onboard_consumer", "fraud_present": False}, "effect": {"decision": "deny", "reason": "consumer_fraud_required", "required_action": "attach_fraud_evidence"}},
	{"name": "merchant_program_required", "description": "Merchants require a BNPL program.", "condition": {"operation": "register_merchant", "program_present": False}, "effect": {"decision": "deny", "reason": "merchant_program_required", "required_action": "select_bnpl_program"}},
	{"name": "merchant_legal_entity_required", "description": "Merchants require legal entity evidence.", "condition": {"operation": "register_merchant", "legal_entity_present": False}, "effect": {"decision": "deny", "reason": "merchant_legal_entity_required", "required_action": "attach_legal_entity"}},
	{"name": "merchant_category_supported", "description": "Merchant category must be supported.", "condition": {"operation": "register_merchant", "merchant_category_supported": False}, "effect": {"decision": "deny", "reason": "merchant_category_not_supported", "required_action": "select_supported_merchant_category"}},
	{"name": "merchant_country_supported", "description": "Merchant country must be supported.", "condition": {"operation": "register_merchant", "country_supported": False}, "effect": {"decision": "deny", "reason": "merchant_country_not_supported", "required_action": "select_supported_country"}},
	{"name": "merchant_risk_tier_required", "description": "Merchants require risk tier.", "condition": {"operation": "register_merchant", "risk_tier_present": False}, "effect": {"decision": "deny", "reason": "merchant_risk_tier_required", "required_action": "assign_risk_tier"}},
	{"name": "merchant_settlement_account_required", "description": "Merchants require settlement account.", "condition": {"operation": "register_merchant", "settlement_account_present": False}, "effect": {"decision": "deny", "reason": "merchant_settlement_account_required", "required_action": "attach_settlement_account"}},
	{"name": "checkout_merchant_required", "description": "Checkout sessions require merchant.", "condition": {"operation": "create_checkout_session", "merchant_present": False}, "effect": {"decision": "deny", "reason": "checkout_merchant_required", "required_action": "select_merchant"}},
	{"name": "checkout_consumer_required", "description": "Checkout sessions require consumer.", "condition": {"operation": "create_checkout_session", "consumer_present": False}, "effect": {"decision": "deny", "reason": "checkout_consumer_required", "required_action": "select_consumer"}},
	{"name": "checkout_channel_supported", "description": "Checkout channel must be supported.", "condition": {"operation": "create_checkout_session", "channel_supported": False}, "effect": {"decision": "deny", "reason": "checkout_channel_not_supported", "required_action": "select_supported_channel"}},
	{"name": "checkout_category_supported", "description": "Checkout category must be supported.", "condition": {"operation": "create_checkout_session", "merchant_category_supported": False}, "effect": {"decision": "deny", "reason": "checkout_category_not_supported", "required_action": "select_supported_category"}},
	{"name": "checkout_amount_positive", "description": "Checkout amount must be positive.", "condition": {"operation": "create_checkout_session", "positive_amount": False}, "effect": {"decision": "deny", "reason": "checkout_amount_positive_required", "required_action": "set_positive_checkout_amount"}},
	{"name": "checkout_currency_supported", "description": "Checkout currency must be supported.", "condition": {"operation": "create_checkout_session", "currency_supported": False}, "effect": {"decision": "deny", "reason": "checkout_currency_not_supported", "required_action": "select_supported_currency"}},
	{"name": "checkout_payment_reference_required", "description": "Checkout requires payment reference.", "condition": {"operation": "create_checkout_session", "payment_reference_present": False}, "effect": {"decision": "deny", "reason": "payment_reference_required", "required_action": "attach_payment_reference"}},
	{"name": "checkout_fraud_required", "description": "Checkout requires fraud evidence.", "condition": {"operation": "create_checkout_session", "fraud_present": False}, "effect": {"decision": "deny", "reason": "checkout_fraud_required", "required_action": "attach_fraud_evidence"}},
	{"name": "checkout_aml_required", "description": "Checkout requires AML evidence.", "condition": {"operation": "create_checkout_session", "aml_present": False}, "effect": {"decision": "deny", "reason": "checkout_aml_required", "required_action": "attach_aml_evidence"}},
	{"name": "checkout_customer_consent_required", "description": "Checkout requires customer consent.", "condition": {"operation": "create_checkout_session", "consent_present": False}, "effect": {"decision": "deny", "reason": "checkout_customer_consent_required", "required_action": "capture_customer_consent"}},
	{"name": "high_value_checkout_requires_review", "description": "High-value checkout requires review.", "condition": {"operation": "create_checkout_session", "high_value": True, "human_review_recorded": False}, "effect": {"decision": "require_review", "reason": "checkout_review_required", "required_action": "record_checkout_review"}},
	{"name": "affordability_checkout_required", "description": "Affordability requires checkout.", "condition": {"operation": "record_affordability_decision", "checkout_present": False}, "effect": {"decision": "deny", "reason": "affordability_checkout_required", "required_action": "select_checkout"}},
	{"name": "affordability_decision_supported", "description": "Affordability decision must be supported.", "condition": {"operation": "record_affordability_decision", "decision_supported": False}, "effect": {"decision": "deny", "reason": "affordability_decision_not_supported", "required_action": "select_supported_decision"}},
	{"name": "affordability_score_in_range", "description": "Affordability score must be in range.", "condition": {"operation": "record_affordability_decision", "score_in_range": False}, "effect": {"decision": "deny", "reason": "affordability_score_out_of_range", "required_action": "set_score_0_to_1000"}},
	{"name": "affordability_evidence_required", "description": "Affordability requires evidence.", "condition": {"operation": "record_affordability_decision", "decision_evidence_present": False}, "effect": {"decision": "deny", "reason": "affordability_evidence_required", "required_action": "attach_decision_evidence"}},
	{"name": "declined_affordability_requires_adverse_reason", "description": "Declines require adverse reason.", "condition": {"operation": "record_affordability_decision", "adverse_decision": True, "adverse_reason_present": False}, "effect": {"decision": "deny", "reason": "adverse_reason_required", "required_action": "record_adverse_reason"}},
	{"name": "final_affordability_requires_approval", "description": "Final affordability decisions require approval.", "condition": {"operation": "record_affordability_decision", "final_decision": True, "human_approval_recorded": False}, "effect": {"decision": "require_review", "reason": "affordability_approval_required", "required_action": "record_affordability_approval"}},
	{"name": "plan_checkout_required", "description": "Plans require checkout.", "condition": {"operation": "create_bnpl_plan", "checkout_present": False}, "effect": {"decision": "deny", "reason": "plan_checkout_required", "required_action": "select_checkout"}},
	{"name": "plan_affordability_approved", "description": "Plans require approved affordability.", "condition": {"operation": "create_bnpl_plan", "affordability_approved": False}, "effect": {"decision": "deny", "reason": "affordability_approval_required", "required_action": "record_approved_affordability"}},
	{"name": "plan_type_supported", "description": "Plan type must be supported.", "condition": {"operation": "create_bnpl_plan", "plan_type_supported": False}, "effect": {"decision": "deny", "reason": "bnpl_plan_type_not_supported", "required_action": "select_supported_plan_type"}},
	{"name": "plan_principal_positive", "description": "Plan principal must be positive.", "condition": {"operation": "create_bnpl_plan", "positive_principal": False}, "effect": {"decision": "deny", "reason": "positive_principal_required", "required_action": "set_positive_principal"}},
	{"name": "plan_currency_supported", "description": "Plan currency must be supported.", "condition": {"operation": "create_bnpl_plan", "currency_supported": False}, "effect": {"decision": "deny", "reason": "plan_currency_not_supported", "required_action": "select_supported_currency"}},
	{"name": "plan_term_valid", "description": "Plan term must be valid.", "condition": {"operation": "create_bnpl_plan", "term_valid": False}, "effect": {"decision": "deny", "reason": "plan_term_invalid", "required_action": "set_valid_term"}},
	{"name": "plan_down_payment_valid", "description": "Plan down payment cannot exceed principal.", "condition": {"operation": "create_bnpl_plan", "down_payment_valid": False}, "effect": {"decision": "deny", "reason": "down_payment_invalid", "required_action": "set_valid_down_payment"}},
	{"name": "plan_fee_disclosure_required", "description": "Plans require fee disclosure.", "condition": {"operation": "create_bnpl_plan", "fee_disclosure_present": False}, "effect": {"decision": "deny", "reason": "plan_fee_disclosure_required", "required_action": "attach_fee_disclosure"}},
	{"name": "plan_customer_acceptance_required", "description": "Plans require customer acceptance.", "condition": {"operation": "create_bnpl_plan", "customer_acceptance_present": False}, "effect": {"decision": "deny", "reason": "customer_acceptance_required", "required_action": "capture_customer_acceptance"}},
	{"name": "installment_plan_required", "description": "Installments require plan.", "condition": {"operation": "schedule_installment", "plan_present": False}, "effect": {"decision": "deny", "reason": "installment_plan_required", "required_action": "select_plan"}},
	{"name": "installment_due_amount_positive", "description": "Installment due amount must be positive.", "condition": {"operation": "schedule_installment", "positive_due_amount": False}, "effect": {"decision": "deny", "reason": "positive_due_amount_required", "required_action": "set_positive_due_amount"}},
	{"name": "installment_due_date_required", "description": "Installments require due date.", "condition": {"operation": "schedule_installment", "due_date_present": False}, "effect": {"decision": "deny", "reason": "installment_due_date_required", "required_action": "set_due_date"}},
	{"name": "installment_status_supported", "description": "Installment status must be supported.", "condition": {"operation": "schedule_installment", "installment_status_supported": False}, "effect": {"decision": "deny", "reason": "installment_status_not_supported", "required_action": "select_supported_status"}},
	{"name": "settlement_merchant_required", "description": "Settlement requires merchant.", "condition": {"operation": "record_merchant_settlement", "merchant_present": False}, "effect": {"decision": "deny", "reason": "settlement_merchant_required", "required_action": "select_merchant"}},
	{"name": "settlement_plan_required", "description": "Settlement requires plan.", "condition": {"operation": "record_merchant_settlement", "plan_present": False}, "effect": {"decision": "deny", "reason": "settlement_plan_required", "required_action": "select_plan"}},
	{"name": "settlement_status_supported", "description": "Settlement status must be supported.", "condition": {"operation": "record_merchant_settlement", "settlement_status_supported": False}, "effect": {"decision": "deny", "reason": "settlement_status_not_supported", "required_action": "select_supported_status"}},
	{"name": "settlement_amounts_valid", "description": "Settlement amounts must be valid.", "condition": {"operation": "record_merchant_settlement", "settlement_amounts_valid": False}, "effect": {"decision": "deny", "reason": "settlement_amounts_invalid", "required_action": "set_valid_settlement_amounts"}},
	{"name": "settlement_reconciliation_required", "description": "Settlement requires reconciliation evidence.", "condition": {"operation": "record_merchant_settlement", "reconciliation_present": False}, "effect": {"decision": "deny", "reason": "settlement_reconciliation_required", "required_action": "attach_reconciliation_evidence"}},
	{"name": "settlement_payment_rail_required", "description": "Settlement requires payment rail reference.", "condition": {"operation": "record_merchant_settlement", "payment_rail_present": False}, "effect": {"decision": "deny", "reason": "settlement_payment_rail_required", "required_action": "attach_payment_rail_reference"}},
	{"name": "settlement_hold_or_high_value_release_requires_approval", "description": "Holds and high-value releases require approval.", "condition": {"operation": "record_merchant_settlement", "approval_required": True, "human_approval_recorded": False}, "effect": {"decision": "require_review", "reason": "settlement_approval_required", "required_action": "record_settlement_approval"}},
	{"name": "dispute_plan_required", "description": "Disputes require plan.", "condition": {"operation": "open_bnpl_dispute", "plan_present": False}, "effect": {"decision": "deny", "reason": "dispute_plan_required", "required_action": "select_plan"}},
	{"name": "dispute_reason_supported", "description": "Dispute reason must be supported.", "condition": {"operation": "open_bnpl_dispute", "dispute_reason_supported": False}, "effect": {"decision": "deny", "reason": "dispute_reason_not_supported", "required_action": "select_supported_dispute_reason"}},
	{"name": "dispute_evidence_required", "description": "Disputes require evidence.", "condition": {"operation": "open_bnpl_dispute", "evidence_present": False}, "effect": {"decision": "deny", "reason": "dispute_evidence_required", "required_action": "attach_dispute_evidence"}},
	{"name": "dispute_reviewer_required", "description": "Disputes require reviewer.", "condition": {"operation": "open_bnpl_dispute", "reviewer_present": False}, "effect": {"decision": "deny", "reason": "dispute_reviewer_required", "required_action": "assign_dispute_reviewer"}},
	{"name": "bnpl_batch_requires_bytewax", "description": "BNPL batches require Bytewax.", "condition": {"operation": "bnpl_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_bnpl_batch_to_bytewax"}},
	{"name": "bnpl_agent_runtime_supported", "description": "BNPL agents must use a supported runtime.", "condition": {"operation": "register_bnpl_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "bnpl_agent_runtime_not_supported", "required_action": "select_supported_agent_runtime"}},
	{"name": "bnpl_agent_role_supported", "description": "BNPL agents must use a supported role.", "condition": {"operation": "register_bnpl_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "bnpl_agent_role_not_supported", "required_action": "select_supported_agent_role"}},
	{"name": "privileged_bnpl_agent_action_requires_human_approval", "description": "Privileged BNPL-agent actions require human approval.", "condition": {"operation": "bnpl_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
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
	return {"capability": CAPABILITY_ID, "display_name": CAPABILITY_NAME, "name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "configuration": configuration, "configuration_schema": _configuration_schema(), "provides": PROVIDES, "requires": REQUIRES, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "api_prefix": "/fintech-bnpl/api/v1", "routes": deepcopy(UI_ROUTES), "requires_theme": True, "view_module": "views.py", "template_roots": ["templates/", "static/"]}, "theme": deepcopy(THEME), "streaming": deepcopy(STREAMING)}


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
