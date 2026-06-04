"""Executable capability contract for APG Agency Banking."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "fintech_agency"
CAPABILITY_NAME = "Agency Banking"
CAPABILITY_VERSION = "1.1.0"
AGENCY_EVENT_STREAM = "apg.fintech.agency.lifecycle"

SUPPORTED_CURRENCIES = ["USD", "EUR", "GBP", "KES", "ZAR", "NGN", "GHS", "UGX", "TZS"]
SUPPORTED_COUNTRIES = ["KE", "UG", "TZ", "RW", "GH", "NG", "ZA", "GB", "US", "AE"]
SUPPORTED_OUTLET_TYPES = ["retail_shop", "pharmacy", "supermarket", "petrol_station", "mobile_money_agent", "post_office", "cooperative", "microfinance", "community_bank", "mobile_agent"]
SUPPORTED_CHANNELS = ["pos_terminal", "mobile_app", "ussd", "sms", "web_portal", "tablet", "feature_phone", "api"]
SUPPORTED_SERVICES = ["cash_in", "cash_out", "money_transfer", "bill_payment", "airtime_topup", "loan_disbursement", "loan_collection", "account_opening", "balance_inquiry", "mini_statement", "card_services", "insurance", "savings_products", "government_payments"]
SUPPORTED_SETTLEMENT_MODELS = ["real_time", "batch_hourly", "batch_daily", "bilateral", "central_switch"]
SUPPORTED_CUSTOMER_TIERS = ["tier_1", "tier_2", "tier_3"]
SUPPORTED_CASH_MOVEMENT_TYPES = ["float_topup", "cash_pickup", "cash_drop", "vault_rebalance", "emergency_liquidity"]
SUPPORTED_TRANSACTION_STATUSES = ["accepted", "posted", "reversed", "held"]
SUPPORTED_DISPUTE_REASONS = ["cash_shortage", "duplicate_transaction", "customer_denial", "failed_reversal", "agent_error", "fraud_suspected", "service_quality"]
SUPPORTED_SUPERVISION_OUTCOMES = ["passed", "remediation_required", "suspended", "closed"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["agency_ops_reviewer", "liquidity_reviewer", "field_supervisor", "agency_compliance_reviewer", "agency_dispute_reviewer", "commission_reviewer", "fraud_reviewer"]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"programs": {"owner_required": True, "supported_countries": SUPPORTED_COUNTRIES, "supported_currencies": SUPPORTED_CURRENCIES, "supported_services": SUPPORTED_SERVICES, "supported_settlement_models": SUPPORTED_SETTLEMENT_MODELS},
	"outlets": {"supported_types": SUPPORTED_OUTLET_TYPES, "license_required": True, "location_required": True, "security_plan_required": True, "minimum_initial_float": 500},
	"agents": {"identity_check_required": True, "training_required": True, "background_check_required": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
	"float_accounts": {"supported_currencies": SUPPORTED_CURRENCIES, "ledger_reference_required": True, "minimum_opening_balance": 0},
	"customers": {"supported_tiers": SUPPORTED_CUSTOMER_TIERS, "kyc_required": True, "consent_required": True, "aml_required": True, "fraud_required": True},
	"transactions": {"supported_services": SUPPORTED_SERVICES, "supported_channels": SUPPORTED_CHANNELS, "supported_statuses": SUPPORTED_TRANSACTION_STATUSES, "supported_currencies": SUPPORTED_CURRENCIES, "daily_limit": 200000, "high_value_threshold": 100000, "risk_reference_required": True},
	"cash_movements": {"supported_types": SUPPORTED_CASH_MOVEMENT_TYPES, "custodian_required": True, "approval_required_for_high_value": True, "high_value_threshold": 100000},
	"commissions": {"reconciliation_required": True, "payment_reference_required": True, "positive_amount_required": True},
	"disputes": {"supported_reasons": SUPPORTED_DISPUTE_REASONS, "evidence_required": True, "reviewer_required": True},
	"supervision": {"supported_outcomes": SUPPORTED_SUPERVISION_OUTCOMES, "evidence_required": True, "remediation_plan_required_for_findings": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_agency_events": True},
	"observability": {"event_stream": AGENCY_EVENT_STREAM, "stream_processor": "bytewax", "emit_program_events": True, "emit_outlet_events": True, "emit_agent_events": True, "emit_float_events": True, "emit_customer_events": True, "emit_transaction_events": True, "emit_cash_movement_events": True, "emit_commission_events": True, "emit_dispute_events": True, "emit_supervision_events": True},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "nlp": "nlpc", "keys": "keym", "payments": "fintech_payments", "wallets": "fintech_wallets", "cards": "fintech_cards", "kyc": "fintech_kyc", "aml": "fintech_aml", "fraud": "fintech_fraud", "remittance": "fintech_remittance", "neobanking": "fintech_neobanking", "lending": "fintech_lending", "event_stream": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_programs": True, "enable_outlets": True, "enable_agents": True, "enable_float_accounts": True, "enable_customers": True, "enable_transactions": True, "enable_cash_movements": True, "enable_commissions": True, "enable_disputes": True, "enable_supervision": True, "enable_ai_agents": True},
	"theme": {"default_theme": "fintech_agency_control", "allow_tenant_overrides": True},
}

PROVIDES = ["agency_program_governance", "agency_outlet_lifecycle", "agency_agent_accreditation", "agency_float_management", "agency_customer_workflow", "agency_transaction_workflow", "agency_cash_movement_workflow", "agency_commission_settlement_workflow", "agency_dispute_workflow", "agency_supervision_workflow", "agency_ai_agent_workflow"]
REQUIRES = ["auth", "audl", "ntfy", "nlpc", "keym", "fintech_payments", "fintech_wallets", "fintech_cards", "fintech_kyc", "fintech_aml", "fintech_fraud", "fintech_remittance", "fintech_neobanking", "fintech_lending"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/fintech-agency/dashboard", "component": "AgencyDashboard", "permission": "fintech_agency:view", "nav_group": "Overview"},
	{"name": "programs", "path": "/fintech-agency/programs", "component": "AgencyProgramConsole", "permission": "fintech_agency:manage_programs", "nav_group": "Programs"},
	{"name": "outlets", "path": "/fintech-agency/outlets", "component": "AgencyOutletWorkbench", "permission": "fintech_agency:manage_outlets", "nav_group": "Network"},
	{"name": "agents", "path": "/fintech-agency/agents", "component": "AccreditedAgentConsole", "permission": "fintech_agency:manage_agents", "nav_group": "Network"},
	{"name": "float_accounts", "path": "/fintech-agency/float-accounts", "component": "FloatAccountConsole", "permission": "fintech_agency:float", "nav_group": "Liquidity"},
	{"name": "customers", "path": "/fintech-agency/customers", "component": "AgencyCustomerWorkbench", "permission": "fintech_agency:customers", "nav_group": "Customers"},
	{"name": "transactions", "path": "/fintech-agency/transactions", "component": "AgencyTransactionConsole", "permission": "fintech_agency:transactions", "nav_group": "Transactions"},
	{"name": "cash_movements", "path": "/fintech-agency/cash-movements", "component": "CashMovementWorkbench", "permission": "fintech_agency:liquidity", "nav_group": "Liquidity"},
	{"name": "commissions", "path": "/fintech-agency/commissions", "component": "CommissionSettlementWorkbench", "permission": "fintech_agency:commissions", "nav_group": "Settlement"},
	{"name": "disputes", "path": "/fintech-agency/disputes", "component": "AgencyDisputeWorkbench", "permission": "fintech_agency:disputes", "nav_group": "Servicing"},
	{"name": "supervision", "path": "/fintech-agency/supervision", "component": "SupervisionVisitConsole", "permission": "fintech_agency:supervision", "nav_group": "Field Control"},
	{"name": "ai_agents", "path": "/fintech-agency/ai-agents", "component": "AgencyAIAgentWorkbench", "permission": "fintech_agency:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/fintech-agency/settings", "component": "AgencySettings", "permission": "fintech_agency:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "fintech_agency_control",
	"tokens": {"color.primary": "#047857", "color.accent": "#2563EB", "color.success": "#15803D", "color.warning": "#B45309", "color.danger": "#B91C1C", "surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF", "text.primary": "#111827", "text.secondary": "#4B5563", "border.radius": "8px", "density": "compact"},
	"components": {"programs": {"icon": "network", "status_indicator": "program-chip"}, "outlets": {"icon": "store", "status_indicator": "outlet-chip"}, "agents": {"icon": "badge-check", "status_indicator": "agent-chip"}, "float_accounts": {"icon": "wallet", "status_indicator": "float-chip"}, "transactions": {"icon": "receipt", "status_indicator": "transaction-chip"}, "cash_movements": {"icon": "truck", "status_indicator": "cash-chip"}, "commissions": {"icon": "percent", "status_indicator": "commission-chip"}, "disputes": {"icon": "circle-alert", "status_indicator": "dispute-chip"}, "supervision": {"icon": "clipboard-check", "status_indicator": "supervision-chip"}, "ai_agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"}},
}

STREAMING = {
	"processor": "bytewax",
	"stream": AGENCY_EVENT_STREAM,
	"key": "tenant_id",
	"events": ["agency_program_registered", "agency_outlet_onboarded", "agency_agent_accredited", "float_account_opened", "agency_customer_onboarded", "agency_transaction_recorded", "cash_movement_recorded", "commission_settlement_recorded", "agency_dispute_opened", "supervision_visit_recorded", "agency_ai_agent_registered"],
	"guardrails": ["agency_batch_requires_bytewax", "privileged_agency_agent_action_requires_human_approval"],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "Agency banking operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "agency_write_requires_policy", "description": "Agency banking writes require policy evidence.", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "policy_evidence_required", "required_action": "attach_policy_evidence"}},
	{"name": "program_owner_required", "description": "Programs require owner.", "condition": {"operation": "register_program", "owner_present": False}, "effect": {"decision": "deny", "reason": "program_owner_required", "required_action": "assign_program_owner"}},
	{"name": "program_country_supported", "description": "Program country must be supported.", "condition": {"operation": "register_program", "country_supported": False}, "effect": {"decision": "deny", "reason": "program_country_not_supported", "required_action": "select_supported_country"}},
	{"name": "program_currency_supported", "description": "Program currency must be supported.", "condition": {"operation": "register_program", "currency_supported": False}, "effect": {"decision": "deny", "reason": "program_currency_not_supported", "required_action": "select_supported_currency"}},
	{"name": "program_settlement_model_supported", "description": "Program settlement model must be supported.", "condition": {"operation": "register_program", "settlement_model_supported": False}, "effect": {"decision": "deny", "reason": "settlement_model_not_supported", "required_action": "select_supported_settlement_model"}},
	{"name": "program_services_required", "description": "Programs require at least one supported service.", "condition": {"operation": "register_program", "services_valid": False}, "effect": {"decision": "deny", "reason": "program_services_invalid", "required_action": "select_supported_services"}},
	{"name": "outlet_program_required", "description": "Outlets require program.", "condition": {"operation": "onboard_outlet", "program_present": False}, "effect": {"decision": "deny", "reason": "outlet_program_required", "required_action": "select_program"}},
	{"name": "outlet_type_supported", "description": "Outlet type must be supported.", "condition": {"operation": "onboard_outlet", "outlet_type_supported": False}, "effect": {"decision": "deny", "reason": "outlet_type_not_supported", "required_action": "select_supported_outlet_type"}},
	{"name": "outlet_country_supported", "description": "Outlet country must be supported.", "condition": {"operation": "onboard_outlet", "country_supported": False}, "effect": {"decision": "deny", "reason": "outlet_country_not_supported", "required_action": "select_supported_country"}},
	{"name": "outlet_license_required", "description": "Outlets require license evidence.", "condition": {"operation": "onboard_outlet", "license_present": False}, "effect": {"decision": "deny", "reason": "outlet_license_required", "required_action": "attach_license"}},
	{"name": "outlet_location_required", "description": "Outlets require location evidence.", "condition": {"operation": "onboard_outlet", "location_present": False}, "effect": {"decision": "deny", "reason": "outlet_location_required", "required_action": "attach_location"}},
	{"name": "outlet_security_required", "description": "Outlets require security plan.", "condition": {"operation": "onboard_outlet", "security_plan_present": False}, "effect": {"decision": "deny", "reason": "outlet_security_plan_required", "required_action": "attach_security_plan"}},
	{"name": "outlet_channel_supported", "description": "Outlet channel must be supported.", "condition": {"operation": "onboard_outlet", "channel_supported": False}, "effect": {"decision": "deny", "reason": "outlet_channel_not_supported", "required_action": "select_supported_channel"}},
	{"name": "outlet_initial_float_valid", "description": "Outlets require minimum initial float.", "condition": {"operation": "onboard_outlet", "initial_float_valid": False}, "effect": {"decision": "deny", "reason": "initial_float_below_minimum", "required_action": "increase_initial_float"}},
	{"name": "agent_outlet_required", "description": "Agents require outlet.", "condition": {"operation": "accredit_agent", "outlet_present": False}, "effect": {"decision": "deny", "reason": "agent_outlet_required", "required_action": "select_outlet"}},
	{"name": "agent_identity_required", "description": "Agents require identity evidence.", "condition": {"operation": "accredit_agent", "identity_present": False}, "effect": {"decision": "deny", "reason": "agent_identity_required", "required_action": "attach_identity_check"}},
	{"name": "agent_training_required", "description": "Agents require training evidence.", "condition": {"operation": "accredit_agent", "training_present": False}, "effect": {"decision": "deny", "reason": "agent_training_required", "required_action": "attach_training_evidence"}},
	{"name": "agent_background_required", "description": "Agents require background check.", "condition": {"operation": "accredit_agent", "background_check_present": False}, "effect": {"decision": "deny", "reason": "agent_background_check_required", "required_action": "attach_background_check"}},
	{"name": "float_outlet_required", "description": "Float accounts require outlet.", "condition": {"operation": "open_float_account", "outlet_present": False}, "effect": {"decision": "deny", "reason": "float_outlet_required", "required_action": "select_outlet"}},
	{"name": "float_currency_supported", "description": "Float currency must be supported.", "condition": {"operation": "open_float_account", "currency_supported": False}, "effect": {"decision": "deny", "reason": "float_currency_not_supported", "required_action": "select_supported_currency"}},
	{"name": "float_balance_non_negative", "description": "Opening balance cannot be negative.", "condition": {"operation": "open_float_account", "balance_non_negative": False}, "effect": {"decision": "deny", "reason": "float_balance_negative", "required_action": "set_non_negative_balance"}},
	{"name": "float_ledger_reference_required", "description": "Float account requires ledger reference.", "condition": {"operation": "open_float_account", "ledger_reference_present": False}, "effect": {"decision": "deny", "reason": "float_ledger_reference_required", "required_action": "attach_ledger_reference"}},
	{"name": "customer_reference_required", "description": "Agency customers require customer reference.", "condition": {"operation": "onboard_customer", "customer_present": False}, "effect": {"decision": "deny", "reason": "customer_reference_required", "required_action": "attach_customer_reference"}},
	{"name": "customer_tier_supported", "description": "Customer tier must be supported.", "condition": {"operation": "onboard_customer", "customer_tier_supported": False}, "effect": {"decision": "deny", "reason": "customer_tier_not_supported", "required_action": "select_supported_customer_tier"}},
	{"name": "customer_kyc_required", "description": "Customers require KYC evidence.", "condition": {"operation": "onboard_customer", "kyc_present": False}, "effect": {"decision": "deny", "reason": "customer_kyc_required", "required_action": "attach_kyc_profile"}},
	{"name": "customer_consent_required", "description": "Customers require consent.", "condition": {"operation": "onboard_customer", "consent_present": False}, "effect": {"decision": "deny", "reason": "customer_consent_required", "required_action": "capture_consent"}},
	{"name": "customer_aml_required", "description": "Customers require AML evidence.", "condition": {"operation": "onboard_customer", "aml_present": False}, "effect": {"decision": "deny", "reason": "customer_aml_required", "required_action": "attach_aml_evidence"}},
	{"name": "customer_fraud_required", "description": "Customers require fraud evidence.", "condition": {"operation": "onboard_customer", "fraud_present": False}, "effect": {"decision": "deny", "reason": "customer_fraud_required", "required_action": "attach_fraud_evidence"}},
	{"name": "transaction_outlet_required", "description": "Transactions require outlet.", "condition": {"operation": "record_transaction", "outlet_present": False}, "effect": {"decision": "deny", "reason": "transaction_outlet_required", "required_action": "select_outlet"}},
	{"name": "transaction_agent_required", "description": "Transactions require accredited agent.", "condition": {"operation": "record_transaction", "agent_present": False}, "effect": {"decision": "deny", "reason": "transaction_agent_required", "required_action": "select_agent"}},
	{"name": "transaction_customer_required", "description": "Transactions require customer.", "condition": {"operation": "record_transaction", "customer_present": False}, "effect": {"decision": "deny", "reason": "transaction_customer_required", "required_action": "select_customer"}},
	{"name": "transaction_float_required", "description": "Transactions require float account.", "condition": {"operation": "record_transaction", "float_account_present": False}, "effect": {"decision": "deny", "reason": "transaction_float_account_required", "required_action": "select_float_account"}},
	{"name": "transaction_service_supported", "description": "Transaction service must be supported.", "condition": {"operation": "record_transaction", "service_supported": False}, "effect": {"decision": "deny", "reason": "transaction_service_not_supported", "required_action": "select_supported_service"}},
	{"name": "transaction_service_allowed_by_program", "description": "Transaction service must be enabled by the outlet program.", "condition": {"operation": "record_transaction", "service_allowed_by_program": False}, "effect": {"decision": "deny", "reason": "transaction_service_not_enabled", "required_action": "enable_service_on_program"}},
	{"name": "transaction_channel_supported", "description": "Transaction channel must be supported.", "condition": {"operation": "record_transaction", "channel_supported": False}, "effect": {"decision": "deny", "reason": "transaction_channel_not_supported", "required_action": "select_supported_channel"}},
	{"name": "transaction_currency_supported", "description": "Transaction currency must be supported.", "condition": {"operation": "record_transaction", "currency_supported": False}, "effect": {"decision": "deny", "reason": "transaction_currency_not_supported", "required_action": "select_supported_currency"}},
	{"name": "transaction_currency_matches_float", "description": "Transaction currency must match the float account.", "condition": {"operation": "record_transaction", "currency_matches_float": False}, "effect": {"decision": "deny", "reason": "transaction_float_currency_mismatch", "required_action": "use_float_account_currency"}},
	{"name": "transaction_amount_positive", "description": "Transaction amount must be positive.", "condition": {"operation": "record_transaction", "positive_amount": False}, "effect": {"decision": "deny", "reason": "positive_transaction_amount_required", "required_action": "set_positive_amount"}},
	{"name": "transaction_limit_valid", "description": "Transaction amount must be within limit.", "condition": {"operation": "record_transaction", "within_limit": False}, "effect": {"decision": "deny", "reason": "transaction_limit_exceeded", "required_action": "split_or_review_transaction"}},
	{"name": "transaction_float_sufficient", "description": "Cash-out transactions require sufficient float.", "condition": {"operation": "record_transaction", "float_sufficient": False}, "effect": {"decision": "deny", "reason": "insufficient_agent_float", "required_action": "rebalance_float"}},
	{"name": "transaction_reference_required", "description": "Transactions require customer reference.", "condition": {"operation": "record_transaction", "customer_reference_present": False}, "effect": {"decision": "deny", "reason": "transaction_customer_reference_required", "required_action": "attach_customer_reference"}},
	{"name": "transaction_risk_reference_required", "description": "Transactions require risk reference.", "condition": {"operation": "record_transaction", "risk_reference_present": False}, "effect": {"decision": "deny", "reason": "transaction_risk_reference_required", "required_action": "attach_risk_reference"}},
	{"name": "high_value_transaction_requires_approval", "description": "High-value transactions require review.", "condition": {"operation": "record_transaction", "high_value": True, "human_approval_recorded": False}, "effect": {"decision": "require_review", "reason": "transaction_approval_required", "required_action": "record_transaction_approval"}},
	{"name": "cash_movement_outlet_required", "description": "Cash movements require outlet.", "condition": {"operation": "record_cash_movement", "outlet_present": False}, "effect": {"decision": "deny", "reason": "cash_movement_outlet_required", "required_action": "select_outlet"}},
	{"name": "cash_movement_type_supported", "description": "Cash movement type must be supported.", "condition": {"operation": "record_cash_movement", "movement_type_supported": False}, "effect": {"decision": "deny", "reason": "cash_movement_type_not_supported", "required_action": "select_supported_type"}},
	{"name": "cash_movement_currency_supported", "description": "Cash movement currency must be supported.", "condition": {"operation": "record_cash_movement", "currency_supported": False}, "effect": {"decision": "deny", "reason": "cash_movement_currency_not_supported", "required_action": "select_supported_currency"}},
	{"name": "cash_movement_amount_positive", "description": "Cash movement amount must be positive.", "condition": {"operation": "record_cash_movement", "positive_amount": False}, "effect": {"decision": "deny", "reason": "cash_movement_amount_required", "required_action": "set_positive_amount"}},
	{"name": "cash_movement_custodian_required", "description": "Cash movements require custodian.", "condition": {"operation": "record_cash_movement", "custodian_present": False}, "effect": {"decision": "deny", "reason": "cash_movement_custodian_required", "required_action": "assign_custodian"}},
	{"name": "high_value_cash_movement_requires_approval", "description": "High-value cash movements require approval.", "condition": {"operation": "record_cash_movement", "high_value": True, "human_approval_recorded": False}, "effect": {"decision": "require_review", "reason": "cash_movement_approval_required", "required_action": "record_cash_movement_approval"}},
	{"name": "commission_outlet_required", "description": "Commission settlement requires outlet.", "condition": {"operation": "settle_commission", "outlet_present": False}, "effect": {"decision": "deny", "reason": "commission_outlet_required", "required_action": "select_outlet"}},
	{"name": "commission_currency_supported", "description": "Commission currency must be supported.", "condition": {"operation": "settle_commission", "currency_supported": False}, "effect": {"decision": "deny", "reason": "commission_currency_not_supported", "required_action": "select_supported_currency"}},
	{"name": "commission_amount_positive", "description": "Commission amount must be positive.", "condition": {"operation": "settle_commission", "positive_amount": False}, "effect": {"decision": "deny", "reason": "commission_amount_required", "required_action": "set_positive_amount"}},
	{"name": "commission_reconciliation_required", "description": "Commission requires reconciliation.", "condition": {"operation": "settle_commission", "reconciliation_present": False}, "effect": {"decision": "deny", "reason": "commission_reconciliation_required", "required_action": "attach_reconciliation"}},
	{"name": "commission_payment_reference_required", "description": "Commission requires payment reference.", "condition": {"operation": "settle_commission", "payment_reference_present": False}, "effect": {"decision": "deny", "reason": "commission_payment_reference_required", "required_action": "attach_payment_reference"}},
	{"name": "dispute_transaction_required", "description": "Disputes require transaction.", "condition": {"operation": "open_dispute", "transaction_present": False}, "effect": {"decision": "deny", "reason": "dispute_transaction_required", "required_action": "select_transaction"}},
	{"name": "dispute_reason_supported", "description": "Dispute reason must be supported.", "condition": {"operation": "open_dispute", "dispute_reason_supported": False}, "effect": {"decision": "deny", "reason": "dispute_reason_not_supported", "required_action": "select_supported_reason"}},
	{"name": "dispute_evidence_required", "description": "Disputes require evidence.", "condition": {"operation": "open_dispute", "evidence_present": False}, "effect": {"decision": "deny", "reason": "dispute_evidence_required", "required_action": "attach_dispute_evidence"}},
	{"name": "dispute_reviewer_required", "description": "Disputes require reviewer.", "condition": {"operation": "open_dispute", "reviewer_present": False}, "effect": {"decision": "deny", "reason": "dispute_reviewer_required", "required_action": "assign_reviewer"}},
	{"name": "supervision_outlet_required", "description": "Supervision requires outlet.", "condition": {"operation": "record_supervision_visit", "outlet_present": False}, "effect": {"decision": "deny", "reason": "supervision_outlet_required", "required_action": "select_outlet"}},
	{"name": "supervision_supervisor_required", "description": "Supervision requires supervisor.", "condition": {"operation": "record_supervision_visit", "supervisor_present": False}, "effect": {"decision": "deny", "reason": "supervision_supervisor_required", "required_action": "assign_supervisor"}},
	{"name": "supervision_outcome_supported", "description": "Supervision outcome must be supported.", "condition": {"operation": "record_supervision_visit", "outcome_supported": False}, "effect": {"decision": "deny", "reason": "supervision_outcome_not_supported", "required_action": "select_supported_outcome"}},
	{"name": "supervision_evidence_required", "description": "Supervision requires evidence.", "condition": {"operation": "record_supervision_visit", "evidence_present": False}, "effect": {"decision": "deny", "reason": "supervision_evidence_required", "required_action": "attach_supervision_evidence"}},
	{"name": "supervision_findings_require_remediation", "description": "Findings require remediation plan.", "condition": {"operation": "record_supervision_visit", "findings_present": True, "remediation_plan_present": False}, "effect": {"decision": "require_review", "reason": "remediation_plan_required", "required_action": "attach_remediation_plan"}},
	{"name": "agency_batch_requires_bytewax", "description": "Agency batches require Bytewax.", "condition": {"operation": "agency_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_agency_batch_to_bytewax"}},
	{"name": "agency_ai_agent_runtime_supported", "description": "Agency AI agents must use a supported runtime.", "condition": {"operation": "register_agency_ai_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "agency_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "agency_ai_agent_role_supported", "description": "Agency AI agents must use a supported role.", "condition": {"operation": "register_agency_ai_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "agency_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "privileged_agency_agent_action_requires_human_approval", "description": "Privileged agency-agent actions require human approval.", "condition": {"operation": "agency_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},

	# Cross-tenant and privilege escalation guards
	{"name": "cross_tenant_agency_access_denied", "description": "Agency banking resources cannot be accessed across tenant boundaries.", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_access_denied", "required_action": "use_tenant_scoped_credentials"}},
	{"name": "privilege_escalation_denied", "description": "Agency banking privilege escalation without approval is denied.", "condition": {"privilege_escalation_attempt": True, "approval_recorded": False}, "effect": {"decision": "deny", "reason": "privilege_escalation_denied", "required_action": "obtain_escalation_approval"}},

	# Africa-specific agency banking rules
	{"name": "ke_cbk_agent_licence_required", "description": "Kenya CBK requires agent banking licence for all agency operations.", "condition": {"operation": "register_agent", "country": "KE", "cbk_agent_licence_present": False}, "effect": {"decision": "deny", "reason": "ke_cbk_agent_licence_required", "required_action": "attach_cbk_agent_licence"}},
	{"name": "mpesa_agent_float_minimum", "description": "M-Pesa agents must maintain minimum float balance.", "condition": {"operation": "process_mobile_money", "provider": "mpesa", "float_below_minimum": True}, "effect": {"decision": "deny", "reason": "mpesa_agent_float_minimum_not_met", "required_action": "top_up_agent_float"}},
	{"name": "mpesa_agent_float_aml_screening", "description": "M-Pesa agent float top-ups above threshold require AML screening.", "condition": {"operation": "float_top_up", "provider": "mpesa", "large_float": True, "aml_screened": False}, "effect": {"decision": "require_review", "reason": "mpesa_agent_float_aml_screening_required", "required_action": "screen_agent_float_top_up"}},
	{"name": "mobile_money_agency_kyc_required", "description": "Mobile money agency customers require tiered KYC per CBK guidelines.", "condition": {"operation": "open_agent_account", "kyc_tier_assigned": False}, "effect": {"decision": "deny", "reason": "mobile_money_kyc_tier_required", "required_action": "assign_kyc_tier"}},
	{"name": "agent_cash_limit_enforced", "description": "Agent daily cash transaction limit is enforced per CBK guidelines.", "condition": {"operation": "process_cash", "daily_cash_limit_exceeded": True}, "effect": {"decision": "deny", "reason": "agent_cash_limit_exceeded", "required_action": "defer_to_next_day_or_branch"}},
	{"name": "ng_cbn_agent_banking_guidelines", "description": "Nigeria CBN agent banking guidelines require CBN-approved principal institution.", "condition": {"operation": "register_agent", "country": "NG", "cbn_principal_approved": False}, "effect": {"decision": "deny", "reason": "ng_cbn_principal_institution_required", "required_action": "attach_cbn_approved_principal"}},
	{"name": "agent_biometric_verification_required", "description": "High-value agency transactions require biometric verification.", "condition": {"operation": "process_transaction", "high_value": True, "biometric_verified": False}, "effect": {"decision": "require_review", "reason": "agent_biometric_verification_required", "required_action": "complete_biometric_verification"}},
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
	return {"capability": CAPABILITY_ID, "display_name": CAPABILITY_NAME, "name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "configuration": configuration, "configuration_schema": _configuration_schema(), "provides": PROVIDES, "requires": REQUIRES, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "api_prefix": "/fintech-agency/api/v1", "routes": deepcopy(UI_ROUTES), "requires_theme": True, "view_module": "views.py", "template_roots": ["templates/", "static/"]}, "theme": deepcopy(THEME), "streaming": deepcopy(STREAMING)}


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
