"""Executable capability contract for APG Embedded Finance."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "fintech_embedded"
CAPABILITY_NAME = "Embedded Finance"
CAPABILITY_VERSION = "1.1.0"
EMBEDDED_EVENT_STREAM = "apg.fintech.embedded.lifecycle"

SUPPORTED_PRODUCTS = ["accounts", "wallet", "payments", "cards", "loans", "bnpl", "remittance", "insurance", "marketplace_finance"]
SUPPORTED_CHANNELS = ["checkout", "marketplace", "mobile_app", "web_app", "pos", "agent", "api"]
SUPPORTED_ENVIRONMENTS = ["sandbox", "pilot", "production"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["partner_risk_reviewer", "placement_reviewer", "consent_reviewer", "settlement_reviewer", "embedded_compliance_reviewer"]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"programs": {"kyb_required": True, "contract_required": True, "risk_review_required": True},
	"applications": {"supported_environments": SUPPORTED_ENVIRONMENTS, "domain_required": True, "terms_required": True},
	"placements": {"supported_products": SUPPORTED_PRODUCTS, "supported_channels": SUPPORTED_CHANNELS, "scope_required": True, "risk_policy_required": True},
	"consents": {"customer_reference_required": True, "scope_required": True, "expiry_required": True},
	"accounts": {"kyc_required": True, "wallet_reference_required": True},
	"payments": {"positive_amount_required": True, "supported_currencies": ["USD", "KES", "EUR", "GBP", "NGN", "GHS", "ZAR"], "risk_reference_required": True},
	"cards": {"positive_limit_required": True, "risk_reference_required": True},
	"lending": {"affordability_required": True, "underwriting_required": True},
	"settlements": {"reconciliation_required": True, "positive_amount_required": True},
	"revenue_share": {"minimum_percent": 0, "maximum_percent": 100},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True},
	"observability": {"event_stream": EMBEDDED_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "nlp": "nlpc", "keys": "keym", "banking_apis": "fintech_apis", "payments": "fintech_payments", "wallets": "fintech_wallets", "cards": "fintech_cards", "lending": "fintech_lending", "bnpl": "fintech_bnpl", "kyc": "fintech_kyc", "aml": "fintech_aml", "fraud": "fintech_fraud", "mobile": "fintech_mobile", "event_stream": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_programs": True, "enable_applications": True, "enable_placements": True, "enable_consents": True, "enable_accounts": True, "enable_payments": True, "enable_cards": True, "enable_lending": True, "enable_settlements": True, "enable_revenue_share": True, "enable_agents": True},
	"theme": {"default_theme": "embedded_finance_control", "allow_tenant_overrides": True},
}

PROVIDES = ["partner_program_workflow", "host_application_workflow", "embedded_product_placement_workflow", "embedded_customer_consent_workflow", "embedded_account_workflow", "embedded_payment_workflow", "embedded_card_workflow", "embedded_lending_workflow", "embedded_settlement_workflow", "embedded_revenue_share_workflow", "embedded_finance_agent_workflow"]
REQUIRES = ["auth", "audl", "ntfy", "nlpc", "keym", "fintech_apis", "fintech_payments", "fintech_wallets", "fintech_cards", "fintech_lending", "fintech_bnpl", "fintech_kyc", "fintech_aml", "fintech_fraud", "fintech_mobile"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/fintech-embedded/dashboard", "component": "EmbeddedFinanceDashboard", "permission": "fintech_embedded:view", "nav_group": "Overview"},
	{"name": "programs", "path": "/fintech-embedded/programs", "component": "PartnerProgramConsole", "permission": "fintech_embedded:programs", "nav_group": "Partners"},
	{"name": "applications", "path": "/fintech-embedded/applications", "component": "HostApplicationConsole", "permission": "fintech_embedded:applications", "nav_group": "Partners"},
	{"name": "placements", "path": "/fintech-embedded/placements", "component": "ProductPlacementConsole", "permission": "fintech_embedded:placements", "nav_group": "Products"},
	{"name": "consents", "path": "/fintech-embedded/consents", "component": "EmbeddedConsentWorkbench", "permission": "fintech_embedded:consents", "nav_group": "Consent"},
	{"name": "accounts", "path": "/fintech-embedded/accounts", "component": "EmbeddedAccountConsole", "permission": "fintech_embedded:accounts", "nav_group": "Journeys"},
	{"name": "payments", "path": "/fintech-embedded/payments", "component": "EmbeddedPaymentConsole", "permission": "fintech_embedded:payments", "nav_group": "Journeys"},
	{"name": "cards", "path": "/fintech-embedded/cards", "component": "EmbeddedCardConsole", "permission": "fintech_embedded:cards", "nav_group": "Journeys"},
	{"name": "lending", "path": "/fintech-embedded/lending", "component": "EmbeddedLendingConsole", "permission": "fintech_embedded:lending", "nav_group": "Journeys"},
	{"name": "settlements", "path": "/fintech-embedded/settlements", "component": "EmbeddedSettlementConsole", "permission": "fintech_embedded:settlements", "nav_group": "Operations"},
	{"name": "revenue_share", "path": "/fintech-embedded/revenue-share", "component": "RevenueShareConsole", "permission": "fintech_embedded:revenue_share", "nav_group": "Operations"},
	{"name": "agents", "path": "/fintech-embedded/agents", "component": "EmbeddedFinanceAgentWorkbench", "permission": "fintech_embedded:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/fintech-embedded/settings", "component": "EmbeddedFinanceSettings", "permission": "fintech_embedded:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "embedded_finance_control",
	"tokens": {"color.primary": "#0F766E", "color.accent": "#1D4ED8", "color.success": "#15803D", "color.warning": "#A16207", "color.danger": "#B91C1C", "surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF", "text.primary": "#111827", "text.secondary": "#4B5563", "border.radius": "8px", "density": "compact"},
	"components": {"programs": {"icon": "handshake", "status_indicator": "program-chip"}, "applications": {"icon": "app-window", "status_indicator": "app-chip"}, "placements": {"icon": "layout-template", "status_indicator": "placement-chip"}, "consents": {"icon": "file-check", "status_indicator": "consent-chip"}, "accounts": {"icon": "wallet", "status_indicator": "account-chip"}, "payments": {"icon": "credit-card", "status_indicator": "payment-chip"}, "cards": {"icon": "badge-credit-card", "status_indicator": "card-chip"}, "lending": {"icon": "landmark", "status_indicator": "loan-chip"}, "settlements": {"icon": "receipt", "status_indicator": "settlement-chip"}, "revenue_share": {"icon": "percent", "status_indicator": "revenue-chip"}, "agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"}},
}

STREAMING = {
	"processor": "bytewax",
	"stream": EMBEDDED_EVENT_STREAM,
	"key": "tenant_id",
	"events": ["partner_program_registered", "host_application_registered", "product_placement_published", "customer_consent_captured", "embedded_account_opened", "embedded_payment_initiated", "embedded_card_offered", "embedded_lending_offer_created", "settlement_batch_closed", "revenue_share_recorded", "embedded_agent_registered"],
	"guardrails": ["embedded_batch_requires_bytewax", "privileged_embedded_agent_action_requires_human_approval"],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "embedded_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "policy_evidence_required", "required_action": "attach_policy_evidence"}},
	{"name": "program_kyb_required", "condition": {"operation": "register_partner_program", "kyb_present": False}, "effect": {"decision": "deny", "reason": "partner_kyb_required", "required_action": "attach_kyb_evidence"}},
	{"name": "program_contract_required", "condition": {"operation": "register_partner_program", "contract_present": False}, "effect": {"decision": "deny", "reason": "partner_contract_required", "required_action": "attach_contract"}},
	{"name": "program_risk_required", "condition": {"operation": "register_partner_program", "risk_present": False}, "effect": {"decision": "deny", "reason": "partner_risk_review_required", "required_action": "attach_risk_review"}},
	{"name": "application_program_required", "condition": {"operation": "register_host_application", "program_present": False}, "effect": {"decision": "deny", "reason": "host_program_required", "required_action": "select_partner_program"}},
	{"name": "application_environment_supported", "condition": {"operation": "register_host_application", "environment_supported": False}, "effect": {"decision": "deny", "reason": "host_environment_not_supported", "required_action": "select_supported_environment"}},
	{"name": "application_domain_required", "condition": {"operation": "register_host_application", "domain_present": False}, "effect": {"decision": "deny", "reason": "host_domain_required", "required_action": "attach_domain"}},
	{"name": "application_terms_required", "condition": {"operation": "register_host_application", "terms_present": False}, "effect": {"decision": "deny", "reason": "host_terms_required", "required_action": "attach_terms"}},
	{"name": "placement_application_required", "condition": {"operation": "publish_product_placement", "application_present": False}, "effect": {"decision": "deny", "reason": "placement_application_required", "required_action": "select_host_application"}},
	{"name": "placement_product_supported", "condition": {"operation": "publish_product_placement", "product_supported": False}, "effect": {"decision": "deny", "reason": "embedded_product_not_supported", "required_action": "select_supported_product"}},
	{"name": "placement_channel_supported", "condition": {"operation": "publish_product_placement", "channel_supported": False}, "effect": {"decision": "deny", "reason": "embedded_channel_not_supported", "required_action": "select_supported_channel"}},
	{"name": "placement_scopes_required", "condition": {"operation": "publish_product_placement", "scopes_present": False}, "effect": {"decision": "deny", "reason": "placement_scopes_required", "required_action": "attach_scopes"}},
	{"name": "placement_risk_policy_required", "condition": {"operation": "publish_product_placement", "risk_policy_present": False}, "effect": {"decision": "deny", "reason": "placement_risk_policy_required", "required_action": "attach_risk_policy"}},
	{"name": "consent_application_required", "condition": {"operation": "capture_customer_consent", "application_present": False}, "effect": {"decision": "deny", "reason": "consent_application_required", "required_action": "select_host_application"}},
	{"name": "consent_customer_required", "condition": {"operation": "capture_customer_consent", "customer_present": False}, "effect": {"decision": "deny", "reason": "consent_customer_required", "required_action": "attach_customer_reference"}},
	{"name": "consent_scopes_required", "condition": {"operation": "capture_customer_consent", "scopes_present": False}, "effect": {"decision": "deny", "reason": "consent_scopes_required", "required_action": "attach_scopes"}},
	{"name": "consent_expiry_required", "condition": {"operation": "capture_customer_consent", "expiry_present": False}, "effect": {"decision": "deny", "reason": "consent_expiry_required", "required_action": "attach_expiry"}},
	{"name": "account_application_required", "condition": {"operation": "open_embedded_account", "application_present": False}, "effect": {"decision": "deny", "reason": "account_application_required", "required_action": "select_host_application"}},
	{"name": "account_kyc_required", "condition": {"operation": "open_embedded_account", "kyc_present": False}, "effect": {"decision": "deny", "reason": "account_kyc_required", "required_action": "attach_kyc_reference"}},
	{"name": "account_wallet_required", "condition": {"operation": "open_embedded_account", "wallet_present": False}, "effect": {"decision": "deny", "reason": "account_wallet_required", "required_action": "attach_wallet_reference"}},
	{"name": "payment_application_required", "condition": {"operation": "initiate_embedded_payment", "application_present": False}, "effect": {"decision": "deny", "reason": "payment_application_required", "required_action": "select_host_application"}},
	{"name": "payment_placement_required", "condition": {"operation": "initiate_embedded_payment", "placement_present": False}, "effect": {"decision": "deny", "reason": "payment_placement_required", "required_action": "select_placement"}},
	{"name": "payment_placement_matches_application", "condition": {"operation": "initiate_embedded_payment", "placement_matches_application": False}, "effect": {"decision": "deny", "reason": "payment_placement_application_mismatch", "required_action": "select_application_placement"}},
	{"name": "payment_consent_required", "condition": {"operation": "initiate_embedded_payment", "consent_present": False}, "effect": {"decision": "deny", "reason": "payment_consent_required", "required_action": "capture_consent"}},
	{"name": "payment_consent_covers_scope", "condition": {"operation": "initiate_embedded_payment", "consent_covers_scope": False}, "effect": {"decision": "deny", "reason": "payment_scope_not_consented", "required_action": "capture_matching_consent"}},
	{"name": "payment_positive_amount", "condition": {"operation": "initiate_embedded_payment", "positive_amount": False}, "effect": {"decision": "deny", "reason": "positive_payment_amount_required", "required_action": "set_positive_amount"}},
	{"name": "payment_currency_supported", "condition": {"operation": "initiate_embedded_payment", "currency_supported": False}, "effect": {"decision": "deny", "reason": "payment_currency_not_supported", "required_action": "select_supported_currency"}},
	{"name": "payment_risk_required", "condition": {"operation": "initiate_embedded_payment", "risk_reference_present": False}, "effect": {"decision": "deny", "reason": "payment_risk_reference_required", "required_action": "attach_risk_reference"}},
	{"name": "card_application_required", "condition": {"operation": "offer_embedded_card", "application_present": False}, "effect": {"decision": "deny", "reason": "card_application_required", "required_action": "select_host_application"}},
	{"name": "card_positive_limit", "condition": {"operation": "offer_embedded_card", "positive_limit": False}, "effect": {"decision": "deny", "reason": "positive_card_limit_required", "required_action": "set_positive_limit"}},
	{"name": "card_risk_required", "condition": {"operation": "offer_embedded_card", "risk_reference_present": False}, "effect": {"decision": "deny", "reason": "card_risk_reference_required", "required_action": "attach_risk_reference"}},
	{"name": "lending_affordability_required", "condition": {"operation": "create_lending_offer", "affordability_present": False}, "effect": {"decision": "deny", "reason": "affordability_evidence_required", "required_action": "attach_affordability"}},
	{"name": "lending_underwriting_required", "condition": {"operation": "create_lending_offer", "underwriting_present": False}, "effect": {"decision": "deny", "reason": "underwriting_evidence_required", "required_action": "attach_underwriting"}},
	{"name": "settlement_reconciliation_required", "condition": {"operation": "close_settlement_batch", "reconciled": False}, "effect": {"decision": "deny", "reason": "settlement_reconciliation_required", "required_action": "attach_reconciliation"}},
	{"name": "settlement_positive_amount", "condition": {"operation": "close_settlement_batch", "positive_amount": False}, "effect": {"decision": "deny", "reason": "positive_settlement_amount_required", "required_action": "set_positive_amount"}},
	{"name": "revenue_share_program_required", "condition": {"operation": "record_revenue_share", "program_present": False}, "effect": {"decision": "deny", "reason": "revenue_share_program_required", "required_action": "select_partner_program"}},
	{"name": "revenue_share_percent_bounded", "condition": {"operation": "record_revenue_share", "percent_bounded": False}, "effect": {"decision": "deny", "reason": "revenue_share_percent_out_of_bounds", "required_action": "set_valid_percent"}},
	{"name": "embedded_batch_requires_bytewax", "condition": {"operation": "embedded_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_embedded_batch_to_bytewax"}},
	{"name": "embedded_agent_runtime_supported", "condition": {"operation": "register_embedded_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "embedded_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "embedded_agent_role_supported", "condition": {"operation": "register_embedded_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "embedded_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "privileged_embedded_agent_action_requires_human_approval", "condition": {"operation": "embedded_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},

	# Cross-tenant and privilege escalation guards
	{"name": "cross_tenant_embedded_access_denied", "description": "Embedded finance resources cannot be accessed across tenant boundaries.", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_access_denied", "required_action": "use_tenant_scoped_credentials"}},
	{"name": "privilege_escalation_denied", "description": "Embedded finance privilege escalation without approval is denied.", "condition": {"privilege_escalation_attempt": True, "approval_recorded": False}, "effect": {"decision": "deny", "reason": "privilege_escalation_denied", "required_action": "obtain_escalation_approval"}},

	# Africa-specific embedded finance rules
	{"name": "ke_cbk_fintech_partnership_required", "description": "Embedded finance in Kenya requires a CBK-licensed fintech partner.", "condition": {"operation": "embed_financial_service", "country": "KE", "cbk_licensed_partner_present": False}, "effect": {"decision": "deny", "reason": "ke_cbk_licensed_partner_required", "required_action": "partner_with_cbk_licensed_entity"}},
	{"name": "mpesa_embedded_shortcode_required", "description": "M-Pesa embedded payment products require registered shortcode.", "condition": {"operation": "embed_mpesa_payment", "mpesa_shortcode_present": False}, "effect": {"decision": "deny", "reason": "mpesa_shortcode_required", "required_action": "register_mpesa_shortcode"}},
	{"name": "mobile_money_embedded_kyc_required", "description": "Embedded mobile money products require customer KYC.", "condition": {"operation": "embed_mobile_money", "customer_kyc_present": False}, "effect": {"decision": "deny", "reason": "mobile_money_embedded_kyc_required", "required_action": "complete_customer_kyc"}},
	{"name": "embedded_aml_passthrough_required", "description": "Embedded finance transactions must pass through AML screening.", "condition": {"operation": "embedded_transaction", "aml_screened": False}, "effect": {"decision": "deny", "reason": "embedded_aml_screening_required", "required_action": "route_through_aml_screening"}},
	{"name": "ke_data_protection_embedded", "description": "Embedded finance data sharing requires Kenya Data Protection Act consent.", "condition": {"operation": "share_customer_data", "country": "KE", "dpa_consent_recorded": False}, "effect": {"decision": "deny", "reason": "ke_data_protection_consent_required", "required_action": "record_dpa_consent"}},
	{"name": "ng_cbn_embedded_finance_framework", "description": "Nigeria CBN embedded finance framework compliance required.", "condition": {"operation": "embed_financial_service", "country": "NG", "cbn_framework_compliant": False}, "effect": {"decision": "deny", "reason": "ng_cbn_embedded_finance_compliance_required", "required_action": "comply_with_cbn_embedded_finance_framework"}},
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
		"configuration_schema": {"type": "object", "required": list(configuration), "properties": {key: {"type": "object"} for key in configuration if key != "tenant_id"} | {"tenant_id": {"type": "string", "minLength": 1}}},
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {"shell": "apg_python", "requires_theme": True, "api_prefix": "/fintech-embedded/api/v1", "template_roots": ["templates/", "static/"], "view_module": "views.py", "routes": deepcopy(UI_ROUTES)},
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
	decision = "require_review" if any(action["decision"] == "require_review" for action in actions) else "deny"
	return {"decision": decision, "actions": actions, "context": dict(context)}


def _matches(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if key.endswith("_ne"):
			if context.get(key[:-3]) == expected:
				return False
			continue
		if context.get(key) != expected:
			return False
	return True
