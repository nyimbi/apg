"""Executable capability contract for APG Crowdfunding Platform."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "fintech_crowdfunding"
CAPABILITY_NAME = "Crowdfunding Platform"
CAPABILITY_VERSION = "1.1.0"
CROWDFUNDING_EVENT_STREAM = "apg.fintech.crowdfunding.lifecycle"

SUPPORTED_CAMPAIGN_TYPES = ["equity", "debt", "reward", "donation", "revenue_share"]
SUPPORTED_CURRENCIES = ["USD", "KES", "EUR", "GBP", "NGN", "GHS", "ZAR"]
SUPPORTED_COMMITMENT_STATUSES = ["pledged", "funded", "cancelled", "refunded"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_DISCLOSURE_TYPES = ["offering_memo", "risk_factors", "financials", "use_of_funds", "issuer_update"]
SUPPORTED_ALERT_SEVERITIES = ["low", "medium", "high", "critical"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = [
	"issuer_due_diligence_reviewer",
	"campaign_disclosure_reviewer",
	"investor_commitment_reviewer",
	"escrow_release_reviewer",
	"crowdfunding_compliance_reviewer",
	"investor_update_reviewer",
]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"issuers": {"kyc_required": True, "beneficial_owner_required": True, "risk_rating_required": True},
	"campaigns": {"supported_types": SUPPORTED_CAMPAIGN_TYPES, "supported_currencies": SUPPORTED_CURRENCIES, "issuer_required": True, "positive_target_required": True, "disclosure_required": True},
	"disclosures": {"supported_types": SUPPORTED_DISCLOSURE_TYPES, "campaign_required": True, "evidence_required": True, "review_required": True},
	"commitments": {"campaign_required": True, "investor_kyc_required": True, "positive_amount_required": True, "risk_acknowledgement_required": True, "supported_statuses": SUPPORTED_COMMITMENT_STATUSES},
	"escrow": {"wallet_reference_required": True, "funded_commitment_required": True, "positive_amount_required": True},
	"milestones": {"campaign_required": True, "evidence_required": True, "review_required": True},
	"payouts": {"campaign_required": True, "milestone_required": True, "positive_amount_required": True, "approval_required": True},
	"investor_updates": {"campaign_required": True, "disclosure_reference_required": True, "recipient_scope_required": True},
	"compliance": {"supported_severities": SUPPORTED_ALERT_SEVERITIES, "evidence_required": True, "review_required": True},
	"reviews": {"supported_statuses": SUPPORTED_REVIEW_STATUSES, "reviewer_required": True, "evidence_required": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True},
	"observability": {"event_stream": CROWDFUNDING_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "nlp": "nlpc", "keys": "keym", "payments": "fintech_payments", "wallets": "fintech_wallets", "kyc": "fintech_kyc", "aml": "fintech_aml", "fraud": "fintech_fraud", "portfolio": "fintech_portfolio", "wealth": "fintech_wealth", "analytics": "bia", "reporting": "fin_rpt", "event_stream": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_issuers": True, "enable_campaigns": True, "enable_disclosures": True, "enable_commitments": True, "enable_escrow": True, "enable_milestones": True, "enable_payouts": True, "enable_updates": True, "enable_compliance": True, "enable_reviews": True, "enable_agents": True},
	"theme": {"default_theme": "crowdfunding_platform_control", "allow_tenant_overrides": True},
}

PROVIDES = [
	"crowdfunding_issuer_workflow",
	"crowdfunding_campaign_workflow",
	"crowdfunding_disclosure_workflow",
	"crowdfunding_commitment_workflow",
	"crowdfunding_escrow_workflow",
	"crowdfunding_milestone_workflow",
	"crowdfunding_payout_workflow",
	"crowdfunding_investor_update_workflow",
	"crowdfunding_compliance_workflow",
	"crowdfunding_review_workflow",
	"crowdfunding_agent_workflow",
]
REQUIRES = [
	"auth",
	"audl",
	"ntfy",
	"nlpc",
	"keym",
	"fintech_payments",
	"fintech_wallets",
	"fintech_kyc",
	"fintech_aml",
	"fintech_fraud",
	"fintech_portfolio",
	"fintech_wealth",
	"bia_anl",
	"fin_rpt",
]

UI_ROUTES = [
	{"name": "dashboard", "path": "/fintech-crowdfunding/dashboard", "component": "CrowdfundingDashboard", "permission": "fintech_crowdfunding:view", "nav_group": "Overview"},
	{"name": "issuers", "path": "/fintech-crowdfunding/issuers", "component": "IssuerConsole", "permission": "fintech_crowdfunding:issuers", "nav_group": "Issuers"},
	{"name": "campaigns", "path": "/fintech-crowdfunding/campaigns", "component": "CampaignConsole", "permission": "fintech_crowdfunding:campaigns", "nav_group": "Campaigns"},
	{"name": "disclosures", "path": "/fintech-crowdfunding/disclosures", "component": "DisclosureWorkbench", "permission": "fintech_crowdfunding:disclosures", "nav_group": "Campaigns"},
	{"name": "commitments", "path": "/fintech-crowdfunding/commitments", "component": "InvestorCommitmentConsole", "permission": "fintech_crowdfunding:commitments", "nav_group": "Investors"},
	{"name": "escrow", "path": "/fintech-crowdfunding/escrow", "component": "EscrowFundingConsole", "permission": "fintech_crowdfunding:escrow", "nav_group": "Funds"},
	{"name": "milestones", "path": "/fintech-crowdfunding/milestones", "component": "MilestoneConsole", "permission": "fintech_crowdfunding:milestones", "nav_group": "Funds"},
	{"name": "payouts", "path": "/fintech-crowdfunding/payouts", "component": "PayoutWorkbench", "permission": "fintech_crowdfunding:payouts", "nav_group": "Funds"},
	{"name": "updates", "path": "/fintech-crowdfunding/updates", "component": "InvestorUpdateConsole", "permission": "fintech_crowdfunding:updates", "nav_group": "Investors"},
	{"name": "compliance", "path": "/fintech-crowdfunding/compliance", "component": "CrowdfundingComplianceConsole", "permission": "fintech_crowdfunding:compliance", "nav_group": "Governance"},
	{"name": "reviews", "path": "/fintech-crowdfunding/reviews", "component": "CrowdfundingReviewConsole", "permission": "fintech_crowdfunding:reviews", "nav_group": "Governance"},
	{"name": "agents", "path": "/fintech-crowdfunding/agents", "component": "CrowdfundingAgentWorkbench", "permission": "fintech_crowdfunding:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/fintech-crowdfunding/settings", "component": "CrowdfundingSettings", "permission": "fintech_crowdfunding:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "crowdfunding_platform_control",
	"tokens": {"color.primary": "#047857", "color.accent": "#2563EB", "color.success": "#15803D", "color.warning": "#A16207", "color.danger": "#B91C1C", "surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF", "text.primary": "#111827", "text.secondary": "#4B5563", "border.radius": "8px", "density": "compact"},
	"components": {"issuers": {"icon": "building-2", "status_indicator": "issuer-chip"}, "campaigns": {"icon": "megaphone", "status_indicator": "campaign-chip"}, "disclosures": {"icon": "file-check-2", "status_indicator": "disclosure-chip"}, "commitments": {"icon": "hand-coins", "status_indicator": "commitment-chip"}, "escrow": {"icon": "landmark", "status_indicator": "escrow-chip"}, "milestones": {"icon": "flag", "status_indicator": "milestone-chip"}, "payouts": {"icon": "circle-dollar-sign", "status_indicator": "payout-chip"}, "updates": {"icon": "send", "status_indicator": "update-chip"}, "compliance": {"icon": "scale", "status_indicator": "alert-chip"}, "reviews": {"icon": "clipboard-check", "status_indicator": "review-chip"}, "agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"}},
}

STREAMING = {
	"processor": "bytewax",
	"stream": CROWDFUNDING_EVENT_STREAM,
	"key": "tenant_id",
	"events": ["issuer_onboarded", "campaign_published", "disclosure_recorded", "investor_commitment_recorded", "escrow_funding_recorded", "milestone_recorded", "payout_authorized", "investor_update_published", "crowdfunding_compliance_alert_recorded", "crowdfunding_review_recorded", "crowdfunding_agent_registered"],
	"guardrails": ["crowdfunding_batch_requires_bytewax", "privileged_crowdfunding_agent_action_requires_human_approval"],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "crowdfunding_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "policy_evidence_required", "required_action": "attach_policy_evidence"}},
	{"name": "issuer_kyc_required", "condition": {"operation": "onboard_issuer", "kyc_present": False}, "effect": {"decision": "deny", "reason": "issuer_kyc_required", "required_action": "attach_issuer_kyc"}},
	{"name": "issuer_owner_required", "condition": {"operation": "onboard_issuer", "beneficial_owner_present": False}, "effect": {"decision": "deny", "reason": "beneficial_owner_required", "required_action": "attach_beneficial_owner"}},
	{"name": "issuer_risk_rating_required", "condition": {"operation": "onboard_issuer", "risk_rating_present": False}, "effect": {"decision": "deny", "reason": "issuer_risk_rating_required", "required_action": "attach_risk_rating"}},
	{"name": "campaign_issuer_required", "condition": {"operation": "publish_campaign", "issuer_present": False}, "effect": {"decision": "deny", "reason": "campaign_issuer_required", "required_action": "select_issuer"}},
	{"name": "campaign_type_supported", "condition": {"operation": "publish_campaign", "campaign_type_supported": False}, "effect": {"decision": "deny", "reason": "campaign_type_not_supported", "required_action": "select_supported_campaign_type"}},
	{"name": "campaign_currency_supported", "condition": {"operation": "publish_campaign", "currency_supported": False}, "effect": {"decision": "deny", "reason": "campaign_currency_not_supported", "required_action": "select_supported_currency"}},
	{"name": "campaign_positive_target", "condition": {"operation": "publish_campaign", "positive_target": False}, "effect": {"decision": "deny", "reason": "positive_campaign_target_required", "required_action": "set_positive_target"}},
	{"name": "campaign_disclosure_required", "condition": {"operation": "publish_campaign", "disclosure_present": False}, "effect": {"decision": "deny", "reason": "campaign_disclosure_required", "required_action": "attach_disclosure"}},
	{"name": "disclosure_campaign_required", "condition": {"operation": "record_disclosure", "campaign_present": False}, "effect": {"decision": "deny", "reason": "disclosure_campaign_required", "required_action": "select_campaign"}},
	{"name": "disclosure_type_supported", "condition": {"operation": "record_disclosure", "disclosure_type_supported": False}, "effect": {"decision": "deny", "reason": "disclosure_type_not_supported", "required_action": "select_supported_disclosure_type"}},
	{"name": "disclosure_evidence_required", "condition": {"operation": "record_disclosure", "evidence_present": False}, "effect": {"decision": "deny", "reason": "disclosure_evidence_required", "required_action": "attach_disclosure_evidence"}},
	{"name": "commitment_campaign_required", "condition": {"operation": "record_commitment", "campaign_present": False}, "effect": {"decision": "deny", "reason": "commitment_campaign_required", "required_action": "select_campaign"}},
	{"name": "commitment_investor_kyc_required", "condition": {"operation": "record_commitment", "investor_kyc_present": False}, "effect": {"decision": "deny", "reason": "investor_kyc_required", "required_action": "attach_investor_kyc"}},
	{"name": "commitment_positive_amount", "condition": {"operation": "record_commitment", "positive_amount": False}, "effect": {"decision": "deny", "reason": "positive_commitment_amount_required", "required_action": "set_positive_commitment_amount"}},
	{"name": "commitment_risk_ack_required", "condition": {"operation": "record_commitment", "risk_ack_present": False}, "effect": {"decision": "deny", "reason": "risk_acknowledgement_required", "required_action": "record_risk_acknowledgement"}},
	{"name": "escrow_commitment_required", "condition": {"operation": "record_escrow_funding", "funded_commitment_present": False}, "effect": {"decision": "deny", "reason": "funded_commitment_required", "required_action": "select_funded_commitment"}},
	{"name": "escrow_wallet_required", "condition": {"operation": "record_escrow_funding", "wallet_reference_present": False}, "effect": {"decision": "deny", "reason": "escrow_wallet_reference_required", "required_action": "attach_wallet_reference"}},
	{"name": "escrow_positive_amount", "condition": {"operation": "record_escrow_funding", "positive_amount": False}, "effect": {"decision": "deny", "reason": "positive_escrow_amount_required", "required_action": "set_positive_amount"}},
	{"name": "milestone_campaign_required", "condition": {"operation": "record_milestone", "campaign_present": False}, "effect": {"decision": "deny", "reason": "milestone_campaign_required", "required_action": "select_campaign"}},
	{"name": "milestone_evidence_required", "condition": {"operation": "record_milestone", "evidence_present": False}, "effect": {"decision": "deny", "reason": "milestone_evidence_required", "required_action": "attach_milestone_evidence"}},
	{"name": "payout_campaign_required", "condition": {"operation": "authorize_payout", "campaign_present": False}, "effect": {"decision": "deny", "reason": "payout_campaign_required", "required_action": "select_campaign"}},
	{"name": "payout_milestone_required", "condition": {"operation": "authorize_payout", "milestone_present": False}, "effect": {"decision": "deny", "reason": "payout_milestone_required", "required_action": "select_milestone"}},
	{"name": "payout_positive_amount", "condition": {"operation": "authorize_payout", "positive_amount": False}, "effect": {"decision": "deny", "reason": "positive_payout_amount_required", "required_action": "set_positive_payout"}},
	{"name": "payout_approval_required", "condition": {"operation": "authorize_payout", "approval_present": False}, "effect": {"decision": "deny", "reason": "payout_approval_required", "required_action": "attach_payout_approval"}},
	{"name": "update_campaign_required", "condition": {"operation": "publish_investor_update", "campaign_present": False}, "effect": {"decision": "deny", "reason": "update_campaign_required", "required_action": "select_campaign"}},
	{"name": "update_disclosure_required", "condition": {"operation": "publish_investor_update", "disclosure_reference_present": False}, "effect": {"decision": "deny", "reason": "update_disclosure_reference_required", "required_action": "attach_disclosure_reference"}},
	{"name": "compliance_severity_supported", "condition": {"operation": "record_compliance_alert", "severity_supported": False}, "effect": {"decision": "deny", "reason": "compliance_severity_not_supported", "required_action": "select_supported_severity"}},
	{"name": "compliance_evidence_required", "condition": {"operation": "record_compliance_alert", "evidence_present": False}, "effect": {"decision": "deny", "reason": "compliance_evidence_required", "required_action": "attach_compliance_evidence"}},
	{"name": "review_status_supported", "condition": {"operation": "record_review", "status_supported": False}, "effect": {"decision": "deny", "reason": "review_status_not_supported", "required_action": "select_supported_status"}},
	{"name": "review_evidence_required", "condition": {"operation": "record_review", "evidence_present": False}, "effect": {"decision": "deny", "reason": "review_evidence_required", "required_action": "attach_review_evidence"}},
	{"name": "crowdfunding_batch_requires_bytewax", "condition": {"operation": "crowdfunding_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_crowdfunding_batch_to_bytewax"}},
	{"name": "crowdfunding_agent_runtime_supported", "condition": {"operation": "register_crowdfunding_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "crowdfunding_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "crowdfunding_agent_role_supported", "condition": {"operation": "register_crowdfunding_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "crowdfunding_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "privileged_crowdfunding_agent_action_requires_human_approval", "condition": {"operation": "crowdfunding_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},

	# Cross-tenant and privilege escalation guards
	{"name": "cross_tenant_crowdfunding_access_denied", "description": "Crowdfunding resources cannot be accessed across tenant boundaries.", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_access_denied", "required_action": "use_tenant_scoped_credentials"}},
	{"name": "privilege_escalation_denied", "description": "Crowdfunding privilege escalation without approval is denied.", "condition": {"privilege_escalation_attempt": True, "approval_recorded": False}, "effect": {"decision": "deny", "reason": "privilege_escalation_denied", "required_action": "obtain_escalation_approval"}},

	# Africa-specific crowdfunding rules
	{"name": "ke_cma_crowdfunding_licence_required", "description": "Kenya CMA investment-based crowdfunding requires CMA licence.", "condition": {"operation": "launch_campaign", "country": "KE", "campaign_type": "investment", "cma_licence_present": False}, "effect": {"decision": "deny", "reason": "ke_cma_crowdfunding_licence_required", "required_action": "obtain_cma_crowdfunding_licence"}},
	{"name": "mpesa_campaign_collection_shortcode_required", "description": "M-Pesa campaign fund collection requires a registered paybill/shortcode.", "condition": {"operation": "launch_campaign", "collection_method": "mpesa", "mpesa_shortcode_present": False}, "effect": {"decision": "deny", "reason": "mpesa_collection_shortcode_required", "required_action": "register_mpesa_collection_shortcode"}},
	{"name": "mobile_money_backer_kyc_required", "description": "Mobile money campaign contributions require backer KYC verification.", "condition": {"operation": "pledge_contribution", "payment_method": "mobile_money", "backer_kyc_present": False}, "effect": {"decision": "deny", "reason": "mobile_money_backer_kyc_required", "required_action": "verify_backer_kyc"}},
	{"name": "ke_campaign_disclosure_required", "description": "Kenya CMA requires full campaign disclosure for investment crowdfunding.", "condition": {"operation": "launch_campaign", "country": "KE", "campaign_type": "investment", "disclosure_present": False}, "effect": {"decision": "deny", "reason": "ke_campaign_disclosure_required", "required_action": "file_campaign_disclosure"}},
	{"name": "campaign_aml_screening_required", "description": "Campaign organiser and large contributors require AML screening.", "condition": {"operation": "launch_campaign", "aml_screened": False}, "effect": {"decision": "deny", "reason": "campaign_aml_screening_required", "required_action": "screen_campaign_organiser"}},
	{"name": "mobile_money_disbursement_limit_enforced", "description": "Mobile money campaign disbursements are subject to CBK daily limits.", "condition": {"operation": "disburse_funds", "method": "mobile_money", "daily_limit_exceeded": True}, "effect": {"decision": "deny", "reason": "mobile_money_disbursement_limit_exceeded", "required_action": "schedule_batch_disbursement"}},
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
		"ui": {"shell": "apg_python", "requires_theme": True, "api_prefix": "/fintech-crowdfunding/api/v1", "template_roots": ["templates/", "static/"], "view_module": "views.py", "routes": deepcopy(UI_ROUTES)},
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
