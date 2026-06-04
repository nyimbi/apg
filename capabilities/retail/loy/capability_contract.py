"""Executable capability contract for APG Loyalty & Rewards."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "retail_loy"
CAPABILITY_NAME = "Loyalty & Rewards"
CAPABILITY_VERSION = "1.0.0"
LOY_EVENT_STREAM = "apg.retail.loy.lifecycle"

SUPPORTED_PROGRAMME_TYPES = ["points", "cashback", "tiered_points", "coalition", "subscription", "hybrid", "stamp_card"]
SUPPORTED_TIER_NAMES = ["bronze", "silver", "gold", "platinum", "diamond", "founder", "vip"]
SUPPORTED_EARN_MECHANISMS = ["purchase_amount", "purchase_quantity", "category_bonus", "double_points", "partner_earn", "referral", "survey", "birthday_bonus", "anniversary_bonus"]
SUPPORTED_REDEEM_MECHANISMS = ["discount", "free_product", "voucher", "transfer_out", "partner_redeem", "charity_donation", "experience_reward"]
SUPPORTED_EXPIRY_POLICIES = ["rolling_activity", "calendar_year", "fixed_date", "never", "tier_based"]
SUPPORTED_PARTNER_ROLES = ["earn_partner", "redeem_partner", "coalition_hub", "bilateral"]
SUPPORTED_TRANSACTION_TYPES = ["earn", "redeem", "adjust", "expire", "transfer_in", "transfer_out", "bonus", "reversal"]
SUPPORTED_MEMBER_STATUSES = ["active", "inactive", "frozen", "pending_verification", "churned"]
SUPPORTED_CAMPAIGN_TYPES = ["bonus_points", "double_points", "category_boost", "referral", "win_back", "tier_upgrade", "partner_promo"]
SUPPORTED_REWARD_STATUSES = ["available", "reserved", "redeemed", "expired", "cancelled"]
SUPPORTED_AUDIT_EVENTS = ["enrolment", "earn", "redeem", "tier_change", "expiry", "adjustment", "partner_sync", "campaign_trigger"]
SUPPORTED_CLV_SEGMENTS = ["high_value", "medium_value", "low_value", "at_risk", "churned", "new", "reactivated"]
SUPPORTED_NOTIFICATION_CHANNELS = ["email", "sms", "push", "in_app", "whatsapp"]
SUPPORTED_AGENT_ROLES = ["enrolment_agent", "earn_agent", "redeem_agent", "tier_agent", "clv_analyst", "campaign_agent"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"programme": {
		"supported_types": SUPPORTED_PROGRAMME_TYPES,
		"points_currency": "PTS",
		"points_value_decimal": 4,
		"max_earn_per_transaction": 100000,
		"max_redeem_per_transaction": 50000,
	},
	"tiers": {
		"supported_names": SUPPORTED_TIER_NAMES,
		"earn_multipliers": {"bronze": 1.0, "silver": 1.5, "gold": 2.0, "platinum": 3.0, "diamond": 5.0},
		"qualification_window_days": 365,
		"downgrade_grace_days": 90,
	},
	"earn": {"supported_mechanisms": SUPPORTED_EARN_MECHANISMS, "receipt_required": True, "pos_integration_required": True},
	"redeem": {"supported_mechanisms": SUPPORTED_REDEEM_MECHANISMS, "minimum_balance_required": True, "approval_required_above": 10000},
	"expiry": {"supported_policies": SUPPORTED_EXPIRY_POLICIES, "default_policy": "rolling_activity", "default_rolling_days": 365},
	"partners": {"supported_roles": SUPPORTED_PARTNER_ROLES, "sla_required": True, "settlement_required": True},
	"campaigns": {"supported_types": SUPPORTED_CAMPAIGN_TYPES, "approval_required": True, "budget_cap_required": True},
	"clv": {"segments": SUPPORTED_CLV_SEGMENTS, "recalculation_frequency_days": 7},
	"notifications": {"supported_channels": SUPPORTED_NOTIFICATION_CHANNELS, "opt_in_required": True},
	"governance": {
		"require_tenant_context": True,
		"policy_attached_for_writes": True,
		"audit_events": True,
		"cross_tenant_access_denied": True,
		"negative_balance_denied": True,
		"redeem_without_earn_denied": True,
		"tier_skip_denied": True,
	},
	"observability": {"event_stream": LOY_EVENT_STREAM, "stream_processor": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_members": True, "enable_tiers": True, "enable_transactions": True, "enable_campaigns": True, "enable_partners": True, "enable_clv": True},
	"theme": {"default_theme": "retail_loy_gold", "allow_tenant_overrides": True},
}

PROVIDES = [
	"loyalty_member_enrolment",
	"loyalty_points_earn",
	"loyalty_points_redeem",
	"loyalty_tier_management",
	"loyalty_campaign_management",
	"loyalty_partner_coalition",
	"loyalty_clv_analytics",
	"loyalty_expiry_management",
	"loyalty_reward_catalogue",
	"loyalty_transaction_ledger",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "mqeb", "moni", "schd"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/retail-loy/dashboard", "component": "LoyDashboard", "permission": "retail_loy:view", "nav_group": "Overview"},
	{"name": "members", "path": "/retail-loy/members", "component": "LoyMemberList", "permission": "retail_loy:view", "nav_group": "Members"},
	{"name": "member_detail", "path": "/retail-loy/members/<id>", "component": "LoyMemberDetail", "permission": "retail_loy:view", "nav_group": "Members"},
	{"name": "enrolment", "path": "/retail-loy/members/enrol", "component": "LoyEnrolmentForm", "permission": "retail_loy:write", "nav_group": "Members"},
	{"name": "transactions", "path": "/retail-loy/transactions", "component": "LoyTransactionLedger", "permission": "retail_loy:view", "nav_group": "Transactions"},
	{"name": "earn", "path": "/retail-loy/earn", "component": "LoyEarnConsole", "permission": "retail_loy:write", "nav_group": "Transactions"},
	{"name": "redeem", "path": "/retail-loy/redeem", "component": "LoyRedeemConsole", "permission": "retail_loy:write", "nav_group": "Transactions"},
	{"name": "tiers", "path": "/retail-loy/tiers", "component": "LoyTierManager", "permission": "retail_loy:admin", "nav_group": "Programme"},
	{"name": "campaigns", "path": "/retail-loy/campaigns", "component": "LoyCampaignList", "permission": "retail_loy:view", "nav_group": "Campaigns"},
	{"name": "campaign_create", "path": "/retail-loy/campaigns/create", "component": "LoyCampaignCreate", "permission": "retail_loy:write", "nav_group": "Campaigns"},
	{"name": "partners", "path": "/retail-loy/partners", "component": "LoyPartnerList", "permission": "retail_loy:admin", "nav_group": "Partners"},
	{"name": "clv", "path": "/retail-loy/clv", "component": "LoyClvAnalytics", "permission": "retail_loy:view", "nav_group": "Analytics"},
	{"name": "rewards", "path": "/retail-loy/rewards", "component": "LoyRewardCatalogue", "permission": "retail_loy:view", "nav_group": "Programme"},
	{"name": "settings", "path": "/retail-loy/settings", "component": "LoySettings", "permission": "retail_loy:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "retail_loy_gold",
	"tokens": {
		"color.primary": "#B45309",
		"color.accent": "#D97706",
		"color.success": "#166534",
		"color.warning": "#92400E",
		"color.danger": "#991B1B",
		"surface.canvas": "#FFFBEB",
		"surface.panel": "#FFFFFF",
		"text.primary": "#1C1917",
		"text.secondary": "#57534E",
		"border.radius": "10px",
		"density": "comfortable",
	},
	"components": {
		"member": {"icon": "user-circle", "status_indicator": "member-status-chip"},
		"tier": {"icon": "trophy", "status_indicator": "tier-badge"},
		"transaction": {"icon": "coins", "status_indicator": "txn-type-chip"},
		"campaign": {"icon": "megaphone", "status_indicator": "campaign-status-chip"},
		"partner": {"icon": "handshake", "status_indicator": "partner-role-chip"},
		"reward": {"icon": "gift", "status_indicator": "reward-status-chip"},
		"clv_segment": {"icon": "chart-line", "status_indicator": "clv-segment-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": LOY_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"member_enrolled",
		"points_earned",
		"points_redeemed",
		"points_expired",
		"points_adjusted",
		"tier_upgraded",
		"tier_downgraded",
		"campaign_triggered",
		"partner_earn_recorded",
		"partner_redeem_recorded",
		"clv_segment_changed",
		"reward_redeemed",
	],
	"guardrails": [
		"batch_earn_requires_bytewax",
		"negative_balance_denied",
		"redeem_exceeds_balance_denied",
		"tier_skip_requires_approval",
		"cross_tenant_earn_denied",
		"expired_points_redeem_denied",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "policy_required", "required_action": "attach_loy_policy"}},
	{"name": "enrolment_requires_consent", "condition": {"operation": "enrol_member", "consent_recorded": False}, "effect": {"decision": "deny", "reason": "member_consent_required", "required_action": "record_member_consent"}},
	{"name": "enrolment_requires_identity", "condition": {"operation": "enrol_member", "identity_verified": False}, "effect": {"decision": "deny", "reason": "identity_verification_required", "required_action": "verify_member_identity"}},
	{"name": "programme_type_supported", "condition": {"operation": "create_programme", "programme_type_supported": False}, "effect": {"decision": "deny", "reason": "programme_type_not_supported", "required_action": "select_supported_programme_type"}},
	{"name": "earn_requires_receipt", "condition": {"operation": "earn_points", "receipt_present": False}, "effect": {"decision": "deny", "reason": "receipt_required_for_earn", "required_action": "attach_receipt_reference"}},
	{"name": "earn_requires_valid_amount", "condition": {"operation": "earn_points", "amount_valid": False}, "effect": {"decision": "deny", "reason": "earn_amount_invalid", "required_action": "provide_valid_earn_amount"}},
	{"name": "earn_exceeds_max_denied", "condition": {"operation": "earn_points", "exceeds_max_earn": True}, "effect": {"decision": "deny", "reason": "earn_exceeds_transaction_max", "required_action": "split_earn_transaction"}},
	{"name": "redeem_requires_sufficient_balance", "condition": {"operation": "redeem_points", "sufficient_balance": False}, "effect": {"decision": "deny", "reason": "insufficient_points_balance", "required_action": "reduce_redeem_amount"}},
	{"name": "redeem_requires_active_member", "condition": {"operation": "redeem_points", "member_status": "inactive"}, "effect": {"decision": "deny", "reason": "inactive_member_cannot_redeem", "required_action": "reactivate_member"}},
	{"name": "redeem_frozen_member_denied", "condition": {"operation": "redeem_points", "member_status": "frozen"}, "effect": {"decision": "deny", "reason": "frozen_member_redeem_denied", "required_action": "unfreeze_member_first"}},
	{"name": "redeem_expired_points_denied", "condition": {"operation": "redeem_points", "points_expired": True}, "effect": {"decision": "deny", "reason": "expired_points_not_redeemable", "required_action": "earn_fresh_points"}},
	{"name": "high_value_redeem_requires_approval", "condition": {"operation": "redeem_points", "exceeds_approval_threshold": True, "approval_present": False}, "effect": {"decision": "deny", "reason": "high_value_redeem_approval_required", "required_action": "obtain_redeem_approval"}},
	{"name": "tier_name_supported", "condition": {"operation": "create_tier", "tier_name_supported": False}, "effect": {"decision": "deny", "reason": "tier_name_not_supported", "required_action": "select_supported_tier_name"}},
	{"name": "tier_skip_denied", "condition": {"operation": "assign_tier", "tier_skip_detected": True, "override_approved": False}, "effect": {"decision": "deny", "reason": "tier_skip_requires_approval", "required_action": "obtain_tier_skip_approval"}},
	{"name": "tier_downgrade_requires_grace", "condition": {"operation": "downgrade_tier", "grace_period_active": True}, "effect": {"decision": "deny", "reason": "tier_downgrade_in_grace_period", "required_action": "wait_for_grace_period"}},
	{"name": "campaign_requires_approval", "condition": {"operation": "activate_campaign", "approval_present": False}, "effect": {"decision": "deny", "reason": "campaign_approval_required", "required_action": "obtain_campaign_approval"}},
	{"name": "campaign_requires_budget_cap", "condition": {"operation": "create_campaign", "budget_cap_set": False}, "effect": {"decision": "deny", "reason": "campaign_budget_cap_required", "required_action": "set_campaign_budget_cap"}},
	{"name": "campaign_type_supported", "condition": {"operation": "create_campaign", "campaign_type_supported": False}, "effect": {"decision": "deny", "reason": "campaign_type_not_supported", "required_action": "select_supported_campaign_type"}},
	{"name": "partner_requires_sla", "condition": {"operation": "register_partner", "sla_present": False}, "effect": {"decision": "deny", "reason": "partner_sla_required", "required_action": "attach_partner_sla"}},
	{"name": "partner_role_supported", "condition": {"operation": "register_partner", "partner_role_supported": False}, "effect": {"decision": "deny", "reason": "partner_role_not_supported", "required_action": "select_supported_partner_role"}},
	{"name": "cross_tenant_access_denied", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_access_denied", "required_action": "use_correct_tenant_context"}},
	{"name": "negative_balance_denied", "condition": {"operation": "adjust_points", "resulting_balance_negative": True}, "effect": {"decision": "deny", "reason": "negative_balance_not_allowed", "required_action": "reduce_adjustment_amount"}},
	{"name": "earn_mechanism_supported", "condition": {"operation": "earn_points", "earn_mechanism_supported": False}, "effect": {"decision": "deny", "reason": "earn_mechanism_not_supported", "required_action": "select_supported_earn_mechanism"}},
	{"name": "redeem_mechanism_supported", "condition": {"operation": "redeem_points", "redeem_mechanism_supported": False}, "effect": {"decision": "deny", "reason": "redeem_mechanism_not_supported", "required_action": "select_supported_redeem_mechanism"}},
	{"name": "expiry_policy_supported", "condition": {"operation": "set_expiry_policy", "expiry_policy_supported": False}, "effect": {"decision": "deny", "reason": "expiry_policy_not_supported", "required_action": "select_supported_expiry_policy"}},
	{"name": "batch_earn_requires_bytewax", "condition": {"operation": "batch_earn", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "batch_earn_requires_bytewax_stream", "required_action": "route_batch_to_bytewax"}},
	{"name": "clv_recalculation_requires_schedule", "condition": {"operation": "recalculate_clv", "schedule_present": False}, "effect": {"decision": "deny", "reason": "clv_schedule_required", "required_action": "attach_clv_schedule"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	return {
		"capability": CAPABILITY_ID,
		"display_name": CAPABILITY_NAME,
		"version": CAPABILITY_VERSION,
		"provides": list(PROVIDES),
		"requires": list(REQUIRES),
		"configuration": configuration,
		"configuration_schema": {
			"type": "object",
			"required": ["tenant_id", "ui", "theme"],
			"properties": {k: {"type": "object"} for k in configuration if k != "tenant_id"} | {"tenant_id": {"type": "string", "minLength": 1}},
		},
		"rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"api_prefix": "/retail-loy/api/v1",
			"requires_theme": True,
			"template_roots": ["templates/", "static/"],
			"routes": deepcopy(UI_ROUTES),
		},
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
