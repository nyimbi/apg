"""Executable capability contract for APG Promotions Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "retail_prm"
CAPABILITY_NAME = "Promotions Management"
CAPABILITY_VERSION = "1.0.0"
PRM_EVENT_STREAM = "apg.retail.prm.lifecycle"

SUPPORTED_PROMOTION_TYPES = ["percentage_off", "fixed_amount_off", "bogo", "multibuy", "bundle", "free_gift", "tiered_spend", "category_discount", "clearance", "flash_sale", "loyalty_multiplier", "threshold_discount"]
SUPPORTED_COUPON_TYPES = ["single_use", "multi_use", "customer_specific", "channel_specific", "partner_issued", "auto_apply", "referral_code"]
SUPPORTED_TRIGGER_TYPES = ["basket_value", "item_quantity", "category_spend", "loyalty_tier", "customer_segment", "channel", "time_window", "geo_location", "event_based"]
SUPPORTED_STACK_POLICIES = ["exclusive", "additive_capped", "best_of", "explicit_priority", "loyalty_plus_one"]
SUPPORTED_MARKDOWN_TYPES = ["end_of_season", "slow_mover", "expiry_based", "aged_stock", "clearance_cascade", "competitive_response"]
SUPPORTED_CHANNEL_RESTRICTIONS = ["all_channels", "store_only", "online_only", "app_only", "specific_stores", "specific_channels"]
SUPPORTED_APPROVAL_STATUSES = ["draft", "pending_review", "approved", "rejected", "active", "paused", "expired", "archived"]
SUPPORTED_EFFECTIVENESS_METRICS = ["redemption_rate", "incremental_revenue", "margin_impact", "basket_uplift", "new_customer_acquisition", "repeat_purchase_rate", "roi"]
SUPPORTED_AUDIENCE_TYPES = ["all_customers", "loyalty_members", "high_value", "at_risk", "new_customers", "specific_segment", "lookalike"]
SUPPORTED_BUDGET_STRATEGIES = ["total_cap", "per_day_cap", "per_customer_cap", "redemption_count_cap", "margin_floor"]
SUPPORTED_AGENT_ROLES = ["promotion_author", "pricing_optimizer", "markdown_agent", "effectiveness_analyst", "coupon_issuer"]
SUPPORTED_EXCLUSION_REASONS = ["already_discounted", "clearance_item", "tobacco", "alcohol", "gift_card", "prescription", "fuel"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"promotions": {
		"supported_types": SUPPORTED_PROMOTION_TYPES,
		"max_active_promotions": 500,
		"approval_required": True,
		"budget_cap_required": True,
		"end_date_required": True,
	},
	"coupons": {
		"supported_types": SUPPORTED_COUPON_TYPES,
		"max_uses_default": 1,
		"expiry_required": True,
		"unique_code_required": True,
	},
	"triggers": {"supported_types": SUPPORTED_TRIGGER_TYPES, "multi_trigger_enabled": True},
	"stacking": {"supported_policies": SUPPORTED_STACK_POLICIES, "default_policy": "best_of", "max_concurrent_promotions": 3},
	"markdown": {"supported_types": SUPPORTED_MARKDOWN_TYPES, "optimisation_enabled": True, "floor_margin_pct": 5},
	"channels": {"supported_restrictions": SUPPORTED_CHANNEL_RESTRICTIONS, "default_restriction": "all_channels"},
	"audience": {"supported_types": SUPPORTED_AUDIENCE_TYPES, "personalisation_enabled": True},
	"budget": {"supported_strategies": SUPPORTED_BUDGET_STRATEGIES, "real_time_tracking_enabled": True},
	"effectiveness": {"supported_metrics": SUPPORTED_EFFECTIVENESS_METRICS, "reporting_lag_hours": 24},
	"exclusions": {"supported_reasons": SUPPORTED_EXCLUSION_REASONS, "exclusion_list_required": True},
	"governance": {
		"require_tenant_context": True,
		"policy_attached_for_writes": True,
		"audit_events": True,
		"unapproved_promotion_activation_denied": True,
		"budget_exceeded_denied": True,
		"margin_floor_breach_denied": True,
		"cross_tenant_access_denied": True,
		"expired_coupon_redemption_denied": True,
	},
	"observability": {"event_stream": PRM_EVENT_STREAM, "stream_processor": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_promotions": True, "enable_coupons": True, "enable_markdown": True, "enable_effectiveness": True},
	"theme": {"default_theme": "retail_prm_vibrant", "allow_tenant_overrides": True},
}

PROVIDES = [
	"promotion_authoring",
	"promotion_activation",
	"pricing_rules_engine",
	"coupon_management",
	"coupon_redemption",
	"markdown_optimisation",
	"promotion_effectiveness_analytics",
	"audience_targeting",
	"promotion_budget_management",
	"promotion_stacking_engine",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "mqeb", "moni", "nlpc", "schd"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/retail-prm/dashboard", "component": "PrmDashboard", "permission": "retail_prm:view", "nav_group": "Overview"},
	{"name": "promotions", "path": "/retail-prm/promotions", "component": "PrmPromotionList", "permission": "retail_prm:view", "nav_group": "Promotions"},
	{"name": "promotion_detail", "path": "/retail-prm/promotions/<id>", "component": "PrmPromotionDetail", "permission": "retail_prm:view", "nav_group": "Promotions"},
	{"name": "promotion_create", "path": "/retail-prm/promotions/create", "component": "PrmPromotionCreate", "permission": "retail_prm:write", "nav_group": "Promotions"},
	{"name": "coupons", "path": "/retail-prm/coupons", "component": "PrmCouponList", "permission": "retail_prm:view", "nav_group": "Coupons"},
	{"name": "coupon_create", "path": "/retail-prm/coupons/create", "component": "PrmCouponCreate", "permission": "retail_prm:write", "nav_group": "Coupons"},
	{"name": "coupon_redemption", "path": "/retail-prm/coupons/redeem", "component": "PrmCouponRedeem", "permission": "retail_prm:write", "nav_group": "Coupons"},
	{"name": "pricing_rules", "path": "/retail-prm/pricing", "component": "PrmPricingRules", "permission": "retail_prm:admin", "nav_group": "Pricing"},
	{"name": "markdown", "path": "/retail-prm/markdown", "component": "PrmMarkdownConsole", "permission": "retail_prm:write", "nav_group": "Pricing"},
	{"name": "effectiveness", "path": "/retail-prm/effectiveness", "component": "PrmEffectivenessReport", "permission": "retail_prm:view", "nav_group": "Analytics"},
	{"name": "audience", "path": "/retail-prm/audience", "component": "PrmAudienceBuilder", "permission": "retail_prm:write", "nav_group": "Targeting"},
	{"name": "approvals", "path": "/retail-prm/approvals", "component": "PrmApprovalQueue", "permission": "retail_prm:approve", "nav_group": "Governance"},
	{"name": "budget_tracker", "path": "/retail-prm/budget", "component": "PrmBudgetTracker", "permission": "retail_prm:view", "nav_group": "Analytics"},
	{"name": "settings", "path": "/retail-prm/settings", "component": "PrmSettings", "permission": "retail_prm:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "retail_prm_vibrant",
	"tokens": {
		"color.primary": "#7C3AED",
		"color.accent": "#EC4899",
		"color.success": "#16A34A",
		"color.warning": "#D97706",
		"color.danger": "#DC2626",
		"surface.canvas": "#FAF5FF",
		"surface.panel": "#FFFFFF",
		"text.primary": "#1E1B4B",
		"text.secondary": "#6B7280",
		"border.radius": "10px",
		"density": "comfortable",
	},
	"components": {
		"promotion": {"icon": "tag", "status_indicator": "promo-status-chip"},
		"coupon": {"icon": "ticket", "status_indicator": "coupon-type-chip"},
		"pricing_rule": {"icon": "sliders", "status_indicator": "rule-active-chip"},
		"markdown": {"icon": "trending-down", "status_indicator": "markdown-type-chip"},
		"audience": {"icon": "users", "status_indicator": "audience-type-chip"},
		"budget": {"icon": "pie-chart", "status_indicator": "budget-health-chip"},
		"effectiveness": {"icon": "bar-chart-2", "status_indicator": "metric-trend-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": PRM_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"promotion_created",
		"promotion_approved",
		"promotion_activated",
		"promotion_paused",
		"promotion_expired",
		"coupon_issued",
		"coupon_redeemed",
		"coupon_voided",
		"markdown_applied",
		"budget_cap_reached",
		"margin_floor_breached",
		"effectiveness_calculated",
	],
	"guardrails": [
		"unapproved_promotion_activation_denied",
		"budget_cap_exceeded_denied",
		"margin_floor_breach_denied",
		"expired_coupon_denied",
		"excluded_item_promotion_denied",
		"batch_promotion_requires_bytewax",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "policy_required", "required_action": "attach_prm_policy"}},
	{"name": "promotion_type_supported", "condition": {"operation": "create_promotion", "promotion_type_supported": False}, "effect": {"decision": "deny", "reason": "promotion_type_not_supported", "required_action": "select_supported_promotion_type"}},
	{"name": "promotion_requires_end_date", "condition": {"operation": "create_promotion", "end_date_present": False}, "effect": {"decision": "deny", "reason": "promotion_end_date_required", "required_action": "set_promotion_end_date"}},
	{"name": "promotion_requires_budget_cap", "condition": {"operation": "create_promotion", "budget_cap_set": False}, "effect": {"decision": "deny", "reason": "promotion_budget_cap_required", "required_action": "set_promotion_budget_cap"}},
	{"name": "unapproved_activation_denied", "condition": {"operation": "activate_promotion", "approval_status": "pending_review"}, "effect": {"decision": "deny", "reason": "approval_required_before_activation", "required_action": "obtain_promotion_approval"}},
	{"name": "budget_exceeded_denied", "condition": {"operation": "activate_promotion", "budget_exceeded": True}, "effect": {"decision": "deny", "reason": "promotion_budget_exceeded", "required_action": "increase_budget_or_pause"}},
	{"name": "margin_floor_breach_denied", "condition": {"operation": "apply_promotion", "margin_floor_breached": True}, "effect": {"decision": "deny", "reason": "margin_floor_would_be_breached", "required_action": "reduce_discount_depth"}},
	{"name": "coupon_type_supported", "condition": {"operation": "create_coupon", "coupon_type_supported": False}, "effect": {"decision": "deny", "reason": "coupon_type_not_supported", "required_action": "select_supported_coupon_type"}},
	{"name": "coupon_requires_expiry", "condition": {"operation": "create_coupon", "expiry_date_present": False}, "effect": {"decision": "deny", "reason": "coupon_expiry_required", "required_action": "set_coupon_expiry"}},
	{"name": "expired_coupon_denied", "condition": {"operation": "redeem_coupon", "coupon_expired": True}, "effect": {"decision": "deny", "reason": "coupon_has_expired", "required_action": "issue_replacement_coupon"}},
	{"name": "coupon_already_redeemed_denied", "condition": {"operation": "redeem_coupon", "coupon_already_redeemed": True}, "effect": {"decision": "deny", "reason": "coupon_already_redeemed", "required_action": "verify_coupon_status"}},
	{"name": "max_concurrent_promotions_exceeded", "condition": {"operation": "apply_promotion", "max_concurrent_exceeded": True, "stack_policy": "exclusive"}, "effect": {"decision": "deny", "reason": "exclusive_promotion_already_applied", "required_action": "apply_best_of_policy"}},
	{"name": "trigger_type_supported", "condition": {"operation": "set_promotion_trigger", "trigger_type_supported": False}, "effect": {"decision": "deny", "reason": "trigger_type_not_supported", "required_action": "select_supported_trigger_type"}},
	{"name": "markdown_type_supported", "condition": {"operation": "apply_markdown", "markdown_type_supported": False}, "effect": {"decision": "deny", "reason": "markdown_type_not_supported", "required_action": "select_supported_markdown_type"}},
	{"name": "markdown_exceeds_floor_margin", "condition": {"operation": "apply_markdown", "floor_margin_breached": True}, "effect": {"decision": "deny", "reason": "markdown_would_breach_floor_margin", "required_action": "reduce_markdown_depth"}},
	{"name": "excluded_item_protection", "condition": {"operation": "apply_promotion", "item_excluded": True}, "effect": {"decision": "deny", "reason": "item_excluded_from_promotion", "required_action": "remove_excluded_item_from_scope"}},
	{"name": "audience_type_supported", "condition": {"operation": "set_audience", "audience_type_supported": False}, "effect": {"decision": "deny", "reason": "audience_type_not_supported", "required_action": "select_supported_audience_type"}},
	{"name": "stack_policy_supported", "condition": {"operation": "set_stack_policy", "stack_policy_supported": False}, "effect": {"decision": "deny", "reason": "stack_policy_not_supported", "required_action": "select_supported_stack_policy"}},
	{"name": "cross_tenant_access_denied", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_access_denied", "required_action": "use_correct_tenant_context"}},
	{"name": "channel_restriction_supported", "condition": {"operation": "set_channel_restriction", "restriction_supported": False}, "effect": {"decision": "deny", "reason": "channel_restriction_not_supported", "required_action": "select_supported_channel_restriction"}},
	{"name": "budget_strategy_supported", "condition": {"operation": "set_budget_strategy", "budget_strategy_supported": False}, "effect": {"decision": "deny", "reason": "budget_strategy_not_supported", "required_action": "select_supported_budget_strategy"}},
	{"name": "batch_promotion_requires_bytewax", "condition": {"operation": "batch_apply_promotion", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "batch_promotion_requires_bytewax", "required_action": "route_batch_to_bytewax"}},
	{"name": "approval_status_supported", "condition": {"operation": "update_approval_status", "approval_status_supported": False}, "effect": {"decision": "deny", "reason": "approval_status_not_supported", "required_action": "select_supported_approval_status"}},
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
			"api_prefix": "/retail-prm/api/v1",
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
