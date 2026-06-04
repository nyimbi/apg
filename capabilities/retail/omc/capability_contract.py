"""Executable capability contract for APG Omnichannel Commerce."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "retail_omc"
CAPABILITY_NAME = "Omnichannel Commerce"
CAPABILITY_VERSION = "1.0.0"
OMC_EVENT_STREAM = "apg.retail.omc.lifecycle"

SUPPORTED_CHANNEL_TYPES = ["store", "ecommerce", "mobile_app", "marketplace", "social_commerce", "kiosk", "call_centre", "catalogue", "pop_up"]
SUPPORTED_FULFILMENT_MODES = ["ship_to_home", "click_and_collect", "curbside_pickup", "ship_from_store", "locker_pickup", "same_day_delivery", "subscription_delivery"]
SUPPORTED_ORDER_STATUSES = ["draft", "confirmed", "payment_pending", "paid", "picking", "packed", "shipped", "out_for_delivery", "delivered", "collection_ready", "collected", "cancelled", "refunded", "partially_refunded"]
SUPPORTED_INVENTORY_VISIBILITY_MODES = ["real_time", "near_real_time", "batch_synced", "manual"]
SUPPORTED_CART_STATES = ["active", "abandoned", "converted", "expired", "saved"]
SUPPORTED_PAYMENT_METHODS = ["card", "mobile_money", "bank_transfer", "cash", "gift_card", "loyalty_points", "buy_now_pay_later", "crypto", "voucher"]
SUPPORTED_RETURN_REASONS = ["wrong_item", "damaged", "not_as_described", "changed_mind", "defective", "late_delivery", "duplicate_order"]
SUPPORTED_JOURNEY_STAGES = ["discovery", "consideration", "intent", "purchase", "fulfilment", "post_purchase", "loyalty"]
SUPPORTED_SEARCH_MODALITIES = ["text", "image", "barcode", "voice", "recommendation"]
SUPPORTED_PRICING_RULES = ["base_price", "promotional", "tiered_volume", "bundle", "dynamic", "personalised", "channel_specific"]
SUPPORTED_NOTIFICATION_TRIGGERS = ["order_confirmed", "payment_received", "order_shipped", "collection_ready", "delivery_failed", "return_initiated", "refund_processed"]
SUPPORTED_AGENT_ROLES = ["inventory_sync_agent", "order_routing_agent", "journey_orchestrator", "search_agent", "recommendation_agent"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"channels": {"supported_types": SUPPORTED_CHANNEL_TYPES, "default_channel": "store", "cross_channel_inventory_visible": True},
	"fulfilment": {"supported_modes": SUPPORTED_FULFILMENT_MODES, "sla_required": True, "carrier_integration_required": True},
	"orders": {"supported_statuses": SUPPORTED_ORDER_STATUSES, "approval_required_above_value": 50000, "fraud_check_required": True},
	"inventory": {"visibility_modes": SUPPORTED_INVENTORY_VISIBILITY_MODES, "safety_stock_enabled": True, "reservation_ttl_seconds": 900},
	"cart": {"supported_states": SUPPORTED_CART_STATES, "abandonment_timeout_minutes": 60, "max_items": 200},
	"payments": {"supported_methods": SUPPORTED_PAYMENT_METHODS, "pci_compliant_required": True, "tokenisation_required": True},
	"returns": {"supported_reasons": SUPPORTED_RETURN_REASONS, "return_window_days": 30, "approval_required_above": 10000},
	"journey": {"stages": SUPPORTED_JOURNEY_STAGES, "session_tracking_enabled": True, "attribution_model": "last_touch"},
	"search": {"supported_modalities": SUPPORTED_SEARCH_MODALITIES, "nlp_enabled": True, "personalisation_enabled": True},
	"pricing": {"supported_rules": SUPPORTED_PRICING_RULES, "real_time_pricing_enabled": True},
	"notifications": {"triggers": SUPPORTED_NOTIFICATION_TRIGGERS, "opt_in_required": True},
	"governance": {
		"require_tenant_context": True,
		"policy_attached_for_writes": True,
		"audit_events": True,
		"cross_tenant_access_denied": True,
		"oversell_denied": True,
		"payment_without_fraud_check_denied": True,
		"channel_price_arbitrage_denied": True,
	},
	"observability": {"event_stream": OMC_EVENT_STREAM, "stream_processor": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_channels": True, "enable_orders": True, "enable_inventory": True, "enable_journey": True, "enable_returns": True},
	"theme": {"default_theme": "retail_omc_unified", "allow_tenant_overrides": True},
}

PROVIDES = [
	"omnichannel_order_management",
	"inventory_visibility",
	"click_and_collect",
	"customer_journey_orchestration",
	"unified_cart",
	"cross_channel_fulfilment",
	"omnichannel_search",
	"return_management",
	"channel_pricing_engine",
	"session_attribution",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "mqeb", "moni", "nlpc", "schd"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/retail-omc/dashboard", "component": "OmcDashboard", "permission": "retail_omc:view", "nav_group": "Overview"},
	{"name": "orders", "path": "/retail-omc/orders", "component": "OmcOrderList", "permission": "retail_omc:view", "nav_group": "Orders"},
	{"name": "order_detail", "path": "/retail-omc/orders/<id>", "component": "OmcOrderDetail", "permission": "retail_omc:view", "nav_group": "Orders"},
	{"name": "order_create", "path": "/retail-omc/orders/create", "component": "OmcOrderCreate", "permission": "retail_omc:write", "nav_group": "Orders"},
	{"name": "inventory", "path": "/retail-omc/inventory", "component": "OmcInventoryView", "permission": "retail_omc:view", "nav_group": "Inventory"},
	{"name": "channels", "path": "/retail-omc/channels", "component": "OmcChannelManager", "permission": "retail_omc:admin", "nav_group": "Channels"},
	{"name": "fulfilment", "path": "/retail-omc/fulfilment", "component": "OmcFulfilmentConsole", "permission": "retail_omc:write", "nav_group": "Fulfilment"},
	{"name": "cart", "path": "/retail-omc/carts", "component": "OmcCartManager", "permission": "retail_omc:view", "nav_group": "Commerce"},
	{"name": "returns", "path": "/retail-omc/returns", "component": "OmcReturnList", "permission": "retail_omc:view", "nav_group": "Post-Sale"},
	{"name": "journey", "path": "/retail-omc/journey", "component": "OmcJourneyMap", "permission": "retail_omc:view", "nav_group": "Analytics"},
	{"name": "search_config", "path": "/retail-omc/search", "component": "OmcSearchConfig", "permission": "retail_omc:admin", "nav_group": "Commerce"},
	{"name": "pricing", "path": "/retail-omc/pricing", "component": "OmcPricingRules", "permission": "retail_omc:admin", "nav_group": "Commerce"},
	{"name": "reports", "path": "/retail-omc/reports", "component": "OmcReports", "permission": "retail_omc:view", "nav_group": "Analytics"},
	{"name": "settings", "path": "/retail-omc/settings", "component": "OmcSettings", "permission": "retail_omc:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "retail_omc_unified",
	"tokens": {
		"color.primary": "#1D4ED8",
		"color.accent": "#7C3AED",
		"color.success": "#166534",
		"color.warning": "#B45309",
		"color.danger": "#DC2626",
		"surface.canvas": "#F0F4FF",
		"surface.panel": "#FFFFFF",
		"text.primary": "#111827",
		"text.secondary": "#4B5563",
		"border.radius": "8px",
		"density": "comfortable",
	},
	"components": {
		"order": {"icon": "shopping-bag", "status_indicator": "order-status-chip"},
		"channel": {"icon": "layers", "status_indicator": "channel-type-chip"},
		"inventory": {"icon": "package", "status_indicator": "stock-level-chip"},
		"cart": {"icon": "shopping-cart", "status_indicator": "cart-state-chip"},
		"fulfilment": {"icon": "truck", "status_indicator": "fulfilment-mode-chip"},
		"return": {"icon": "rotate-ccw", "status_indicator": "return-reason-chip"},
		"journey": {"icon": "map", "status_indicator": "journey-stage-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": OMC_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"order_created",
		"order_paid",
		"order_shipped",
		"order_delivered",
		"order_collected",
		"order_cancelled",
		"inventory_reserved",
		"inventory_released",
		"cart_abandoned",
		"cart_converted",
		"return_initiated",
		"refund_processed",
		"journey_stage_advanced",
	],
	"guardrails": [
		"oversell_denied",
		"payment_without_fraud_check_denied",
		"cross_tenant_order_denied",
		"channel_price_arbitrage_denied",
		"batch_inventory_requires_bytewax",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "policy_required", "required_action": "attach_omc_policy"}},
	{"name": "channel_type_supported", "condition": {"operation": "create_channel", "channel_type_supported": False}, "effect": {"decision": "deny", "reason": "channel_type_not_supported", "required_action": "select_supported_channel_type"}},
	{"name": "order_requires_channel", "condition": {"operation": "create_order", "channel_present": False}, "effect": {"decision": "deny", "reason": "channel_required_for_order", "required_action": "assign_order_channel"}},
	{"name": "order_fulfilment_mode_supported", "condition": {"operation": "create_order", "fulfilment_mode_supported": False}, "effect": {"decision": "deny", "reason": "fulfilment_mode_not_supported", "required_action": "select_supported_fulfilment_mode"}},
	{"name": "oversell_denied", "condition": {"operation": "reserve_inventory", "available_stock": 0}, "effect": {"decision": "deny", "reason": "oversell_not_permitted", "required_action": "waitlist_or_backorder"}},
	{"name": "payment_method_supported", "condition": {"operation": "process_payment", "payment_method_supported": False}, "effect": {"decision": "deny", "reason": "payment_method_not_supported", "required_action": "select_supported_payment_method"}},
	{"name": "payment_requires_fraud_check", "condition": {"operation": "process_payment", "fraud_check_passed": False}, "effect": {"decision": "deny", "reason": "fraud_check_required", "required_action": "complete_fraud_screening"}},
	{"name": "high_value_order_requires_approval", "condition": {"operation": "confirm_order", "exceeds_approval_threshold": True, "approval_present": False}, "effect": {"decision": "deny", "reason": "high_value_order_approval_required", "required_action": "obtain_order_approval"}},
	{"name": "return_reason_supported", "condition": {"operation": "initiate_return", "return_reason_supported": False}, "effect": {"decision": "deny", "reason": "return_reason_not_supported", "required_action": "select_supported_return_reason"}},
	{"name": "return_window_expired", "condition": {"operation": "initiate_return", "within_return_window": False}, "effect": {"decision": "deny", "reason": "return_window_expired", "required_action": "obtain_manager_exception"}},
	{"name": "click_and_collect_requires_store", "condition": {"operation": "create_order", "fulfilment_mode": "click_and_collect", "store_id_present": False}, "effect": {"decision": "deny", "reason": "store_required_for_click_and_collect", "required_action": "select_collection_store"}},
	{"name": "inventory_visibility_mode_supported", "condition": {"operation": "set_inventory_visibility", "visibility_mode_supported": False}, "effect": {"decision": "deny", "reason": "visibility_mode_not_supported", "required_action": "select_supported_visibility_mode"}},
	{"name": "cart_max_items_exceeded", "condition": {"operation": "add_to_cart", "cart_max_items_exceeded": True}, "effect": {"decision": "deny", "reason": "cart_item_limit_reached", "required_action": "remove_items_before_adding"}},
	{"name": "cross_tenant_access_denied", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_access_denied", "required_action": "use_correct_tenant_context"}},
	{"name": "channel_price_arbitrage_denied", "condition": {"operation": "apply_pricing", "channel_price_arbitrage_detected": True}, "effect": {"decision": "deny", "reason": "channel_price_arbitrage_not_permitted", "required_action": "align_channel_prices"}},
	{"name": "inventory_reservation_ttl_enforced", "condition": {"operation": "reserve_inventory", "reservation_ttl_set": False}, "effect": {"decision": "deny", "reason": "inventory_reservation_ttl_required", "required_action": "set_reservation_ttl"}},
	{"name": "pci_compliance_required", "condition": {"operation": "process_payment", "pci_compliant": False}, "effect": {"decision": "deny", "reason": "pci_compliance_required", "required_action": "route_through_pci_compliant_processor"}},
	{"name": "search_modality_supported", "condition": {"operation": "search_catalogue", "search_modality_supported": False}, "effect": {"decision": "deny", "reason": "search_modality_not_supported", "required_action": "select_supported_search_modality"}},
	{"name": "journey_stage_supported", "condition": {"operation": "record_journey_event", "journey_stage_supported": False}, "effect": {"decision": "deny", "reason": "journey_stage_not_supported", "required_action": "select_supported_journey_stage"}},
	{"name": "batch_inventory_requires_bytewax", "condition": {"operation": "batch_inventory_sync", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "batch_inventory_requires_bytewax", "required_action": "route_batch_to_bytewax"}},
	{"name": "pricing_rule_supported", "condition": {"operation": "apply_pricing_rule", "pricing_rule_supported": False}, "effect": {"decision": "deny", "reason": "pricing_rule_not_supported", "required_action": "select_supported_pricing_rule"}},
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
			"api_prefix": "/retail-omc/api/v1",
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
