"""Executable capability contract for APG Point of Sale."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "retail_pos"
CAPABILITY_NAME = "Point of Sale"
CAPABILITY_VERSION = "1.0.0"
POS_EVENT_STREAM = "apg.retail.pos.lifecycle"

SUPPORTED_TERMINAL_TYPES = ["fixed_counter", "mobile_pos", "self_checkout", "kiosk", "drive_through", "pop_up", "tablet_pos"]
SUPPORTED_PAYMENT_METHODS = ["cash", "card_chip", "card_tap", "card_swipe", "mobile_money", "qr_code", "loyalty_points", "gift_card", "voucher", "mixed"]
SUPPORTED_TRANSACTION_TYPES = ["sale", "refund", "void", "exchange", "layaway", "deposit", "partial_payment", "no_sale"]
SUPPORTED_TENDER_STATUSES = ["pending", "authorised", "captured", "declined", "voided", "refunded"]
SUPPORTED_SESSION_STATUSES = ["open", "suspended", "reconciling", "closed", "discrepancy_review"]
SUPPORTED_CASH_EVENTS = ["open_float", "petty_cash_out", "safe_drop", "pickup", "close_float", "reconcile"]
SUPPORTED_RECEIPT_TYPES = ["printed", "email", "sms", "digital_wallet", "no_receipt"]
SUPPORTED_DISCOUNT_TYPES = ["percentage", "fixed_amount", "bogo", "loyalty_discount", "employee_discount", "manager_override", "promotional"]
SUPPORTED_OFFLINE_MODES = ["queued_sync", "store_and_forward", "floor_limit", "offline_loyalty"]
SUPPORTED_PRINTER_STATUSES = ["online", "offline", "paper_low", "paper_out", "error"]
SUPPORTED_AUDIT_EVENTS = ["session_open", "session_close", "sale_posted", "refund_posted", "void_posted", "cash_event", "override_applied", "reconciliation"]
SUPPORTED_AGENT_ROLES = ["transaction_agent", "reconciliation_agent", "fraud_screen_agent", "receipt_agent"]
SUPPORTED_VOID_REASONS = ["operator_error", "customer_changed_mind", "price_check", "duplicate", "system_error", "manager_override"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"terminals": {"supported_types": SUPPORTED_TERMINAL_TYPES, "offline_capable": True, "heartbeat_interval_seconds": 30},
	"transactions": {
		"supported_types": SUPPORTED_TRANSACTION_TYPES,
		"max_items_per_transaction": 500,
		"max_value_per_transaction": 1000000,
		"void_window_minutes": 30,
		"manager_override_required_above": 5000,
	},
	"payments": {"supported_methods": SUPPORTED_PAYMENT_METHODS, "mixed_tender_enabled": True, "change_giving_enabled": True},
	"cash": {"supported_events": SUPPORTED_CASH_EVENTS, "starting_float_required": True, "safe_drop_threshold": 50000},
	"sessions": {"supported_statuses": SUPPORTED_SESSION_STATUSES, "auto_suspend_after_minutes": 15, "reconciliation_required_on_close": True},
	"receipts": {"supported_types": SUPPORTED_RECEIPT_TYPES, "digital_default": False, "logo_enabled": True},
	"discounts": {"supported_types": SUPPORTED_DISCOUNT_TYPES, "manager_override_required_above_pct": 20, "max_discount_pct": 50},
	"offline": {"supported_modes": SUPPORTED_OFFLINE_MODES, "floor_limit": 5000, "queue_max_transactions": 1000},
	"governance": {
		"require_tenant_context": True,
		"policy_attached_for_writes": True,
		"audit_events": True,
		"unsigned_transaction_denied": True,
		"void_without_reason_denied": True,
		"discount_exceeds_max_denied": True,
		"unreconciled_session_carry_over_denied": True,
		"cross_terminal_void_denied": True,
	},
	"observability": {"event_stream": POS_EVENT_STREAM, "stream_processor": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_sessions": True, "enable_transactions": True, "enable_cash": True, "enable_reconciliation": True, "enable_reports": True},
	"theme": {"default_theme": "retail_pos_dark", "allow_tenant_overrides": True},
}

PROVIDES = [
	"pos_transaction_processing",
	"pos_session_management",
	"pos_cash_management",
	"pos_till_reconciliation",
	"pos_receipt_management",
	"pos_discount_management",
	"pos_offline_resilience",
	"pos_payment_processing",
	"pos_void_management",
	"pos_audit_trail",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "mqeb", "moni", "comp"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/retail-pos/dashboard", "component": "PosDashboard", "permission": "retail_pos:view", "nav_group": "Overview"},
	{"name": "terminal", "path": "/retail-pos/terminal", "component": "PosTerminalScreen", "permission": "retail_pos:transact", "nav_group": "POS"},
	{"name": "sessions", "path": "/retail-pos/sessions", "component": "PosSessionList", "permission": "retail_pos:view", "nav_group": "Sessions"},
	{"name": "session_detail", "path": "/retail-pos/sessions/<id>", "component": "PosSessionDetail", "permission": "retail_pos:view", "nav_group": "Sessions"},
	{"name": "transactions", "path": "/retail-pos/transactions", "component": "PosTransactionList", "permission": "retail_pos:view", "nav_group": "Transactions"},
	{"name": "transaction_detail", "path": "/retail-pos/transactions/<id>", "component": "PosTransactionDetail", "permission": "retail_pos:view", "nav_group": "Transactions"},
	{"name": "cash_events", "path": "/retail-pos/cash", "component": "PosCashEventList", "permission": "retail_pos:view", "nav_group": "Cash"},
	{"name": "reconciliation", "path": "/retail-pos/reconcile", "component": "PosReconciliationConsole", "permission": "retail_pos:reconcile", "nav_group": "Cash"},
	{"name": "refunds", "path": "/retail-pos/refunds", "component": "PosRefundConsole", "permission": "retail_pos:refund", "nav_group": "Transactions"},
	{"name": "voids", "path": "/retail-pos/voids", "component": "PosVoidConsole", "permission": "retail_pos:void", "nav_group": "Transactions"},
	{"name": "receipts", "path": "/retail-pos/receipts", "component": "PosReceiptManager", "permission": "retail_pos:view", "nav_group": "Operations"},
	{"name": "terminals", "path": "/retail-pos/terminals", "component": "PosTerminalManager", "permission": "retail_pos:admin", "nav_group": "Administration"},
	{"name": "reports", "path": "/retail-pos/reports", "component": "PosReports", "permission": "retail_pos:view", "nav_group": "Analytics"},
	{"name": "settings", "path": "/retail-pos/settings", "component": "PosSettings", "permission": "retail_pos:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "retail_pos_dark",
	"tokens": {
		"color.primary": "#1E3A5F",
		"color.accent": "#0EA5E9",
		"color.success": "#16A34A",
		"color.warning": "#F59E0B",
		"color.danger": "#EF4444",
		"surface.canvas": "#0F172A",
		"surface.panel": "#1E293B",
		"text.primary": "#F1F5F9",
		"text.secondary": "#94A3B8",
		"border.radius": "6px",
		"density": "compact",
	},
	"components": {
		"terminal": {"icon": "monitor", "status_indicator": "terminal-status-chip"},
		"session": {"icon": "clock", "status_indicator": "session-status-chip"},
		"transaction": {"icon": "receipt", "status_indicator": "txn-type-chip"},
		"cash_event": {"icon": "banknote", "status_indicator": "cash-event-chip"},
		"receipt": {"icon": "file-text", "status_indicator": "receipt-type-chip"},
		"payment": {"icon": "credit-card", "status_indicator": "tender-status-chip"},
		"void": {"icon": "x-circle", "status_indicator": "void-reason-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": POS_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"session_opened",
		"session_closed",
		"transaction_posted",
		"refund_posted",
		"void_posted",
		"cash_event_recorded",
		"reconciliation_completed",
		"terminal_offline",
		"terminal_online",
		"discount_applied",
		"manager_override_recorded",
	],
	"guardrails": [
		"unsigned_transaction_denied",
		"void_without_reason_denied",
		"discount_exceeds_max_denied",
		"unreconciled_carry_over_denied",
		"cross_terminal_void_denied",
		"offline_floor_limit_enforced",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "policy_required", "required_action": "attach_pos_policy"}},
	{"name": "transaction_requires_open_session", "condition": {"operation": "post_transaction", "session_status": "closed"}, "effect": {"decision": "deny", "reason": "session_not_open", "required_action": "open_pos_session"}},
	{"name": "terminal_type_supported", "condition": {"operation": "register_terminal", "terminal_type_supported": False}, "effect": {"decision": "deny", "reason": "terminal_type_not_supported", "required_action": "select_supported_terminal_type"}},
	{"name": "transaction_type_supported", "condition": {"operation": "post_transaction", "transaction_type_supported": False}, "effect": {"decision": "deny", "reason": "transaction_type_not_supported", "required_action": "select_supported_transaction_type"}},
	{"name": "payment_method_supported", "condition": {"operation": "tender_payment", "payment_method_supported": False}, "effect": {"decision": "deny", "reason": "payment_method_not_supported", "required_action": "select_supported_payment_method"}},
	{"name": "unsigned_transaction_denied", "condition": {"operation": "post_transaction", "transaction_signed": False}, "effect": {"decision": "deny", "reason": "transaction_signature_required", "required_action": "sign_transaction"}},
	{"name": "void_requires_reason", "condition": {"operation": "void_transaction", "void_reason_present": False}, "effect": {"decision": "deny", "reason": "void_reason_required", "required_action": "select_void_reason"}},
	{"name": "void_reason_supported", "condition": {"operation": "void_transaction", "void_reason_supported": False}, "effect": {"decision": "deny", "reason": "void_reason_not_supported", "required_action": "select_supported_void_reason"}},
	{"name": "void_window_expired", "condition": {"operation": "void_transaction", "within_void_window": False}, "effect": {"decision": "deny", "reason": "void_window_expired", "required_action": "process_as_refund"}},
	{"name": "cross_terminal_void_denied", "condition": {"operation": "void_transaction", "same_terminal": False}, "effect": {"decision": "deny", "reason": "cross_terminal_void_not_permitted", "required_action": "void_on_originating_terminal"}},
	{"name": "discount_exceeds_max_denied", "condition": {"operation": "apply_discount", "discount_exceeds_max": True}, "effect": {"decision": "deny", "reason": "discount_exceeds_maximum", "required_action": "reduce_discount_amount"}},
	{"name": "large_discount_requires_manager", "condition": {"operation": "apply_discount", "requires_manager_override": True, "manager_override_present": False}, "effect": {"decision": "deny", "reason": "manager_override_required", "required_action": "obtain_manager_override"}},
	{"name": "discount_type_supported", "condition": {"operation": "apply_discount", "discount_type_supported": False}, "effect": {"decision": "deny", "reason": "discount_type_not_supported", "required_action": "select_supported_discount_type"}},
	{"name": "refund_requires_original_transaction", "condition": {"operation": "post_refund", "original_transaction_present": False}, "effect": {"decision": "deny", "reason": "original_transaction_required", "required_action": "attach_original_transaction_id"}},
	{"name": "cash_event_type_supported", "condition": {"operation": "record_cash_event", "cash_event_type_supported": False}, "effect": {"decision": "deny", "reason": "cash_event_type_not_supported", "required_action": "select_supported_cash_event_type"}},
	{"name": "session_reconciliation_required", "condition": {"operation": "close_session", "reconciliation_completed": False}, "effect": {"decision": "deny", "reason": "reconciliation_required_before_close", "required_action": "complete_reconciliation"}},
	{"name": "unreconciled_carry_over_denied", "condition": {"operation": "open_session", "previous_session_unreconciled": True}, "effect": {"decision": "deny", "reason": "previous_session_must_be_reconciled", "required_action": "reconcile_previous_session"}},
	{"name": "offline_floor_limit_enforced", "condition": {"operation": "post_transaction", "terminal_offline": True, "exceeds_floor_limit": True}, "effect": {"decision": "deny", "reason": "offline_floor_limit_exceeded", "required_action": "obtain_authorisation_code"}},
	{"name": "session_status_supported", "condition": {"operation": "update_session_status", "session_status_supported": False}, "effect": {"decision": "deny", "reason": "session_status_not_supported", "required_action": "select_supported_session_status"}},
	{"name": "receipt_type_supported", "condition": {"operation": "issue_receipt", "receipt_type_supported": False}, "effect": {"decision": "deny", "reason": "receipt_type_not_supported", "required_action": "select_supported_receipt_type"}},
	{"name": "transaction_exceeds_max_value", "condition": {"operation": "post_transaction", "exceeds_max_value": True}, "effect": {"decision": "deny", "reason": "transaction_exceeds_maximum_value", "required_action": "split_transaction"}},
	{"name": "cross_tenant_access_denied", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_access_denied", "required_action": "use_correct_tenant_context"}},
	{"name": "safe_drop_required", "condition": {"operation": "record_cash_event", "safe_drop_threshold_exceeded": True, "safe_drop_pending": True}, "effect": {"decision": "deny", "reason": "safe_drop_required_before_continuing", "required_action": "perform_safe_drop"}},
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
			"api_prefix": "/retail-pos/api/v1",
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
