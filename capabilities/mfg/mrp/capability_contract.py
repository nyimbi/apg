"""Executable capability contract for APG Material Requirements Planning."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "mfg_mrp"
CAPABILITY_NAME = "Material Requirements Planning"
CAPABILITY_VERSION = "1.0.0"
CAPABILITY_DOMAIN = "mfg"
CAPABILITY_DESCRIPTION = "MRP-II: demand-driven production orders, purchase requisitions, BOM explosion, pegging, and net change planning."

MRP_EVENT_STREAM = "apg.mfg.mrp.lifecycle"

SUPPORTED_ORDER_TYPES = ["planned", "firm_planned", "released", "completed", "cancelled"]
SUPPORTED_REQUISITION_STATUSES = ["draft", "submitted", "approved", "ordered", "received", "cancelled"]
SUPPORTED_PLANNING_HORIZONS = ["day", "week", "month", "quarter"]
SUPPORTED_LOT_SIZING_RULES = ["lot_for_lot", "fixed_order_qty", "economic_order_qty", "min_max", "period_order_qty"]
SUPPORTED_PEGGING_MODES = ["single_level", "full_peg", "summarized"]
SUPPORTED_SEVERITIES = ["low", "medium", "high", "critical"]
SUPPORTED_MESSAGE_TYPES = ["expedite", "defer", "cancel", "new_order", "quantity_change"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"planning": {
		"supported_horizons": SUPPORTED_PLANNING_HORIZONS,
		"supported_lot_sizing_rules": SUPPORTED_LOT_SIZING_RULES,
		"default_horizon": "week",
		"default_lot_sizing": "lot_for_lot",
		"safety_stock_enabled": True,
		"reorder_point_enabled": True,
		"mrp_explosion_depth": 10,
	},
	"orders": {
		"supported_order_types": SUPPORTED_ORDER_TYPES,
		"auto_firm_threshold_days": 3,
		"require_bom": True,
		"require_routing": False,
	},
	"requisitions": {
		"supported_statuses": SUPPORTED_REQUISITION_STATUSES,
		"approval_required": True,
		"lead_time_buffer_pct": 10,
	},
	"pegging": {
		"supported_modes": SUPPORTED_PEGGING_MODES,
		"default_mode": "single_level",
	},
	"messages": {
		"supported_types": SUPPORTED_MESSAGE_TYPES,
		"auto_process_low_value": False,
	},
	"governance": {
		"require_tenant_context": True,
		"audit_events": True,
		"cross_tenant_denied": True,
	},
	"adapters": {
		"auth": "auth",
		"audit": "audl",
		"bom": "mfg_bom",
		"inventory": "inv",
		"purchasing": "prc",
		"event_stream": "bytewax",
	},
}

PROVIDES = [
	"mrp_planning_run",
	"production_order_workflow",
	"purchase_requisition_workflow",
	"demand_pegging",
	"exception_message_workflow",
	"net_change_planning",
]

REQUIRES = ["auth", "audl", "mfg_bom", "inv"]

PUBLISHES = [
	"apg.mfg.mrp.production_order_created",
	"apg.mfg.mrp.production_order_released",
	"apg.mfg.mrp.purchase_requisition_created",
	"apg.mfg.mrp.planning_run_completed",
	"apg.mfg.mrp.exception_message_raised",
]

SUBSCRIBES = [
	"apg.mfg.bom.bom_changed",
	"apg.inv.inventory_adjusted",
	"apg.crm.sales_order_created",
	"apg.mfg.ppl.master_schedule_updated",
]

UI_ROUTES = [
	{"name": "dashboard", "path": "/mfg-mrp/dashboard", "component": "MfgMrpDashboard", "permission": "mfg_mrp:view", "nav_group": "Overview"},
	{"name": "planning_runs", "path": "/mfg-mrp/planning-runs", "component": "MfgMrpPlanningRuns", "permission": "mfg_mrp:plan", "nav_group": "Planning"},
	{"name": "production_orders", "path": "/mfg-mrp/production-orders", "component": "MfgMrpProductionOrders", "permission": "mfg_mrp:manage", "nav_group": "Orders"},
	{"name": "purchase_requisitions", "path": "/mfg-mrp/purchase-requisitions", "component": "MfgMrpPurchaseRequisitions", "permission": "mfg_mrp:manage", "nav_group": "Orders"},
	{"name": "pegging", "path": "/mfg-mrp/pegging", "component": "MfgMrpPegging", "permission": "mfg_mrp:view", "nav_group": "Analysis"},
	{"name": "exception_messages", "path": "/mfg-mrp/exceptions", "component": "MfgMrpExceptions", "permission": "mfg_mrp:manage", "nav_group": "Exceptions"},
	{"name": "settings", "path": "/mfg-mrp/settings", "component": "MfgMrpSettings", "permission": "mfg_mrp:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "mfg_mrp_theme",
	"tokens": {
		"color.primary": "#1A3A5C",
		"color.accent": "#F59E0B",
		"color.success": "#166534",
		"color.warning": "#A16207",
		"color.danger": "#991B1B",
		"surface.canvas": "#F0F4F8",
		"surface.panel": "#FFFFFF",
		"text.primary": "#111827",
		"text.secondary": "#4B5563",
		"border.radius": "6px",
		"density": "compact",
	},
	"components": {
		"production_orders": {"icon": "clipboard-list", "status_indicator": "order-status-chip"},
		"purchase_requisitions": {"icon": "shopping-cart", "status_indicator": "requisition-status-chip"},
		"planning_runs": {"icon": "refresh-cw", "status_indicator": "run-status-chip"},
		"exception_messages": {"icon": "alert-triangle", "status_indicator": "severity-chip"},
	},
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "production_order_requires_item", "condition": {"operation": "create_production_order", "item_present": False}, "effect": {"decision": "deny", "reason": "item_required", "required_action": "specify_item"}},
	{"name": "production_order_requires_quantity", "condition": {"operation": "create_production_order", "quantity_valid": False}, "effect": {"decision": "deny", "reason": "positive_quantity_required", "required_action": "set_valid_quantity"}},
	{"name": "production_order_requires_bom", "condition": {"operation": "release_production_order", "bom_present": False}, "effect": {"decision": "deny", "reason": "bom_required_for_release", "required_action": "attach_bom"}},
	{"name": "purchase_requisition_requires_item", "condition": {"operation": "create_purchase_requisition", "item_present": False}, "effect": {"decision": "deny", "reason": "item_required", "required_action": "specify_item"}},
	{"name": "purchase_requisition_approval_required", "condition": {"operation": "submit_purchase_requisition", "approval_present": False}, "effect": {"decision": "deny", "reason": "approval_required", "required_action": "submit_for_approval"}},
	{"name": "planning_run_requires_horizon", "condition": {"operation": "run_mrp", "horizon_valid": False}, "effect": {"decision": "deny", "reason": "valid_planning_horizon_required", "required_action": "set_planning_horizon"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	cfg = deepcopy(DEFAULT_CONFIGURATION)
	cfg["tenant_id"] = tenant_id
	return {
		"capability": CAPABILITY_ID,
		"name": CAPABILITY_NAME,
		"display_name": CAPABILITY_NAME,
		"version": CAPABILITY_VERSION,
		"domain": CAPABILITY_DOMAIN,
		"description": CAPABILITY_DESCRIPTION,
		"provides": list(PROVIDES),
		"requires": list(REQUIRES),
		"publishes": list(PUBLISHES),
		"subscribes": list(SUBSCRIBES),
		"configuration": cfg,
		"rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)},
		"ui": {"shell": "apg_python", "api_prefix": "/mfg-mrp/api/v1", "requires_theme": True, "routes": deepcopy(UI_ROUTES)},
		"theme": deepcopy(THEME),
		"streaming": {"processor": "bytewax", "stream": MRP_EVENT_STREAM, "key": "tenant_id", "events": list(PUBLISHES)},
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
		if context.get(key) != expected:
			return False
	return True
