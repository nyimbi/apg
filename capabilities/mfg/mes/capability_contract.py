"""Executable capability contract for APG Manufacturing Execution System."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "mfg_mes"
CAPABILITY_NAME = "Manufacturing Execution System"
CAPABILITY_VERSION = "1.0.0"
CAPABILITY_DOMAIN = "mfg"
CAPABILITY_DESCRIPTION = "Real-time production tracking: work orders, OEE, resource monitoring, production events, labour and material transactions."

MES_EVENT_STREAM = "apg.mfg.mes.lifecycle"

SUPPORTED_WORK_ORDER_STATUSES = ["created", "released", "started", "in_progress", "paused", "completed", "closed", "cancelled"]
SUPPORTED_RESOURCE_STATUSES = ["available", "busy", "maintenance", "breakdown", "offline"]
SUPPORTED_EVENT_TYPES = ["start", "pause", "resume", "scrap", "rework", "complete", "downtime", "setup", "teardown"]
SUPPORTED_DOWNTIME_CATEGORIES = ["planned", "unplanned", "maintenance", "changeover", "breakdown", "quality"]
SUPPORTED_OEE_PILLARS = ["availability", "performance", "quality"]
SUPPORTED_SEVERITIES = ["low", "medium", "high", "critical"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"work_orders": {
		"supported_statuses": SUPPORTED_WORK_ORDER_STATUSES,
		"require_routing": False,
		"require_bom": True,
		"auto_backflush_materials": True,
	},
	"resources": {
		"supported_statuses": SUPPORTED_RESOURCE_STATUSES,
		"oee_calculation_enabled": True,
		"downtime_categories": SUPPORTED_DOWNTIME_CATEGORIES,
	},
	"production_events": {
		"supported_types": SUPPORTED_EVENT_TYPES,
		"real_time_tracking": True,
		"barcode_scan_enabled": True,
	},
	"oee": {
		"target_oee": 0.85,
		"target_availability": 0.90,
		"target_performance": 0.95,
		"target_quality": 0.99,
		"calculation_interval_minutes": 60,
	},
	"governance": {
		"require_tenant_context": True,
		"audit_events": True,
	},
	"adapters": {
		"auth": "auth",
		"audit": "audl",
		"mrp": "mfg_mrp",
		"sfc": "mfg_sfc",
		"qms": "mfg_qms",
		"event_stream": "bytewax",
	},
}

PROVIDES = [
	"work_order_execution",
	"production_event_tracking",
	"oee_calculation",
	"resource_monitoring",
	"labour_transaction",
	"material_transaction",
]

REQUIRES = ["auth", "audl", "mfg_mrp"]

PUBLISHES = [
	"apg.mfg.mes.work_order_started",
	"apg.mfg.mes.work_order_completed",
	"apg.mfg.mes.production_event_recorded",
	"apg.mfg.mes.downtime_recorded",
	"apg.mfg.mes.oee_calculated",
	"apg.mfg.mes.scrap_recorded",
]

SUBSCRIBES = [
	"apg.mfg.mrp.production_order_released",
	"apg.mfg.sfc.operation_completed",
	"apg.mfg.qms.ncr_raised",
]

UI_ROUTES = [
	{"name": "dashboard", "path": "/mfg-mes/dashboard", "component": "MfgMesDashboard", "permission": "mfg_mes:view", "nav_group": "Overview"},
	{"name": "work_orders", "path": "/mfg-mes/work-orders", "component": "MfgMesWorkOrders", "permission": "mfg_mes:manage", "nav_group": "Execution"},
	{"name": "production_events", "path": "/mfg-mes/events", "component": "MfgMesProductionEvents", "permission": "mfg_mes:view", "nav_group": "Execution"},
	{"name": "resources", "path": "/mfg-mes/resources", "component": "MfgMesResources", "permission": "mfg_mes:view", "nav_group": "Resources"},
	{"name": "oee", "path": "/mfg-mes/oee", "component": "MfgMesOee", "permission": "mfg_mes:view", "nav_group": "Analytics"},
	{"name": "downtime", "path": "/mfg-mes/downtime", "component": "MfgMesDowntime", "permission": "mfg_mes:manage", "nav_group": "Analytics"},
	{"name": "settings", "path": "/mfg-mes/settings", "component": "MfgMesSettings", "permission": "mfg_mes:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "mfg_mes_theme",
	"tokens": {
		"color.primary": "#0F4C75",
		"color.accent": "#1B9AAA",
		"color.success": "#06A77D",
		"color.warning": "#D62246",
		"color.danger": "#B91C1C",
		"surface.canvas": "#EFF2F7",
		"surface.panel": "#FFFFFF",
		"text.primary": "#1A1A2E",
		"text.secondary": "#4A4A6A",
		"border.radius": "6px",
		"density": "compact",
	},
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "work_order_requires_item", "condition": {"operation": "create_work_order", "item_present": False}, "effect": {"decision": "deny", "reason": "item_required", "required_action": "specify_item"}},
	{"name": "work_order_requires_quantity", "condition": {"operation": "create_work_order", "quantity_valid": False}, "effect": {"decision": "deny", "reason": "positive_quantity_required", "required_action": "set_valid_quantity"}},
	{"name": "event_requires_work_order", "condition": {"operation": "record_production_event", "work_order_present": False}, "effect": {"decision": "deny", "reason": "work_order_required", "required_action": "specify_work_order"}},
	{"name": "complete_requires_started", "condition": {"operation": "complete_work_order", "order_started": False}, "effect": {"decision": "deny", "reason": "work_order_must_be_started", "required_action": "start_work_order_first"}},
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
		"ui": {"shell": "apg_python", "api_prefix": "/mfg-mes/api/v1", "requires_theme": True, "routes": deepcopy(UI_ROUTES)},
		"theme": deepcopy(THEME),
		"streaming": {"processor": "bytewax", "stream": MES_EVENT_STREAM, "key": "tenant_id", "events": list(PUBLISHES)},
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
