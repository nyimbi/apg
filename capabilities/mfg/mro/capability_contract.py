"""Executable capability contract for APG Maintenance, Repair & Overhaul."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

CAPABILITY_ID = "mfg_mro"
CAPABILITY_NAME = "Maintenance, Repair and Overhaul"
CAPABILITY_VERSION = "1.0.0"
CAPABILITY_DOMAIN = "mfg"
CAPABILITY_DESCRIPTION = "Production asset maintenance: work orders, PM schedules, failure analysis, spare parts management, and uptime tracking."

MRO_EVENT_STREAM = "apg.mfg.mro.lifecycle"

SUPPORTED_WORK_ORDER_TYPES = ["corrective", "preventive", "predictive", "inspection", "calibration"]
SUPPORTED_WORK_ORDER_STATUSES = ["open", "assigned", "in_progress", "on_hold", "completed", "cancelled"]
SUPPORTED_PRIORITY_LEVELS = ["emergency", "high", "medium", "low"]
SUPPORTED_ASSET_STATUSES = ["operational", "degraded", "under_maintenance", "decommissioned"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"work_orders": {"supported_types": SUPPORTED_WORK_ORDER_TYPES, "supported_statuses": SUPPORTED_WORK_ORDER_STATUSES, "supported_priorities": SUPPORTED_PRIORITY_LEVELS},
	"preventive_maintenance": {"enabled": True, "trigger_types": ["calendar", "meter", "condition"]},
	"spare_parts": {"min_stock_alerts": True, "auto_requisition": False},
	"governance": {"require_tenant_context": True, "audit_events": True},
	"adapters": {"auth": "auth", "audit": "audl", "mes": "mfg_mes", "event_stream": "bytewax"},
}

PROVIDES = ["maintenance_work_order", "pm_scheduling", "failure_analysis", "spare_parts_management", "asset_uptime_tracking"]
REQUIRES = ["auth", "audl"]
PUBLISHES = ["apg.mfg.mro.work_order_created", "apg.mfg.mro.work_order_completed", "apg.mfg.mro.asset_downtime_recorded", "apg.mfg.mro.pm_triggered"]
SUBSCRIBES = ["apg.mfg.mes.downtime_recorded", "apg.eam.asset_condition_updated"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/mfg-mro/dashboard", "component": "MfgMroDashboard", "permission": "mfg_mro:view", "nav_group": "Overview"},
	{"name": "work_orders", "path": "/mfg-mro/work-orders", "component": "MfgMroWorkOrders", "permission": "mfg_mro:manage", "nav_group": "Work Orders"},
	{"name": "assets", "path": "/mfg-mro/assets", "component": "MfgMroAssets", "permission": "mfg_mro:view", "nav_group": "Assets"},
	{"name": "pm_schedule", "path": "/mfg-mro/pm-schedule", "component": "MfgMroPmSchedule", "permission": "mfg_mro:manage", "nav_group": "Preventive Maintenance"},
	{"name": "spare_parts", "path": "/mfg-mro/spare-parts", "component": "MfgMroSpareParts", "permission": "mfg_mro:view", "nav_group": "Spare Parts"},
	{"name": "settings", "path": "/mfg-mro/settings", "component": "MfgMroSettings", "permission": "mfg_mro:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "mfg_mro_theme",
	"tokens": {"color.primary": "#34495E", "color.accent": "#E74C3C", "color.success": "#10B981", "color.danger": "#EF4444", "surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF", "text.primary": "#111827", "border.radius": "6px", "density": "compact"},
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "work_order_requires_asset", "condition": {"operation": "create_work_order", "asset_present": False}, "effect": {"decision": "deny", "reason": "asset_required", "required_action": "specify_asset"}},
	{"name": "work_order_requires_type", "condition": {"operation": "create_work_order", "type_valid": False}, "effect": {"decision": "deny", "reason": "valid_work_order_type_required", "required_action": "specify_type"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	cfg = deepcopy(DEFAULT_CONFIGURATION); cfg["tenant_id"] = tenant_id
	return {"capability": CAPABILITY_ID, "name": CAPABILITY_NAME, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "domain": CAPABILITY_DOMAIN, "description": CAPABILITY_DESCRIPTION, "provides": list(PROVIDES), "requires": list(REQUIRES), "publishes": list(PUBLISHES), "subscribes": list(SUBSCRIBES), "configuration": cfg, "rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "api_prefix": "/mfg-mro/api/v1", "requires_theme": True, "routes": deepcopy(UI_ROUTES)}, "theme": deepcopy(THEME), "streaming": {"processor": "bytewax", "stream": MRO_EVENT_STREAM, "key": "tenant_id", "events": list(PUBLISHES)}}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	actions = [rule["effect"] | {"rule": rule["name"]} for rule in RULES if all(context.get(k) == v for k, v in rule["condition"].items())]
	return {"decision": "deny", "actions": actions, "context": dict(context)} if actions else {"decision": "allow", "actions": [], "context": dict(context)}
