"""Executable capability contract for APG Lot and Batch Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

CAPABILITY_ID = "mfg_lmt"
CAPABILITY_NAME = "Lot and Batch Management"
CAPABILITY_VERSION = "1.0.0"
CAPABILITY_DOMAIN = "mfg"
CAPABILITY_DESCRIPTION = "Lot creation, traceability genealogy, shelf-life and expiry management, lot recall, and forward/backward tracing."

LMT_EVENT_STREAM = "apg.mfg.lmt.lifecycle"

SUPPORTED_LOT_STATUSES = ["available", "quarantine", "on_hold", "rejected", "expired", "consumed"]
SUPPORTED_LOT_TYPES = ["production", "purchase", "process", "sub_lot"]
SUPPORTED_TRACE_DIRECTIONS = ["forward", "backward", "bidirectional"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"lots": {"supported_statuses": SUPPORTED_LOT_STATUSES, "supported_types": SUPPORTED_LOT_TYPES, "expiry_tracking": True, "shelf_life_days_default": None},
	"traceability": {"supported_directions": SUPPORTED_TRACE_DIRECTIONS, "genealogy_depth": 10},
	"recall": {"enabled": True, "auto_quarantine_on_recall": True},
	"governance": {"require_tenant_context": True, "audit_events": True},
	"adapters": {"auth": "auth", "audit": "audl", "qms": "mfg_qms", "event_stream": "bytewax"},
}

PROVIDES = ["lot_creation", "lot_traceability", "shelf_life_management", "lot_recall", "genealogy_query"]
REQUIRES = ["auth", "audl"]
PUBLISHES = ["apg.mfg.lmt.lot_created", "apg.mfg.lmt.lot_quarantined", "apg.mfg.lmt.lot_expired", "apg.mfg.lmt.recall_initiated"]
SUBSCRIBES = ["apg.mfg.mes.work_order_completed", "apg.mfg.qms.ncr_raised"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/mfg-lmt/dashboard", "component": "MfgLmtDashboard", "permission": "mfg_lmt:view", "nav_group": "Overview"},
	{"name": "lots", "path": "/mfg-lmt/lots", "component": "MfgLmtLots", "permission": "mfg_lmt:manage", "nav_group": "Lots"},
	{"name": "traceability", "path": "/mfg-lmt/traceability", "component": "MfgLmtTraceability", "permission": "mfg_lmt:view", "nav_group": "Traceability"},
	{"name": "recall", "path": "/mfg-lmt/recall", "component": "MfgLmtRecall", "permission": "mfg_lmt:admin", "nav_group": "Recall"},
	{"name": "settings", "path": "/mfg-lmt/settings", "component": "MfgLmtSettings", "permission": "mfg_lmt:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "mfg_lmt_theme",
	"tokens": {"color.primary": "#2D4059", "color.accent": "#F07B3F", "color.success": "#10B981", "color.danger": "#EF4444", "surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF", "text.primary": "#111827", "border.radius": "6px", "density": "compact"},
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "lot_requires_item", "condition": {"operation": "create_lot", "item_present": False}, "effect": {"decision": "deny", "reason": "item_required", "required_action": "specify_item"}},
	{"name": "recall_requires_lot", "condition": {"operation": "initiate_recall", "lot_present": False}, "effect": {"decision": "deny", "reason": "lot_required_for_recall", "required_action": "specify_lot"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	cfg = deepcopy(DEFAULT_CONFIGURATION); cfg["tenant_id"] = tenant_id
	return {"capability": CAPABILITY_ID, "name": CAPABILITY_NAME, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "domain": CAPABILITY_DOMAIN, "description": CAPABILITY_DESCRIPTION, "provides": list(PROVIDES), "requires": list(REQUIRES), "publishes": list(PUBLISHES), "subscribes": list(SUBSCRIBES), "configuration": cfg, "rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "api_prefix": "/mfg-lmt/api/v1", "requires_theme": True, "routes": deepcopy(UI_ROUTES)}, "theme": deepcopy(THEME), "streaming": {"processor": "bytewax", "stream": LMT_EVENT_STREAM, "key": "tenant_id", "events": list(PUBLISHES)},
		"configuration_schema": {
			"type": "object",
			"required": ['tenant_id'],
			"properties": {
				"tenant_id": {"type": "string"},
				"lots": {"type": "object"},
				"traceability": {"type": "object"},
				"recall": {"type": "object"},
				"governance": {"type": "object"},
				"adapters": {"type": "object"},
			},
		},
}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	actions = [rule["effect"] | {"rule": rule["name"]} for rule in RULES if all(context.get(k) == v for k, v in rule["condition"].items())]
	return {"decision": "deny", "actions": actions, "context": dict(context)} if actions else {"decision": "allow", "actions": [], "context": dict(context)}
