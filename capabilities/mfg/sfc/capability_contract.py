"""Executable capability contract for APG Shop Floor Control."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

CAPABILITY_ID = "mfg_sfc"
CAPABILITY_NAME = "Shop Floor Control"
CAPABILITY_VERSION = "1.0.0"
CAPABILITY_DOMAIN = "mfg"
CAPABILITY_DESCRIPTION = "Routing management, work centre dispatch, operation tracking, and labour time recording."

SFC_EVENT_STREAM = "apg.mfg.sfc.lifecycle"

SUPPORTED_OPERATION_STATUSES = ["queued", "setup", "in_progress", "completed", "scrapped"]
SUPPORTED_WC_TYPES = ["machine", "labour", "subcontract", "inspection"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"routing": {"version_control": True, "operation_overlap_pct": 0},
	"work_centres": {"supported_types": SUPPORTED_WC_TYPES, "capacity_planning_enabled": True},
	"governance": {"require_tenant_context": True, "audit_events": True},
	"adapters": {"auth": "auth", "audit": "audl", "mes": "mfg_mes", "event_stream": "bytewax"},
}

PROVIDES = ["routing_management", "work_centre_dispatch", "operation_tracking", "labour_recording"]
REQUIRES = ["auth", "audl"]
PUBLISHES = ["apg.mfg.sfc.operation_started", "apg.mfg.sfc.operation_completed", "apg.mfg.sfc.labour_recorded"]
SUBSCRIBES = ["apg.mfg.mes.work_order_started"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/mfg-sfc/dashboard", "component": "MfgSfcDashboard", "permission": "mfg_sfc:view", "nav_group": "Overview"},
	{"name": "dispatch", "path": "/mfg-sfc/dispatch", "component": "MfgSfcDispatch", "permission": "mfg_sfc:manage", "nav_group": "Dispatch"},
	{"name": "routings", "path": "/mfg-sfc/routings", "component": "MfgSfcRoutings", "permission": "mfg_sfc:manage", "nav_group": "Configuration"},
	{"name": "work_centres", "path": "/mfg-sfc/work-centres", "component": "MfgSfcWorkCentres", "permission": "mfg_sfc:manage", "nav_group": "Configuration"},
	{"name": "settings", "path": "/mfg-sfc/settings", "component": "MfgSfcSettings", "permission": "mfg_sfc:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "mfg_sfc_theme",
	"tokens": {"color.primary": "#2C3E50", "color.accent": "#E67E22", "border.radius": "6px", "density": "compact"},
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "operation_requires_routing", "condition": {"operation": "create_operation", "routing_present": False}, "effect": {"decision": "deny", "reason": "routing_required", "required_action": "specify_routing"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	cfg = deepcopy(DEFAULT_CONFIGURATION); cfg["tenant_id"] = tenant_id
	return {"capability": CAPABILITY_ID, "name": CAPABILITY_NAME, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "domain": CAPABILITY_DOMAIN, "description": CAPABILITY_DESCRIPTION, "provides": list(PROVIDES), "requires": list(REQUIRES), "publishes": list(PUBLISHES), "subscribes": list(SUBSCRIBES), "configuration": cfg, "rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "api_prefix": "/mfg-sfc/api/v1", "requires_theme": True, "routes": deepcopy(UI_ROUTES)}, "theme": deepcopy(THEME), "streaming": {"processor": "bytewax", "stream": SFC_EVENT_STREAM, "key": "tenant_id", "events": list(PUBLISHES)}}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	actions = [rule["effect"] | {"rule": rule["name"]} for rule in RULES if all(context.get(k) == v for k, v in rule["condition"].items())]
	return {"decision": "deny", "actions": actions, "context": dict(context)} if actions else {"decision": "allow", "actions": [], "context": dict(context)}
