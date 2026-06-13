"""Executable capability contract for APG Capacity Planning."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

CAPABILITY_ID = "mfg_cap"
CAPABILITY_NAME = "Capacity Planning"
CAPABILITY_VERSION = "1.0.0"
CAPABILITY_DOMAIN = "mfg"
CAPABILITY_DESCRIPTION = "Work centre capacity modelling, load vs capacity analysis, constraint identification, and what-if simulation."

CAP_EVENT_STREAM = "apg.mfg.cap.lifecycle"

SUPPORTED_CAPACITY_TYPES = ["machine", "labour", "subcontract"]
SUPPORTED_LOAD_SOURCES = ["production_order", "planned_order", "forecast"]
SUPPORTED_HORIZON_UNITS = ["day", "week", "month"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"capacity": {"supported_types": SUPPORTED_CAPACITY_TYPES, "calendar_based": True, "efficiency_factor_enabled": True},
	"load": {"supported_sources": SUPPORTED_LOAD_SOURCES, "overload_threshold_pct": 100},
	"simulation": {"what_if_enabled": True, "max_scenarios": 5},
	"governance": {"require_tenant_context": True, "audit_events": True},
	"adapters": {"auth": "auth", "audit": "audl", "sfc": "mfg_sfc", "mrp": "mfg_mrp", "event_stream": "bytewax"},
}

PROVIDES = ["work_centre_capacity", "capacity_load_analysis", "constraint_identification", "capacity_simulation"]
REQUIRES = ["auth", "audl"]
PUBLISHES = ["apg.mfg.cap.capacity_updated", "apg.mfg.cap.overload_detected"]
SUBSCRIBES = ["apg.mfg.mrp.production_order_released", "apg.mfg.ppl.master_schedule_updated"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/mfg-cap/dashboard", "component": "MfgCapDashboard", "permission": "mfg_cap:view", "nav_group": "Overview"},
	{"name": "load_chart", "path": "/mfg-cap/load-chart", "component": "MfgCapLoadChart", "permission": "mfg_cap:view", "nav_group": "Analysis"},
	{"name": "work_centres", "path": "/mfg-cap/work-centres", "component": "MfgCapWorkCentres", "permission": "mfg_cap:manage", "nav_group": "Configuration"},
	{"name": "simulation", "path": "/mfg-cap/simulation", "component": "MfgCapSimulation", "permission": "mfg_cap:manage", "nav_group": "Simulation"},
	{"name": "settings", "path": "/mfg-cap/settings", "component": "MfgCapSettings", "permission": "mfg_cap:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "mfg_cap_theme",
	"tokens": {"color.primary": "#1C4E6A", "color.accent": "#F0A500", "border.radius": "6px", "density": "compact"},
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "capacity_requires_work_centre", "condition": {"operation": "define_capacity", "work_centre_present": False}, "effect": {"decision": "deny", "reason": "work_centre_required", "required_action": "specify_work_centre"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	cfg = deepcopy(DEFAULT_CONFIGURATION); cfg["tenant_id"] = tenant_id
	return {"capability": CAPABILITY_ID, "name": CAPABILITY_NAME, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "domain": CAPABILITY_DOMAIN, "description": CAPABILITY_DESCRIPTION, "provides": list(PROVIDES), "requires": list(REQUIRES), "publishes": list(PUBLISHES), "subscribes": list(SUBSCRIBES), "configuration": cfg, "rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "api_prefix": "/mfg-cap/api/v1", "requires_theme": True, "routes": deepcopy(UI_ROUTES)}, "theme": deepcopy(THEME), "streaming": {"processor": "bytewax", "stream": CAP_EVENT_STREAM, "key": "tenant_id", "events": list(PUBLISHES)}}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	actions = [rule["effect"] | {"rule": rule["name"]} for rule in RULES if all(context.get(k) == v for k, v in rule["condition"].items())]
	return {"decision": "deny", "actions": actions, "context": dict(context)} if actions else {"decision": "allow", "actions": [], "context": dict(context)}
