"""Executable capability contract for APG Production Planning."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

CAPABILITY_ID = "mfg_ppl"
CAPABILITY_NAME = "Production Planning"
CAPABILITY_VERSION = "1.0.0"
CAPABILITY_DOMAIN = "mfg"
CAPABILITY_DESCRIPTION = "S&OP, master production schedule (MPS), rough-cut capacity planning (RCCP), and demand management."

PPL_EVENT_STREAM = "apg.mfg.ppl.lifecycle"

SUPPORTED_PLAN_TYPES = ["mps", "sop", "rccp", "demand_forecast"]
SUPPORTED_HORIZON_UNITS = ["week", "month", "quarter"]
SUPPORTED_PLAN_STATUSES = ["draft", "submitted", "approved", "active", "closed"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"planning": {"supported_plan_types": SUPPORTED_PLAN_TYPES, "supported_horizons": SUPPORTED_HORIZON_UNITS, "default_horizon_weeks": 13},
	"sop": {"cycle": "monthly", "approval_required": True},
	"rccp": {"enabled": True, "work_centre_groups_enabled": True},
	"governance": {"require_tenant_context": True, "audit_events": True},
	"adapters": {"auth": "auth", "audit": "audl", "mrp": "mfg_mrp", "cap": "mfg_cap", "event_stream": "bytewax"},
}

PROVIDES = ["master_production_schedule", "sop_process", "rccp", "demand_management"]
REQUIRES = ["auth", "audl"]
PUBLISHES = ["apg.mfg.ppl.master_schedule_updated", "apg.mfg.ppl.sop_approved", "apg.mfg.ppl.demand_plan_updated"]
SUBSCRIBES = ["apg.crm.forecast_updated", "apg.mfg.mrp.planning_run_completed"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/mfg-ppl/dashboard", "component": "MfgPplDashboard", "permission": "mfg_ppl:view", "nav_group": "Overview"},
	{"name": "mps", "path": "/mfg-ppl/mps", "component": "MfgPplMps", "permission": "mfg_ppl:manage", "nav_group": "Master Schedule"},
	{"name": "sop", "path": "/mfg-ppl/sop", "component": "MfgPplSop", "permission": "mfg_ppl:manage", "nav_group": "S&OP"},
	{"name": "demand", "path": "/mfg-ppl/demand", "component": "MfgPplDemand", "permission": "mfg_ppl:view", "nav_group": "Demand"},
	{"name": "rccp", "path": "/mfg-ppl/rccp", "component": "MfgPplRccp", "permission": "mfg_ppl:view", "nav_group": "Capacity"},
	{"name": "settings", "path": "/mfg-ppl/settings", "component": "MfgPplSettings", "permission": "mfg_ppl:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "mfg_ppl_theme",
	"tokens": {"color.primary": "#1A3C5E", "color.accent": "#E8A838", "color.success": "#10B981", "color.danger": "#EF4444", "surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF", "text.primary": "#111827", "border.radius": "6px", "density": "compact"},
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "mps_approval_required", "condition": {"operation": "activate_mps", "approval_present": False}, "effect": {"decision": "deny", "reason": "mps_approval_required", "required_action": "obtain_mps_approval"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	cfg = deepcopy(DEFAULT_CONFIGURATION); cfg["tenant_id"] = tenant_id
	return {"capability": CAPABILITY_ID, "name": CAPABILITY_NAME, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "domain": CAPABILITY_DOMAIN, "description": CAPABILITY_DESCRIPTION, "provides": list(PROVIDES), "requires": list(REQUIRES), "publishes": list(PUBLISHES), "subscribes": list(SUBSCRIBES), "configuration": cfg, "rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "api_prefix": "/mfg-ppl/api/v1", "requires_theme": True, "routes": deepcopy(UI_ROUTES)}, "theme": deepcopy(THEME), "streaming": {"processor": "bytewax", "stream": PPL_EVENT_STREAM, "key": "tenant_id", "events": list(PUBLISHES)}}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	actions = [rule["effect"] | {"rule": rule["name"]} for rule in RULES if all(context.get(k) == v for k, v in rule["condition"].items())]
	return {"decision": "deny", "actions": actions, "context": dict(context)} if actions else {"decision": "allow", "actions": [], "context": dict(context)}
