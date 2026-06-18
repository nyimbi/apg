"""Executable capability contract for APG Advanced Planning & Scheduling."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

CAPABILITY_ID = "mfg_aps"
CAPABILITY_NAME = "Advanced Planning and Scheduling"
CAPABILITY_VERSION = "1.0.0"
CAPABILITY_DOMAIN = "mfg"
CAPABILITY_DESCRIPTION = "Finite capacity scheduling, Gantt chart visualisation, sequencing optimisation, and constraint-based dispatch."

APS_EVENT_STREAM = "apg.mfg.aps.lifecycle"

SUPPORTED_SCHEDULING_METHODS = ["forward", "backward", "bidirectional", "constraint_based"]
SUPPORTED_SEQUENCING_RULES = ["earliest_due_date", "shortest_processing_time", "critical_ratio", "johnson_rule"]
SUPPORTED_OPTIMISATION_OBJECTIVES = ["minimize_makespan", "minimize_tardiness", "maximize_throughput", "minimize_wip"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"scheduling": {"supported_methods": SUPPORTED_SCHEDULING_METHODS, "default_method": "forward", "finite_capacity": True},
	"sequencing": {"supported_rules": SUPPORTED_SEQUENCING_RULES, "default_rule": "earliest_due_date"},
	"optimisation": {"objectives": SUPPORTED_OPTIMISATION_OBJECTIVES, "max_iterations": 1000},
	"governance": {"require_tenant_context": True, "audit_events": True},
	"adapters": {"auth": "auth", "audit": "audl", "mrp": "mfg_mrp", "cap": "mfg_cap", "sfc": "mfg_sfc", "event_stream": "bytewax"},
}

PROVIDES = ["finite_capacity_scheduling", "gantt_visualisation", "sequence_optimisation", "constraint_dispatch"]
REQUIRES = ["auth", "audl", "mfg_cap"]
PUBLISHES = ["apg.mfg.aps.schedule_published", "apg.mfg.aps.sequence_optimised"]
SUBSCRIBES = ["apg.mfg.mrp.production_order_released", "apg.mfg.cap.capacity_updated"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/mfg-aps/dashboard", "component": "MfgApsDashboard", "permission": "mfg_aps:view", "nav_group": "Overview"},
	{"name": "gantt", "path": "/mfg-aps/gantt", "component": "MfgApsGantt", "permission": "mfg_aps:view", "nav_group": "Schedule"},
	{"name": "sequence", "path": "/mfg-aps/sequence", "component": "MfgApsSequence", "permission": "mfg_aps:manage", "nav_group": "Scheduling"},
	{"name": "constraints", "path": "/mfg-aps/constraints", "component": "MfgApsConstraints", "permission": "mfg_aps:view", "nav_group": "Analysis"},
	{"name": "settings", "path": "/mfg-aps/settings", "component": "MfgApsSettings", "permission": "mfg_aps:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "mfg_aps_theme",
	"tokens": {"color.primary": "#1D3557", "color.accent": "#E63946", "color.success": "#10B981", "color.danger": "#EF4444", "surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF", "text.primary": "#111827", "border.radius": "6px", "density": "compact"},
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "schedule_requires_orders", "condition": {"operation": "run_aps", "orders_present": False}, "effect": {"decision": "deny", "reason": "production_orders_required", "required_action": "load_production_orders"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	cfg = deepcopy(DEFAULT_CONFIGURATION); cfg["tenant_id"] = tenant_id
	return {"capability": CAPABILITY_ID, "name": CAPABILITY_NAME, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "domain": CAPABILITY_DOMAIN, "description": CAPABILITY_DESCRIPTION, "provides": list(PROVIDES), "requires": list(REQUIRES), "publishes": list(PUBLISHES), "subscribes": list(SUBSCRIBES), "configuration": cfg, "rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "api_prefix": "/mfg-aps/api/v1", "requires_theme": True, "routes": deepcopy(UI_ROUTES)}, "theme": deepcopy(THEME), "streaming": {"processor": "bytewax", "stream": APS_EVENT_STREAM, "key": "tenant_id", "events": list(PUBLISHES)}}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	actions = [rule["effect"] | {"rule": rule["name"]} for rule in RULES if all(context.get(k) == v for k, v in rule["condition"].items())]
	return {"decision": "deny", "actions": actions, "context": dict(context)} if actions else {"decision": "allow", "actions": [], "context": dict(context)}
