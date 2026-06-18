"""Executable capability contract for APG Product Costing."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

CAPABILITY_ID = "mfg_pco"
CAPABILITY_NAME = "Product Costing"
CAPABILITY_VERSION = "1.0.0"
CAPABILITY_DOMAIN = "mfg"
CAPABILITY_DESCRIPTION = "Standard costing, cost rollup from BOM and routing, variance analysis (price, quantity, efficiency), and period-end costing close."

PCO_EVENT_STREAM = "apg.mfg.pco.lifecycle"

SUPPORTED_COST_TYPES = ["standard", "actual", "average", "target", "simulated"]
SUPPORTED_COST_ELEMENTS = ["material", "labour", "overhead", "subcontract", "tooling"]
SUPPORTED_VARIANCE_TYPES = ["price", "quantity", "efficiency", "overhead_absorption", "mix"]
SUPPORTED_COSTING_STATUSES = ["draft", "active", "frozen", "archived"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"costing": {
		"supported_types": SUPPORTED_COST_TYPES,
		"default_type": "standard",
		"supported_elements": SUPPORTED_COST_ELEMENTS,
		"currency": "USD",
		"rollup_depth": 10,
	},
	"variance": {"supported_types": SUPPORTED_VARIANCE_TYPES, "auto_post_to_gl": False},
	"period_close": {"enabled": True, "approval_required": True},
	"governance": {"require_tenant_context": True, "audit_events": True},
	"adapters": {"auth": "auth", "audit": "audl", "bom": "mfg_bom", "sfc": "mfg_sfc", "gl": "fin_gl", "event_stream": "bytewax"},
}

PROVIDES = ["standard_cost_management", "cost_rollup", "variance_analysis", "period_costing_close"]
REQUIRES = ["auth", "audl", "mfg_bom"]
PUBLISHES = ["apg.mfg.pco.cost_rolled_up", "apg.mfg.pco.variance_posted", "apg.mfg.pco.period_closed"]
SUBSCRIBES = ["apg.mfg.bom.bom_changed", "apg.mfg.mes.work_order_completed", "apg.mfg.mrp.production_order_released"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/mfg-pco/dashboard", "component": "MfgPcoDashboard", "permission": "mfg_pco:view", "nav_group": "Overview"},
	{"name": "cost_records", "path": "/mfg-pco/cost-records", "component": "MfgPcoCostRecords", "permission": "mfg_pco:view", "nav_group": "Costing"},
	{"name": "rollup", "path": "/mfg-pco/rollup", "component": "MfgPcoRollup", "permission": "mfg_pco:manage", "nav_group": "Costing"},
	{"name": "variance", "path": "/mfg-pco/variance", "component": "MfgPcoVariance", "permission": "mfg_pco:view", "nav_group": "Analysis"},
	{"name": "period_close", "path": "/mfg-pco/period-close", "component": "MfgPcoPeriodClose", "permission": "mfg_pco:admin", "nav_group": "Period Close"},
	{"name": "settings", "path": "/mfg-pco/settings", "component": "MfgPcoSettings", "permission": "mfg_pco:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "mfg_pco_theme",
	"tokens": {"color.primary": "#1B3A4B", "color.accent": "#C9A227", "color.success": "#10B981", "color.danger": "#EF4444", "surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF", "text.primary": "#111827", "border.radius": "6px", "density": "compact"},
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "cost_record_requires_item", "condition": {"operation": "create_cost_record", "item_present": False}, "effect": {"decision": "deny", "reason": "item_required", "required_action": "specify_item"}},
	{"name": "period_close_requires_approval", "condition": {"operation": "close_period", "approval_present": False}, "effect": {"decision": "deny", "reason": "period_close_approval_required", "required_action": "obtain_approval"}},
	{"name": "frozen_cost_immutable", "condition": {"operation": "update_cost_record", "cost_status_frozen": True}, "effect": {"decision": "deny", "reason": "frozen_cost_record_immutable", "required_action": "create_new_cost_version"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	cfg = deepcopy(DEFAULT_CONFIGURATION); cfg["tenant_id"] = tenant_id
	return {"capability": CAPABILITY_ID, "name": CAPABILITY_NAME, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "domain": CAPABILITY_DOMAIN, "description": CAPABILITY_DESCRIPTION, "provides": list(PROVIDES), "requires": list(REQUIRES), "publishes": list(PUBLISHES), "subscribes": list(SUBSCRIBES), "configuration": cfg, "rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "api_prefix": "/mfg-pco/api/v1", "requires_theme": True, "routes": deepcopy(UI_ROUTES)}, "theme": deepcopy(THEME), "streaming": {"processor": "bytewax", "stream": PCO_EVENT_STREAM, "key": "tenant_id", "events": list(PUBLISHES)},
		"configuration_schema": {
			"type": "object",
			"required": ['tenant_id'],
			"properties": {
				"tenant_id": {"type": "string"},
				"costing": {"type": "object"},
				"variance": {"type": "object"},
				"period_close": {"type": "object"},
				"governance": {"type": "object"},
				"adapters": {"type": "object"},
			},
		},
}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	actions = [rule["effect"] | {"rule": rule["name"]} for rule in RULES if all(context.get(k) == v for k, v in rule["condition"].items())]
	return {"decision": "deny", "actions": actions, "context": dict(context)} if actions else {"decision": "allow", "actions": [], "context": dict(context)}
