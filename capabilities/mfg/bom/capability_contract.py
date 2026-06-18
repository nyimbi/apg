"""Executable capability contract for APG Bill of Materials."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

CAPABILITY_ID = "mfg_bom"
CAPABILITY_NAME = "Bill of Materials"
CAPABILITY_VERSION = "1.0.0"
CAPABILITY_DOMAIN = "mfg"
CAPABILITY_DESCRIPTION = "Multi-level BOM management: engineering BOM, manufacturing BOM, ECO/ECN workflow, BOM comparison and cost rollup."

BOM_EVENT_STREAM = "apg.mfg.bom.lifecycle"

SUPPORTED_BOM_TYPES = ["engineering", "manufacturing", "phantom", "sales", "service"]
SUPPORTED_ITEM_TYPES = ["make", "buy", "phantom", "reference"]
SUPPORTED_ECO_STATUSES = ["draft", "review", "approved", "released", "rejected"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"bom": {
		"supported_types": SUPPORTED_BOM_TYPES,
		"max_levels": 15,
		"version_control": True,
		"effectivity_dates": True,
	},
	"eco": {
		"supported_statuses": SUPPORTED_ECO_STATUSES,
		"approval_required": True,
	},
	"governance": {"require_tenant_context": True, "audit_events": True},
	"adapters": {"auth": "auth", "audit": "audl", "event_stream": "bytewax"},
}

PROVIDES = ["bom_structure", "bom_explosion", "eco_workflow", "bom_comparison", "cost_rollup"]
REQUIRES = ["auth", "audl"]
PUBLISHES = ["apg.mfg.bom.bom_created", "apg.mfg.bom.bom_changed", "apg.mfg.bom.eco_released"]
SUBSCRIBES = ["apg.mfg.plm.design_released"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/mfg-bom/dashboard", "component": "MfgBomDashboard", "permission": "mfg_bom:view", "nav_group": "Overview"},
	{"name": "boms", "path": "/mfg-bom/boms", "component": "MfgBomList", "permission": "mfg_bom:manage", "nav_group": "BOMs"},
	{"name": "structure", "path": "/mfg-bom/structure", "component": "MfgBomStructure", "permission": "mfg_bom:view", "nav_group": "BOMs"},
	{"name": "ecos", "path": "/mfg-bom/ecos", "component": "MfgBomEcos", "permission": "mfg_bom:manage", "nav_group": "Changes"},
	{"name": "settings", "path": "/mfg-bom/settings", "component": "MfgBomSettings", "permission": "mfg_bom:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "mfg_bom_theme",
	"tokens": {"color.primary": "#1E3A5F", "color.accent": "#2E86AB", "color.success": "#10B981", "color.danger": "#EF4444", "surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF", "text.primary": "#111827", "border.radius": "6px", "density": "compact"},
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "bom_requires_parent_item", "condition": {"operation": "create_bom", "parent_item_present": False}, "effect": {"decision": "deny", "reason": "parent_item_required", "required_action": "specify_parent_item"}},
	{"name": "eco_approval_required", "condition": {"operation": "release_eco", "approval_present": False}, "effect": {"decision": "deny", "reason": "eco_approval_required", "required_action": "obtain_eco_approval"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	cfg = deepcopy(DEFAULT_CONFIGURATION)
	cfg["tenant_id"] = tenant_id
	return {
		"capability": CAPABILITY_ID, "name": CAPABILITY_NAME, "display_name": CAPABILITY_NAME,
		"version": CAPABILITY_VERSION, "domain": CAPABILITY_DOMAIN, "description": CAPABILITY_DESCRIPTION,
		"provides": list(PROVIDES), "requires": list(REQUIRES), "publishes": list(PUBLISHES), "subscribes": list(SUBSCRIBES),
		"configuration": cfg,
		"rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)},
		"ui": {"shell": "apg_python", "api_prefix": "/mfg-bom/api/v1", "requires_theme": True, "routes": deepcopy(UI_ROUTES)},
		"theme": deepcopy(THEME),
		"streaming": {"processor": "bytewax", "stream": BOM_EVENT_STREAM, "key": "tenant_id", "events": list(PUBLISHES)},
	
		"configuration_schema": {
			"type": "object",
			"required": ['tenant_id'],
			"properties": {
				"tenant_id": {"type": "string"},
				"bom": {"type": "object"},
				"eco": {"type": "object"},
				"governance": {"type": "object"},
				"adapters": {"type": "object"},
			},
		},
}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	actions = [rule["effect"] | {"rule": rule["name"]} for rule in RULES if all(context.get(k) == v for k, v in rule["condition"].items())]
	return {"decision": "deny", "actions": actions, "context": dict(context)} if actions else {"decision": "allow", "actions": [], "context": dict(context)}
