"""Executable capability contract for APG Product Lifecycle Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

CAPABILITY_ID = "mfg_plm"
CAPABILITY_NAME = "Product Lifecycle Management"
CAPABILITY_VERSION = "1.0.0"
CAPABILITY_DOMAIN = "mfg"
CAPABILITY_DESCRIPTION = "Product portfolio management, stage-gate NPI workflow, design release, and product discontinuation."

PLM_EVENT_STREAM = "apg.mfg.plm.lifecycle"

SUPPORTED_LIFECYCLE_STAGES = ["concept", "design", "prototype", "pilot", "production", "maturity", "decline", "discontinued"]
SUPPORTED_GATE_DECISIONS = ["pass", "conditional_pass", "hold", "kill"]
SUPPORTED_PRODUCT_TYPES = ["standard", "configurable", "variant", "service", "kit"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"products": {"supported_types": SUPPORTED_PRODUCT_TYPES, "revision_control": True},
	"stage_gates": {"supported_stages": SUPPORTED_LIFECYCLE_STAGES, "supported_decisions": SUPPORTED_GATE_DECISIONS, "approval_required": True},
	"governance": {"require_tenant_context": True, "audit_events": True},
	"adapters": {"auth": "auth", "audit": "audl", "bom": "mfg_bom", "event_stream": "bytewax"},
}

PROVIDES = ["product_portfolio", "npi_stage_gate", "design_release", "product_discontinuation"]
REQUIRES = ["auth", "audl"]
PUBLISHES = ["apg.mfg.plm.product_created", "apg.mfg.plm.design_released", "apg.mfg.plm.product_discontinued"]
SUBSCRIBES = ["apg.mfg.bom.eco_released"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/mfg-plm/dashboard", "component": "MfgPlmDashboard", "permission": "mfg_plm:view", "nav_group": "Overview"},
	{"name": "portfolio", "path": "/mfg-plm/portfolio", "component": "MfgPlmPortfolio", "permission": "mfg_plm:view", "nav_group": "Products"},
	{"name": "npi", "path": "/mfg-plm/npi", "component": "MfgPlmNpi", "permission": "mfg_plm:manage", "nav_group": "NPI"},
	{"name": "settings", "path": "/mfg-plm/settings", "component": "MfgPlmSettings", "permission": "mfg_plm:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "mfg_plm_theme",
	"tokens": {"color.primary": "#2E4057", "color.accent": "#048A81", "color.success": "#10B981", "color.danger": "#EF4444", "surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF", "text.primary": "#111827", "border.radius": "6px", "density": "comfortable"},
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "stage_gate_approval_required", "condition": {"operation": "record_gate_decision", "approval_present": False}, "effect": {"decision": "deny", "reason": "gate_approval_required", "required_action": "obtain_gate_approval"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	cfg = deepcopy(DEFAULT_CONFIGURATION); cfg["tenant_id"] = tenant_id
	return {"capability": CAPABILITY_ID, "name": CAPABILITY_NAME, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "domain": CAPABILITY_DOMAIN, "description": CAPABILITY_DESCRIPTION, "provides": list(PROVIDES), "requires": list(REQUIRES), "publishes": list(PUBLISHES), "subscribes": list(SUBSCRIBES), "configuration": cfg, "rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "api_prefix": "/mfg-plm/api/v1", "requires_theme": True, "routes": deepcopy(UI_ROUTES)}, "theme": deepcopy(THEME), "streaming": {"processor": "bytewax", "stream": PLM_EVENT_STREAM, "key": "tenant_id", "events": list(PUBLISHES)},
		"configuration_schema": {
			"type": "object",
			"required": ['tenant_id'],
			"properties": {
				"tenant_id": {"type": "string"},
				"products": {"type": "object"},
				"stage_gates": {"type": "object"},
				"governance": {"type": "object"},
				"adapters": {"type": "object"},
			},
		},
}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	actions = [rule["effect"] | {"rule": rule["name"]} for rule in RULES if all(context.get(k) == v for k, v in rule["condition"].items())]
	return {"decision": "deny", "actions": actions, "context": dict(context)} if actions else {"decision": "allow", "actions": [], "context": dict(context)}
