"""Executable capability contract for APG Actuarial Tools."""
from __future__ import annotations
from typing import Any

CAPABILITY_ID = 'ins_act'
CAPABILITY_NAME = 'Actuarial Tools'
CAPABILITY_VERSION = '1.0.0'
CAPABILITY_DOMAIN = 'insurance'
CAPABILITY_DESCRIPTION = 'Reserve calculations, loss ratio analysis, pricing models, stress testing'

SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["ins_act_manager", "ins_act_viewer", "ins_act_admin"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
    "tenant_id": "default",
    "governance": {
        "require_tenant_context": True,
        "audit_events": True,
        "human_approval_required_for_high_impact_actions": True,
    },
    "agents": {
        "enabled": True,
        "supported_runtimes": SUPPORTED_AGENT_RUNTIMES,
        "supported_roles": SUPPORTED_AGENT_ROLES,
        "human_approval_required_for_privileged_actions": True,
    },
    "theme": {"default_theme": "ins_act_theme", "allow_tenant_overrides": True},
}

PROVIDES = ['reserve_calculation', 'loss_ratio_analysis', 'pricing_models', 'stress_testing', 'actuarial_reporting']
REQUIRES = ['auth', 'audl', 'ins_pol', 'ins_clm']

# NATS integration declarations
PUBLISHES = ['reserve.calculated', 'pricing_model.updated']
SUBSCRIBES = []

UI_ROUTES = [{'name': 'dashboard', 'path': '/ins-act/dashboard', 'component': 'InsActDashboard', 'permission': 'ins_act:view', 'nav_group': 'Overview'}, {'name': 'list', 'path': '/ins-act/list', 'component': 'InsActList', 'permission': 'ins_act:view', 'nav_group': 'Overview'}, {'name': 'settings', 'path': '/ins-act/settings', 'component': 'InsActSettings', 'permission': 'ins_act:admin', 'nav_group': 'Administration'}]

THEME = {
    "name": "ins_act_theme",
    "tokens": {
        "color.primary": '#4A1942',
        "color.accent": '#C77DFF',
        "color.success": "#15803D",
        "color.warning": "#B45309",
        "color.danger": "#B91C1C",
        "surface.canvas": "#F8FAFC",
        "surface.panel": "#FFFFFF",
        "text.primary": "#111827",
        "text.secondary": "#4B5563",
        "border.radius": "8px",
        "density": "compact",
    },
}


def get_capability_contract() -> dict[str, Any]:
    return {
        "id": CAPABILITY_ID,
        "name": CAPABILITY_NAME,
        "version": CAPABILITY_VERSION,
        "domain": CAPABILITY_DOMAIN,
        "description": CAPABILITY_DESCRIPTION,
        "provides": PROVIDES,
        "requires": REQUIRES,
        "publishes": PUBLISHES,
        "subscribes": SUBSCRIBES,
        "ui_routes": UI_ROUTES,
        "theme": THEME,
        "configuration": DEFAULT_CONFIGURATION,
    }


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
    """Evaluate APG capability rules for Actuarial Tools."""
    reasons: list[str] = []
    actions: list[dict[str, Any]] = []

    if not context.get("tenant_context_present"):
        reasons.append("tenant_id required")
        actions.append({"type": "deny", "reason": "missing_tenant_context"})
        return {"decision": "deny", "matched_rules": ["tenant_required"], "actions": actions}

    return {"decision": "allow", "matched_rules": [], "actions": []}
