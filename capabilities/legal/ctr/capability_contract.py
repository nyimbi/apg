"""Executable capability contract for APG Contract Lifecycle Management."""
from __future__ import annotations
from typing import Any

CAPABILITY_ID = 'leg_ctr'
CAPABILITY_NAME = 'Contract Lifecycle Management'
CAPABILITY_VERSION = '1.0.0'
CAPABILITY_DOMAIN = 'legal'
CAPABILITY_DESCRIPTION = 'Contract drafting, negotiation, signing, obligation tracking, renewal alerts'

SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["leg_ctr_manager", "leg_ctr_viewer", "leg_ctr_admin"]

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
    "theme": {"default_theme": "leg_ctr_theme", "allow_tenant_overrides": True},
}

PROVIDES = ['contract_drafting', 'negotiation_management', 'contract_execution', 'obligation_tracking', 'renewal_management']
REQUIRES = ['auth', 'audl', 'ntfy', 'esig', 'ckm_wfa']

# NATS integration declarations
PUBLISHES = ['contract.drafted', 'contract.signed', 'obligation.due']
SUBSCRIBES = []

UI_ROUTES = [{'name': 'dashboard', 'path': '/leg-ctr/dashboard', 'component': 'LegCtrDashboard', 'permission': 'leg_ctr:view', 'nav_group': 'Overview'}, {'name': 'list', 'path': '/leg-ctr/list', 'component': 'LegCtrList', 'permission': 'leg_ctr:view', 'nav_group': 'Overview'}, {'name': 'settings', 'path': '/leg-ctr/settings', 'component': 'LegCtrSettings', 'permission': 'leg_ctr:admin', 'nav_group': 'Administration'}]

THEME = {
    "name": "leg_ctr_theme",
    "tokens": {
        "color.primary": '#2D3561',
        "color.accent": '#A239CA',
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


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
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
    """Evaluate APG capability rules for Contract Lifecycle Management."""
    reasons: list[str] = []
    actions: list[dict[str, Any]] = []

    if not context.get("tenant_context_present"):
        reasons.append("tenant_id required")
        actions.append({"type": "deny", "reason": "missing_tenant_context"})
        return {"decision": "deny", "matched_rules": ["tenant_required"], "actions": actions}

    return {"decision": "allow", "matched_rules": [], "actions": []}
