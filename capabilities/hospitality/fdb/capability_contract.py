"""Executable capability contract for APG F&B Management."""
from __future__ import annotations
from typing import Any

CAPABILITY_ID = 'hos_fdb'
CAPABILITY_NAME = 'F&B Management'
CAPABILITY_VERSION = '1.0.0'
CAPABILITY_DOMAIN = 'hospitality'
CAPABILITY_DESCRIPTION = 'Restaurant POS, menu management, table reservations, cost control, inventory'

SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["hos_fdb_manager", "hos_fdb_viewer", "hos_fdb_admin"]

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
    "theme": {"default_theme": "hos_fdb_theme", "allow_tenant_overrides": True},
}

PROVIDES = ['restaurant_pos', 'menu_management', 'table_reservations', 'cost_control', 'fb_inventory']
REQUIRES = ['auth', 'audl', 'fintech_payments', 'scm_inv']

# NATS integration declarations
PUBLISHES = ['order.placed', 'table.reserved', 'inventory.low']
SUBSCRIBES = []

UI_ROUTES = [{'name': 'dashboard', 'path': '/hos-fdb/dashboard', 'component': 'HosFdbDashboard', 'permission': 'hos_fdb:view', 'nav_group': 'Overview'}, {'name': 'list', 'path': '/hos-fdb/list', 'component': 'HosFdbList', 'permission': 'hos_fdb:view', 'nav_group': 'Overview'}, {'name': 'settings', 'path': '/hos-fdb/settings', 'component': 'HosFdbSettings', 'permission': 'hos_fdb:admin', 'nav_group': 'Administration'}]

THEME = {
    "name": "hos_fdb_theme",
    "tokens": {
        "color.primary": '#7B2D00',
        "color.accent": '#F4A261',
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
    """Evaluate APG capability rules for F&B Management."""
    reasons: list[str] = []
    actions: list[dict[str, Any]] = []

    if not context.get("tenant_context_present"):
        reasons.append("tenant_id required")
        actions.append({"type": "deny", "reason": "missing_tenant_context"})
        return {"decision": "deny", "matched_rules": ["tenant_required"], "actions": actions}

    return {"decision": "allow", "matched_rules": [], "actions": []}
