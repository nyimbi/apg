"""Executable capability contract for APG Donor Relationship Management."""
from __future__ import annotations
from typing import Any

CAPABILITY_ID = 'ngo_don'
CAPABILITY_NAME = 'Donor Relationship Management'
CAPABILITY_VERSION = '1.0.0'
CAPABILITY_DOMAIN = 'ngo'
CAPABILITY_DESCRIPTION = 'Donor profiles, giving history, stewardship activities, pledge tracking'

SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["ngo_don_manager", "ngo_don_viewer", "ngo_don_admin"]

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
    "theme": {"default_theme": "ngo_don_theme", "allow_tenant_overrides": True},
}

PROVIDES = ['donor_profiles', 'giving_history', 'pledge_management', 'stewardship_activities', 'donation_receipting']
REQUIRES = ['auth', 'audl', 'ntfy', 'crm_adv']

# NATS integration declarations
PUBLISHES = ['donation.received', 'pledge.created', 'stewardship.completed']
SUBSCRIBES = []

UI_ROUTES = [{'name': 'dashboard', 'path': '/ngo-don/dashboard', 'component': 'NgoDonDashboard', 'permission': 'ngo_don:view', 'nav_group': 'Overview'}, {'name': 'list', 'path': '/ngo-don/list', 'component': 'NgoDonList', 'permission': 'ngo_don:view', 'nav_group': 'Overview'}, {'name': 'settings', 'path': '/ngo-don/settings', 'component': 'NgoDonSettings', 'permission': 'ngo_don:admin', 'nav_group': 'Administration'}]

THEME = {
    "name": "ngo_don_theme",
    "tokens": {
        "color.primary": '#2C6E49',
        "color.accent": '#4C956C',
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
    """Evaluate APG capability rules for Donor Relationship Management."""
    reasons: list[str] = []
    actions: list[dict[str, Any]] = []

    if not context.get("tenant_context_present"):
        reasons.append("tenant_id required")
        actions.append({"type": "deny", "reason": "missing_tenant_context"})
        return {"decision": "deny", "matched_rules": ["tenant_required"], "actions": actions}

    return {"decision": "allow", "matched_rules": [], "actions": []}
