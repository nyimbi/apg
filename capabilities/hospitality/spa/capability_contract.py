"""Executable capability contract for APG Spa & Activities Management."""
from __future__ import annotations
from typing import Any

CAPABILITY_ID = 'hos_spa'
CAPABILITY_NAME = 'Spa & Activities Management'
CAPABILITY_VERSION = '1.0.0'
CAPABILITY_DOMAIN = 'hospitality'
CAPABILITY_DESCRIPTION = 'Spa appointment booking, treatment management, activity scheduling, gift vouchers'

SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["hos_spa_manager", "hos_spa_viewer", "hos_spa_admin"]

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
    "theme": {"default_theme": "hos_spa_theme", "allow_tenant_overrides": True},
}

PROVIDES = ['spa_booking', 'treatment_management', 'activity_scheduling', 'gift_vouchers', 'therapist_management']
REQUIRES = ['auth', 'audl', 'ntfy', 'fintech_payments']

# NATS integration declarations
PUBLISHES = ['appointment.booked', 'treatment.completed']
SUBSCRIBES = []

UI_ROUTES = [{'name': 'dashboard', 'path': '/hos-spa/dashboard', 'component': 'HosSpaDashboard', 'permission': 'hos_spa:view', 'nav_group': 'Overview'}, {'name': 'list', 'path': '/hos-spa/list', 'component': 'HosSpaList', 'permission': 'hos_spa:view', 'nav_group': 'Overview'}, {'name': 'settings', 'path': '/hos-spa/settings', 'component': 'HosSpaSettings', 'permission': 'hos_spa:admin', 'nav_group': 'Administration'}]

THEME = {
    "name": "hos_spa_theme",
    "tokens": {
        "color.primary": '#6B4C82',
        "color.accent": '#C9ADE7',
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
    """Evaluate APG capability rules for Spa & Activities Management."""
    reasons: list[str] = []
    actions: list[dict[str, Any]] = []

    if not context.get("tenant_context_present"):
        reasons.append("tenant_id required")
        actions.append({"type": "deny", "reason": "missing_tenant_context"})
        return {"decision": "deny", "matched_rules": ["tenant_required"], "actions": actions}

    return {"decision": "allow", "matched_rules": [], "actions": []}
