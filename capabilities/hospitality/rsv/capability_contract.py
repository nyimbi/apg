"""Executable capability contract for APG Reservations & Channel Manager."""
from __future__ import annotations
from typing import Any

CAPABILITY_ID = 'hos_rsv'
CAPABILITY_NAME = 'Reservations & Channel Manager'
CAPABILITY_VERSION = '1.0.0'
CAPABILITY_DOMAIN = 'hospitality'
CAPABILITY_DESCRIPTION = 'Booking engine, OTA channel management, group bookings, availability sync'

SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["hos_rsv_manager", "hos_rsv_viewer", "hos_rsv_admin"]

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
    "theme": {"default_theme": "hos_rsv_theme", "allow_tenant_overrides": True},
}

PROVIDES = ['booking_engine', 'channel_management', 'group_bookings', 'availability_sync', 'cancellation_management']
REQUIRES = ['auth', 'audl', 'ntfy', 'hos_pms', 'fintech_payments']

# NATS integration declarations
PUBLISHES = ['reservation.confirmed', 'reservation.cancelled', 'availability.synced']
SUBSCRIBES = []

UI_ROUTES = [{'name': 'dashboard', 'path': '/hos-rsv/dashboard', 'component': 'HosRsvDashboard', 'permission': 'hos_rsv:view', 'nav_group': 'Overview'}, {'name': 'list', 'path': '/hos-rsv/list', 'component': 'HosRsvList', 'permission': 'hos_rsv:view', 'nav_group': 'Overview'}, {'name': 'settings', 'path': '/hos-rsv/settings', 'component': 'HosRsvSettings', 'permission': 'hos_rsv:admin', 'nav_group': 'Administration'}]

THEME = {
    "name": "hos_rsv_theme",
    "tokens": {
        "color.primary": '#0077B6',
        "color.accent": '#00B4D8',
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
    
        "rule_engine": {
            "type": "deterministic",
            "default_decision": "deny",
            "rules": [
                {"name": "tenant_required", "condition": {"tenant_context_present": True}, "effect": {"decision": "allow"}},
                {"name": "write_policy", "condition": {"write_requires_policy": True}, "effect": {"decision": "allow"}},
                {"name": "cross_tenant_denied", "condition": {"cross_tenant_access": "cross_tenant"}, "effect": {"decision": "deny"}},
            ],
        },
        "ui": {
            "shell": "apg_python",
            "requires_theme": True,
            "template_roots": ["templates"],
            "routes": [{'name': 'dashboard', 'path': '/hos-rsv/dashboard', 'component': 'HosRsvDashboard', 'permission': 'hos_rsv:view', 'nav_group': 'Overview'}, {'name': 'list', 'path': '/hos-rsv/list', 'component': 'HosRsvList', 'permission': 'hos_rsv:view', 'nav_group': 'Overview'}, {'name': 'settings', 'path': '/hos-rsv/settings', 'component': 'HosRsvSettings', 'permission': 'hos_rsv:admin', 'nav_group': 'Administration'}],
        },
        "configuration_schema": {
            "type": "object",
            "required": ['tenant_id'],
            "properties": {
                "tenant_id": {"type": "string"},
                "governance": {"type": "object"},
                "agents": {"type": "object"},
                "theme": {"type": "object"},
            },
        },
}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
    """Evaluate APG capability rules for Reservations & Channel Manager."""
    reasons: list[str] = []
    actions: list[dict[str, Any]] = []

    if not context.get("tenant_context_present"):
        reasons.append("tenant_id required")
        actions.append({"type": "deny", "reason": "missing_tenant_context"})
        return {"decision": "deny", "matched_rules": ["tenant_required"], "actions": actions}

    return {"decision": "allow", "matched_rules": [], "actions": []}
