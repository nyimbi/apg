"""Executable capability contract for APG Property Management System."""
from __future__ import annotations
from typing import Any

CAPABILITY_ID = 'hos_pms'
CAPABILITY_NAME = 'Property Management System'
CAPABILITY_VERSION = '1.0.0'
CAPABILITY_DOMAIN = 'hospitality'
CAPABILITY_DESCRIPTION = 'Room inventory, check-in/check-out, housekeeping, folio management'

SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["hos_pms_manager", "hos_pms_viewer", "hos_pms_admin"]

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
    "theme": {"default_theme": "hos_pms_theme", "allow_tenant_overrides": True},
}

PROVIDES = ['room_management', 'check_in_out', 'housekeeping_management', 'folio_management', 'room_assignment']
REQUIRES = ['auth', 'audl', 'ntfy', 'fintech_payments']

# NATS integration declarations
PUBLISHES = ['guest.checked_in', 'guest.checked_out', 'room.cleaned']
SUBSCRIBES = [{'source_capability': 'hos_rsv', 'event_type': 'reservation.confirmed', 'handler': 'on_reservation_confirmed'}]

UI_ROUTES = [{'name': 'dashboard', 'path': '/hos-pms/dashboard', 'component': 'HosPmsDashboard', 'permission': 'hos_pms:view', 'nav_group': 'Overview'}, {'name': 'list', 'path': '/hos-pms/list', 'component': 'HosPmsList', 'permission': 'hos_pms:view', 'nav_group': 'Overview'}, {'name': 'settings', 'path': '/hos-pms/settings', 'component': 'HosPmsSettings', 'permission': 'hos_pms:admin', 'nav_group': 'Administration'}]

THEME = {
    "name": "hos_pms_theme",
    "tokens": {
        "color.primary": '#8B1A1A',
        "color.accent": '#D4A017',
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
    """Evaluate APG capability rules for Property Management System."""
    reasons: list[str] = []
    actions: list[dict[str, Any]] = []

    if not context.get("tenant_context_present"):
        reasons.append("tenant_id required")
        actions.append({"type": "deny", "reason": "missing_tenant_context"})
        return {"decision": "deny", "matched_rules": ["tenant_required"], "actions": actions}

    return {"decision": "allow", "matched_rules": [], "actions": []}
