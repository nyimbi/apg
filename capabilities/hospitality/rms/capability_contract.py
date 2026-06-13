"""Executable capability contract for APG Revenue Management & Rates."""
from __future__ import annotations
from typing import Any

CAPABILITY_ID = 'hos_rms'
CAPABILITY_NAME = 'Revenue Management & Rates'
CAPABILITY_VERSION = '1.0.0'
CAPABILITY_DOMAIN = 'hospitality'
CAPABILITY_DESCRIPTION = 'Dynamic pricing, demand forecasting, rate management, channel rate parity'

SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["hos_rms_manager", "hos_rms_viewer", "hos_rms_admin"]

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
    "theme": {"default_theme": "hos_rms_theme", "allow_tenant_overrides": True},
}

PROVIDES = ['dynamic_pricing', 'demand_forecasting', 'rate_management', 'rate_parity', 'yield_optimisation']
REQUIRES = ['auth', 'audl', 'hos_pms', 'mlx']

# NATS integration declarations
PUBLISHES = ['rate.updated', 'demand_forecast.generated']
SUBSCRIBES = []

UI_ROUTES = [{'name': 'dashboard', 'path': '/hos-rms/dashboard', 'component': 'HosRmsDashboard', 'permission': 'hos_rms:view', 'nav_group': 'Overview'}, {'name': 'list', 'path': '/hos-rms/list', 'component': 'HosRmsList', 'permission': 'hos_rms:view', 'nav_group': 'Overview'}, {'name': 'settings', 'path': '/hos-rms/settings', 'component': 'HosRmsSettings', 'permission': 'hos_rms:admin', 'nav_group': 'Administration'}]

THEME = {
    "name": "hos_rms_theme",
    "tokens": {
        "color.primary": '#1A472A',
        "color.accent": '#2ECC71',
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
    """Evaluate APG capability rules for Revenue Management & Rates."""
    reasons: list[str] = []
    actions: list[dict[str, Any]] = []

    if not context.get("tenant_context_present"):
        reasons.append("tenant_id required")
        actions.append({"type": "deny", "reason": "missing_tenant_context"})
        return {"decision": "deny", "matched_rules": ["tenant_required"], "actions": actions}

    return {"decision": "allow", "matched_rules": [], "actions": []}
