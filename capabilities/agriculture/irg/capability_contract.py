"""Executable capability contract for APG Irrigation Management."""
from __future__ import annotations
from typing import Any

CAPABILITY_ID = 'agr_irg'
CAPABILITY_NAME = 'Irrigation Management'
CAPABILITY_VERSION = '1.0.0'
CAPABILITY_DOMAIN = 'agriculture'
CAPABILITY_DESCRIPTION = 'Sensor integration, irrigation schedule optimisation, water usage tracking'

SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["agr_irg_manager", "agr_irg_viewer", "agr_irg_admin"]

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
    "theme": {"default_theme": "agr_irg_theme", "allow_tenant_overrides": True},
}

PROVIDES = ['irrigation_scheduling', 'water_usage_tracking', 'sensor_integration', 'schedule_optimisation']
REQUIRES = ['auth', 'audl', 'agr_iot', 'agr_crp']

# NATS integration declarations
PUBLISHES = ['irrigation.scheduled', 'irrigation.completed', 'water_alert.triggered']
SUBSCRIBES = []

UI_ROUTES = [{'name': 'dashboard', 'path': '/agr-irg/dashboard', 'component': 'AgrIrgDashboard', 'permission': 'agr_irg:view', 'nav_group': 'Overview'}, {'name': 'list', 'path': '/agr-irg/list', 'component': 'AgrIrgList', 'permission': 'agr_irg:view', 'nav_group': 'Overview'}, {'name': 'settings', 'path': '/agr-irg/settings', 'component': 'AgrIrgSettings', 'permission': 'agr_irg:admin', 'nav_group': 'Administration'}]

THEME = {
    "name": "agr_irg_theme",
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
    """Evaluate APG capability rules for Irrigation Management."""
    reasons: list[str] = []
    actions: list[dict[str, Any]] = []

    if not context.get("tenant_context_present"):
        reasons.append("tenant_id required")
        actions.append({"type": "deny", "reason": "missing_tenant_context"})
        return {"decision": "deny", "matched_rules": ["tenant_required"], "actions": actions}

    return {"decision": "allow", "matched_rules": [], "actions": []}
