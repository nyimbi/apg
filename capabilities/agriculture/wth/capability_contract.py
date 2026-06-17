"""Executable capability contract for APG Weather & Climate Analytics."""
from __future__ import annotations
from typing import Any

CAPABILITY_ID = 'agr_wth'
CAPABILITY_NAME = 'Weather & Climate Analytics'
CAPABILITY_VERSION = '1.0.0'
CAPABILITY_DOMAIN = 'agriculture'
CAPABILITY_DESCRIPTION = 'Weather forecast integration, climate data analysis, agri-alert thresholds'

SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["agr_wth_manager", "agr_wth_viewer", "agr_wth_admin"]

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
    "theme": {"default_theme": "agr_wth_theme", "allow_tenant_overrides": True},
}

PROVIDES = ['weather_forecast', 'climate_analytics', 'agri_alerts', 'temperature_monitoring', 'rainfall_tracking']
REQUIRES = ['auth', 'audl', 'ntfy']

# NATS integration declarations
PUBLISHES = ['weather.alert_triggered', 'frost.warning', 'drought.alert']
SUBSCRIBES = []

UI_ROUTES = [{'name': 'dashboard', 'path': '/agr-wth/dashboard', 'component': 'AgrWthDashboard', 'permission': 'agr_wth:view', 'nav_group': 'Overview'}, {'name': 'list', 'path': '/agr-wth/list', 'component': 'AgrWthList', 'permission': 'agr_wth:view', 'nav_group': 'Overview'}, {'name': 'settings', 'path': '/agr-wth/settings', 'component': 'AgrWthSettings', 'permission': 'agr_wth:admin', 'nav_group': 'Administration'}]

THEME = {
    "name": "agr_wth_theme",
    "tokens": {
        "color.primary": '#023E8A',
        "color.accent": '#48CAE4',
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
    """Evaluate APG capability rules for Weather & Climate Analytics."""
    reasons: list[str] = []
    actions: list[dict[str, Any]] = []

    if not context.get("tenant_context_present"):
        reasons.append("tenant_id required")
        actions.append({"type": "deny", "reason": "missing_tenant_context"})
        return {"decision": "deny", "matched_rules": ["tenant_required"], "actions": actions}

    return {"decision": "allow", "matched_rules": [], "actions": []}
