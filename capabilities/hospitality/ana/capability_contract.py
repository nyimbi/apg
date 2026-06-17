"""Executable capability contract for APG Hospitality Analytics."""
from __future__ import annotations
from typing import Any

CAPABILITY_ID = 'hos_ana'
CAPABILITY_NAME = 'Hospitality Analytics'
CAPABILITY_VERSION = '1.0.0'
CAPABILITY_DOMAIN = 'hospitality'
CAPABILITY_DESCRIPTION = 'RevPAR, occupancy analytics, guest satisfaction, competitor benchmarking'

SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["hos_ana_manager", "hos_ana_viewer", "hos_ana_admin"]

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
    "theme": {"default_theme": "hos_ana_theme", "allow_tenant_overrides": True},
}

PROVIDES = ['revpar_analytics', 'occupancy_reporting', 'guest_satisfaction', 'competitor_benchmarking', 'forecast_reporting']
REQUIRES = ['auth', 'audl', 'hos_pms', 'hos_rms', 'bia_anl']

# NATS integration declarations
PUBLISHES = ['report.generated', 'benchmark.updated']
SUBSCRIBES = []

UI_ROUTES = [{'name': 'dashboard', 'path': '/hos-ana/dashboard', 'component': 'HosAnaDashboard', 'permission': 'hos_ana:view', 'nav_group': 'Overview'}, {'name': 'list', 'path': '/hos-ana/list', 'component': 'HosAnaList', 'permission': 'hos_ana:view', 'nav_group': 'Overview'}, {'name': 'settings', 'path': '/hos-ana/settings', 'component': 'HosAnaSettings', 'permission': 'hos_ana:admin', 'nav_group': 'Administration'}]

THEME = {
    "name": "hos_ana_theme",
    "tokens": {
        "color.primary": '#2C3E50',
        "color.accent": '#3498DB',
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
    """Evaluate APG capability rules for Hospitality Analytics."""
    reasons: list[str] = []
    actions: list[dict[str, Any]] = []

    if not context.get("tenant_context_present"):
        reasons.append("tenant_id required")
        actions.append({"type": "deny", "reason": "missing_tenant_context"})
        return {"decision": "deny", "matched_rules": ["tenant_required"], "actions": actions}

    return {"decision": "allow", "matched_rules": [], "actions": []}
