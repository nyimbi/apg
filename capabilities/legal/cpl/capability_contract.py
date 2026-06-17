"""Executable capability contract for APG Legal Compliance Management."""
from __future__ import annotations
from typing import Any

CAPABILITY_ID = 'leg_cpl'
CAPABILITY_NAME = 'Legal Compliance Management'
CAPABILITY_VERSION = '1.0.0'
CAPABILITY_DOMAIN = 'legal'
CAPABILITY_DESCRIPTION = 'Regulatory compliance calendar, obligations register, compliance testing, reporting'

SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["leg_cpl_manager", "leg_cpl_viewer", "leg_cpl_admin"]

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
    "theme": {"default_theme": "leg_cpl_theme", "allow_tenant_overrides": True},
}

PROVIDES = ['compliance_calendar', 'obligations_register', 'compliance_testing', 'compliance_reporting', 'breach_management']
REQUIRES = ['auth', 'audl', 'ntfy', 'grc_pol']

# NATS integration declarations
PUBLISHES = ['obligation.due', 'breach.reported', 'compliance.tested']
SUBSCRIBES = []

UI_ROUTES = [{'name': 'dashboard', 'path': '/leg-cpl/dashboard', 'component': 'LegCplDashboard', 'permission': 'leg_cpl:view', 'nav_group': 'Overview'}, {'name': 'list', 'path': '/leg-cpl/list', 'component': 'LegCplList', 'permission': 'leg_cpl:view', 'nav_group': 'Overview'}, {'name': 'settings', 'path': '/leg-cpl/settings', 'component': 'LegCplSettings', 'permission': 'leg_cpl:admin', 'nav_group': 'Administration'}]

THEME = {
    "name": "leg_cpl_theme",
    "tokens": {
        "color.primary": '#3B1F2B',
        "color.accent": '#F06543',
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
    """Evaluate APG capability rules for Legal Compliance Management."""
    reasons: list[str] = []
    actions: list[dict[str, Any]] = []

    if not context.get("tenant_context_present"):
        reasons.append("tenant_id required")
        actions.append({"type": "deny", "reason": "missing_tenant_context"})
        return {"decision": "deny", "matched_rules": ["tenant_required"], "actions": actions}

    return {"decision": "allow", "matched_rules": [], "actions": []}
