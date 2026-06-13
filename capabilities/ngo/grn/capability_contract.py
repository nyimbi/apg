"""Executable capability contract for APG Grant Management."""
from __future__ import annotations
from typing import Any

CAPABILITY_ID = 'ngo_grn'
CAPABILITY_NAME = 'Grant Management'
CAPABILITY_VERSION = '1.0.0'
CAPABILITY_DOMAIN = 'ngo'
CAPABILITY_DESCRIPTION = 'Grant application, award management, disbursement tracking, donor reporting'

SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["ngo_grn_manager", "ngo_grn_viewer", "ngo_grn_admin"]

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
    "theme": {"default_theme": "ngo_grn_theme", "allow_tenant_overrides": True},
}

PROVIDES = ['grant_application', 'award_management', 'disbursement_tracking', 'donor_reporting', 'grant_compliance']
REQUIRES = ['auth', 'audl', 'ntfy', 'fin_gl']

# NATS integration declarations
PUBLISHES = ['grant.awarded', 'disbursement.processed', 'report.submitted']
SUBSCRIBES = []

UI_ROUTES = [{'name': 'dashboard', 'path': '/ngo-grn/dashboard', 'component': 'NgoGrnDashboard', 'permission': 'ngo_grn:view', 'nav_group': 'Overview'}, {'name': 'list', 'path': '/ngo-grn/list', 'component': 'NgoGrnList', 'permission': 'ngo_grn:view', 'nav_group': 'Overview'}, {'name': 'settings', 'path': '/ngo-grn/settings', 'component': 'NgoGrnSettings', 'permission': 'ngo_grn:admin', 'nav_group': 'Administration'}]

THEME = {
    "name": "ngo_grn_theme",
    "tokens": {
        "color.primary": '#1D4E89',
        "color.accent": '#48A9A6',
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
    """Evaluate APG capability rules for Grant Management."""
    reasons: list[str] = []
    actions: list[dict[str, Any]] = []

    if not context.get("tenant_context_present"):
        reasons.append("tenant_id required")
        actions.append({"type": "deny", "reason": "missing_tenant_context"})
        return {"decision": "deny", "matched_rules": ["tenant_required"], "actions": actions}

    return {"decision": "allow", "matched_rules": [], "actions": []}
