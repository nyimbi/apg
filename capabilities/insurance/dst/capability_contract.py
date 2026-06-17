"""Executable capability contract for APG Distribution & Agency Management."""
from __future__ import annotations
from typing import Any

CAPABILITY_ID = 'ins_dst'
CAPABILITY_NAME = 'Distribution & Agency Management'
CAPABILITY_VERSION = '1.0.0'
CAPABILITY_DOMAIN = 'insurance'
CAPABILITY_DESCRIPTION = 'Agent onboarding, commission tracking, performance management, licensing'

SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["ins_dst_manager", "ins_dst_viewer", "ins_dst_admin"]

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
    "theme": {"default_theme": "ins_dst_theme", "allow_tenant_overrides": True},
}

PROVIDES = ['agent_management', 'commission_tracking', 'performance_reporting', 'licensing_registry', 'channel_management']
REQUIRES = ['auth', 'audl', 'ntfy', 'fin_gl']

# NATS integration declarations
PUBLISHES = ['agent.onboarded', 'commission.calculated', 'licence.renewed']
SUBSCRIBES = []

UI_ROUTES = [{'name': 'dashboard', 'path': '/ins-dst/dashboard', 'component': 'InsDstDashboard', 'permission': 'ins_dst:view', 'nav_group': 'Overview'}, {'name': 'list', 'path': '/ins-dst/list', 'component': 'InsDstList', 'permission': 'ins_dst:view', 'nav_group': 'Overview'}, {'name': 'settings', 'path': '/ins-dst/settings', 'component': 'InsDstSettings', 'permission': 'ins_dst:admin', 'nav_group': 'Administration'}]

THEME = {
    "name": "ins_dst_theme",
    "tokens": {
        "color.primary": '#344E41',
        "color.accent": '#A3B18A',
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
    """Evaluate APG capability rules for Distribution & Agency Management."""
    reasons: list[str] = []
    actions: list[dict[str, Any]] = []

    if not context.get("tenant_context_present"):
        reasons.append("tenant_id required")
        actions.append({"type": "deny", "reason": "missing_tenant_context"})
        return {"decision": "deny", "matched_rules": ["tenant_required"], "actions": actions}

    return {"decision": "allow", "matched_rules": [], "actions": []}
