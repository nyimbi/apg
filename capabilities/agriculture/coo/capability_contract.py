"""Executable capability contract for APG Cooperative Management."""
from __future__ import annotations
from typing import Any

CAPABILITY_ID = 'agr_coo'
CAPABILITY_NAME = 'Cooperative Management'
CAPABILITY_VERSION = '1.0.0'
CAPABILITY_DOMAIN = 'agriculture'
CAPABILITY_DESCRIPTION = 'Member registry, pooled inputs, bulk sales, dividend distribution'

SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["agr_coo_manager", "agr_coo_viewer", "agr_coo_admin"]

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
    "theme": {"default_theme": "agr_coo_theme", "allow_tenant_overrides": True},
}

PROVIDES = ['member_management', 'pooled_input_procurement', 'bulk_sales', 'dividend_distribution', 'cooperative_accounting']
REQUIRES = ['auth', 'audl', 'ntfy', 'fin_gl']

# NATS integration declarations
PUBLISHES = ['member.joined', 'dividend.declared', 'bulk_sale.completed']
SUBSCRIBES = []

UI_ROUTES = [{'name': 'dashboard', 'path': '/agr-coo/dashboard', 'component': 'AgrCooDashboard', 'permission': 'agr_coo:view', 'nav_group': 'Overview'}, {'name': 'list', 'path': '/agr-coo/list', 'component': 'AgrCooList', 'permission': 'agr_coo:view', 'nav_group': 'Overview'}, {'name': 'settings', 'path': '/agr-coo/settings', 'component': 'AgrCooSettings', 'permission': 'agr_coo:admin', 'nav_group': 'Administration'}]

THEME = {
    "name": "agr_coo_theme",
    "tokens": {
        "color.primary": '#3D405B',
        "color.accent": '#81B29A',
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
    """Evaluate APG capability rules for Cooperative Management."""
    reasons: list[str] = []
    actions: list[dict[str, Any]] = []

    if not context.get("tenant_context_present"):
        reasons.append("tenant_id required")
        actions.append({"type": "deny", "reason": "missing_tenant_context"})
        return {"decision": "deny", "matched_rules": ["tenant_required"], "actions": actions}

    return {"decision": "allow", "matched_rules": [], "actions": []}
