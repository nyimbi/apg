"""Executable capability contract for APG Intellectual Property Registry."""
from __future__ import annotations
from typing import Any

CAPABILITY_ID = 'leg_ip'
CAPABILITY_NAME = 'Intellectual Property Registry'
CAPABILITY_VERSION = '1.0.0'
CAPABILITY_DOMAIN = 'legal'
CAPABILITY_DESCRIPTION = 'Patent, trademark, copyright and design registrations, renewal management'

SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["leg_ip_manager", "leg_ip_viewer", "leg_ip_admin"]

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
    "theme": {"default_theme": "leg_ip_theme", "allow_tenant_overrides": True},
}

PROVIDES = ['ip_registration', 'renewal_management', 'ip_portfolio', 'infringement_tracking', 'ip_valuation']
REQUIRES = ['auth', 'audl', 'ntfy']

# NATS integration declarations
PUBLISHES = ['ip.registered', 'ip.renewal_due', 'infringement.reported']
SUBSCRIBES = []

UI_ROUTES = [{'name': 'dashboard', 'path': '/leg-ip/dashboard', 'component': 'LegIpDashboard', 'permission': 'leg_ip:view', 'nav_group': 'Overview'}, {'name': 'list', 'path': '/leg-ip/list', 'component': 'LegIpList', 'permission': 'leg_ip:view', 'nav_group': 'Overview'}, {'name': 'settings', 'path': '/leg-ip/settings', 'component': 'LegIpSettings', 'permission': 'leg_ip:admin', 'nav_group': 'Administration'}]

THEME = {
    "name": "leg_ip_theme",
    "tokens": {
        "color.primary": '#4A0E8F',
        "color.accent": '#9B5DE5',
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
            "routes": [{'name': 'dashboard', 'path': '/leg-ip/dashboard', 'component': 'LegIpDashboard', 'permission': 'leg_ip:view', 'nav_group': 'Overview'}, {'name': 'list', 'path': '/leg-ip/list', 'component': 'LegIpList', 'permission': 'leg_ip:view', 'nav_group': 'Overview'}, {'name': 'settings', 'path': '/leg-ip/settings', 'component': 'LegIpSettings', 'permission': 'leg_ip:admin', 'nav_group': 'Administration'}],
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
    """Evaluate APG capability rules for Intellectual Property Registry."""
    reasons: list[str] = []
    actions: list[dict[str, Any]] = []

    if not context.get("tenant_context_present"):
        reasons.append("tenant_id required")
        actions.append({"type": "deny", "reason": "missing_tenant_context"})
        return {"decision": "deny", "matched_rules": ["tenant_required"], "actions": actions}

    return {"decision": "allow", "matched_rules": [], "actions": []}
