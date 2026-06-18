"""Executable capability contract for APG M&E — Monitoring & Evaluation."""
from __future__ import annotations
from typing import Any

CAPABILITY_ID = 'ngo_me'
CAPABILITY_NAME = 'M&E — Monitoring & Evaluation'
CAPABILITY_VERSION = '1.0.0'
CAPABILITY_DOMAIN = 'ngo'
CAPABILITY_DESCRIPTION = 'Indicator tracking, data collection, evaluation frameworks, impact reporting'

SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["ngo_me_manager", "ngo_me_viewer", "ngo_me_admin"]

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
    "theme": {"default_theme": "ngo_me_theme", "allow_tenant_overrides": True},
}

PROVIDES = ['indicator_tracking', 'data_collection', 'evaluation_management', 'impact_reporting', 'log_frame']
REQUIRES = ['auth', 'audl', 'ngo_prg', 'bia_anl']

# NATS integration declarations
PUBLISHES = ['indicator.updated', 'evaluation.completed', 'impact.reported']
SUBSCRIBES = []

UI_ROUTES = [{'name': 'dashboard', 'path': '/ngo-me/dashboard', 'component': 'NgoMeDashboard', 'permission': 'ngo_me:view', 'nav_group': 'Overview'}, {'name': 'list', 'path': '/ngo-me/list', 'component': 'NgoMeList', 'permission': 'ngo_me:view', 'nav_group': 'Overview'}, {'name': 'settings', 'path': '/ngo-me/settings', 'component': 'NgoMeSettings', 'permission': 'ngo_me:admin', 'nav_group': 'Administration'}]

THEME = {
    "name": "ngo_me_theme",
    "tokens": {
        "color.primary": '#355070',
        "color.accent": '#6D9DC5',
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
            "routes": [{'name': 'dashboard', 'path': '/ngo-me/dashboard', 'component': 'NgoMeDashboard', 'permission': 'ngo_me:view', 'nav_group': 'Overview'}, {'name': 'list', 'path': '/ngo-me/list', 'component': 'NgoMeList', 'permission': 'ngo_me:view', 'nav_group': 'Overview'}, {'name': 'settings', 'path': '/ngo-me/settings', 'component': 'NgoMeSettings', 'permission': 'ngo_me:admin', 'nav_group': 'Administration'}],
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
    """Evaluate APG capability rules for M&E — Monitoring & Evaluation."""
    reasons: list[str] = []
    actions: list[dict[str, Any]] = []

    if not context.get("tenant_context_present"):
        reasons.append("tenant_id required")
        actions.append({"type": "deny", "reason": "missing_tenant_context"})
        return {"decision": "deny", "matched_rules": ["tenant_required"], "actions": actions}

    return {"decision": "allow", "matched_rules": [], "actions": []}
