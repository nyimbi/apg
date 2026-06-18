"""Executable capability contract for APG Entity & Corporate Secretary."""
from __future__ import annotations
from typing import Any

CAPABILITY_ID = 'leg_ent'
CAPABILITY_NAME = 'Entity & Corporate Secretary'
CAPABILITY_VERSION = '1.0.0'
CAPABILITY_DOMAIN = 'legal'
CAPABILITY_DESCRIPTION = 'Corporate entity registry, shareholder register, board resolutions, filings'

SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["leg_ent_manager", "leg_ent_viewer", "leg_ent_admin"]

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
    "theme": {"default_theme": "leg_ent_theme", "allow_tenant_overrides": True},
}

PROVIDES = ['entity_registry', 'shareholder_register', 'board_resolutions', 'statutory_filings', 'cap_table']
REQUIRES = ['auth', 'audl', 'esig']

# NATS integration declarations
PUBLISHES = ['entity.incorporated', 'resolution.passed', 'filing.submitted']
SUBSCRIBES = []

UI_ROUTES = [{'name': 'dashboard', 'path': '/leg-ent/dashboard', 'component': 'LegEntDashboard', 'permission': 'leg_ent:view', 'nav_group': 'Overview'}, {'name': 'list', 'path': '/leg-ent/list', 'component': 'LegEntList', 'permission': 'leg_ent:view', 'nav_group': 'Overview'}, {'name': 'settings', 'path': '/leg-ent/settings', 'component': 'LegEntSettings', 'permission': 'leg_ent:admin', 'nav_group': 'Administration'}]

THEME = {
    "name": "leg_ent_theme",
    "tokens": {
        "color.primary": '#2F4858',
        "color.accent": '#86BBD8',
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
            "routes": [{'name': 'dashboard', 'path': '/leg-ent/dashboard', 'component': 'LegEntDashboard', 'permission': 'leg_ent:view', 'nav_group': 'Overview'}, {'name': 'list', 'path': '/leg-ent/list', 'component': 'LegEntList', 'permission': 'leg_ent:view', 'nav_group': 'Overview'}, {'name': 'settings', 'path': '/leg-ent/settings', 'component': 'LegEntSettings', 'permission': 'leg_ent:admin', 'nav_group': 'Administration'}],
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
    """Evaluate APG capability rules for Entity & Corporate Secretary."""
    reasons: list[str] = []
    actions: list[dict[str, Any]] = []

    if not context.get("tenant_context_present"):
        reasons.append("tenant_id required")
        actions.append({"type": "deny", "reason": "missing_tenant_context"})
        return {"decision": "deny", "matched_rules": ["tenant_required"], "actions": actions}

    return {"decision": "allow", "matched_rules": [], "actions": []}
