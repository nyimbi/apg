"""Executable capability contract for APG Programme & Project Monitoring."""
from __future__ import annotations
from typing import Any

CAPABILITY_ID = 'ngo_prg'
CAPABILITY_NAME = 'Programme & Project Monitoring'
CAPABILITY_VERSION = '1.0.0'
CAPABILITY_DOMAIN = 'ngo'
CAPABILITY_DESCRIPTION = 'Programme design, activity scheduling, budget tracking, results framework'

SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["ngo_prg_manager", "ngo_prg_viewer", "ngo_prg_admin"]

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
    "theme": {"default_theme": "ngo_prg_theme", "allow_tenant_overrides": True},
}

PROVIDES = ['programme_management', 'activity_scheduling', 'budget_tracking', 'results_framework', 'milestone_tracking']
REQUIRES = ['auth', 'audl', 'ntfy', 'ngo_grn']

# NATS integration declarations
PUBLISHES = ['activity.completed', 'milestone.reached', 'budget.alert']
SUBSCRIBES = []

UI_ROUTES = [{'name': 'dashboard', 'path': '/ngo-prg/dashboard', 'component': 'NgoPrgDashboard', 'permission': 'ngo_prg:view', 'nav_group': 'Overview'}, {'name': 'list', 'path': '/ngo-prg/list', 'component': 'NgoPrgList', 'permission': 'ngo_prg:view', 'nav_group': 'Overview'}, {'name': 'settings', 'path': '/ngo-prg/settings', 'component': 'NgoPrgSettings', 'permission': 'ngo_prg:admin', 'nav_group': 'Administration'}]

THEME = {
    "name": "ngo_prg_theme",
    "tokens": {
        "color.primary": '#264653',
        "color.accent": '#2A9D8F',
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
            "routes": [{'name': 'dashboard', 'path': '/ngo-prg/dashboard', 'component': 'NgoPrgDashboard', 'permission': 'ngo_prg:view', 'nav_group': 'Overview'}, {'name': 'list', 'path': '/ngo-prg/list', 'component': 'NgoPrgList', 'permission': 'ngo_prg:view', 'nav_group': 'Overview'}, {'name': 'settings', 'path': '/ngo-prg/settings', 'component': 'NgoPrgSettings', 'permission': 'ngo_prg:admin', 'nav_group': 'Administration'}],
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
    """Evaluate APG capability rules for Programme & Project Monitoring."""
    reasons: list[str] = []
    actions: list[dict[str, Any]] = []

    if not context.get("tenant_context_present"):
        reasons.append("tenant_id required")
        actions.append({"type": "deny", "reason": "missing_tenant_context"})
        return {"decision": "deny", "matched_rules": ["tenant_required"], "actions": actions}

    return {"decision": "allow", "matched_rules": [], "actions": []}
