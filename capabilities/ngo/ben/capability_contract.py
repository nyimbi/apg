"""Executable capability contract for APG Beneficiary Registry."""
from __future__ import annotations
from typing import Any

CAPABILITY_ID = 'ngo_ben'
CAPABILITY_NAME = 'Beneficiary Registry'
CAPABILITY_VERSION = '1.0.0'
CAPABILITY_DOMAIN = 'ngo'
CAPABILITY_DESCRIPTION = 'Beneficiary registration, vulnerability assessment, case management, deduplication'

SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["ngo_ben_manager", "ngo_ben_viewer", "ngo_ben_admin"]

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
    "theme": {"default_theme": "ngo_ben_theme", "allow_tenant_overrides": True},
}

PROVIDES = ['beneficiary_registration', 'vulnerability_assessment', 'case_management', 'deduplication', 'beneficiary_analytics']
REQUIRES = ['auth', 'audl', 'mdm']

# NATS integration declarations
PUBLISHES = ['beneficiary.registered', 'case.opened', 'service.delivered']
SUBSCRIBES = []

UI_ROUTES = [{'name': 'dashboard', 'path': '/ngo-ben/dashboard', 'component': 'NgoBenDashboard', 'permission': 'ngo_ben:view', 'nav_group': 'Overview'}, {'name': 'list', 'path': '/ngo-ben/list', 'component': 'NgoBenList', 'permission': 'ngo_ben:view', 'nav_group': 'Overview'}, {'name': 'settings', 'path': '/ngo-ben/settings', 'component': 'NgoBenSettings', 'permission': 'ngo_ben:admin', 'nav_group': 'Administration'}]

THEME = {
    "name": "ngo_ben_theme",
    "tokens": {
        "color.primary": '#4A4E69',
        "color.accent": '#9A8C98',
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
            "routes": [{'name': 'dashboard', 'path': '/ngo-ben/dashboard', 'component': 'NgoBenDashboard', 'permission': 'ngo_ben:view', 'nav_group': 'Overview'}, {'name': 'list', 'path': '/ngo-ben/list', 'component': 'NgoBenList', 'permission': 'ngo_ben:view', 'nav_group': 'Overview'}, {'name': 'settings', 'path': '/ngo-ben/settings', 'component': 'NgoBenSettings', 'permission': 'ngo_ben:admin', 'nav_group': 'Administration'}],
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
    """Evaluate APG capability rules for Beneficiary Registry."""
    reasons: list[str] = []
    actions: list[dict[str, Any]] = []

    if not context.get("tenant_context_present"):
        reasons.append("tenant_id required")
        actions.append({"type": "deny", "reason": "missing_tenant_context"})
        return {"decision": "deny", "matched_rules": ["tenant_required"], "actions": actions}

    return {"decision": "allow", "matched_rules": [], "actions": []}
