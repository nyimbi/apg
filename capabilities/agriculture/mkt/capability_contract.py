"""Executable capability contract for APG Agri-Marketplace."""
from __future__ import annotations
from typing import Any

CAPABILITY_ID = 'agr_mkt'
CAPABILITY_NAME = 'Agri-Marketplace'
CAPABILITY_VERSION = '1.0.0'
CAPABILITY_DOMAIN = 'agriculture'
CAPABILITY_DESCRIPTION = 'Farmer listing, price discovery, escrow payments, buyer matching'

SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["agr_mkt_manager", "agr_mkt_viewer", "agr_mkt_admin"]

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
    "theme": {"default_theme": "agr_mkt_theme", "allow_tenant_overrides": True},
}

PROVIDES = ['produce_listing', 'price_discovery', 'buyer_matching', 'escrow_payments', 'market_analytics']
REQUIRES = ['auth', 'audl', 'ntfy', 'fintech_payments']

# NATS integration declarations
PUBLISHES = ['listing.created', 'bid.accepted', 'payment.escrowed']
SUBSCRIBES = []

UI_ROUTES = [{'name': 'dashboard', 'path': '/agr-mkt/dashboard', 'component': 'AgrMktDashboard', 'permission': 'agr_mkt:view', 'nav_group': 'Overview'}, {'name': 'list', 'path': '/agr-mkt/list', 'component': 'AgrMktList', 'permission': 'agr_mkt:view', 'nav_group': 'Overview'}, {'name': 'settings', 'path': '/agr-mkt/settings', 'component': 'AgrMktSettings', 'permission': 'agr_mkt:admin', 'nav_group': 'Administration'}]

THEME = {
    "name": "agr_mkt_theme",
    "tokens": {
        "color.primary": '#E9C46A',
        "color.accent": '#F4A261',
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
            "routes": [{'name': 'dashboard', 'path': '/agr-mkt/dashboard', 'component': 'AgrMktDashboard', 'permission': 'agr_mkt:view', 'nav_group': 'Overview'}, {'name': 'list', 'path': '/agr-mkt/list', 'component': 'AgrMktList', 'permission': 'agr_mkt:view', 'nav_group': 'Overview'}, {'name': 'settings', 'path': '/agr-mkt/settings', 'component': 'AgrMktSettings', 'permission': 'agr_mkt:admin', 'nav_group': 'Administration'}],
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
    """Evaluate APG capability rules for Agri-Marketplace."""
    reasons: list[str] = []
    actions: list[dict[str, Any]] = []

    if not context.get("tenant_context_present"):
        reasons.append("tenant_id required")
        actions.append({"type": "deny", "reason": "missing_tenant_context"})
        return {"decision": "deny", "matched_rules": ["tenant_required"], "actions": actions}

    return {"decision": "allow", "matched_rules": [], "actions": []}
