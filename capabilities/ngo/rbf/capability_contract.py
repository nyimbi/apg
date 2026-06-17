"""Executable capability contract for APG Results-Based Financing."""
from __future__ import annotations
from typing import Any

CAPABILITY_ID = 'ngo_rbf'
CAPABILITY_NAME = 'Results-Based Financing'
CAPABILITY_VERSION = '1.0.0'
CAPABILITY_DOMAIN = 'ngo'
CAPABILITY_DESCRIPTION = 'RBF contract management, result verification, disbursement triggers, World Bank/USAID compliance'

SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["ngo_rbf_manager", "ngo_rbf_viewer", "ngo_rbf_admin"]

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
    "theme": {"default_theme": "ngo_rbf_theme", "allow_tenant_overrides": True},
}

PROVIDES = ['rbf_contract_management', 'result_verification', 'disbursement_triggers', 'compliance_reporting', 'independent_verification']
REQUIRES = ['auth', 'audl', 'ngo_me', 'ngo_grn', 'fintech_payments']

# NATS integration declarations
PUBLISHES = ['result.verified', 'disbursement.triggered', 'rbf_report.submitted']
SUBSCRIBES = []

UI_ROUTES = [{'name': 'dashboard', 'path': '/ngo-rbf/dashboard', 'component': 'NgoRbfDashboard', 'permission': 'ngo_rbf:view', 'nav_group': 'Overview'}, {'name': 'list', 'path': '/ngo-rbf/list', 'component': 'NgoRbfList', 'permission': 'ngo_rbf:view', 'nav_group': 'Overview'}, {'name': 'settings', 'path': '/ngo-rbf/settings', 'component': 'NgoRbfSettings', 'permission': 'ngo_rbf:admin', 'nav_group': 'Administration'}]

THEME = {
    "name": "ngo_rbf_theme",
    "tokens": {
        "color.primary": '#0A2342',
        "color.accent": '#2CA58D',
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
    """Evaluate APG capability rules for Results-Based Financing."""
    reasons: list[str] = []
    actions: list[dict[str, Any]] = []

    if not context.get("tenant_context_present"):
        reasons.append("tenant_id required")
        actions.append({"type": "deny", "reason": "missing_tenant_context"})
        return {"decision": "deny", "matched_rules": ["tenant_required"], "actions": actions}

    return {"decision": "allow", "matched_rules": [], "actions": []}
