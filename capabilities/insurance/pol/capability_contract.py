"""Executable capability contract for APG Policy Administration."""
from __future__ import annotations
from typing import Any

CAPABILITY_ID = 'ins_pol'
CAPABILITY_NAME = 'Policy Administration'
CAPABILITY_VERSION = '1.0.0'
CAPABILITY_DOMAIN = 'insurance'
CAPABILITY_DESCRIPTION = 'Policy lifecycle management, endorsements, renewals, cancellations'

SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["ins_pol_manager", "ins_pol_viewer", "ins_pol_admin"]

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
    "theme": {"default_theme": "ins_pol_theme", "allow_tenant_overrides": True},
}

PROVIDES = ['policy_issuance', 'endorsement_management', 'renewal_processing', 'cancellation_management', 'policy_inquiry']
REQUIRES = ['auth', 'audl', 'ntfy', 'ins_und', 'fintech_payments']

# NATS integration declarations
PUBLISHES = ['policy.issued', 'policy.renewed', 'policy.cancelled']
SUBSCRIBES = []

UI_ROUTES = [{'name': 'dashboard', 'path': '/ins-pol/dashboard', 'component': 'InsPolDashboard', 'permission': 'ins_pol:view', 'nav_group': 'Overview'}, {'name': 'list', 'path': '/ins-pol/list', 'component': 'InsPolList', 'permission': 'ins_pol:view', 'nav_group': 'Overview'}, {'name': 'settings', 'path': '/ins-pol/settings', 'component': 'InsPolSettings', 'permission': 'ins_pol:admin', 'nav_group': 'Administration'}]

THEME = {
    "name": "ins_pol_theme",
    "tokens": {
        "color.primary": '#1D3557',
        "color.accent": '#457B9D',
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
    """Evaluate APG capability rules for Policy Administration."""
    reasons: list[str] = []
    actions: list[dict[str, Any]] = []

    if not context.get("tenant_context_present"):
        reasons.append("tenant_id required")
        actions.append({"type": "deny", "reason": "missing_tenant_context"})
        return {"decision": "deny", "matched_rules": ["tenant_required"], "actions": actions}

    return {"decision": "allow", "matched_rules": [], "actions": []}
