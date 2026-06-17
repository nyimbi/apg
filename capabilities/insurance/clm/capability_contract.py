"""Executable capability contract for APG Claims Management."""
from __future__ import annotations
from typing import Any

CAPABILITY_ID = 'ins_clm'
CAPABILITY_NAME = 'Claims Management'
CAPABILITY_VERSION = '1.0.0'
CAPABILITY_DOMAIN = 'insurance'
CAPABILITY_DESCRIPTION = 'First notice of loss, claims adjudication, payment, fraud detection'

SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["ins_clm_manager", "ins_clm_viewer", "ins_clm_admin"]

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
    "theme": {"default_theme": "ins_clm_theme", "allow_tenant_overrides": True},
}

PROVIDES = ['fnol_intake', 'claims_adjudication', 'claims_payment', 'fraud_referral', 'claims_analytics']
REQUIRES = ['auth', 'audl', 'ntfy', 'ins_pol', 'fintech_payments']

# NATS integration declarations
PUBLISHES = ['claim.filed', 'claim.approved', 'claim.paid']
SUBSCRIBES = [{'source_capability': 'ins_pol', 'event_type': 'policy.issued', 'handler': 'on_policy_issued'}]

UI_ROUTES = [{'name': 'dashboard', 'path': '/ins-clm/dashboard', 'component': 'InsClmDashboard', 'permission': 'ins_clm:view', 'nav_group': 'Overview'}, {'name': 'list', 'path': '/ins-clm/list', 'component': 'InsClmList', 'permission': 'ins_clm:view', 'nav_group': 'Overview'}, {'name': 'settings', 'path': '/ins-clm/settings', 'component': 'InsClmSettings', 'permission': 'ins_clm:admin', 'nav_group': 'Administration'}]

THEME = {
    "name": "ins_clm_theme",
    "tokens": {
        "color.primary": '#C1121F',
        "color.accent": '#E63946',
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
    """Evaluate APG capability rules for Claims Management."""
    reasons: list[str] = []
    actions: list[dict[str, Any]] = []

    if not context.get("tenant_context_present"):
        reasons.append("tenant_id required")
        actions.append({"type": "deny", "reason": "missing_tenant_context"})
        return {"decision": "deny", "matched_rules": ["tenant_required"], "actions": actions}

    return {"decision": "allow", "matched_rules": [], "actions": []}
