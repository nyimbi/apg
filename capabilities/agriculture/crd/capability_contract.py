"""Executable capability contract for APG Agricultural Credit Scoring."""
from __future__ import annotations
from typing import Any

CAPABILITY_ID = 'agr_crd'
CAPABILITY_NAME = 'Agricultural Credit Scoring'
CAPABILITY_VERSION = '1.0.0'
CAPABILITY_DOMAIN = 'agriculture'
CAPABILITY_DESCRIPTION = 'Yield-based credit scoring, group lending, smallholder credit profiles'

SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["agr_crd_manager", "agr_crd_viewer", "agr_crd_admin"]

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
    "theme": {"default_theme": "agr_crd_theme", "allow_tenant_overrides": True},
}

PROVIDES = ['credit_scoring', 'loan_eligibility', 'group_lending', 'credit_profile', 'repayment_tracking']
REQUIRES = ['auth', 'audl', 'agr_fms', 'agr_crp', 'fintech_kyc']

# NATS integration declarations
PUBLISHES = ['credit.scored', 'loan.approved', 'repayment.received']
SUBSCRIBES = []

UI_ROUTES = [{'name': 'dashboard', 'path': '/agr-crd/dashboard', 'component': 'AgrCrdDashboard', 'permission': 'agr_crd:view', 'nav_group': 'Overview'}, {'name': 'list', 'path': '/agr-crd/list', 'component': 'AgrCrdList', 'permission': 'agr_crd:view', 'nav_group': 'Overview'}, {'name': 'settings', 'path': '/agr-crd/settings', 'component': 'AgrCrdSettings', 'permission': 'agr_crd:admin', 'nav_group': 'Administration'}]

THEME = {
    "name": "agr_crd_theme",
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
    }


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
    """Evaluate APG capability rules for Agricultural Credit Scoring."""
    reasons: list[str] = []
    actions: list[dict[str, Any]] = []

    if not context.get("tenant_context_present"):
        reasons.append("tenant_id required")
        actions.append({"type": "deny", "reason": "missing_tenant_context"})
        return {"decision": "deny", "matched_rules": ["tenant_required"], "actions": actions}

    return {"decision": "allow", "matched_rules": [], "actions": []}
