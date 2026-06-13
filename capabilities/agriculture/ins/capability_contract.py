"""Executable capability contract for APG Crop Insurance."""
from __future__ import annotations
from typing import Any

CAPABILITY_ID = 'agr_ins'
CAPABILITY_NAME = 'Crop Insurance'
CAPABILITY_VERSION = '1.0.0'
CAPABILITY_DOMAIN = 'agriculture'
CAPABILITY_DESCRIPTION = 'Parametric index insurance, satellite verification, mobile money claims'

SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["agr_ins_manager", "agr_ins_viewer", "agr_ins_admin"]

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
    "theme": {"default_theme": "agr_ins_theme", "allow_tenant_overrides": True},
}

PROVIDES = ['policy_issuance', 'parametric_claims', 'satellite_verification', 'mobile_payout', 'index_calculation']
REQUIRES = ['auth', 'audl', 'agr_wth', 'agr_crp', 'fintech_payments']

# NATS integration declarations
PUBLISHES = ['policy.issued', 'claim.triggered', 'payout.disbursed']
SUBSCRIBES = [{'source_capability': 'agr_wth', 'event_type': 'drought.alert', 'handler': 'on_drought_alert'}]

UI_ROUTES = [{'name': 'dashboard', 'path': '/agr-ins/dashboard', 'component': 'AgrInsDashboard', 'permission': 'agr_ins:view', 'nav_group': 'Overview'}, {'name': 'list', 'path': '/agr-ins/list', 'component': 'AgrInsList', 'permission': 'agr_ins:view', 'nav_group': 'Overview'}, {'name': 'settings', 'path': '/agr-ins/settings', 'component': 'AgrInsSettings', 'permission': 'agr_ins:admin', 'nav_group': 'Administration'}]

THEME = {
    "name": "agr_ins_theme",
    "tokens": {
        "color.primary": '#5C4033',
        "color.accent": '#E07A5F',
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
    """Evaluate APG capability rules for Crop Insurance."""
    reasons: list[str] = []
    actions: list[dict[str, Any]] = []

    if not context.get("tenant_context_present"):
        reasons.append("tenant_id required")
        actions.append({"type": "deny", "reason": "missing_tenant_context"})
        return {"decision": "deny", "matched_rules": ["tenant_required"], "actions": actions}

    return {"decision": "allow", "matched_rules": [], "actions": []}
