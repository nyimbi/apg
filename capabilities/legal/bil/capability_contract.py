"""Executable capability contract for APG Legal Billing & Time Tracking."""
from __future__ import annotations
from typing import Any

CAPABILITY_ID = 'leg_bil'
CAPABILITY_NAME = 'Legal Billing & Time Tracking'
CAPABILITY_VERSION = '1.0.0'
CAPABILITY_DOMAIN = 'legal'
CAPABILITY_DESCRIPTION = 'Time entry, expense capture, invoice generation, LEDES billing, client trust accounts'

SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["leg_bil_manager", "leg_bil_viewer", "leg_bil_admin"]

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
    "theme": {"default_theme": "leg_bil_theme", "allow_tenant_overrides": True},
}

PROVIDES = ['time_entry', 'expense_capture', 'invoice_generation', 'trust_accounting', 'billing_analytics']
REQUIRES = ['auth', 'audl', 'fin_gl', 'leg_mat']

# NATS integration declarations
PUBLISHES = ['time.entered', 'invoice.generated', 'trust.deposit_received']
SUBSCRIBES = []

UI_ROUTES = [{'name': 'dashboard', 'path': '/leg-bil/dashboard', 'component': 'LegBilDashboard', 'permission': 'leg_bil:view', 'nav_group': 'Overview'}, {'name': 'list', 'path': '/leg-bil/list', 'component': 'LegBilList', 'permission': 'leg_bil:view', 'nav_group': 'Overview'}, {'name': 'settings', 'path': '/leg-bil/settings', 'component': 'LegBilSettings', 'permission': 'leg_bil:admin', 'nav_group': 'Administration'}]

THEME = {
    "name": "leg_bil_theme",
    "tokens": {
        "color.primary": '#0D3349',
        "color.accent": '#00A8E8',
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
    """Evaluate APG capability rules for Legal Billing & Time Tracking."""
    reasons: list[str] = []
    actions: list[dict[str, Any]] = []

    if not context.get("tenant_context_present"):
        reasons.append("tenant_id required")
        actions.append({"type": "deny", "reason": "missing_tenant_context"})
        return {"decision": "deny", "matched_rules": ["tenant_required"], "actions": actions}

    return {"decision": "allow", "matched_rules": [], "actions": []}
