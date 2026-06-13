"""Executable capability contract for APG Farm Management System."""
from __future__ import annotations
from typing import Any

CAPABILITY_ID = 'agr_fms'
CAPABILITY_NAME = 'Farm Management System'
CAPABILITY_VERSION = '1.0.0'
CAPABILITY_DOMAIN = 'agriculture'
CAPABILITY_DESCRIPTION = 'Parcel registry, input recording, labour management, farm operations'

SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["agr_fms_manager", "agr_fms_viewer", "agr_fms_admin"]

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
    "theme": {"default_theme": "agr_fms_theme", "allow_tenant_overrides": True},
}

PROVIDES = ['farm_registration', 'parcel_management', 'input_recording', 'labour_tracking', 'farm_operations']
REQUIRES = ['auth', 'audl', 'agr_crp', 'agr_lnd']

# NATS integration declarations
PUBLISHES = ['farm.registered', 'input.applied', 'operation.completed']
SUBSCRIBES = []

UI_ROUTES = [{'name': 'dashboard', 'path': '/agr-fms/dashboard', 'component': 'AgrFmsDashboard', 'permission': 'agr_fms:view', 'nav_group': 'Overview'}, {'name': 'list', 'path': '/agr-fms/list', 'component': 'AgrFmsList', 'permission': 'agr_fms:view', 'nav_group': 'Overview'}, {'name': 'settings', 'path': '/agr-fms/settings', 'component': 'AgrFmsSettings', 'permission': 'agr_fms:admin', 'nav_group': 'Administration'}]

THEME = {
    "name": "agr_fms_theme",
    "tokens": {
        "color.primary": '#1B4332',
        "color.accent": '#40916C',
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
    """Evaluate APG capability rules for Farm Management System."""
    reasons: list[str] = []
    actions: list[dict[str, Any]] = []

    if not context.get("tenant_context_present"):
        reasons.append("tenant_id required")
        actions.append({"type": "deny", "reason": "missing_tenant_context"})
        return {"decision": "deny", "matched_rules": ["tenant_required"], "actions": actions}

    return {"decision": "allow", "matched_rules": [], "actions": []}
