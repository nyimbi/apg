"""Executable capability contract for APG Micro-Insurance Platform."""
from __future__ import annotations
from typing import Any

CAPABILITY_ID = 'ins_mic'
CAPABILITY_NAME = 'Micro-Insurance Platform'
CAPABILITY_VERSION = '1.0.0'
CAPABILITY_DOMAIN = 'insurance'
CAPABILITY_DESCRIPTION = 'Mobile-first micro-insurance, USSD enrolment, parametric payouts, feature phones'

SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["ins_mic_manager", "ins_mic_viewer", "ins_mic_admin"]

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
    "theme": {"default_theme": "ins_mic_theme", "allow_tenant_overrides": True},
}

PROVIDES = ['micro_policy_issuance', 'ussd_enrolment', 'parametric_claims', 'mobile_money_payout', 'group_insurance']
REQUIRES = ['auth', 'audl', 'ntfy', 'ussd', 'fintech_payments']

# NATS integration declarations
PUBLISHES = ['micro_policy.issued', 'micro_claim.triggered', 'mobile_payout.sent']
SUBSCRIBES = []

UI_ROUTES = [{'name': 'dashboard', 'path': '/ins-mic/dashboard', 'component': 'InsMicDashboard', 'permission': 'ins_mic:view', 'nav_group': 'Overview'}, {'name': 'list', 'path': '/ins-mic/list', 'component': 'InsMicList', 'permission': 'ins_mic:view', 'nav_group': 'Overview'}, {'name': 'settings', 'path': '/ins-mic/settings', 'component': 'InsMicSettings', 'permission': 'ins_mic:admin', 'nav_group': 'Administration'}]

THEME = {
    "name": "ins_mic_theme",
    "tokens": {
        "color.primary": '#FF6B35',
        "color.accent": '#FFD166',
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
            "routes": [{'name': 'dashboard', 'path': '/ins-mic/dashboard', 'component': 'InsMicDashboard', 'permission': 'ins_mic:view', 'nav_group': 'Overview'}, {'name': 'list', 'path': '/ins-mic/list', 'component': 'InsMicList', 'permission': 'ins_mic:view', 'nav_group': 'Overview'}, {'name': 'settings', 'path': '/ins-mic/settings', 'component': 'InsMicSettings', 'permission': 'ins_mic:admin', 'nav_group': 'Administration'}],
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
    """Evaluate APG capability rules for Micro-Insurance Platform."""
    reasons: list[str] = []
    actions: list[dict[str, Any]] = []

    if not context.get("tenant_context_present"):
        reasons.append("tenant_id required")
        actions.append({"type": "deny", "reason": "missing_tenant_context"})
        return {"decision": "deny", "matched_rules": ["tenant_required"], "actions": actions}

    return {"decision": "allow", "matched_rules": [], "actions": []}
