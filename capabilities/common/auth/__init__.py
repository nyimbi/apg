"""APG Authentication & RBAC capability.

Standalone package: ``pip install apg-common-auth``

Quick start::

    from apg_common_auth import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : auth
Provides      : identity_registry, role_governance, session_control, access_decisions, privacy_budget_governance, security_agents
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-common-auth"
__capability_id__ = "auth"

from .capability_contract import (  # noqa: E402
    get_capability_contract,
    evaluate_capability_rules,
)

__all__ = [
    "__version__",
    "__capability_id__",
    "get_capability_contract",
    "evaluate_capability_rules",
]

# ── ABAC stubs ────────────────────────────────────────────────────────────────
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AccessRequest:
    subject_id: str
    resource: str
    action: str
    tenant_id: str = ""
    ip_address: str | None = None
    environment: dict[str, Any] = field(default_factory=dict)
    # legacy alias
    subject: str = ""
    context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.subject:
            self.subject = self.subject_id


@dataclass
class Policy:
    name: str
    effect: str = "allow"
    priority: int = 50
    resource_conditions: list[dict[str, Any]] = field(default_factory=list)
    action_conditions: list[dict[str, Any]] = field(default_factory=list)
    subject_conditions: list[dict[str, Any]] = field(default_factory=list)
    environment_conditions: list[dict[str, Any]] = field(default_factory=list)
    # legacy fields
    id: str = ""
    conditions: dict[str, Any] = field(default_factory=dict)


@dataclass
class AccessDecision:
    decision: str
    reason: str
    policies_evaluated: list[str] = field(default_factory=list)


def _check_condition(condition: dict[str, Any], attrs: dict[str, Any]) -> bool:
    """Evaluate a single condition dict against an attribute map."""
    attr = condition.get("attribute", "")
    op = condition.get("operator", "equals")
    expected = condition.get("value")
    actual = attrs.get(attr)
    if actual is None:
        return False
    if op == "equals":
        return str(actual) == str(expected)
    if op == "starts_with":
        return str(actual).startswith(str(expected))
    if op == "contains":
        return str(expected) in str(actual)
    return False


def _policy_matches(policy: Policy, request: "AccessRequest", subject_attrs: dict[str, Any]) -> bool:
    """Return True if all conditions on the policy match the request."""
    request_attrs = {
        "subject_id": request.subject_id,
        "resource": request.resource,
        "action": request.action,
        "tenant_id": request.tenant_id,
        **subject_attrs,
    }
    env_attrs = {
        "ip_address": request.ip_address,
        **(request.environment or {}),
    }

    for cond in policy.resource_conditions:
        if not _check_condition(cond, request_attrs):
            return False
    for cond in policy.action_conditions:
        if not _check_condition(cond, request_attrs):
            return False
    for cond in policy.subject_conditions:
        if not _check_condition(cond, {**request_attrs, **subject_attrs}):
            return False
    for cond in policy.environment_conditions:
        if not _check_condition(cond, env_attrs):
            return False
    return True


class ABACEngine:
    def __init__(self) -> None:
        self._policies: list[Policy] = []
        self._subject_attrs: dict[str, dict[str, Any]] = {}

    def add_policy(self, policy: Policy) -> None:
        self._policies.append(policy)
        # keep sorted ascending by priority so lowest number = highest priority
        self._policies.sort(key=lambda p: p.priority)

    def set_attributes(self, subject_id: str, attrs: dict[str, Any]) -> None:
        self._subject_attrs[subject_id] = dict(attrs)

    async def evaluate_access(self, request: "AccessRequest") -> AccessDecision:
        subject_attrs = self._subject_attrs.get(request.subject_id, {})
        for policy in self._policies:
            if _policy_matches(policy, request, subject_attrs):
                return AccessDecision(
                    decision=policy.effect,
                    reason=f"Policy {policy.name} evaluated to {policy.effect}",
                    policies_evaluated=[policy.name],
                )
        return AccessDecision(
            decision="deny",
            reason="No matching policy",
            policies_evaluated=[],
        )

    # legacy sync evaluate
    def evaluate(self, request: "AccessRequest") -> bool:
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            d = loop.run_until_complete(self.evaluate_access(request))
            return d.decision == "allow"
        finally:
            loop.close()
