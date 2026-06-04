"""Deterministic domain rules for Point of Sale.

These rules are evaluated by the capability rule engine and are the single
source of truth for all governance decisions within this capability.
"""
from __future__ import annotations
from typing import Any


class RuleViolation(Exception):
    """Raised when a business rule is violated."""
    def __init__(self, rule_name: str, reason: str, required_action: str = "") -> None:
        self.rule_name = rule_name
        self.reason = reason
        self.required_action = required_action
        super().__init__(f"Rule '{rule_name}' violated: {reason}")


def assert_tenant_context(context: dict[str, Any]) -> None:
    """All operations require a tenant context."""
    if not context.get("tenant_id"):
        raise RuleViolation("tenant_context_required", "tenant_id is required", "attach_tenant_context")


def assert_write_policy(context: dict[str, Any]) -> None:
    """Write operations require an attached policy."""
    if context.get("operation_type") == "write" and not context.get("policy_attached"):
        raise RuleViolation("write_requires_policy", "write operations require an attached policy", "attach_policy")


def assert_no_cross_tenant_access(actor_tenant: str, resource_tenant: str) -> None:
    """Cross-tenant access is always denied."""
    if actor_tenant != resource_tenant:
        raise RuleViolation("cross_tenant_access_denied", "cross-tenant access is not permitted", "use_own_tenant_resources")
