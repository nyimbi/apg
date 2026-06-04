"""Deterministic domain rules for Time and Attendance Tracking.

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

def assert_annualised_hours_deficit_manageable(
    contracted_hours: float,
    worked_hours: float,
    max_deficit_pct: float = 0.10,
) -> None:
    """Assert that the annualised hours deficit is within acceptable limits."""
    if contracted_hours <= 0:
        return
    deficit = contracted_hours - worked_hours
    deficit_pct = deficit / contracted_hours
    if deficit_pct > max_deficit_pct:
        from .rules import RuleViolation
        raise RuleViolation(
            "annualised_hours_deficit_too_large",
            f"hours deficit {deficit_pct:.1%} exceeds maximum {max_deficit_pct:.0%}",
            "increase_hours_worked_or_adjust_contract",
        )

def assert_biometric_confidence(
    confidence_score: float,
    min_confidence: float = 0.80,
    context: str = "clock_in",
) -> None:
    """Assert that biometric verification confidence meets the minimum threshold."""
    if confidence_score < min_confidence:
        raise RuleViolation(
            "biometric_confidence_too_low",
            f"biometric confidence {confidence_score:.0%} below minimum {min_confidence:.0%} for {context}",
            "retry_biometric_verification_or_use_pin",
        )
