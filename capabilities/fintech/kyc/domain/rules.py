"""Deterministic domain rules for Know Your Customer.

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


# ─────────────────────────────────────────────────────────────
# KYC-specific rule functions
# ─────────────────────────────────────────────────────────────

def assert_identity_document_present(context: dict[str, Any]) -> None:
	if not context.get("identity_document_id"):
		raise RuleViolation("identity_document_required", "identity document is required for KYC", "upload_identity_document")


def assert_address_document_present(context: dict[str, Any]) -> None:
	if not context.get("address_document_id") and context.get("require_address_proof"):
		raise RuleViolation("address_document_required", "proof of address required", "upload_address_document")


def assert_biometric_match(context: dict[str, Any]) -> None:
	score = float(context.get("biometric_match_score", 0))
	threshold = float(context.get("biometric_threshold", 0.8))
	if score < threshold:
		raise RuleViolation("biometric_mismatch", f"biometric match score {score:.2f} below threshold {threshold}", "retake_biometrics")


def assert_consent_recorded(context: dict[str, Any]) -> None:
	if not context.get("consent_recorded"):
		raise RuleViolation("consent_required", "customer consent must be recorded before KYC", "record_consent")


def assert_liveness_check_passed(context: dict[str, Any]) -> None:
	if not context.get("liveness_passed"):
		raise RuleViolation("liveness_check_failed", "liveness check must pass before proceeding", "redo_liveness_check")


def assert_screening_completed(context: dict[str, Any]) -> None:
	if not context.get("screening_completed"):
		raise RuleViolation("screening_required", "PEP/sanctions screening must be completed", "run_screening")


def assert_no_unresolved_sanction(context: dict[str, Any]) -> None:
	if context.get("has_active_sanction"):
		raise RuleViolation("active_sanction", "customer has active sanction match — cannot proceed", "resolve_sanction_match")


def assert_edd_for_pep(context: dict[str, Any]) -> None:
	if context.get("is_pep") and not context.get("edd_completed"):
		raise RuleViolation("edd_required_for_pep", "enhanced due diligence required for politically exposed persons", "complete_edd")


def assert_edd_for_high_risk(context: dict[str, Any]) -> None:
	risk = context.get("risk_level", "low")
	if risk in ("high", "very_high") and not context.get("edd_completed"):
		raise RuleViolation("edd_required_high_risk", f"enhanced due diligence required for {risk} risk customers", "complete_edd")


def assert_kyc_not_expired(context: dict[str, Any]) -> None:
	if context.get("kyc_expired"):
		raise RuleViolation("kyc_expired", "customer KYC has expired — re-KYC required", "initiate_re_kyc")


def assert_risk_score_range(context: dict[str, Any]) -> None:
	score = int(context.get("risk_score", 0))
	if not (0 <= score <= 100):
		raise RuleViolation("invalid_risk_score", f"risk score {score} must be between 0 and 100", "recalculate_risk_score")


def assert_ubo_declared(context: dict[str, Any]) -> None:
	if context.get("is_corporate") and not context.get("ubo_declared"):
		raise RuleViolation("ubo_declaration_required", "ultimate beneficial owners must be declared for corporate customers", "declare_ubos")


def assert_no_open_reviews(context: dict[str, Any]) -> None:
	if context.get("has_open_reviews"):
		raise RuleViolation("open_reviews_exist", "customer has open KYC reviews that must be resolved first", "resolve_open_reviews")


def assert_no_deceased_id(context: dict[str, Any]) -> None:
	if context.get("id_belongs_to_deceased"):
		raise RuleViolation("deceased_id_fraud", "identity document belongs to a deceased person", "escalate_to_fraud_team")


def assert_no_synthetic_identity(context: dict[str, Any]) -> None:
	if context.get("synthetic_identity_detected"):
		raise RuleViolation("synthetic_identity", "synthetic identity indicators detected", "escalate_to_fraud_team")
