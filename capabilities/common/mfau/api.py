"""Dependency-light API helpers for the MFAU generated-app runtime."""

from __future__ import annotations

from typing import Any

from .mfa_runtime import MfauGuardrailError, MfauService


def create_service(tenant_id: str = "default", configuration_overrides: dict[str, Any] | None = None) -> MfauService:
	"""Create an MFAU runtime service for generated APG applications."""
	return MfauService(tenant_id=tenant_id, configuration_overrides=configuration_overrides)


def health(service: MfauService, tenant_id: str = "default") -> dict[str, Any]:
	"""Return a serializable health payload."""
	summary = service.dashboard_summary(tenant_id)
	return {
		"status": "ok",
		"capability": "mfau",
		"tenant_id": tenant_id,
		"rule_count": service.describe()["rule_count"],
		"summary": summary,
	}


def register_profile_endpoint(service: MfauService, payload: dict[str, Any]) -> dict[str, Any]:
	return _wrap(lambda: service.create_user_profile(
		profile_id=payload["profile_id"],
		tenant_id=payload["tenant_id"],
		user_id=payload["user_id"],
		policy_id=payload["policy_id"],
		primary_channel=payload["primary_channel"],
		status=payload.get("status", "active"),
	))


def enroll_method_endpoint(service: MfauService, payload: dict[str, Any]) -> dict[str, Any]:
	return _wrap(lambda: service.enroll_method(
		method_id=payload["method_id"],
		tenant_id=payload["tenant_id"],
		user_id=payload["user_id"],
		method_type=payload["method_type"],
		channel_verified=payload.get("channel_verified", True),
		biometric_consent_recorded=payload.get("biometric_consent_recorded", True),
		template_encrypted=payload.get("template_encrypted", True),
		secret_encrypted=payload.get("secret_encrypted", True),
		device_id=payload.get("device_id", ""),
		phishing_resistant=payload.get("phishing_resistant", False),
		review_recorded=payload.get("review_recorded", False),
	))


def bind_device_endpoint(service: MfauService, payload: dict[str, Any]) -> dict[str, Any]:
	return _wrap(lambda: service.bind_device(
		device_id=payload["device_id"],
		tenant_id=payload["tenant_id"],
		user_id=payload["user_id"],
		trust_score=payload["trust_score"],
		reviewed=payload.get("reviewed", False),
	))


def assess_risk_endpoint(service: MfauService, payload: dict[str, Any]) -> dict[str, Any]:
	return _wrap(lambda: service.assess_risk(
		assessment_id=payload["assessment_id"],
		tenant_id=payload["tenant_id"],
		user_id=payload["user_id"],
		risk_score=payload["risk_score"],
		device_trust_score=payload["device_trust_score"],
		external_signal=payload.get("external_signal", False),
		review_recorded=payload.get("review_recorded", False),
	))


def create_challenge_endpoint(service: MfauService, payload: dict[str, Any]) -> dict[str, Any]:
	return _wrap(lambda: service.create_challenge(
		challenge_id=payload["challenge_id"],
		tenant_id=payload["tenant_id"],
		user_id=payload["user_id"],
		method_id=payload["method_id"],
		assessment_id=payload["assessment_id"],
		action_risk=payload.get("action_risk", "normal"),
		step_up_completed=payload.get("step_up_completed", True),
		phishing_resistant_factor_present=payload.get("phishing_resistant_factor_present", True),
		token_unexpired=payload.get("token_unexpired", True),
		token_reused=payload.get("token_reused", False),
		verification_evidence=payload.get("verification_evidence", True),
		failed_attempts=payload.get("failed_attempts", 0),
		user_locked=payload.get("user_locked", False),
		device_review_recorded=payload.get("device_review_recorded", False),
		risk_override_approved=payload.get("risk_override_approved", False),
	))


def complete_challenge_endpoint(service: MfauService, payload: dict[str, Any]) -> dict[str, Any]:
	return _wrap(lambda: service.complete_challenge(
		challenge_id=payload["challenge_id"],
		tenant_id=payload["tenant_id"],
		verification_evidence=payload.get("verification_evidence", True),
	))


def recover_account_endpoint(service: MfauService, payload: dict[str, Any]) -> dict[str, Any]:
	return _wrap(lambda: service.recover_account(
		recovery_id=payload["recovery_id"],
		tenant_id=payload["tenant_id"],
		user_id=payload["user_id"],
		verified_recovery_channel=payload.get("verified_recovery_channel", True),
		audit_event_recorded=payload.get("audit_event_recorded", True),
		admin_recovery=payload.get("admin_recovery", False),
		admin_approval_recorded=payload.get("admin_approval_recorded", True),
		recovery_evidence_present=payload.get("recovery_evidence_present", True),
	))


def backup_codes_endpoint(service: MfauService, payload: dict[str, Any]) -> dict[str, Any]:
	return _wrap(lambda: service.generate_backup_codes(
		code_set_id=payload["code_set_id"],
		tenant_id=payload["tenant_id"],
		user_id=payload["user_id"],
		code_count=payload.get("code_count", 10),
	))


def use_backup_code_endpoint(service: MfauService, payload: dict[str, Any]) -> dict[str, Any]:
	return _wrap(lambda: service.use_backup_code(
		code_set_id=payload["code_set_id"],
		tenant_id=payload["tenant_id"],
		user_id=payload["user_id"],
		code_value=payload["code_value"],
	))


def policy_endpoint(service: MfauService, payload: dict[str, Any]) -> dict[str, Any]:
	return _wrap(lambda: service.create_policy(
		policy_id=payload["policy_id"],
		tenant_id=payload["tenant_id"],
		name=payload["name"],
		audit_event_recorded=payload.get("audit_event_recorded", True),
	))


def dashboard_endpoint(service: MfauService, tenant_id: str) -> dict[str, Any]:
	return {"ok": True, "data": service.package(tenant_id)}


def _wrap(operation) -> dict[str, Any]:
	try:
		return {"ok": True, "data": operation()}
	except MfauGuardrailError as exc:
		return {"ok": False, "error": exc.result}
