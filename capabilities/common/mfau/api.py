"""Dependency-light API helpers for the MFAU generated-app runtime."""

from __future__ import annotations

import inspect
import os
from typing import Any

from .mfa_runtime import MfauGuardrailError, MfauService


def _clean_text(value: Any) -> str | None:
	if value is None:
		return None
	text = str(value).strip()
	return text or None


def _resolve_current_user_id() -> str:
	"""Resolve user id from Flask request context.

	Priority: request.current_user > X-APG-User-ID header > session > query params > env
	"""
	try:
		from flask import g, request, session
		# 1. request.current_user attribute
		cu = getattr(request, "current_user", None)
		if isinstance(cu, dict) and cu.get("user_id"):
			return str(cu["user_id"])
		# 2. g.current_user
		g_cu = getattr(g, "current_user", None)
		if isinstance(g_cu, dict) and g_cu.get("user_id"):
			return str(g_cu["user_id"])
		# 3. Headers
		h = request.headers.get("X-APG-User-ID") or request.headers.get("X-User-ID")
		if h:
			return h
		# 4. Session
		s = session.get("user_id")
		if s:
			return str(s)
		# 5. Query params
		q = request.args.get("user_id")
		if q:
			return q
	except Exception:
		pass
	return os.getenv("APG_DEFAULT_USER_ID", "anonymous")


def _resolve_current_tenant_id() -> str:
	"""Resolve tenant id from Flask request context.

	Priority: request.current_user > X-APG-Tenant-ID header > session > query params > env
	"""
	try:
		from flask import g, request, session
		# 1. request.current_user attribute
		cu = getattr(request, "current_user", None)
		if isinstance(cu, dict) and cu.get("tenant_id"):
			return str(cu["tenant_id"])
		# 2. g.current_user
		g_cu = getattr(g, "current_user", None)
		if isinstance(g_cu, dict) and g_cu.get("tenant_id"):
			return str(g_cu["tenant_id"])
		# 3. Headers
		h = request.headers.get("X-APG-Tenant-ID") or request.headers.get("X-Tenant-ID")
		if h:
			return h
		# 4. Session
		s = session.get("tenant_id")
		if s:
			return str(s)
		# 5. Query params
		q = request.args.get("tenant_id") or request.args.get("tenant")
		if q:
			return q
	except Exception:
		pass
	return os.getenv("APG_DEFAULT_TENANT_ID", "default")


# Rate limiting decorator
def rate_limit(max_per_minute: int = 60):
	"""Decorator that enforces a per-minute call rate limit."""
	import functools
	import time
	_calls: list[float] = []

	def decorator(f):
		@functools.wraps(f)
		async def async_wrapper(*args, **kwargs):
			now = time.time()
			_calls[:] = [t for t in _calls if now - t < 60]
			if len(_calls) >= max_per_minute:
				raise RuntimeError("Rate limit exceeded")
			_calls.append(now)
			return await f(*args, **kwargs)

		@functools.wraps(f)
		def sync_wrapper(*args, **kwargs):
			now = time.time()
			_calls[:] = [t for t in _calls if now - t < 60]
			if len(_calls) >= max_per_minute:
				raise RuntimeError("Rate limit exceeded")
			_calls.append(now)
			return f(*args, **kwargs)

		if inspect.iscoroutinefunction(f):
			return async_wrapper
		return sync_wrapper

	return decorator


class MFAUserProfileView:
	"""Flask-Appbuilder view stub for MFA user profile management."""

	async def post(self):
		user_id = _resolve_current_user_id()
		tenant_id = _resolve_current_tenant_id()
		return {"user_id": user_id, "tenant_id": tenant_id}


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
		"agents": service.describe()["agents"],
		"streaming": service.describe()["streaming"],
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


def register_mfa_agent_endpoint(service: MfauService, payload: dict[str, Any]) -> dict[str, Any]:
	return _wrap(lambda: service.register_mfa_agent(
		agent_id=payload["agent_id"],
		tenant_id=payload["tenant_id"],
		name=payload["name"],
		runtime=payload["runtime"],
		role=payload["role"],
		scope=payload["scope"],
		owner=payload["owner"],
		purpose=payload["purpose"],
		contribution_disclosed=payload.get("contribution_disclosed", True),
		human_approval_required=payload.get("human_approval_required", False),
	))


def validate_mfa_lifecycle_batch_endpoint(service: MfauService, payload: dict[str, Any]) -> dict[str, Any]:
	return _wrap(lambda: service.validate_mfa_lifecycle_batch(
		tenant_id=payload["tenant_id"],
		event_stream=payload.get("event_stream", "bytewax"),
		mutation_count=payload.get("mutation_count", 1),
		operation=payload.get("operation", "mfa_agent_batch"),
		batch_id=payload.get("batch_id"),
	))


def dashboard_endpoint(service: MfauService, tenant_id: str) -> dict[str, Any]:
	return {"ok": True, "data": service.package(tenant_id)}


def _wrap(operation) -> dict[str, Any]:
	try:
		return {"ok": True, "data": operation()}
	except MfauGuardrailError as exc:
		return {"ok": False, "error": exc.result}
