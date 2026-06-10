"""Dependency-light API helpers for Sustainability and ESG Management."""

from __future__ import annotations

import os
from typing import Any

try:
	from fastapi import Request
	from fastapi.security import HTTPAuthorizationCredentials
except ImportError:  # pragma: no cover
	Request = object  # type: ignore
	HTTPAuthorizationCredentials = object  # type: ignore


def _clean_text(value):
	if value is None:
		return None
	text = str(value).strip()
	return text or None


def _object_value(source, name):
	if source is None:
		return None
	if isinstance(source, dict):
		return source.get(name)
	return getattr(source, name, None)


def _mapping_value(source, name):
	if source is None:
		return None
	getter = getattr(source, "get", None)
	return getter(name) if getter else None


def _decode_bearer_claims(credentials):
	try:
		import base64 as _b64, json as _json
		token = getattr(credentials, "credentials", None) or str(credentials)
		parts = token.split(".")
		if len(parts) >= 2:
			padded = parts[1] + "=" * (4 - len(parts[1]) % 4)
			return _json.loads(_b64.urlsafe_b64decode(padded))
	except Exception as _exc:
		_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)
	return {}


async def get_current_user(request: Request, credentials=None):
	"""Resolve current user from JWT claims, request state, headers, or env vars.

	Priority: Bearer JWT > request.state.current_user > X-APG-* headers > env vars
	"""
	# 1. JWT Bearer token claims
	if credentials is not None:
		scheme = getattr(credentials, "scheme", "")
		if str(scheme).lower() == "bearer":
			claims = _decode_bearer_claims(credentials)
			if claims.get("sub") or claims.get("user_id"):
				return {
					"user_id": claims.get("sub") or claims.get("user_id", ""),
					"tenant_id": claims.get("tenant_id") or claims.get("tid", os.getenv("APG_DEFAULT_TENANT_ID", "default")),
					"permissions": claims.get("permissions", ["esg:read"]),
				}

	# 2. Check request.state.current_user
	state = getattr(request, "state", None)
	current_user = _object_value(state, "current_user")
	if isinstance(current_user, dict) and current_user.get("user_id"):
		return {
			"user_id": current_user["user_id"],
			"tenant_id": current_user.get("tenant_id", "default"),
			"permissions": current_user.get("permissions", ["esg:read"]),
		}

	# 3. Headers
	headers = getattr(request, "headers", {})
	header_user = headers.get("X-APG-User-ID") or headers.get("X-User-ID")
	header_tenant = headers.get("X-APG-Tenant-ID") or headers.get("X-Tenant-ID")
	header_perms_raw = headers.get("X-APG-Permissions")
	if header_user:
		permissions = [p.strip() for p in header_perms_raw.split(",")] if header_perms_raw else ["esg:read"]
		return {
			"user_id": header_user,
			"tenant_id": header_tenant or os.getenv("APG_DEFAULT_TENANT_ID", "default"),
			"permissions": permissions,
		}

	# 4. Query string / query params
	try:
		query = getattr(request, "query_params", {}) or {}
		q_user = _mapping_value(query, "user_id")
		q_tenant = _mapping_value(query, "tenant_id") or _mapping_value(query, "tenant")
		if q_user:
			return {"user_id": q_user, "tenant_id": q_tenant or os.getenv("APG_DEFAULT_TENANT_ID", "default"), "permissions": ["esg:read"]}
	except Exception as _exc:
		_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)

	# 5. Env fallback
	return {
		"user_id": os.getenv("APG_DEFAULT_USER_ID", "anonymous"),
		"tenant_id": os.getenv("APG_DEFAULT_TENANT_ID", "default"),
		"permissions": ["esg:read"],
	}


async def get_esg_service():
	"""FastAPI dependency that returns the shared ESG service instance."""
	return _service_singleton()


# ============================================================================
# Dependency Injection
# ============================================================================


try:
	from .service import ESGManagementLifecycleService
except ImportError:  # pragma: no cover
	from service import ESGManagementLifecycleService  # type: ignore


_SERVICE = ESGManagementLifecycleService()


def _service_singleton():
	return _SERVICE


def service():
	return _SERVICE


def _resolve_tenant_from_request(request=None):
	"""Resolve tenant id from request context."""
	try:
		import importlib as _il
		_mod = _il.import_module("capabilities.common.request_context")
		return _mod.get_tenant_id_from_context()
	except Exception as _exc:
		_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)
	if request is not None:
		headers = getattr(request, "headers", {})
		t = headers.get("X-APG-Tenant-ID") or headers.get("X-Tenant-ID")
		if t:
			return str(t)
	return os.getenv("APG_DEFAULT_TENANT_ID", "default")


def get_tenant_id(request=None, credentials=None):
	"""Resolve tenant id — compatibility wrapper."""
	return _resolve_tenant_from_request(request)


def create_esg_profile(payload: dict[str, Any]) -> dict[str, Any]:
	return service().create_esg_profile(payload.get("id", ""), payload.get("tenant_id", "default"), payload.get("name", ""), payload.get("industry", ""), payload.get("country", ""), payload.get("reporting_year"), payload.get("owner_id", ""))


def add_framework(payload: dict[str, Any]) -> dict[str, Any]:
	return service().add_framework(payload.get("id", ""), payload.get("tenant_id", "default"), payload.get("profile_id", ""), payload.get("code", "gri"), payload.get("version", ""), payload.get("mandatory", True), payload.get("owner_id", ""))


def define_metric(payload: dict[str, Any]) -> dict[str, Any]:
	return service().define_metric(payload.get("id", ""), payload.get("tenant_id", "default"), payload.get("profile_id", ""), payload.get("pillar", "environmental"), payload.get("metric_type", "emissions"), payload.get("unit", "tco2e"), payload.get("name", ""), payload.get("owner_id", ""))


def record_measurement(payload: dict[str, Any]) -> dict[str, Any]:
	return service().record_measurement(payload.get("id", ""), payload.get("tenant_id", "default"), payload.get("metric_id", ""), payload.get("period", ""), payload.get("value"), payload.get("source", "manual"), payload.get("evidence_id", ""), payload.get("reviewed_by"))


def set_target(payload: dict[str, Any]) -> dict[str, Any]:
	return service().set_target(payload.get("id", ""), payload.get("tenant_id", "default"), payload.get("metric_id", ""), payload.get("target_type", "absolute"), payload.get("baseline_value"), payload.get("target_value"), payload.get("due_date", ""), payload.get("owner_id", ""))


def record_supplier_assessment(payload: dict[str, Any]) -> dict[str, Any]:
	return service().record_supplier_assessment(payload.get("id", ""), payload.get("tenant_id", "default"), payload.get("supplier_id", ""), payload.get("period", ""), payload.get("score", 0), payload.get("risk_tier", "low"), payload.get("evidence_id", ""), payload.get("owner_id"))


def record_initiative(payload: dict[str, Any]) -> dict[str, Any]:
	return service().record_initiative(payload.get("id", ""), payload.get("tenant_id", "default"), payload.get("profile_id", ""), payload.get("name", ""), payload.get("pillar", "environmental"), payload.get("budget", 0), payload.get("owner_id", ""), payload.get("expected_impact", ""))


def record_risk(payload: dict[str, Any]) -> dict[str, Any]:
	return service().record_risk(payload.get("id", ""), payload.get("tenant_id", "default"), payload.get("profile_id", ""), payload.get("tier", "medium"), payload.get("category", "climate"), payload.get("description", ""), payload.get("owner_id"))


def create_report(payload: dict[str, Any]) -> dict[str, Any]:
	return service().create_report(payload.get("id", ""), payload.get("tenant_id", "default"), payload.get("profile_id", ""), payload.get("report_type", "annual"), payload.get("period", ""), payload.get("framework_ids", []), payload.get("measurement_ids", []), payload.get("approved_by", ""))


def register_stakeholder(payload: dict[str, Any]) -> dict[str, Any]:
	return service().register_stakeholder(payload.get("id", ""), payload.get("tenant_id", "default"), payload.get("profile_id", ""), payload.get("stakeholder_type", "investor"), payload.get("name", ""), payload.get("channel", ""), payload.get("consent_recorded", False))


def record_engagement(payload: dict[str, Any]) -> dict[str, Any]:
	return service().record_engagement(payload.get("id", ""), payload.get("tenant_id", "default"), payload.get("stakeholder_id", ""), payload.get("topic", ""), payload.get("channel", ""), payload.get("sentiment", "neutral"), payload.get("owner_id"))


def register_esg_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return service().register_esg_agent(payload.get("tenant_id", "default"), payload.get("name", "ESG Agent"), payload.get("runtime", "codex"), payload.get("role", "sustainability_reviewer"), payload.get("purpose", "review ESG records"), payload.get("owner_id"))


def dashboard_summary(tenant_id: str = "default") -> dict[str, Any]:
	return service().dashboard_summary(tenant_id)


def audit_events(tenant_id: str = "default") -> list[dict[str, Any]]:
	return service().audit_events(tenant_id)
