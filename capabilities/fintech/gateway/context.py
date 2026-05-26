"""Request context helpers for Payment Gateway APIs."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from flask import g, has_request_context, request, session

from .auth import get_current_user


def _clean_text(value: Any) -> Optional[str]:
	if value is None:
		return None
	text = str(value).strip()
	return text or None


def _gateway_user() -> Dict[str, Any]:
	try:
		user = get_current_user()
	except Exception:
		return {}
	return user if isinstance(user, dict) else {}


def get_tenant_id_from_request(payload: Optional[Dict[str, Any]] = None) -> str:
	"""Resolve tenant ID from payload, gateway auth, Flask context, request metadata, or fallback."""
	default_tenant = os.getenv("APG_DEFAULT_TENANT_ID", os.getenv("APG_TENANT_ID", "default"))
	gateway_user = _gateway_user()
	if not has_request_context():
		return _clean_text(gateway_user.get("tenant_id")) or default_tenant

	current_tenant = getattr(g, "current_tenant", None)
	current_user = getattr(g, "current_user", None)
	fab_user = getattr(g, "user", None)
	candidates: List[Any] = []
	if payload:
		candidates.extend([
			payload.get("tenant_id"),
			payload.get("tenant"),
			payload.get("organization_id"),
		])

	candidates.extend([
		gateway_user.get("tenant_id"),
		getattr(g, "tenant_id", None),
		getattr(current_tenant, "tenant_id", current_tenant),
		getattr(current_user, "tenant_id", None),
		getattr(fab_user, "tenant_id", None),
		session.get("tenant_id"),
		request.headers.get("X-Tenant-ID"),
		request.headers.get("X-APG-Tenant-ID"),
		request.headers.get("X-Organization-ID"),
		request.args.get("tenant_id"),
		request.args.get("tenant"),
		request.environ.get("APG_TENANT_ID"),
	])

	for candidate in candidates:
		tenant_id = _clean_text(candidate)
		if tenant_id:
			return tenant_id
	return default_tenant


def get_current_user_id(payload: Optional[Dict[str, Any]] = None) -> str:
	"""Resolve user ID from payload, gateway auth, Flask context, request metadata, or fallback."""
	default_user = os.getenv("APG_DEFAULT_USER_ID", os.getenv("APG_USER_ID", "system"))
	gateway_user = _gateway_user()
	if not has_request_context():
		return _clean_text(gateway_user.get("id")) or default_user

	current_user = getattr(g, "current_user", None)
	fab_user = getattr(g, "user", None)
	candidates: List[Any] = []
	if payload:
		candidates.extend([
			payload.get("user_id"),
			payload.get("current_user_id"),
			payload.get("created_by"),
			payload.get("updated_by"),
		])

	candidates.extend([
		gateway_user.get("id"),
		getattr(g, "user_id", None),
		getattr(current_user, "username", None),
		getattr(current_user, "id", None),
		getattr(fab_user, "username", None),
		getattr(fab_user, "id", None),
		session.get("user_id"),
		request.headers.get("X-User-ID"),
		request.headers.get("X-APG-User-ID"),
		request.args.get("user_id"),
		request.environ.get("APG_USER_ID"),
	])

	for candidate in candidates:
		user_id = _clean_text(candidate)
		if user_id:
			return user_id
	return default_user
