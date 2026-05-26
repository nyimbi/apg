"""Request context helpers for Employee Data Management APIs and views."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from flask import g, has_request_context, request, session


def _clean_text(value: Any) -> Optional[str]:
	if value is None:
		return None
	text = str(value).strip()
	return text or None


def _current_user_candidates() -> List[Any]:
	"""Return Flask-Login/AppBuilder current-user candidates when available."""
	try:
		from flask_login import current_user
	except Exception:
		try:
			from flask_appbuilder.security import current_user
		except Exception:
			return []

	try:
		if not current_user:
			return []
	except Exception:
		return []
	try:
		is_authenticated = getattr(current_user, "is_authenticated", False)
	except Exception:
		return []
	if callable(is_authenticated):
		is_authenticated = is_authenticated()
	if not is_authenticated:
		return []
	return [
		getattr(current_user, "username", None),
		getattr(current_user, "id", None),
	]


def get_tenant_id_from_request(payload: Optional[Dict[str, Any]] = None) -> str:
	"""Resolve tenant ID from payload, Flask context, session, request metadata, or fallback."""
	default_tenant = os.getenv("APG_DEFAULT_TENANT_ID", "default_tenant")
	if not has_request_context():
		return default_tenant

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
	"""Resolve user ID from payload, Flask context, session, request metadata, or fallback."""
	default_user = os.getenv("APG_DEFAULT_USER_ID", os.getenv("APG_USER_ID", "system"))
	if not has_request_context():
		return default_user

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
		getattr(g, "user_id", None),
		getattr(current_user, "username", None),
		getattr(current_user, "id", None),
		getattr(fab_user, "username", None),
		getattr(fab_user, "id", None),
	])
	candidates.extend(_current_user_candidates())
	candidates.extend([
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
