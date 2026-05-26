"""Request context helpers for Expiry Date Management surfaces."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from flask import g, has_request_context, request


def _clean_text(value: Any) -> Optional[str]:
	"""Return a non-empty stripped string or None."""
	if value is None:
		return None
	text = str(value).strip()
	return text or None


def get_tenant_id_from_request(payload: Optional[Dict[str, Any]] = None) -> str:
	"""Resolve tenant ID from request/auth context with a configured fallback."""
	default_tenant = os.getenv("APG_DEFAULT_TENANT_ID", os.getenv("APG_TENANT_ID", "default"))
	if not has_request_context():
		return default_tenant

	current_user = getattr(g, "current_user", None)
	fab_user = getattr(g, "user", None)
	candidates: List[Any] = []
	if payload:
		candidates.extend([
			payload.get("tenant_id"),
			payload.get("tenant"),
		])

	candidates.extend([
		getattr(g, "tenant_id", None),
		getattr(g, "current_tenant", None),
		getattr(current_user, "tenant_id", None),
		getattr(fab_user, "tenant_id", None),
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
	"""Resolve current user ID from request/auth context with a system fallback."""
	default_user = os.getenv("APG_DEFAULT_USER_ID", os.getenv("APG_USER_ID", "system"))
	if not has_request_context():
		return default_user

	current_user = getattr(g, "current_user", None)
	fab_user = getattr(g, "user", None)
	candidates: List[Any] = []
	if payload:
		candidates.extend([
			payload.get("user_id"),
			payload.get("current_user"),
			payload.get("approved_by"),
			payload.get("acknowledged_by"),
		])

	candidates.extend([
		getattr(g, "user_id", None),
		getattr(current_user, "username", None),
		getattr(current_user, "id", None),
		getattr(fab_user, "username", None),
		getattr(fab_user, "id", None),
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
