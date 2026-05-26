"""Request context helpers for Geo-Spatial Services APIs."""

from __future__ import annotations

import os
from typing import Any, Iterable, Optional


def _clean_text(value: Any) -> Optional[str]:
	if value is None:
		return None
	text = str(value).strip()
	return text or None


def _first_text(candidates: Iterable[Any], fallback: str) -> str:
	for candidate in candidates:
		text = _clean_text(candidate)
		if text:
			return text
	return fallback


def _object_value(source: Any, name: str) -> Any:
	if source is None:
		return None
	if isinstance(source, dict):
		return source.get(name)
	return getattr(source, name, None)


def _state_user(request: Any) -> Any:
	state = getattr(request, "state", None)
	return (
		_object_value(state, "current_user")
		or _object_value(state, "user")
		or _object_value(state, "auth_user")
	)


def resolve_current_user_id(request: Any) -> str:
	"""Resolve user ID from FastAPI request state, headers, query, or fallback."""
	default_user = os.getenv("APG_DEFAULT_USER_ID", os.getenv("APG_USER_ID", "system"))
	state_user = _state_user(request)
	state = getattr(request, "state", None)
	headers = getattr(request, "headers", {}) or {}
	query_params = getattr(request, "query_params", {}) or {}

	return _first_text([
		_object_value(state_user, "user_id"),
		_object_value(state_user, "id"),
		_object_value(state_user, "username"),
		_object_value(state, "user_id"),
		headers.get("X-User-ID"),
		headers.get("X-APG-User-ID"),
		query_params.get("user_id"),
		os.getenv("APG_USER_ID"),
	], default_user)


def resolve_tenant_id(request: Any) -> str:
	"""Resolve tenant ID from FastAPI request state, headers, query, or fallback."""
	default_tenant = os.getenv("APG_DEFAULT_TENANT_ID", os.getenv("APG_TENANT_ID", "default"))
	state_user = _state_user(request)
	state = getattr(request, "state", None)
	headers = getattr(request, "headers", {}) or {}
	query_params = getattr(request, "query_params", {}) or {}

	return _first_text([
		_object_value(state_user, "tenant_id"),
		_object_value(state, "tenant_id"),
		headers.get("X-Tenant-ID"),
		headers.get("X-APG-Tenant-ID"),
		headers.get("X-Organization-ID"),
		query_params.get("tenant_id"),
		query_params.get("tenant"),
		os.getenv("APG_TENANT_ID"),
	], default_tenant)
