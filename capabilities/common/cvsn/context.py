"""Request context helpers for Computer Vision services."""

from __future__ import annotations

import os
from typing import Any, Iterable, Optional


DEFAULT_PERMISSIONS = ["cv:read", "cv:write", "cv:admin"]


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


def _mapping_value(source: Any, name: str) -> Any:
	if source is None:
		return None
	getter = getattr(source, "get", None)
	if getter is None:
		return None
	return getter(name)


def _state_user(request: Any) -> Any:
	state = _object_value(request, "state")
	return (
		_object_value(state, "current_user")
		or _object_value(state, "user")
		or _object_value(state, "auth_user")
	)


def _split_permissions(value: Any) -> list[str]:
	if value is None:
		return []
	if isinstance(value, str):
		values = value.split(",")
	else:
		values = value
	return [text for item in values if (text := _clean_text(item))]


def resolve_current_user_info(
	request: Any = None,
	session: Any = None,
	g: Any = None,
	credentials: Any = None,
) -> dict[str, Any]:
	"""Resolve APG CVSN user context from request state, headers, session, or environment."""
	default_user = os.getenv("APG_DEFAULT_USER_ID", os.getenv("APG_USER_ID", "system"))
	default_tenant = os.getenv("APG_DEFAULT_TENANT_ID", "default_tenant")
	default_permissions = _split_permissions(os.getenv("APG_CVSN_PERMISSIONS")) or DEFAULT_PERMISSIONS

	state_user = _state_user(request)
	state = _object_value(request, "state")
	headers = _object_value(request, "headers") or {}
	query_params = _object_value(request, "query_params") or _object_value(request, "args") or {}

	user_id = _first_text([
		_object_value(state_user, "user_id"),
		_object_value(state_user, "id"),
		_object_value(state_user, "username"),
		_object_value(state, "user_id"),
		_object_value(g, "user_id"),
		_object_value(_object_value(g, "current_user"), "user_id"),
		_object_value(_object_value(g, "current_user"), "id"),
		_mapping_value(headers, "X-User-ID"),
		_mapping_value(headers, "X-APG-User-ID"),
		_mapping_value(query_params, "user_id"),
		_mapping_value(session, "user_id"),
		os.getenv("APG_USER_ID"),
	], default_user)

	tenant_id = _first_text([
		_object_value(state_user, "tenant_id"),
		_object_value(state, "tenant_id"),
		_object_value(g, "tenant_id"),
		_mapping_value(headers, "X-Tenant-ID"),
		_mapping_value(headers, "X-APG-Tenant-ID"),
		_mapping_value(headers, "X-Organization-ID"),
		_mapping_value(query_params, "tenant_id"),
		_mapping_value(query_params, "tenant"),
		_mapping_value(session, "tenant_id"),
		os.getenv("APG_TENANT_ID"),
	], default_tenant)

	permissions = (
		_split_permissions(_object_value(state_user, "permissions"))
		or _split_permissions(_object_value(state, "permissions"))
		or _split_permissions(_object_value(g, "user_permissions"))
		or _split_permissions(_mapping_value(headers, "X-APG-Permissions"))
		or _split_permissions(_mapping_value(headers, "X-Permissions"))
		or default_permissions
	)

	return {
		"user_id": user_id,
		"tenant_id": tenant_id,
		"permissions": permissions,
	}
