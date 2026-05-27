"""Request context helpers for Product Information Management."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from flask import g, has_request_context, request, session


def _clean_text(value: Any) -> Optional[str]:
	if value is None:
		return None
	text = str(value).strip()
	return text or None


def _object_value(source: Any, name: str) -> Any:
	if source is None:
		return None
	if isinstance(source, dict):
		return source.get(name)
	return getattr(source, name, None)


def _split_permissions(value: Any) -> List[str]:
	if value is None:
		return []
	if isinstance(value, (list, tuple, set)):
		return [text for item in value if (text := _clean_text(item))]
	return [text for item in str(value).replace(",", " ").split() if (text := _clean_text(item))]


def permission_matches(granted_permission: str, required_permission: str) -> bool:
	"""Return whether a granted APG permission covers the required permission."""
	granted = _clean_text(granted_permission)
	required = _clean_text(required_permission)
	if not granted or not required:
		return False
	if granted in {"*", "admin", "plm.admin"}:
		return True
	if granted == required:
		return True
	for wildcard_suffix, separator in ((".*", "."), (":*", ":")):
		if granted.endswith(wildcard_suffix):
			prefix = granted[: -len(wildcard_suffix)]
			return required == prefix or required.startswith(f"{prefix}{separator}")
	return False


def get_tenant_id_from_request(payload: Optional[Dict[str, Any]] = None) -> str:
	"""Resolve tenant ID from payload, Flask context, session, request metadata, or fallback."""
	default_tenant = os.getenv("APG_DEFAULT_TENANT_ID", os.getenv("APG_TENANT_ID", "default"))
	if not has_request_context():
		return default_tenant

	current_tenant = getattr(g, "current_tenant", None)
	current_user = getattr(g, "current_user", None)
	fab_user = getattr(g, "user", None)
	candidates: List[Any] = []
	if payload:
		candidates.extend(
			[
				payload.get("tenant_id"),
				payload.get("tenant"),
				payload.get("organization_id"),
			]
		)

	candidates.extend(
		[
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
		]
	)

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
		candidates.extend(
			[
				payload.get("user_id"),
				payload.get("current_user_id"),
				payload.get("created_by"),
				payload.get("updated_by"),
			]
		)

	candidates.extend(
		[
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
		]
	)

	for candidate in candidates:
		user_id = _clean_text(candidate)
		if user_id:
			return user_id
	return default_user


def get_current_permissions(payload: Optional[Dict[str, Any]] = None) -> List[str]:
	"""Resolve APG permissions from payload, Flask context, session, headers, or environment."""
	candidates: List[Any] = []
	if payload:
		candidates.extend(
			[
				payload.get("permissions"),
				payload.get("permission"),
				payload.get("scopes"),
				payload.get("scope"),
			]
		)

	if has_request_context():
		current_user = getattr(g, "current_user", None)
		fab_user = getattr(g, "user", None)
		candidates.extend(
			[
				getattr(g, "permissions", None),
				getattr(g, "user_permissions", None),
				_object_value(current_user, "permissions"),
				_object_value(current_user, "permission"),
				_object_value(current_user, "scopes"),
				_object_value(current_user, "scope"),
				_object_value(fab_user, "permissions"),
				_object_value(fab_user, "permission"),
				_object_value(fab_user, "scopes"),
				_object_value(fab_user, "scope"),
				session.get("permissions"),
				session.get("user_permissions"),
				request.headers.get("X-APG-Permissions"),
				request.headers.get("X-Permissions"),
			]
		)

	candidates.append(os.getenv("APG_DEFAULT_PERMISSIONS"))
	for candidate in candidates:
		permissions = _split_permissions(candidate)
		if permissions:
			return permissions
	return []


def has_current_permission(permission: str, payload: Optional[Dict[str, Any]] = None) -> bool:
	"""Check the resolved APG permission set for a required permission."""
	return any(permission_matches(granted, permission) for granted in get_current_permissions(payload))
