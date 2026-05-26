"""Request context helpers for Accounts Payable APIs."""

from __future__ import annotations

import os
from typing import Any, Iterable, Optional


DEFAULT_PERMISSIONS = [
	"ap.read",
	"ap.write",
	"ap.approve_invoice",
	"ap.process_payment",
	"ap.vendor_admin",
	"ap.admin",
]
DEFAULT_ROLES = ["ap_manager"]


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


def _split_values(value: Any) -> list[str]:
	if value is None:
		return []
	if isinstance(value, str):
		values = value.split(",")
	else:
		values = value
	return [text for item in values if (text := _clean_text(item))]


def resolve_apg_user_context(request: Any = None) -> dict[str, Any]:
	"""Resolve Accounts Payable user context from APG request state, headers, query, or environment."""
	default_user = os.getenv("APG_DEFAULT_USER_ID", os.getenv("APG_USER_ID", "system"))
	default_tenant = os.getenv("APG_DEFAULT_TENANT_ID", "default_tenant")
	default_permissions = _split_values(os.getenv("APG_APY_PERMISSIONS")) or DEFAULT_PERMISSIONS
	default_roles = _split_values(os.getenv("APG_APY_ROLES")) or DEFAULT_ROLES

	state_user = _state_user(request)
	state = _object_value(request, "state")
	headers = _object_value(request, "headers") or {}
	query_params = _object_value(request, "query_params") or {}

	return {
		"user_id": _first_text([
			_object_value(state_user, "user_id"),
			_object_value(state_user, "id"),
			_object_value(state_user, "username"),
			_object_value(state, "user_id"),
			_mapping_value(headers, "X-User-ID"),
			_mapping_value(headers, "X-APG-User-ID"),
			_mapping_value(query_params, "user_id"),
			os.getenv("APG_USER_ID"),
		], default_user),
		"tenant_id": _first_text([
			_object_value(state_user, "tenant_id"),
			_object_value(state, "tenant_id"),
			_mapping_value(headers, "X-Tenant-ID"),
			_mapping_value(headers, "X-APG-Tenant-ID"),
			_mapping_value(headers, "X-Organization-ID"),
			_mapping_value(query_params, "tenant_id"),
			_mapping_value(query_params, "tenant"),
			os.getenv("APG_TENANT_ID"),
		], default_tenant),
		"permissions": (
			_split_values(_object_value(state_user, "permissions"))
			or _split_values(_object_value(state, "permissions"))
			or _split_values(_mapping_value(headers, "X-APG-Permissions"))
			or _split_values(_mapping_value(headers, "X-Permissions"))
			or default_permissions
		),
		"roles": (
			_split_values(_object_value(state_user, "roles"))
			or _split_values(_object_value(state, "roles"))
			or _split_values(_mapping_value(headers, "X-APG-Roles"))
			or _split_values(_mapping_value(headers, "X-Roles"))
			or default_roles
		),
	}
