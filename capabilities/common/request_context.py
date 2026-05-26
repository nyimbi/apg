"""Shared request-context helpers for lightweight capability blueprints."""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Optional


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


def _mapping_value(source: Any, name: str) -> Any:
	if source is None:
		return None
	getter = getattr(source, "get", None)
	if getter is None:
		return None
	return getter(name)


def _payload_values(payload: Optional[Dict[str, Any]], keys: Iterable[str]) -> list[Any]:
	if not payload:
		return []
	return [payload.get(key) for key in keys]


def get_tenant_id_from_context(payload: Optional[Dict[str, Any]] = None) -> str:
	"""Resolve tenant identity from payload, Flask context, request metadata, or fallback."""
	candidates: list[Any] = _payload_values(payload, ["tenant_id", "tenant", "organization_id"])

	try:
		from flask import g, has_request_context, request, session
	except Exception:
		g = None
		has_request_context = lambda: False
		request = None
		session = None

	if has_request_context():
		current_tenant = _object_value(g, "current_tenant")
		current_user = _object_value(g, "current_user")
		fab_user = _object_value(g, "user")
		candidates.extend([
			_mapping_value(session, "tenant_id"),
			_object_value(g, "tenant_id"),
			_object_value(current_tenant, "tenant_id") or current_tenant,
			_object_value(current_user, "tenant_id"),
			_object_value(fab_user, "tenant_id"),
			_mapping_value(request.headers, "X-Tenant-ID"),
			_mapping_value(request.headers, "X-APG-Tenant-ID"),
			_mapping_value(request.headers, "X-Organization-ID"),
			_mapping_value(request.args, "tenant_id"),
			_mapping_value(request.args, "tenant"),
			_mapping_value(request.environ, "APG_TENANT_ID"),
		])

	try:
		from apg.core.context import get_current_context
		context = get_current_context()
	except Exception:
		context = None
	candidates.append(_object_value(context, "tenant_id"))

	candidates.extend([
		os.getenv("APG_DEFAULT_TENANT_ID"),
		os.getenv("APG_TENANT_ID"),
		"default",
	])

	for candidate in candidates:
		tenant_id = _clean_text(candidate)
		if tenant_id:
			return tenant_id
	return "default"
