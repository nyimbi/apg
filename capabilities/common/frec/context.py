"""Request context helpers for Facial Recognition APIs."""

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


def _mapping_value(source: Any, name: str) -> Any:
	if source is None:
		return None
	getter = getattr(source, "get", None)
	if getter is None:
		return None
	return getter(name)


def resolve_tenant_id(request: Any = None, g: Any = None) -> str:
	"""Resolve tenant ID from Flask request context, APG headers, query args, or environment."""
	default_tenant = os.getenv("APG_DEFAULT_TENANT_ID", os.getenv("APG_TENANT_ID", "default"))
	headers = _object_value(request, "headers") or {}
	args = _object_value(request, "args") or {}

	return _first_text([
		_object_value(g, "tenant_id"),
		_mapping_value(headers, "X-Tenant-ID"),
		_mapping_value(headers, "X-APG-Tenant-ID"),
		_mapping_value(headers, "X-Organization-ID"),
		_mapping_value(args, "tenant_id"),
		_mapping_value(args, "tenant"),
		os.getenv("APG_TENANT_ID"),
	], default_tenant)
