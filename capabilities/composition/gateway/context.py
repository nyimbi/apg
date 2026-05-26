"""Request context helpers for the APG API Service Mesh gateway."""

from __future__ import annotations

import os
from typing import Any, Iterable, Optional


def _clean_text(value: Any) -> Optional[str]:
	"""Return a non-empty stripped string or None."""
	if value is None:
		return None
	text = str(value).strip()
	return text or None


def _mapping_get(mapping: Any, keys: Iterable[str]) -> Optional[str]:
	"""Read the first present key from a dict-like or Starlette header/query object."""
	if mapping is None:
		return None
	for key in keys:
		try:
			value = mapping.get(key)
		except AttributeError:
			value = None
		except Exception:
			value = None
		clean_value = _clean_text(value)
		if clean_value:
			return clean_value
	return None


def get_tenant_id_from_request(request: Any = None) -> str:
	"""Resolve tenant ID from a FastAPI request with an environment fallback."""
	default_tenant = os.getenv("APG_DEFAULT_TENANT_ID", "default_tenant")
	if request is None:
		return default_tenant

	for candidate in (
		getattr(getattr(request, "state", None), "tenant_id", None),
		getattr(getattr(request, "state", None), "current_tenant", None),
		_mapping_get(
			getattr(request, "headers", None),
			("X-Tenant-ID", "X-APG-Tenant-ID", "X-Organization-ID"),
		),
		_mapping_get(
			getattr(request, "query_params", None),
			("tenant_id", "tenant"),
		),
		_mapping_get(
			getattr(request, "scope", None),
			("apg_tenant_id", "tenant_id"),
		),
	):
		tenant_id = _clean_text(candidate)
		if tenant_id:
			return tenant_id

	return default_tenant
