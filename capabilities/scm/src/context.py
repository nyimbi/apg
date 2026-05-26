"""Request context helpers for Sourcing & Supplier Selection."""

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
	default_tenant = os.getenv("APG_DEFAULT_TENANT_ID", "default_tenant")
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
