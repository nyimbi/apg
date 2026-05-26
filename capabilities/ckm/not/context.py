"""Request context helpers for CKM notification capabilities."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from flask import g, has_request_context, request


def _clean_text(value: Any) -> Optional[str]:
	if value is None:
		return None
	text = str(value).strip()
	return text or None


def _payload_values(payload: Optional[Dict[str, Any]], keys: List[str]) -> List[Any]:
	if not payload:
		return []
	return [payload.get(key) for key in keys]


def get_tenant_id_from_context(payload: Optional[Dict[str, Any]] = None) -> str:
	"""Resolve tenant identity from payload, Flask context, request metadata, or fallback."""
	candidates: List[Any] = _payload_values(payload, ["tenant_id", "tenant", "organization_id"])

	if has_request_context():
		current_tenant = getattr(g, "current_tenant", None)
		current_user = getattr(g, "current_user", None)
		fab_user = getattr(g, "user", None)
		candidates.extend(
			[
				getattr(g, "tenant_id", None),
				getattr(current_tenant, "tenant_id", current_tenant),
				getattr(current_user, "tenant_id", None),
				getattr(fab_user, "tenant_id", None),
				request.headers.get("X-Tenant-ID"),
				request.headers.get("X-APG-Tenant-ID"),
				request.headers.get("X-Organization-ID"),
				request.args.get("tenant_id"),
				request.args.get("tenant"),
				request.environ.get("APG_TENANT_ID"),
			]
		)

	candidates.extend([os.getenv("APG_DEFAULT_TENANT_ID"), os.getenv("APG_TENANT_ID")])

	for candidate in candidates:
		tenant_id = _clean_text(candidate)
		if tenant_id:
			return tenant_id
	return "default"


def get_current_user_id(payload: Optional[Dict[str, Any]] = None) -> str:
	"""Resolve user identity from payload, Flask context, request metadata, or fallback."""
	candidates: List[Any] = _payload_values(
		payload,
		["user_id", "current_user_id", "recipient_id", "performed_by"],
	)

	if has_request_context():
		current_user = getattr(g, "current_user", None)
		fab_user = getattr(g, "user", None)
		candidates.extend(
			[
				getattr(g, "user_id", None),
				getattr(current_user, "username", None),
				getattr(current_user, "id", None),
				getattr(fab_user, "username", None),
				getattr(fab_user, "id", None),
				request.headers.get("X-User-ID"),
				request.headers.get("X-APG-User-ID"),
				request.args.get("user_id"),
				request.environ.get("APG_USER_ID"),
			]
		)

	candidates.extend([os.getenv("APG_DEFAULT_USER_ID"), os.getenv("APG_USER_ID")])

	for candidate in candidates:
		user_id = _clean_text(candidate)
		if user_id:
			return user_id
	return "system"
