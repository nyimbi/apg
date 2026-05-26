"""Request context helpers for Stock Tracking & Control."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from flask import g, has_request_context, request


def _clean_text(value: Any) -> Optional[str]:
	if value is None:
		return None
	text = str(value).strip()
	return text or None


def _candidate_payload_values(payload: Optional[Dict[str, Any]], keys: List[str]) -> List[Any]:
	if not payload:
		return []
	return [payload.get(key) for key in keys]


def get_tenant_id_from_request(payload: Optional[Dict[str, Any]] = None) -> str:
	"""Resolve tenant identity from payload, Flask context, request metadata, or fallback."""
	default_tenant = os.getenv("APG_DEFAULT_TENANT_ID", "default_tenant")
	if not has_request_context():
		return default_tenant

	candidates: List[Any] = _candidate_payload_values(payload, ["tenant_id", "tenant", "organization_id"])

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

	for candidate in candidates:
		resolved = _clean_text(candidate)
		if resolved:
			return resolved
	return default_tenant


def get_current_user_id(payload: Optional[Dict[str, Any]] = None) -> str:
	"""Resolve actor identity from payload, Flask context, request metadata, or fallback."""
	default_user = os.getenv("APG_DEFAULT_USER_ID", os.getenv("APG_USER_ID", "system"))
	if not has_request_context():
		return default_user

	candidates: List[Any] = _candidate_payload_values(
		payload,
		["user_id", "current_user_id", "performed_by", "acknowledged_by", "resolved_by"],
	)

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

	for candidate in candidates:
		resolved = _clean_text(candidate)
		if resolved:
			return resolved
	return default_user
