"""Dependency-light API helpers for APG enterprise asset management."""

from __future__ import annotations

import os as _ctx_os
import base64 as _ctx_b64
import binascii as _ctx_binascii
import json as _ctx_json
from typing import Any, Any as _Any, Dict as _Dict, List as _List, Optional as _Optional

try:
	from .capability_contract import get_capability_contract
	from .service import EnterpriseAssetManagementService
except ImportError:
	from capability_contract import get_capability_contract
	from service import EnterpriseAssetManagementService


def _clean_text(value: _Any) -> _Optional[str]:
	if value is None:
		return None
	text = str(value).strip()
	return text or None


_ctx_os = __import__("os")
_ctx_b64 = __import__("base64")
_ctx_binascii = __import__("binascii")
_ctx_json = __import__("json")


def _object_value(source: _Any, name: str) -> _Any:
	if source is None:
		return None
	if isinstance(source, dict):
		return source.get(name)
	return getattr(source, name, None)


def _mapping_value(source: _Any, name: str) -> _Any:
	if source is None:
		return None
	getter = getattr(source, "get", None)
	return getter(name) if getter else None


def _decode_bearer_claims(credentials: _Any) -> dict[str, _Any]:
	try:
		token = getattr(credentials, "credentials", None) or str(credentials)
		parts = token.split(".")
		if len(parts) >= 2:
			padded = parts[1] + "=" * (4 - len(parts[1]) % 4)
			return _ctx_json.loads(_ctx_b64.urlsafe_b64decode(padded))
	except Exception as _exc:
		_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)
	return {}


async def get_current_user(request: _Any, credentials: _Any = None) -> _Dict[str, _Any]:
	"""Resolve current user from JWT claims, request state, headers, or env vars.

	Priority: Bearer JWT > request.state.current_user > X-APG-* headers > env vars
	"""
	# 1. JWT Bearer token claims
	if credentials is not None:
		scheme = getattr(credentials, "scheme", "")
		if str(scheme).lower() == "bearer":
			claims = _decode_bearer_claims(credentials)
			if claims.get("sub") or claims.get("user_id"):
				return {
					"user_id": claims.get("sub") or claims.get("user_id", ""),
					"tenant_id": claims.get("tenant_id") or claims.get("tid", _ctx_os.getenv("APG_DEFAULT_TENANT_ID", "default")),
					"permissions": claims.get("permissions", []) or ["eam.asset.view"],
				}

	# 2. Check request.state.current_user
	state = getattr(request, "state", None)
	current_user = _object_value(state, "current_user")
	if isinstance(current_user, dict) and current_user.get("user_id"):
		return {
			"user_id": current_user["user_id"],
			"tenant_id": current_user.get("tenant_id", "default"),
			"permissions": current_user.get("permissions", []) or ["eam.asset.view"],
		}

	# 3. Headers — X-APG-* take priority, fallback to X-* variants
	headers = getattr(request, "headers", {})
	header_user = headers.get("X-APG-User-ID") or headers.get("X-User-ID")
	header_tenant = headers.get("X-APG-Tenant-ID") or headers.get("X-Tenant-ID")
	header_perms_raw = headers.get("X-APG-Permissions")
	if header_user:
		permissions = [p.strip() for p in header_perms_raw.split(",")] if header_perms_raw else ["eam.asset.view"]
		return {
			"user_id": header_user,
			"tenant_id": header_tenant or _ctx_os.getenv("APG_DEFAULT_TENANT_ID", "default"),
			"permissions": permissions,
		}

	# 4. Query string / query params
	try:
		query = getattr(request, "query_params", {}) or {}
		q_user = _mapping_value(query, "user_id")
		q_tenant = _mapping_value(query, "tenant_id") or _mapping_value(query, "tenant")
		if q_user:
			return {
				"user_id": q_user,
				"tenant_id": q_tenant or _ctx_os.getenv("APG_DEFAULT_TENANT_ID", "default"),
				"permissions": ["eam.asset.view"],
			}
	except Exception as _exc:
		_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)

	# 5. Env fallback
	return {
		"user_id": _ctx_os.getenv("APG_DEFAULT_USER_ID", "anonymous"),
		"tenant_id": _ctx_os.getenv("APG_DEFAULT_TENANT_ID", "default"),
		"permissions": ["eam.asset.view"],
	}


def _resolve_tenant_from_request(request: _Any = None) -> str:
	"""Resolve tenant id from request context."""
	try:
		from capabilities.common.request_context import get_tenant_id_from_context
		return get_tenant_id_from_context()
	except Exception as _exc:
		_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)
	if request is not None:
		headers = getattr(request, "headers", {})
		t = headers.get("X-APG-Tenant-ID") or headers.get("X-Tenant-ID")
		if t:
			return str(t)
	return _ctx_os.getenv("APG_DEFAULT_TENANT_ID", "default")


def get_tenant_id(request: _Any = None, credentials: _Any = None) -> str:
	"""Resolve tenant id — compatibility wrapper."""
	return _resolve_tenant_from_request(request)


class Request:  # noqa: D101 — minimal shim so `request: Request` resolves at import time
	"""Minimal request shim for contexts where FastAPI/Starlette is not installed."""
	headers: dict = {}
	query_params: dict = {}
	state: object = object()


async def get_database_session(request: Request = None) -> _Any:
	"""Stub database session dependency."""
	return None


# ============================================================================
# Dependency Injection
# ============================================================================

_SERVICE = EnterpriseAssetManagementService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"ok": True,
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"provides": contract["provides"],
		"requires": contract["requires"],
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		"streaming": contract["streaming"],
		"summary": _SERVICE.dashboard_summary(tenant_id),
	}


def register_location(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.register_location(
		payload["location_id"],
		payload.get("tenant_id", "default"),
		payload["name"],
		payload["location_type"],
		payload.get("parent_location_id"),
	)


def register_asset(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.register_asset(
		payload["asset_id"],
		payload.get("tenant_id", "default"),
		payload["name"],
		payload["owner"],
		payload["category"],
		payload["location_id"],
		payload["criticality"],
		payload.get("health_score", 100),
		payload.get("capitalized", False),
		payload.get("fixed_asset_ref"),
	)


def create_maintenance_plan(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.create_maintenance_plan(
		payload["plan_id"],
		payload.get("tenant_id", "default"),
		payload["asset_record_id"],
		payload["strategy"],
		payload["interval_days"],
		payload.get("condition_source"),
	)


def open_work_order(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.open_work_order(
		payload["work_order_id"],
		payload.get("tenant_id", "default"),
		payload["asset_record_id"],
		payload["title"],
		payload["priority"],
		payload["safety_plan"],
		payload.get("approved_by"),
	)


def complete_work_order(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.complete_work_order(
		payload.get("tenant_id", "default"),
		payload["work_order_record_id"],
		payload["outcome"],
		payload["completed_by"],
	)


def record_inspection(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.record_inspection(
		payload["inspection_id"],
		payload.get("tenant_id", "default"),
		payload["asset_record_id"],
		payload["result"],
		payload["inspector"],
		payload.get("condition_score"),
	)


def record_condition_reading(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.record_condition_reading(
		payload["reading_id"],
		payload.get("tenant_id", "default"),
		payload["asset_record_id"],
		payload["metric"],
		payload["value"],
		payload["unit"],
		payload.get("review_recorded", False),
		payload.get("alert_threshold"),
	)


def reserve_inventory(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.reserve_inventory(
		payload["reservation_id"],
		payload.get("tenant_id", "default"),
		payload["part_id"],
		payload["quantity"],
		payload.get("work_order_record_id"),
	)


def register_eam_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.register_eam_agent(
		payload.get("tenant_id", "default"),
		payload["name"],
		payload["runtime"],
		payload["role"],
		payload.get("instructions", ""),
	)


def create_record(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.create_record(payload)


def list_records(tenant_id: str = "default") -> list[dict[str, Any]]:
	return _SERVICE.list_records(tenant_id)


def service() -> EnterpriseAssetManagementService:
	return _SERVICE
