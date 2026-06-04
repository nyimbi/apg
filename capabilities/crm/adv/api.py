"""Dependency-light API helpers for APG advanced CRM analytics."""

from __future__ import annotations

import os as _os
from typing import Any, Dict, List, Optional
try:
	from starlette.requests import Request
except ImportError:
	Request = Any  # type: ignore

try:
	from .capability_contract import get_capability_contract
	from .service import AdvancedCRMService
except ImportError:
	from capability_contract import get_capability_contract
	from service import AdvancedCRMService


import os  # noqa: E401 — needed for test exec context


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
	return getter(name) if getter else None


async def get_current_user(request: Request, credentials: Any = None) -> Dict[str, Any]:
	"""Resolve user identity from request state, headers, or environment.

	Priority: state.current_user > X-APG-* headers > APG_DEFAULT_* env vars
	"""
	# 1. Check state.current_user (set by auth middleware)
	state = getattr(request, "state", None)
	current_user = _object_value(state, "current_user")
	if isinstance(current_user, dict) and current_user.get("user_id"):
		return {
			"user_id": current_user["user_id"],
			"tenant_id": current_user.get("tenant_id", "default"),
			"roles": current_user.get("roles", ["crm_user"]),
		}

	# 2. Check headers
	headers = getattr(request, "headers", {})
	header_user = _mapping_value(headers, "X-APG-User-ID")
	header_tenant = _mapping_value(headers, "X-APG-Tenant-ID")
	header_roles_raw = _mapping_value(headers, "X-APG-Roles")
	if header_user:
		roles = [r.strip() for r in header_roles_raw.split(",")] if header_roles_raw else ["crm_user"]
		return {
			"user_id": header_user,
			"tenant_id": header_tenant or os.getenv("APG_DEFAULT_TENANT_ID", "default"),
			"roles": roles,
		}

	# 3. Env vars fallback
	return {
		"user_id": os.getenv("APG_DEFAULT_USER_ID", "anonymous"),
		"tenant_id": os.getenv("APG_DEFAULT_TENANT_ID", "default"),
		"roles": ["crm_user"],
	}


def get_tenant_id(request: Any = None) -> str:
	"""Resolve tenant id from request context or environment."""
	try:
		from ..common.request_context import get_tenant_id_from_context
		return get_tenant_id_from_context()
	except Exception:
		return os.getenv("APG_DEFAULT_TENANT_ID", "default")


_SERVICE = AdvancedCRMService()


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


def create_record(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.create_record(payload)


def list_records(tenant_id: str = "default") -> list[dict[str, Any]]:
	return _SERVICE.list_records(tenant_id)


def create_account(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.create_account(payload["account_id"], payload.get("tenant_id", "default"), payload["name"], payload["owner"], payload["segment"], payload.get("territory"))


def create_lead(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.create_lead(payload["lead_id"], payload.get("tenant_id", "default"), payload["name"], payload["source"], payload.get("score"))


def create_opportunity(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.create_opportunity(payload["opportunity_id"], payload.get("tenant_id", "default"), payload["account_id"], payload["name"], payload["stage"], payload["amount"], payload["close_date"])


def register_crm_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.register_crm_agent(payload.get("tenant_id", "default"), payload["name"], payload["runtime"], payload["role"], payload.get("instructions", ""))


def service() -> AdvancedCRMService:
	return _SERVICE
