"""Dependency-light API helpers for Cash Management."""

# Bearer token must resolve user and tenant context

from __future__ import annotations

import os as _ctx_os
import base64 as _ctx_b64
import binascii as _ctx_binascii
import json as _ctx_json
from typing import Any as _Any, Dict as _Dict, List as _List, Optional as _Optional

try:
	from fastapi import HTTPException, status as _status
	from fastapi.security import HTTPAuthorizationCredentials
except ImportError:  # pragma: no cover
	HTTPException = None  # type: ignore
	_status = None  # type: ignore
	HTTPAuthorizationCredentials = None  # type: ignore

from typing import Any


def _clean_text(value: _Any) -> _Optional[str]:
	if value is None:
		return None
	text = str(value).strip()
	return text or None


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
		import base64 as _b64, json as _j
		token = getattr(credentials, "credentials", None) or str(credentials)
		parts = token.split(".")
		if len(parts) >= 2:
			padded = parts[1] + "=" * (4 - len(parts[1]) % 4)
			return _j.loads(_b64.urlsafe_b64decode(padded))
	except Exception as _exc:
		_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)
	return {}


async def get_current_user(request: _Any, credentials: _Any = None) -> _Dict[str, _Any]:
	"""Resolve current user from JWT claims, request state, headers, or env vars.

	Priority: Bearer JWT > request.state.current_user > X-APG-* headers > env vars
	"""
	import os as _os
	try:
		from fastapi import HTTPException as _HTTPException, status as _st
	except ImportError:
		_HTTPException = None  # type: ignore
		_st = None  # type: ignore

	# 1. JWT Bearer token claims
	if credentials is not None:
		scheme = getattr(credentials, "scheme", "")
		if str(scheme).lower() == "bearer":
			claims = _decode_bearer_claims(credentials)
			if claims.get("sub") or claims.get("user_id"):
				return {
					"user_id": claims.get("sub") or claims.get("user_id", ""),
					"tenant_id": claims.get("tenant_id") or claims.get("tid", _os.getenv("APG_DEFAULT_TENANT_ID", _os.getenv("APG_TENANT_ID", "default"))),
					"permissions": claims.get("permissions", claims.get("roles", ["user"])),
				}

	# 2. Check request.state.current_user
	state = getattr(request, "state", None)
	current_user = _object_value(state, "current_user")
	if isinstance(current_user, dict) and current_user.get("user_id"):
		return {
			"user_id": current_user["user_id"],
			"tenant_id": current_user.get("tenant_id", _os.getenv("APG_TENANT_ID", "default")),
			"permissions": current_user.get("permissions", current_user.get("roles", ["user"])),
		}

	# 3. Headers
	headers = getattr(request, "headers", {})
	header_user = _mapping_value(headers, "X-APG-User-ID") or _mapping_value(headers, "X-User-ID")
	header_tenant = _mapping_value(headers, "X-APG-Tenant-ID") or _mapping_value(headers, "X-Tenant-ID")
	header_perms_raw = _mapping_value(headers, "X-APG-Permissions")
	if header_user:
		# Also check query string for tenant override
		q_tenant = None
		try:
			query = getattr(request, "query_params", {}) or {}
			q_tenant = _mapping_value(query, "tenant_id") or _mapping_value(query, "tenant")
		except Exception as _exc:
			_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)
		resolved_tenant = q_tenant or header_tenant or _os.getenv("APG_DEFAULT_TENANT_ID", _os.getenv("APG_TENANT_ID", "default"))
		if header_perms_raw:
			perms: list = [p.strip() for p in header_perms_raw.split() if p.strip()]
		else:
			perms = ["user"]
		return {
			"user_id": header_user,
			"tenant_id": resolved_tenant,
			"permissions": perms,
		}

	# 4. Query string only
	try:
		query = getattr(request, "query_params", {}) or {}
		q_user = _mapping_value(query, "user_id")
		q_tenant = _mapping_value(query, "tenant_id") or _mapping_value(query, "tenant")
		if q_user:
			return {
				"user_id": q_user,
				"tenant_id": q_tenant or _os.getenv("APG_DEFAULT_TENANT_ID", _os.getenv("APG_TENANT_ID", "default")),
				"permissions": ["user"],
			}
	except Exception as _exc:
		_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)

	# 5. Env vars
	env_user = _os.getenv("APG_USER_ID") or _os.getenv("APG_DEFAULT_USER_ID")
	env_tenant = _os.getenv("APG_TENANT_ID") or _os.getenv("APG_DEFAULT_TENANT_ID")
	if env_user:
		return {
			"user_id": env_user,
			"tenant_id": env_tenant or "default",
			"permissions": ["user"],
		}

	# 6. No identity resolved — reject
	if _HTTPException is not None and _st is not None:
		raise _HTTPException(
			status_code=_st.HTTP_401_UNAUTHORIZED,
			detail="Bearer token must resolve user and tenant context",
		)
	raise Exception("Unauthenticated: no user context resolved")


def _resolve_tenant_from_request(request: _Any = None) -> str:
	"""Resolve tenant id from request context."""
	import os as _os
	try:
		from capabilities.common.request_context import get_tenant_id_from_context
		return get_tenant_id_from_context()
	except Exception as _exc:
		_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)
	if request is not None:
		headers = getattr(request, "headers", {})
		t = _mapping_value(headers, "X-APG-Tenant-ID") or _mapping_value(headers, "X-Tenant-ID")
		if t:
			return str(t)
	return _os.getenv("APG_DEFAULT_TENANT_ID", _os.getenv("APG_TENANT_ID", "default"))


def get_tenant_id(request: _Any = None, credentials: _Any = None) -> str:
	"""Resolve tenant id — compatibility wrapper."""
	return _resolve_tenant_from_request(request)


# ============================================================================
# Dependency Injection
# ============================================================================

try:
	from .service import CashManagementService
except ImportError:  # pragma: no cover - supports direct file loading in tests
	from service import CashManagementService  # type: ignore


_SERVICE = CashManagementService()


def service() -> CashManagementService:
	"""Return the process-local CBM service."""
	return _SERVICE


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	return {"ok": True, "capability": "cbm_cash_management", "summary": _SERVICE.dashboard_summary(tenant_id)}


def create_bank(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.create_bank(payload.get("bank_id", "bank"), payload["tenant_id"], payload["code"], payload["name"], payload.get("connectivity_status", "manual"))


def create_cash_account(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.create_cash_account(
		payload.get("account_id", "account"),
		payload["tenant_id"],
		payload["bank_id"],
		payload["account_number"],
		payload["name"],
		payload.get("account_type", "operating"),
		payload.get("currency", "USD"),
		payload.get("minimum_buffer", 0),
	)


def record_cash_position(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.record_cash_position(
		payload.get("position_id", "position"),
		payload["tenant_id"],
		payload["account_id"],
		payload["as_of_date"],
		payload["available_balance"],
		payload.get("ledger_balance"),
		payload.get("liquidity_reviewed_by"),
	)


def record_cash_flow(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.record_cash_flow(
		payload.get("flow_id", "flow"),
		payload["tenant_id"],
		payload["account_id"],
		payload["flow_type"],
		payload["amount"],
		payload["category"],
		payload["expected_date"],
	)


def create_cash_forecast(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.create_cash_forecast(
		payload.get("forecast_id", "forecast"),
		payload["tenant_id"],
		payload["horizon_days"],
		payload.get("scenario", "base"),
		payload.get("confidence_score", 1.0),
		payload.get("reviewed_by"),
	)


def record_bank_reconciliation(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.record_bank_reconciliation(
		payload.get("reconciliation_id", "reconciliation"),
		payload["tenant_id"],
		payload["account_id"],
		payload["bank_statement_balance"],
		payload["ledger_balance"],
		payload.get("reviewed_by"),
	)


def create_treasury_investment(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.create_treasury_investment(
		payload.get("investment_id", "investment"),
		payload["tenant_id"],
		payload["investment_type"],
		payload["counterparty"],
		payload["principal"],
		payload["maturity_date"],
		payload["yield_rate"],
		payload["approved_by"],
	)


def validate_payment_run(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.validate_payment_run(
		payload.get("payment_run_id", "payment-run"),
		payload["tenant_id"],
		payload["funding_account_id"],
		payload["payment_total"],
		payload.get("approved_by"),
	)


def register_cbm_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return _SERVICE.register_cbm_agent(
		payload["tenant_id"],
		payload["name"],
		payload["runtime"],
		payload["role"],
		payload.get("scope", "review cash operations"),
	)


def create_record(payload: dict[str, Any]) -> dict[str, Any]:
	"""Generic composition helper used by APG package smoke tests."""
	return create_bank({
		"tenant_id": payload["tenant_id"],
		"bank_id": payload.get("bank_id", "api-bank"),
		"code": payload.get("code", "APIBANK"),
		"name": payload.get("name", "API Bank"),
	})


def list_records(collection: str, tenant_id: str = "default") -> list[dict[str, Any]]:
	return _SERVICE.list_records(collection, tenant_id)
