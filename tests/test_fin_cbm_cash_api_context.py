"""Authentication context regressions for Cash Management API."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.requests import Request


REPO_ROOT = Path(__file__).resolve().parents[1]
API_PATH = REPO_ROOT / "capabilities" / "fin" / "cbm" / "cash_management" / "api.py"


def _request(path: str = "/cash", headers: dict[str, str] | None = None) -> Request:
	raw_headers = [
		(name.lower().encode("latin-1"), value.encode("latin-1"))
		for name, value in (headers or {}).items()
	]
	path_part, _, query = path.partition("?")
	return Request(
		{
			"type": "http",
			"method": "GET",
			"path": path_part,
			"headers": raw_headers,
			"query_string": query.encode("latin-1"),
		}
	)


def _jwt_token(claims: dict[str, Any]) -> str:
	header = {"alg": "none", "typ": "JWT"}
	parts = []
	for value in (header, claims):
		encoded = base64.urlsafe_b64encode(json.dumps(value).encode("utf-8")).decode("ascii")
		parts.append(encoded.rstrip("="))
	return f"{parts[0]}.{parts[1]}."


def _helpers() -> dict[str, Any]:
	source = API_PATH.read_text(encoding="utf-8")
	start = source.index("def _clean_text")
	end = source.index("# ============================================================================\n# Dependency Injection")
	namespace: dict[str, Any] = {
		"Any": Any,
		"Depends": Depends,
		"Dict": Dict,
		"HTTPAuthorizationCredentials": HTTPAuthorizationCredentials,
		"HTTPBearer": HTTPBearer,
		"HTTPException": HTTPException,
		"List": List,
		"Optional": Optional,
		"Request": Request,
		"base64": __import__("base64"),
		"binascii": __import__("binascii"),
		"json": json,
		"os": __import__("os"),
		"security": HTTPBearer(),
		"status": status,
	}
	exec(compile(source[start:end], str(API_PATH), "exec"), namespace)
	return namespace


def test_cash_management_api_no_longer_returns_fixed_auth_context():
	source = API_PATH.read_text(encoding="utf-8")
	assert "'api_" + "user'" not in source
	assert "'default_" + "tenant'" not in source
	assert "This would validate JWT" not in source
	assert "Bearer token must resolve user and tenant context" in source


def test_cash_management_current_user_resolves_claims_headers_and_env(monkeypatch):
	resolve_user = _helpers()["get_current_user"]
	credentials = HTTPAuthorizationCredentials(
		scheme="Bearer",
		credentials=_jwt_token(
			{
				"sub": "claim-user",
				"tenant_id": "claim-tenant",
				"permissions": ["cash_management.read", "cash_management.write"],
			}
		),
	)

	import asyncio

	user = asyncio.run(resolve_user(_request("/cash"), credentials))
	assert user["user_id"] == "claim-user"
	assert user["tenant_id"] == "claim-tenant"
	assert user["permissions"] == ["cash_management.read", "cash_management.write"]

	monkeypatch.setenv("APG_USER_ID", "env-user")
	monkeypatch.setenv("APG_TENANT_ID", "env-tenant")
	credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="opaque-token")
	user = asyncio.run(
		resolve_user(
			_request(
				"/cash?tenant_id=query-tenant",
				{"X-APG-User-ID": "header-user", "X-APG-Permissions": "cash_management.read cash_management.write"},
			),
			credentials,
		)
	)
	assert user["user_id"] == "header-user"
	assert user["tenant_id"] == "query-tenant"
	assert user["permissions"] == ["cash_management.read", "cash_management.write"]


def test_cash_management_current_user_requires_user_and_tenant_context():
	resolve_user = _helpers()["get_current_user"]
	credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="opaque-token")

	import asyncio
	import pytest

	with pytest.raises(HTTPException) as exc_info:
		asyncio.run(resolve_user(_request("/cash"), credentials))

	assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
