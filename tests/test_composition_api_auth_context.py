"""Authentication context regressions for composition APIs."""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from starlette.requests import Request


REPO_ROOT = Path(__file__).resolve().parents[1]
EVENTS_API_PATH = REPO_ROOT / "capabilities" / "composition" / "events" / "api.py"
CONFIG_API_PATH = REPO_ROOT / "capabilities" / "composition" / "config" / "api.py"


class AuthenticationError(Exception):
	pass


class JWTError(Exception):
	pass


class _JWT:
	def decode(self, token: str, key: str, algorithms: list[str]) -> dict[str, Any]:
		_ = key, algorithms
		return _jwt_claims(token)


def _request(path: str = "/api", headers: dict[str, str] | None = None) -> Request:
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


def _jwt_claims(token: str) -> dict[str, Any]:
	payload = token.split(".")[1]
	padding = "=" * (-len(payload) % 4)
	return json.loads(base64.urlsafe_b64decode(f"{payload}{padding}".encode("ascii")).decode("utf-8"))


def _event_helpers() -> dict[str, Any]:
	source = EVENTS_API_PATH.read_text(encoding="utf-8")
	start = source.index("def _clean_text")
	end = source.index("async def get_event_streaming_service")
	namespace: dict[str, Any] = {
		"Any": Any,
		"Dict": Dict,
		"Depends": Depends,
		"HTTPAuthorizationCredentials": HTTPAuthorizationCredentials,
		"HTTPBearer": HTTPBearer,
		"HTTPException": HTTPException,
		"Optional": Optional,
		"Request": Request,
		"base64": __import__("base64"),
		"binascii": __import__("binascii"),
		"json": json,
		"os": __import__("os"),
		"security": HTTPBearer(),
	}
	exec(compile(source[start:end], str(EVENTS_API_PATH), "exec"), namespace)
	return namespace


def _config_helpers() -> dict[str, Any]:
	source = CONFIG_API_PATH.read_text(encoding="utf-8")
	start = source.index("def _clean_text")
	end = source.index("# ==================== Dependency Injection")
	namespace: dict[str, Any] = {
		"Any": Any,
		"APIKeyHeader": APIKeyHeader,
		"AuthenticationError": AuthenticationError,
		"Depends": Depends,
		"Dict": Dict,
		"HTTPException": HTTPException,
		"JWTError": JWTError,
		"Optional": Optional,
		"Request": Request,
		"Security": Security,
		"api_key_header": APIKeyHeader(name="X-API-Key", auto_error=False),
		"jwt": _JWT(),
		"oauth2_scheme": object(),
		"os": __import__("os"),
	}
	exec(compile(source[start:end], str(CONFIG_API_PATH), "exec"), namespace)
	return namespace


def test_composition_api_auth_no_longer_returns_fixed_identity_placeholders():
	for path in (EVENTS_API_PATH, CONFIG_API_PATH):
		source = path.read_text(encoding="utf-8")
		assert '"api_user"' not in source
		assert '"default_tenant"' not in source
		assert "your-secret-key-here" not in source
		assert "For now, simple validation" not in source


def test_event_api_current_user_resolves_bearer_claims_and_request_context(monkeypatch):
	resolve_user = _event_helpers()["get_current_user"]
	credentials = HTTPAuthorizationCredentials(
		scheme="Bearer",
		credentials=_jwt_token({"sub": "claim-user", "tenant_id": "claim-tenant", "permissions": ["events:write"]}),
	)

	user = asyncio.run(resolve_user(_request("/events"), credentials))
	assert user["user_id"] == "claim-user"
	assert user["tenant_id"] == "claim-tenant"
	assert user["permissions"] == ["events:write"]

	monkeypatch.setenv("APG_USER_ID", "env-user")
	monkeypatch.setenv("APG_TENANT_ID", "env-tenant")
	credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="opaque-token")
	user = asyncio.run(resolve_user(_request("/events", {"X-APG-User-ID": "header-user"}), credentials))
	assert user["user_id"] == "header-user"
	assert user["tenant_id"] == "env-tenant"


def test_config_api_key_context_resolves_headers_query_and_environment(monkeypatch):
	verify_api_key = _config_helpers()["verify_api_key"]

	user = asyncio.run(
		verify_api_key(
			_request(
				"/config?tenant_id=query-tenant",
				{"X-API-Key": "cc_test", "X-APG-User-ID": "header-user"},
			),
			"cc_test",
		)
	)
	assert user["user_id"] == "header-user"
	assert user["tenant_id"] == "query-tenant"

	monkeypatch.setenv("APG_API_KEY_USER_ID", "env-user")
	monkeypatch.setenv("APG_API_KEY_TENANT_ID", "env-tenant")
	user = asyncio.run(verify_api_key(_request("/config"), "cc_test"))
	assert user["user_id"] == "env-user"
	assert user["tenant_id"] == "env-tenant"
