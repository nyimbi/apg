"""Context resolution regressions for Cache Management API dependencies."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from starlette.requests import Request


REPO_ROOT = Path(__file__).resolve().parents[1]
API_PATH = REPO_ROOT / "capabilities" / "common" / "cach" / "api.py"


def _request(path: str = "/cache", headers: dict[str, str] | None = None) -> Request:
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


def _helpers() -> dict[str, Any]:
	source = API_PATH.read_text(encoding="utf-8")
	start = source.index("def _clean_text")
	end = source.index("# Create FastAPI app")
	namespace: dict[str, Any] = {
		"Any": Any,
		"Optional": Optional,
		"Request": Request,
		"os": __import__("os"),
	}
	exec(compile(source[start:end], str(API_PATH), "exec"), namespace)
	return namespace


def test_cache_api_no_longer_uses_fixed_auth_context_placeholders():
	source = API_PATH.read_text(encoding="utf-8")
	assert 'return "api_' + 'user"' not in source
	assert "In production: extract from JWT token" not in source
	assert "Request" in source


def test_cache_api_context_resolves_state_headers_query_scope_and_env(monkeypatch):
	helpers = _helpers()
	resolve_tenant = helpers["get_current_tenant"]
	resolve_user = helpers["get_current_user"]

	monkeypatch.setenv("APG_DEFAULT_TENANT_ID", "env-tenant")
	monkeypatch.setenv("APG_DEFAULT_USER_ID", "env-user")
	assert resolve_tenant(_request("/cache")) == "env-tenant"
	assert resolve_user(_request("/cache")) == "env-user"

	request = _request(
		"/cache?tenant_id=query-tenant&user_id=query-user",
		{"X-APG-Tenant-ID": "header-tenant", "X-APG-User-ID": "header-user"},
	)
	request.scope["tenant_id"] = "scope-tenant"
	request.scope["user_id"] = "scope-user"
	request.state.tenant_id = "state-tenant"
	request.state.user_id = "state-user"
	assert resolve_tenant(request) == "state-tenant"
	assert resolve_user(request) == "state-user"

	request = _request(
		"/cache?tenant=query-tenant&user=query-user",
		{"X-Tenant-ID": "header-tenant", "X-User-ID": "header-user"},
	)
	assert resolve_tenant(request) == "header-tenant"
	assert resolve_user(request) == "header-user"

	request = _request("/cache?tenant=query-tenant&user=query-user")
	request.scope["apg_tenant_id"] = "scope-tenant"
	request.scope["apg_user_id"] = "scope-user"
	assert resolve_tenant(request) == "query-tenant"
	assert resolve_user(request) == "query-user"
