"""Context resolution regressions for Geo-Spatial Services APIs."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
CAPABILITY_PATH = REPO_ROOT / "capabilities" / "common" / "geos"
CONTEXT_PATH = CAPABILITY_PATH / "context.py"
API_PATH = CAPABILITY_PATH / "api.py"


def _context_helpers() -> dict[str, Any]:
	source = CONTEXT_PATH.read_text(encoding="utf-8")
	start = source.index("def _clean_text")
	namespace: dict[str, Any] = {
		"Any": Any,
		"Iterable": Iterable,
		"Optional": Optional,
		"os": __import__("os"),
	}
	exec(compile(source[start:], str(CONTEXT_PATH), "exec"), namespace)
	return namespace


def _request(headers: dict[str, str] | None = None, query: dict[str, str] | None = None, state: Any = None):
	return SimpleNamespace(
		headers=headers or {},
		query_params=query or {},
		state=state or SimpleNamespace(),
	)


def test_geos_api_delegates_request_context_resolution():
	source = API_PATH.read_text(encoding="utf-8")

	assert "from .context import resolve_current_user_id, resolve_tenant_id" in source
	assert "Request" in source
	assert 'return "user_123"' not in source
	assert 'return "tenant_123"' not in source
	assert "decode JWT and extract user ID" not in source
	assert "decode JWT and extract tenant ID" not in source
	assert "return resolve_current_user_id(request)" in source
	assert "return resolve_tenant_id(request)" in source


def test_geos_context_resolves_user_and_tenant(monkeypatch):
	helpers = _context_helpers()
	resolve_user = helpers["resolve_current_user_id"]
	resolve_tenant = helpers["resolve_tenant_id"]

	monkeypatch.setenv("APG_DEFAULT_USER_ID", "geos-env-user")
	monkeypatch.setenv("APG_DEFAULT_TENANT_ID", "geos-env-tenant")
	assert resolve_user(_request()) == "geos-env-user"
	assert resolve_tenant(_request()) == "geos-env-tenant"

	state = SimpleNamespace(current_user={"user_id": "state-user", "tenant_id": "state-tenant"})
	request = _request(headers={"X-User-ID": "header-user", "X-Tenant-ID": "header-tenant"}, state=state)
	assert resolve_user(request) == "state-user"
	assert resolve_tenant(request) == "state-tenant"

	request = _request(headers={"X-APG-User-ID": "header-user", "X-APG-Tenant-ID": "header-tenant"})
	assert resolve_user(request) == "header-user"
	assert resolve_tenant(request) == "header-tenant"

	request = _request(query={"user_id": "query-user", "tenant_id": "query-tenant"})
	assert resolve_user(request) == "query-user"
	assert resolve_tenant(request) == "query-tenant"
