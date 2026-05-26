"""Tenant context regressions for API Service Mesh gateway dependencies."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
API_PATH = REPO_ROOT / "capabilities" / "composition" / "gateway" / "api.py"
CONTEXT_PATH = REPO_ROOT / "capabilities" / "composition" / "gateway" / "context.py"


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


def test_gateway_api_delegates_tenant_resolution():
	source = API_PATH.read_text(encoding="utf-8")

	assert 'return "default_tenant"' not in source
	assert "from .context import get_tenant_id_from_request" in source
	assert "async def get_tenant_id(request: Request) -> str:" in source
	assert "return get_tenant_id_from_request(request)" in source
	assert "from contextlib import asynccontextmanager" in source
	assert "from datetime import datetime, timedelta, timezone" in source


def test_gateway_tenant_resolver_precedence(monkeypatch):
	resolver = _context_helpers()["get_tenant_id_from_request"]

	class State:
		pass

	class Request:
		def __init__(self):
			self.state = State()
			self.headers = {}
			self.query_params = {}
			self.scope = {}

	monkeypatch.setenv("APG_DEFAULT_TENANT_ID", "gateway-env-tenant")
	assert resolver() == "gateway-env-tenant"

	request = Request()
	request.headers["X-Tenant-ID"] = "header-tenant"
	request.query_params["tenant_id"] = "query-tenant"
	request.scope["tenant_id"] = "scope-tenant"
	request.state.tenant_id = "state-tenant"
	assert resolver(request) == "state-tenant"

	request = Request()
	request.headers["X-APG-Tenant-ID"] = "apg-header-tenant"
	request.query_params["tenant_id"] = "query-tenant"
	assert resolver(request) == "apg-header-tenant"

	request = Request()
	request.query_params["tenant"] = "query-tenant"
	request.scope["tenant_id"] = "scope-tenant"
	assert resolver(request) == "query-tenant"

	request = Request()
	request.scope["apg_tenant_id"] = "scope-tenant"
	assert resolver(request) == "scope-tenant"
