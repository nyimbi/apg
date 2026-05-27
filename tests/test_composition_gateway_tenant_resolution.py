"""Tenant context regressions for API Service Mesh gateway dependencies."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Optional
import asyncio

from fastapi import HTTPException


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


def _api_dependency_helpers() -> dict[str, Any]:
	source = API_PATH.read_text(encoding="utf-8")
	start = source.index("async def get_db_session")
	end = source.index("# =============================================================================\n# Service Management Endpoints")
	namespace: dict[str, Any] = {
		"ASMService": object,
		"Any": Any,
		"AsyncSession": object,
		"HTTPException": HTTPException,
		"Request": object,
		"asyncio": asyncio,
	}
	exec(compile(source[start:end], str(API_PATH), "exec"), namespace)
	return namespace


def test_gateway_api_delegates_tenant_resolution():
	source = API_PATH.read_text(encoding="utf-8")

	assert 'return "default_tenant"' not in source
	assert '"api_user"' not in source
	assert "from .context import get_current_user_id_from_request, get_tenant_id_from_request" in source
	assert "async def get_tenant_id(request: Request) -> str:" in source
	assert "return get_tenant_id_from_request(request)" in source
	assert "async def get_user_id(request: Request) -> str:" in source
	assert "return get_current_user_id_from_request(request)" in source
	assert "created_by=user_id" in source
	assert "updated_by=user_id" in source
	assert "from contextlib import asynccontextmanager" in source
	assert "from datetime import datetime, timedelta, timezone" in source
	assert "return None as placeholder" not in source
	assert "async def get_db_session(request: Request) -> AsyncSession:" in source
	assert "async def get_asm_service(request: Request) -> ASMService:" in source
	assert "async def _resolve_app_state_dependency(" in source


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


def test_gateway_user_resolver_precedence(monkeypatch):
	resolver = _context_helpers()["get_current_user_id_from_request"]

	class State:
		pass

	class Request:
		def __init__(self):
			self.state = State()
			self.headers = {}
			self.query_params = {}
			self.scope = {}

	monkeypatch.setenv("APG_DEFAULT_USER_ID", "gateway-env-user")
	assert resolver() == "gateway-env-user"

	request = Request()
	request.headers["X-User-ID"] = "header-user"
	request.query_params["user_id"] = "query-user"
	request.scope["user_id"] = "scope-user"
	request.state.user_id = "state-user"
	assert resolver(request) == "state-user"

	request = Request()
	request.headers["X-APG-User-ID"] = "apg-header-user"
	request.query_params["user_id"] = "query-user"
	assert resolver(request) == "apg-header-user"

	request = Request()
	request.query_params["user"] = "query-user"
	request.scope["user_id"] = "scope-user"
	assert resolver(request) == "query-user"

	request = Request()
	request.scope["apg_user_id"] = "scope-user"
	assert resolver(request) == "scope-user"


def test_gateway_dependencies_resolve_from_app_state():
	helpers = _api_dependency_helpers()
	resolve = helpers["_resolve_app_state_dependency"]
	get_db_session = helpers["get_db_session"]
	get_asm_service = helpers["get_asm_service"]

	db_session = object()
	asm_service = object()
	request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
		db_session=db_session,
		asm_service_factory=lambda: asm_service,
	)))

	assert asyncio.run(resolve(request, ("db_session",), ())) is db_session
	assert asyncio.run(get_db_session(request)) is db_session
	assert asyncio.run(get_asm_service(request)) is asm_service


def test_gateway_dependencies_fail_fast_when_unconfigured():
	helpers = _api_dependency_helpers()
	get_asm_service = helpers["get_asm_service"]
	request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))

	try:
		asyncio.run(get_asm_service(request))
	except HTTPException as exc:
		assert exc.status_code == 503
		assert "ASM service provider is not configured" in exc.detail
	else:
		raise AssertionError("Expected missing ASM service provider to raise HTTPException")
