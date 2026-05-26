"""Context resolution regressions for ESG views."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from flask import Flask, g, has_request_context, request


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTEXT_PATH = REPO_ROOT / "capabilities" / "ecd" / "esg" / "context.py"
API_PATH = REPO_ROOT / "capabilities" / "ecd" / "esg" / "api.py"
VIEWS_PATH = REPO_ROOT / "capabilities" / "ecd" / "esg" / "views.py"


def _context_helpers() -> dict[str, Any]:
	source = CONTEXT_PATH.read_text(encoding="utf-8")
	start = source.index("def _clean_text")
	namespace: dict[str, Any] = {
		"Any": Any,
		"Dict": Dict,
		"List": List,
		"Optional": Optional,
		"g": g,
		"has_request_context": has_request_context,
		"os": __import__("os"),
		"request": request,
	}
	exec(compile(source[start:], str(CONTEXT_PATH), "exec"), namespace)
	return namespace


def _api_context_helpers() -> dict[str, Any]:
	source = API_PATH.read_text(encoding="utf-8")
	start = source.index("def _clean_text")
	end = source.index("async def get_esg_service")
	namespace: dict[str, Any] = {
		"Any": Any,
		"Dict": Dict,
		"List": List,
		"Optional": Optional,
		"Depends": lambda dependency=None: None,
		"HTTPAuthorizationCredentials": SimpleNamespace,
		"Request": SimpleNamespace,
		"os": __import__("os"),
		"security": object(),
	}
	exec(compile(source[start:end], str(API_PATH), "exec"), namespace)
	return namespace


def test_esg_views_delegate_context_resolution():
	source = VIEWS_PATH.read_text(encoding="utf-8")
	api_source = API_PATH.read_text(encoding="utf-8")

	for stale_text in (
		'return "default_tenant"',
		"str(self.appbuilder.sm.get_user().id)",
		"user session/profile",
		'"demo_user"',
		'"demo_tenant"',
		'"esg:admin"',
	):
		assert stale_text not in source
		assert stale_text not in api_source

	assert "from .context import get_current_user_id, get_tenant_id_from_request" in source
	assert source.count("get_tenant_id_from_request()") >= 7
	assert source.count("get_current_user_id(self.appbuilder)") >= 4
	assert "async def get_current_user(" in api_source
	assert "request: Request" in api_source
	assert "headers.get(\"X-APG-User-ID\")" in api_source
	assert "headers.get(\"X-APG-Tenant-ID\")" in api_source
	assert "headers.get(\"X-APG-Permissions\")" in api_source


def test_esg_context_resolves_tenant_and_user_from_request_and_appbuilder(monkeypatch):
	helpers = _context_helpers()
	resolve_tenant = helpers["get_tenant_id_from_request"]
	resolve_user = helpers["get_current_user_id"]
	app = Flask(__name__)

	monkeypatch.setenv("APG_DEFAULT_TENANT_ID", "esg-env-tenant")
	assert resolve_tenant() == "esg-env-tenant"
	assert resolve_user() is None

	with app.test_request_context("/esg?tenant_id=query-tenant", headers={"X-Tenant-ID": "header-tenant"}):
		assert resolve_tenant({"tenant_id": "payload-tenant"}) == "payload-tenant"

	with app.test_request_context("/esg?tenant_id=query-tenant", headers={"X-Tenant-ID": "header-tenant"}):
		g.tenant_id = "context-tenant"
		assert resolve_tenant({}) == "context-tenant"

	with app.test_request_context("/esg?tenant=query-tenant"):
		assert resolve_tenant({}) == "query-tenant"

	with app.test_request_context("/esg", headers={"X-APG-User-ID": "header-user"}):
		assert resolve_user(type("AppBuilder", (), {})()) == "header-user"

	fake_appbuilder = type(
		"AppBuilder",
		(),
		{"sm": type("SM", (), {"get_user": lambda self: type("User", (), {"id": "fab-user"})()})()},
	)()
	assert resolve_user(fake_appbuilder) == "fab-user"


def test_esg_api_auth_context_resolves_from_state_headers_query_and_env(monkeypatch):
	helpers = _api_context_helpers()
	resolve_user = helpers["get_current_user"]

	monkeypatch.setenv("APG_DEFAULT_USER_ID", "env-user")
	monkeypatch.setenv("APG_DEFAULT_TENANT_ID", "env-tenant")
	request = SimpleNamespace(state=SimpleNamespace(), headers={}, query_params={})
	context = asyncio.run(resolve_user(request, SimpleNamespace(scheme="Bearer")))
	assert context["user_id"] == "env-user"
	assert context["tenant_id"] == "env-tenant"
	assert context["permissions"] == ["esg:read"]

	request = SimpleNamespace(
		state=SimpleNamespace(current_user={"user_id": "state-user", "tenant_id": "state-tenant", "permissions": ["esg:write"]}),
		headers={"X-APG-User-ID": "header-user", "X-APG-Tenant-ID": "header-tenant"},
		query_params={"tenant": "query-tenant"},
	)
	context = asyncio.run(resolve_user(request, SimpleNamespace(scheme="Bearer")))
	assert context["user_id"] == "state-user"
	assert context["tenant_id"] == "state-tenant"
	assert context["permissions"] == ["esg:write"]

	request = SimpleNamespace(
		state=SimpleNamespace(),
		headers={"X-APG-User-ID": "header-user", "X-APG-Tenant-ID": "header-tenant", "X-APG-Permissions": "esg:read, esg:write"},
		query_params={"tenant": "query-tenant"},
	)
	context = asyncio.run(resolve_user(request, SimpleNamespace(scheme="Bearer")))
	assert context["user_id"] == "header-user"
	assert context["tenant_id"] == "header-tenant"
	assert context["permissions"] == ["esg:read", "esg:write"]
