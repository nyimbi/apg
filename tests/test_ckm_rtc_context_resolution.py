"""Context resolution regressions for CKM real-time collaboration surfaces."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from flask import Flask, g, request, session


REPO_ROOT = Path(__file__).resolve().parents[1]
API_PATH = REPO_ROOT / "capabilities" / "ckm" / "rtc" / "api.py"
VIEWS_PATH = REPO_ROOT / "capabilities" / "ckm" / "rtc" / "views.py"
WEBSOCKET_PATH = REPO_ROOT / "capabilities" / "ckm" / "rtc" / "websocket_manager.py"


def _api_context_helpers() -> dict[str, Any]:
	source = API_PATH.read_text(encoding="utf-8")
	start = source.index("def _clean_text")
	end = source.index("async def require_permission")
	namespace: dict[str, Any] = {
		"Any": Any,
		"Dict": Dict,
		"List": List,
		"Optional": Optional,
		"Request": SimpleNamespace,
		"os": __import__("os"),
	}
	exec(compile(source[start:end], str(API_PATH), "exec"), namespace)
	return namespace


def _view_context_helpers() -> dict[str, Any]:
	source = VIEWS_PATH.read_text(encoding="utf-8")
	start = source.index("def _clean_text")
	end = source.index("# Pydantic models for view forms")
	namespace: dict[str, Any] = {
		"Any": Any,
		"Dict": Dict,
		"List": List,
		"Optional": Optional,
		"g": g,
		"request": request,
		"session": session,
	}
	exec(compile(source[start:end], str(VIEWS_PATH), "exec"), namespace)
	return namespace


def _websocket_context_helpers() -> dict[str, Any]:
	source = WEBSOCKET_PATH.read_text(encoding="utf-8")
	start = source.index("def _clean_text")
	end = source.index("@dataclass\nclass WebSocketConnection")
	namespace: dict[str, Any] = {
		"Any": Any,
		"Dict": Dict,
		"List": List,
		"Optional": Optional,
		"WebSocketServerProtocol": object,
		"os": __import__("os"),
		"parse_qs": __import__("urllib.parse").parse.parse_qs,
		"unquote": __import__("urllib.parse").parse.unquote,
		"urlparse": __import__("urllib.parse").parse.urlparse,
	}
	exec(compile(source[start:end], str(WEBSOCKET_PATH), "exec"), namespace)
	return namespace


def test_ckm_rtc_surfaces_no_longer_use_fixed_collaboration_identity():
	for path in (API_PATH, VIEWS_PATH, WEBSOCKET_PATH):
		source = path.read_text(encoding="utf-8")
		for stale_text in (
			"'user123'",
			"'tenant123'",
			'"current_user_id"',
			'"current_tenant_id"',
			"Mock current user from APG auth",
			"return mock data",
			"rtc:*",
			"http://localhost:5000/some/page",
		):
			assert stale_text not in source

	assert "request: Request" in API_PATH.read_text(encoding="utf-8")
	assert "resolve_connection_context(websocket, path)" in WEBSOCKET_PATH.read_text(encoding="utf-8")


def test_ckm_rtc_api_auth_resolves_state_headers_and_env(monkeypatch):
	helpers = _api_context_helpers()
	resolve_user = helpers["get_current_user"]

	monkeypatch.setenv("APG_DEFAULT_USER_ID", "env-user")
	monkeypatch.setenv("APG_DEFAULT_TENANT_ID", "env-tenant")
	request_obj = SimpleNamespace(state=SimpleNamespace(), headers={}, query_params={})
	context = asyncio.run(resolve_user(request_obj))
	assert context["user_id"] == "env-user"
	assert context["tenant_id"] == "env-tenant"
	assert context["permissions"] == ["rtc:read"]

	request_obj = SimpleNamespace(
		state=SimpleNamespace(current_user={"user_id": "state-user", "tenant_id": "state-tenant", "permissions": ["rtc:write"]}),
		headers={"X-APG-User-ID": "header-user", "X-APG-Tenant-ID": "header-tenant"},
		query_params={"tenant": "query-tenant"},
	)
	context = asyncio.run(resolve_user(request_obj))
	assert context["user_id"] == "state-user"
	assert context["tenant_id"] == "state-tenant"
	assert context["permissions"] == ["rtc:write"]

	request_obj = SimpleNamespace(
		state=SimpleNamespace(),
		headers={"X-APG-User-ID": "header-user", "X-APG-Tenant-ID": "header-tenant", "X-APG-Permissions": "rtc:read, rtc:write"},
		query_params={},
	)
	context = asyncio.run(resolve_user(request_obj))
	assert context["user_id"] == "header-user"
	assert context["tenant_id"] == "header-tenant"
	assert context["permissions"] == ["rtc:read", "rtc:write"]


def test_ckm_rtc_views_resolve_flask_context_before_headers():
	helpers = _view_context_helpers()
	resolve_user = helpers["_resolve_current_user_id"]
	resolve_tenant = helpers["_resolve_current_tenant_id"]
	app = Flask(__name__)
	app.secret_key = "test"

	with app.test_request_context("/rtc/join?user_id=query-user&tenant=query-tenant", headers={"X-APG-User-ID": "header-user"}):
		session["user_id"] = "session-user"
		session["tenant_id"] = "session-tenant"
		g.current_user = {"user_id": "g-user", "tenant_id": "g-tenant"}
		assert resolve_user() == "g-user"
		assert resolve_tenant() == "g-tenant"

	with app.test_request_context("/rtc/join", headers={"X-APG-User-ID": "header-user", "X-APG-Tenant-ID": "header-tenant"}):
		assert resolve_user() == "header-user"
		assert resolve_tenant() == "header-tenant"


def test_ckm_rtc_websocket_context_resolves_path_headers_and_query(monkeypatch):
	helpers = _websocket_context_helpers()
	resolve = helpers["resolve_connection_context"]

	socket = SimpleNamespace(request_headers={})
	assert resolve(socket, "/ws/rtc/path-tenant/path-user?page_url=/inventory") == (
		"path-user",
		"path-tenant",
		"/inventory",
	)

	socket = SimpleNamespace(request_headers={"X-APG-User-ID": "header-user", "X-APG-Tenant-ID": "header-tenant", "Referer": "/from-header"})
	assert resolve(socket, "/ws/rtc/path-tenant/path-user") == (
		"header-user",
		"header-tenant",
		"/from-header",
	)

	monkeypatch.setenv("APG_DEFAULT_USER_ID", "env-user")
	monkeypatch.setenv("APG_DEFAULT_TENANT_ID", "env-tenant")
	socket = SimpleNamespace(request_headers={})
	assert resolve(socket, "/ws/rtc") == ("env-user", "env-tenant", "/ws/rtc")
