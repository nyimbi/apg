"""Context and executable-syntax regressions for common MFA surfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, g, request, session


REPO_ROOT = Path(__file__).resolve().parents[1]
API_PATH = REPO_ROOT / "capabilities" / "common" / "mfau" / "api.py"
VIEWS_PATH = REPO_ROOT / "capabilities" / "common" / "mfau" / "views.py"


def _api_context_helpers() -> dict[str, Any]:
	source = API_PATH.read_text(encoding="utf-8")
	start = source.index("def _clean_text")
	end = source.index("# Rate limiting decorator")
	namespace: dict[str, Any] = {
		"Any": Any,
		"Dict": Dict,
		"List": List,
		"Optional": Optional,
		"g": g,
		"os": __import__("os"),
		"request": request,
		"session": session,
	}
	exec(compile(source[start:end], str(API_PATH), "exec"), namespace)
	return namespace


def _view_context_helpers() -> dict[str, Any]:
	source = VIEWS_PATH.read_text(encoding="utf-8")
	start = source.index("def _clean_text")
	end = source.index("class MFAUserProfileView")
	namespace: dict[str, Any] = {
		"g": g,
		"os": __import__("os"),
		"request": request,
		"session": session,
	}
	exec(compile(source[start:end], str(VIEWS_PATH), "exec"), namespace)
	return namespace


def test_mfau_surfaces_no_longer_use_fixed_demo_identity():
	for path in (API_PATH, VIEWS_PATH):
		source = path.read_text(encoding="utf-8")
		for stale_text in (
			"'demo_user'",
			"'demo_tenant'",
			'"demo_user"',
			'"demo_tenant"',
			"request.headers.get('X-User-ID', 'demo_user')",
			"request.headers.get('X-Tenant-ID', 'demo_tenant')",
		):
			assert stale_text not in source

	api_source = API_PATH.read_text(encoding="utf-8")
	assert "inspect.iscoroutinefunction(f)" in api_source
	assert "return await f(*args, **kwargs)" in api_source
	assert "async def post(self):" in api_source


def test_mfau_api_context_resolves_request_session_headers_and_env(monkeypatch):
	helpers = _api_context_helpers()
	resolve_user = helpers["_resolve_current_user_id"]
	resolve_tenant = helpers["_resolve_current_tenant_id"]
	app = Flask(__name__)
	app.secret_key = "test"

	monkeypatch.setenv("APG_DEFAULT_USER_ID", "env-user")
	monkeypatch.setenv("APG_DEFAULT_TENANT_ID", "env-tenant")
	with app.test_request_context("/mfa"):
		assert resolve_user() == "env-user"
		assert resolve_tenant() == "env-tenant"

	with app.test_request_context("/mfa?user_id=query-user&tenant=query-tenant", headers={"X-APG-User-ID": "header-user"}):
		session["user_id"] = "session-user"
		session["tenant_id"] = "session-tenant"
		request.current_user = {"user_id": "request-user", "tenant_id": "request-tenant"}
		assert resolve_user() == "request-user"
		assert resolve_tenant() == "request-tenant"

	with app.test_request_context("/mfa", headers={"X-APG-User-ID": "header-user", "X-APG-Tenant-ID": "header-tenant"}):
		assert resolve_user() == "header-user"
		assert resolve_tenant() == "header-tenant"


def test_mfau_views_context_resolves_flask_context_before_headers():
	helpers = _view_context_helpers()
	resolve_user = helpers["_resolve_current_user_id"]
	resolve_tenant = helpers["_resolve_current_tenant_id"]
	app = Flask(__name__)
	app.secret_key = "test"

	with app.test_request_context("/mfa/dashboard?user_id=query-user&tenant=query-tenant", headers={"X-APG-User-ID": "header-user"}):
		session["user_id"] = "session-user"
		session["tenant_id"] = "session-tenant"
		g.current_user = {"user_id": "g-user", "tenant_id": "g-tenant"}
		assert resolve_user() == "g-user"
		assert resolve_tenant() == "g-tenant"

	with app.test_request_context("/mfa/dashboard", headers={"X-APG-User-ID": "header-user", "X-APG-Tenant-ID": "header-tenant"}):
		assert resolve_user() == "header-user"
		assert resolve_tenant() == "header-tenant"
