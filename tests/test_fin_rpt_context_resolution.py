"""Context resolution regressions for Financial Reporting APIs and views."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, g, has_request_context, request, session


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTEXT_PATH = REPO_ROOT / "capabilities" / "fin" / "rpt" / "context.py"
API_PATH = REPO_ROOT / "capabilities" / "fin" / "rpt" / "api.py"
VIEWS_PATH = REPO_ROOT / "capabilities" / "fin" / "rpt" / "views.py"


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
		"session": session,
	}
	exec(compile(source[start:], str(CONTEXT_PATH), "exec"), namespace)
	return namespace


def test_financial_reporting_surfaces_delegate_context_resolution():
	api_source = API_PATH.read_text(encoding="utf-8")
	views_source = VIEWS_PATH.read_text(encoding="utf-8")
	combined = api_source + "\n" + views_source

	assert "from .context import get_tenant_id_from_request" in api_source
	assert "from .context import get_current_user_id, get_tenant_id_from_request" in views_source
	assert "request.headers.get('X-Tenant-ID', 'default_tenant')" not in api_source
	assert 'return "default_tenant"' not in combined
	assert 'return "default_user"' not in combined
	assert "Implementation depends on APG auth system" not in combined
	assert "Simplified for demonstration" not in combined
	assert api_source.count("get_tenant_id_from_request(") >= 1
	assert views_source.count("get_tenant_id_from_request()") >= 6
	assert views_source.count("get_current_user_id()") >= 2


def test_financial_reporting_context_resolves_tenant_and_user(monkeypatch):
	helpers = _context_helpers()
	resolve_tenant = helpers["get_tenant_id_from_request"]
	resolve_user = helpers["get_current_user_id"]
	app = Flask(__name__)
	app.secret_key = "test-secret"

	monkeypatch.setenv("APG_DEFAULT_TENANT_ID", "rpt-env-tenant")
	monkeypatch.setenv("APG_DEFAULT_USER_ID", "rpt-env-user")
	assert resolve_tenant() == "rpt-env-tenant"
	assert resolve_user() == "rpt-env-user"

	with app.test_request_context("/rpt?tenant_id=query-tenant", headers={"X-Tenant-ID": "header-tenant"}):
		session["tenant_id"] = "session-tenant"
		session["user_id"] = "session-user"
		assert resolve_tenant({"tenant_id": "payload-tenant"}) == "payload-tenant"
		assert resolve_user({"user_id": "payload-user"}) == "payload-user"

	with app.test_request_context("/rpt?tenant_id=query-tenant", headers={"X-Tenant-ID": "header-tenant"}):
		session["tenant_id"] = "session-tenant"
		g.tenant_id = "context-tenant"
		assert resolve_tenant({}) == "context-tenant"

	with app.test_request_context("/rpt", headers={"X-Tenant-ID": "header-tenant"}):
		session["tenant_id"] = "session-tenant"
		session["user_id"] = "session-user"
		g.user = type("User", (), {"tenant_id": "fab-user-tenant", "username": "fab-user"})()
		assert resolve_tenant({}) == "fab-user-tenant"
		assert resolve_user({}) == "fab-user"

	with app.test_request_context("/rpt", headers={"X-APG-User-ID": "header-user"}):
		assert resolve_user({}) == "header-user"

	with app.test_request_context("/rpt", environ_base={"APG_TENANT_ID": "request-env-tenant"}):
		assert resolve_tenant({}) == "request-env-tenant"
