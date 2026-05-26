"""Context resolution regressions for Stock Tracking & Control surfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, g, has_request_context, request


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTEXT_PATH = REPO_ROOT / "capabilities" / "scm" / "inv" / "stock_tracking_control" / "context.py"
API_PATH = REPO_ROOT / "capabilities" / "scm" / "inv" / "stock_tracking_control" / "api.py"
VIEWS_PATH = REPO_ROOT / "capabilities" / "scm" / "inv" / "stock_tracking_control" / "views.py"


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


def test_stock_tracking_surfaces_delegate_context_resolution():
	api_source = API_PATH.read_text(encoding="utf-8")
	views_source = VIEWS_PATH.read_text(encoding="utf-8")

	for stale_text in (
		'return "default_tenant"',
		'return "current_user"',
		"TODO: Implement proper tenant resolution from request context",
		"TODO: Implement proper user resolution from request context",
	):
		assert stale_text not in views_source
		assert stale_text not in api_source

	assert "from .context import get_current_user_id, get_tenant_id_from_request" in views_source
	assert "from .context import get_current_user_id, get_tenant_id_from_request" in api_source
	assert views_source.count("get_tenant_id_from_request()") >= 10
	assert views_source.count("get_current_user_id()") >= 1
	assert "get_tenant_id_from_request(payload)" in api_source
	assert "get_current_user_id(payload)" in api_source
	assert "self._get_current_user_id(data)" in api_source


def test_stock_tracking_context_resolves_tenant_and_user(monkeypatch):
	helpers = _context_helpers()
	resolve_tenant = helpers["get_tenant_id_from_request"]
	resolve_user = helpers["get_current_user_id"]
	app = Flask(__name__)

	monkeypatch.setenv("APG_DEFAULT_TENANT_ID", "stc-env-tenant")
	monkeypatch.setenv("APG_DEFAULT_USER_ID", "stc-env-user")
	assert resolve_tenant() == "stc-env-tenant"
	assert resolve_user() == "stc-env-user"

	with app.test_request_context("/stock?tenant_id=query-tenant", headers={"X-Tenant-ID": "header-tenant"}):
		assert resolve_tenant({"tenant_id": "payload-tenant"}) == "payload-tenant"

	with app.test_request_context("/stock?tenant_id=query-tenant", headers={"X-Tenant-ID": "header-tenant"}):
		g.tenant_id = "context-tenant"
		assert resolve_tenant({}) == "context-tenant"

	with app.test_request_context("/stock", headers={"X-Tenant-ID": "header-tenant"}):
		g.user = type("User", (), {"tenant_id": "fab-user-tenant", "username": "fab-user"})()
		assert resolve_tenant({}) == "fab-user-tenant"
		assert resolve_user({}) == "fab-user"

	with app.test_request_context("/stock", headers={"X-APG-User-ID": "header-user"}):
		assert resolve_user({}) == "header-user"

	with app.test_request_context("/stock", environ_base={"APG_USER_ID": "request-env-user"}):
		assert resolve_user({}) == "request-env-user"
