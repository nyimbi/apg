"""Context resolution regressions for Requisitioning surfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, g, has_request_context, request


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTEXT_PATH = REPO_ROOT / "capabilities" / "scm" / "req" / "context.py"
API_PATH = REPO_ROOT / "capabilities" / "scm" / "req" / "api.py"
VIEWS_PATH = REPO_ROOT / "capabilities" / "scm" / "req" / "views.py"


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


def test_requisition_surfaces_delegate_context_resolution():
	api_source = API_PATH.read_text(encoding="utf-8")
	views_source = VIEWS_PATH.read_text(encoding="utf-8")

	for stale_text in (
		'return "default_tenant"',
		'return "current_user"',
		"TODO: Implement tenant resolution",
		"TODO: Get from Flask-Login or similar",
	):
		assert stale_text not in views_source
		assert stale_text not in api_source

	assert "from .context import get_current_user_id, get_tenant_id_from_request" in views_source
	assert "from .context import get_current_user_id, get_tenant_id_from_request" in api_source
	assert views_source.count("get_tenant_id_from_request()") >= 2
	assert views_source.count("get_current_user_id()") >= 2
	assert api_source.count("get_tenant_id_from_request(payload)") >= 1
	assert api_source.count("get_current_user_id(payload)") >= 1
	assert "request.get_json(silent=True) if request.is_json else None" in api_source


def test_requisition_context_resolves_tenant_and_user(monkeypatch):
	helpers = _context_helpers()
	resolve_tenant = helpers["get_tenant_id_from_request"]
	resolve_user = helpers["get_current_user_id"]
	app = Flask(__name__)

	monkeypatch.setenv("APG_DEFAULT_TENANT_ID", "req-env-tenant")
	monkeypatch.setenv("APG_DEFAULT_USER_ID", "req-env-user")
	assert resolve_tenant() == "req-env-tenant"
	assert resolve_user() == "req-env-user"

	with app.test_request_context("/requisitions?tenant_id=query-tenant", headers={"X-Tenant-ID": "header-tenant"}):
		assert resolve_tenant({"tenant_id": "payload-tenant"}) == "payload-tenant"

	with app.test_request_context("/requisitions?tenant_id=query-tenant", headers={"X-Tenant-ID": "header-tenant"}):
		g.tenant_id = "context-tenant"
		assert resolve_tenant({}) == "context-tenant"

	with app.test_request_context("/requisitions", headers={"X-Tenant-ID": "header-tenant"}):
		g.user = type("User", (), {"tenant_id": "fab-user-tenant", "username": "fab-user"})()
		assert resolve_tenant({}) == "fab-user-tenant"
		assert resolve_user({}) == "fab-user"

	with app.test_request_context("/requisitions", headers={"X-APG-User-ID": "header-user"}):
		assert resolve_user({}) == "header-user"

	with app.test_request_context("/requisitions", environ_base={"APG_USER_ID": "request-env-user"}):
		assert resolve_user({}) == "request-env-user"
