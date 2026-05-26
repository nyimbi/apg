"""Context resolution regressions for Budgeting & Forecasting surfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, g, has_request_context, request, session


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTEXT_PATH = REPO_ROOT / "capabilities" / "fin" / "bfc" / "budgeting_forecasting" / "context.py"
API_PATH = REPO_ROOT / "capabilities" / "fin" / "bfc" / "budgeting_forecasting" / "api.py"
VIEWS_PATH = REPO_ROOT / "capabilities" / "fin" / "bfc" / "budgeting_forecasting" / "views.py"
BLUEPRINT_PATH = REPO_ROOT / "capabilities" / "fin" / "bfc" / "budgeting_forecasting" / "blueprint.py"


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


def _api_identity_helpers(jwt_identity: Any = None, jwt_error: Exception | None = None) -> dict[str, Any]:
	source = API_PATH.read_text(encoding="utf-8")
	start = source.index("def get_tenant_id")
	end = source.index("def handle_api_error")
	context_helpers = _context_helpers()

	def get_jwt_identity_stub():
		if jwt_error:
			raise jwt_error
		return jwt_identity

	namespace: dict[str, Any] = {
		"Any": Any,
		"Dict": Dict,
		"get_current_user_id": context_helpers["get_current_user_id"],
		"get_jwt_identity": get_jwt_identity_stub,
		"get_tenant_id_from_request": context_helpers["get_tenant_id_from_request"],
		"request": request,
	}
	exec(compile(source[start:end], str(API_PATH), "exec"), namespace)
	return namespace


def test_budgeting_forecasting_surfaces_delegate_context_resolution():
	api_source = API_PATH.read_text(encoding="utf-8")
	views_source = VIEWS_PATH.read_text(encoding="utf-8")
	context_source = CONTEXT_PATH.read_text(encoding="utf-8")
	blueprint_source = BLUEPRINT_PATH.read_text(encoding="utf-8")

	for stale_text in (
		'return "default_tenant"',
		'return "current_user"',
		'"default_tenant"',
		"request.headers.get('X-Tenant-ID', 'default_tenant')",
	):
		assert stale_text not in api_source
		assert stale_text not in views_source
		assert stale_text not in context_source
		assert stale_text not in blueprint_source

	assert "from .context import get_current_user_id, get_tenant_id_from_request" in api_source
	assert "from .context import get_current_user_id, get_tenant_id_from_request" in views_source
	assert "from .context import get_current_user_id, get_tenant_id_from_request" in blueprint_source
	assert "get_tenant_id_from_request(payload)" in api_source
	assert "return 'api_user'" not in api_source
	assert "return get_current_user_id(payload)" in api_source
	assert "get_tenant_id_from_request()" in views_source
	assert "get_current_user_id()" in views_source
	assert "def _build_tenant_context(payload: Optional[Dict[str, Any]] = None) -> APGTenantContext:" in blueprint_source
	assert "tenant_id=get_tenant_id_from_request(payload)" in blueprint_source
	assert "user_id=get_current_user_id(payload)" in blueprint_source
	assert blueprint_source.count("return _build_tenant_context()") >= 9


def test_budgeting_forecasting_context_resolves_tenant_and_user(monkeypatch):
	helpers = _context_helpers()
	resolve_tenant = helpers["get_tenant_id_from_request"]
	resolve_user = helpers["get_current_user_id"]
	app = Flask(__name__)
	app.secret_key = "test-secret"

	monkeypatch.setenv("APG_DEFAULT_TENANT_ID", "bfc-env-tenant")
	monkeypatch.setenv("APG_DEFAULT_USER_ID", "bfc-env-user")
	assert resolve_tenant() == "bfc-env-tenant"
	assert resolve_user() == "bfc-env-user"

	with app.test_request_context("/budget?tenant_id=query-tenant", headers={"X-Tenant-ID": "header-tenant"}):
		session["tenant_id"] = "session-tenant"
		session["user_id"] = "session-user"
		assert resolve_tenant({"tenant_id": "payload-tenant"}) == "payload-tenant"
		assert resolve_user({"user_id": "payload-user"}) == "payload-user"

	with app.test_request_context("/budget?tenant_id=query-tenant", headers={"X-Tenant-ID": "header-tenant"}):
		session["tenant_id"] = "session-tenant"
		g.tenant_id = "context-tenant"
		assert resolve_tenant({}) == "context-tenant"

	with app.test_request_context("/budget", headers={"X-Tenant-ID": "header-tenant"}):
		session["tenant_id"] = "session-tenant"
		session["user_id"] = "session-user"
		g.user = type("User", (), {"tenant_id": "fab-user-tenant", "username": "fab-user"})()
		assert resolve_tenant({}) == "fab-user-tenant"
		assert resolve_user({}) == "fab-user"

	with app.test_request_context("/budget", headers={"X-APG-User-ID": "header-user"}):
		assert resolve_user({}) == "header-user"

	with app.test_request_context("/budget", environ_base={"APG_TENANT_ID": "request-env-tenant"}):
		assert resolve_tenant({}) == "request-env-tenant"


def test_budgeting_forecasting_api_user_id_falls_back_to_request_context(monkeypatch):
	app = Flask(__name__)
	app.secret_key = "test-secret"

	with app.test_request_context("/budget", json={"user_id": "payload-user"}):
		assert _api_identity_helpers(jwt_identity="jwt-user")["get_user_id"]() == "jwt-user"

	monkeypatch.setenv("APG_DEFAULT_USER_ID", "bfc-env-user")
	with app.test_request_context("/budget", json={"user_id": "payload-user"}):
		assert _api_identity_helpers(jwt_identity=None)["get_user_id"]() == "payload-user"

	with app.test_request_context("/budget", headers={"X-APG-User-ID": "header-user"}):
		assert _api_identity_helpers(jwt_error=RuntimeError("missing jwt"))["get_user_id"]() == "header-user"

	with app.test_request_context("/budget"):
		assert _api_identity_helpers(jwt_identity=None)["get_user_id"]() == "bfc-env-user"
