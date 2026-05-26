"""Context resolution regressions for Demand Planning surfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, g, has_request_context, request


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTEXT_PATH = REPO_ROOT / "capabilities" / "scm" / "dpl" / "demand_planning" / "context.py"
API_PATH = REPO_ROOT / "capabilities" / "scm" / "dpl" / "demand_planning" / "api.py"
VIEWS_PATH = REPO_ROOT / "capabilities" / "scm" / "dpl" / "demand_planning" / "views.py"


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


def test_demand_planning_surfaces_delegate_context_resolution():
	api_source = API_PATH.read_text(encoding="utf-8")
	views_source = VIEWS_PATH.read_text(encoding="utf-8")

	assert 'return "default_tenant"' not in views_source
	assert "request.headers.get('X-Tenant-ID', 'default')" not in api_source
	assert "request.headers.get('X-User-ID', 'api_user')" not in api_source
	assert "Implementation depends on your multi-tenancy setup" not in views_source
	assert "from .context import get_current_user_id, get_tenant_id_from_request" in api_source
	assert "from .context import get_current_user_id, get_tenant_id_from_request" in views_source
	assert api_source.count("get_tenant_id_from_request(") >= 1
	assert api_source.count("get_current_user_id(") >= 1
	assert views_source.count("get_tenant_id_from_request()") >= 1
	assert views_source.count("get_current_user_id(self.appbuilder)") >= 1
	assert "class SCDPForecastAccuracyView(SCDPBaseView):" in views_source
	assert "class SCDPDashboardView(SCDPBaseView):" in views_source


def test_demand_planning_context_resolves_tenant_and_user(monkeypatch):
	helpers = _context_helpers()
	resolve_tenant = helpers["get_tenant_id_from_request"]
	resolve_user = helpers["get_current_user_id"]
	app = Flask(__name__)

	monkeypatch.setenv("APG_DEFAULT_TENANT_ID", "dpl-env-tenant")
	monkeypatch.setenv("APG_DEFAULT_USER_ID", "dpl-env-user")
	assert resolve_tenant() == "dpl-env-tenant"
	assert resolve_user() == "dpl-env-user"

	with app.test_request_context("/dpl?tenant_id=query-tenant", headers={"X-Tenant-ID": "header-tenant"}):
		assert resolve_tenant({"tenant_id": "payload-tenant"}) == "payload-tenant"

	with app.test_request_context("/dpl?tenant_id=query-tenant", headers={"X-Tenant-ID": "header-tenant"}):
		g.tenant_id = "context-tenant"
		assert resolve_tenant({}) == "context-tenant"

	with app.test_request_context("/dpl", headers={"X-Tenant-ID": "header-tenant"}):
		g.user = type("User", (), {"tenant_id": "fab-user-tenant", "username": "fab-user"})()
		assert resolve_tenant({}) == "fab-user-tenant"
		assert resolve_user(None, {}) == "fab-user"

	with app.test_request_context("/dpl", headers={"X-APG-User-ID": "header-user"}):
		assert resolve_user(None, {}) == "header-user"

	with app.test_request_context("/dpl", environ_base={"APG_USER_ID": "request-env-user"}):
		assert resolve_user(None, {}) == "request-env-user"

	with app.test_request_context("/dpl"):
		appbuilder = type(
			"AppBuilder",
			(),
			{"sm": type("SecurityManager", (), {"user": type("User", (), {"username": "appbuilder-user"})()})()},
		)()
		assert resolve_user(appbuilder, {}) == "appbuilder-user"
