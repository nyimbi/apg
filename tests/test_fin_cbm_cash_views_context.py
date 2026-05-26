"""Tenant context regressions for Cash Management FAB views."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional

from flask import Flask, g, request, session


REPO_ROOT = Path(__file__).resolve().parents[1]
VIEWS_PATH = REPO_ROOT / "capabilities" / "fin" / "cbm" / "cash_management" / "views.py"


def _helpers() -> dict[str, Any]:
	source = VIEWS_PATH.read_text(encoding="utf-8")
	start = source.index("def _clean_text")
	end = source.index("# ============================================================================\n# Custom Widgets")
	namespace: dict[str, Any] = {
		"Any": Any,
		"List": List,
		"Optional": Optional,
		"g": g,
		"os": __import__("os"),
		"request": request,
		"session": session,
	}
	exec(compile(source[start:end], str(VIEWS_PATH), "exec"), namespace)
	return namespace


def test_cash_management_views_no_longer_use_fixed_tenant_fallback():
	source = VIEWS_PATH.read_text(encoding="utf-8")
	assert "'default_" + "tenant'" not in source
	assert "Integration with APG authentication" not in source
	assert "return resolve_tenant_id(self.appbuilder)" in source


def test_cash_management_view_tenant_context_resolves_apg_flask_and_appbuilder(monkeypatch):
	resolve_tenant = _helpers()["resolve_tenant_id"]
	app = Flask(__name__)
	app.secret_key = "test"
	appbuilder = SimpleNamespace(sm=SimpleNamespace(user=SimpleNamespace(tenant_id="appbuilder-tenant")))

	monkeypatch.setenv("APG_DEFAULT_TENANT_ID", "env-tenant")
	with app.test_request_context("/cash-management"):
		assert resolve_tenant(None) == "env-tenant"

	with app.test_request_context("/cash-management"):
		assert resolve_tenant(appbuilder) == "appbuilder-tenant"

	with app.test_request_context(
		"/cash-management?tenant=query-tenant",
		headers={"X-APG-Tenant-ID": "header-tenant"},
	):
		session["tenant_id"] = "session-tenant"
		g.current_user = {"tenant_id": "g-user-tenant"}
		assert resolve_tenant(appbuilder) == "g-user-tenant"

	with app.test_request_context(
		"/cash-management?tenant=query-tenant",
		headers={"X-Tenant-ID": "header-tenant"},
	):
		assert resolve_tenant(None) == "header-tenant"

	with app.test_request_context("/cash-management?tenant=query-tenant"):
		assert resolve_tenant(None) == "query-tenant"
