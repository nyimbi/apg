"""Context resolution regressions for Manufacturing Production Planning."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

from flask import Flask, g, request, session


REPO_ROOT = Path(__file__).resolve().parents[1]
API_PATH = REPO_ROOT / "capabilities" / "mfg" / "ppl" / "api.py"
VIEWS_PATH = REPO_ROOT / "capabilities" / "mfg" / "ppl" / "views.py"


def _api_helpers() -> dict[str, Any]:
	source = API_PATH.read_text(encoding="utf-8")
	start = source.index("def _clean_text")
	end = source.index("# Master Production Schedule endpoints")
	namespace: dict[str, Any] = {
		"Any": Any,
		"List": List,
		"Optional": Optional,
		"g": g,
		"os": __import__("os"),
		"request": request,
		"session": session,
	}
	exec(compile(source[start:end], str(API_PATH), "exec"), namespace)
	return namespace


def _view_helpers() -> dict[str, Any]:
	source = VIEWS_PATH.read_text(encoding="utf-8")
	start = source.index("def _clean_text")
	end = source.index("class MasterProductionScheduleView")
	namespace: dict[str, Any] = {
		"Any": Any,
		"Optional": Optional,
		"g": g,
		"os": __import__("os"),
		"request": request,
		"session": session,
	}
	exec(compile(source[start:end], str(VIEWS_PATH), "exec"), namespace)
	return namespace


def test_mfg_ppl_surfaces_no_longer_use_fixed_context_literals():
	for path in (API_PATH, VIEWS_PATH):
		source = path.read_text(encoding="utf-8")
		for stale_text in (
			"'default-tenant'",
			"'current-user'",
			'"default-tenant"',
			'"current-user"',
			"Replace with actual tenant resolution",
			"Replace with actual user resolution",
		):
			assert stale_text not in source

	assert "resolve_current_tenant_id()" in VIEWS_PATH.read_text(encoding="utf-8")
	assert "get_current_tenant_id()" in API_PATH.read_text(encoding="utf-8")


def test_mfg_ppl_api_context_resolves_request_context_and_headers(monkeypatch):
	helpers = _api_helpers()
	resolve_tenant = helpers["get_current_tenant_id"]
	resolve_user = helpers["get_current_user_id"]
	app = Flask(__name__)
	app.secret_key = "test"

	monkeypatch.setenv("APG_DEFAULT_USER_ID", "env-user")
	monkeypatch.setenv("APG_DEFAULT_TENANT_ID", "env-tenant")
	with app.test_request_context("/ppl"):
		assert resolve_user() == "env-user"
		assert resolve_tenant() == "env-tenant"

	with app.test_request_context("/ppl?user_id=query-user&tenant=query-tenant", headers={"X-APG-User-ID": "header-user"}):
		session["user_id"] = "session-user"
		session["tenant_id"] = "session-tenant"
		g.current_user = {"user_id": "g-user", "tenant_id": "g-tenant"}
		assert resolve_user() == "g-user"
		assert resolve_tenant() == "g-tenant"

	with app.test_request_context("/ppl", headers={"X-APG-User-ID": "header-user", "X-APG-Tenant-ID": "header-tenant"}):
		assert resolve_user() == "header-user"
		assert resolve_tenant() == "header-tenant"


def test_mfg_ppl_view_context_resolves_request_context_and_headers(monkeypatch):
	helpers = _view_helpers()
	resolve_tenant = helpers["resolve_current_tenant_id"]
	resolve_user = helpers["resolve_current_user_id"]
	app = Flask(__name__)
	app.secret_key = "test"

	monkeypatch.setenv("APG_DEFAULT_USER_ID", "env-user")
	monkeypatch.setenv("APG_DEFAULT_TENANT_ID", "env-tenant")
	with app.test_request_context("/ppl"):
		assert resolve_user() == "env-user"
		assert resolve_tenant() == "env-tenant"

	with app.test_request_context("/ppl", headers={"X-User-ID": "header-user", "X-Tenant-ID": "header-tenant"}):
		assert resolve_user() == "header-user"
		assert resolve_tenant() == "header-tenant"
