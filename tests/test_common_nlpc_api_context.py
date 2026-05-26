"""Context resolution regressions for NLPC REST API helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

from flask import Flask, g, request, session
from werkzeug.exceptions import BadRequest


REPO_ROOT = Path(__file__).resolve().parents[1]
API_PATH = REPO_ROOT / "capabilities" / "common" / "nlpc" / "api.py"


def _helpers() -> dict[str, Any]:
	source = API_PATH.read_text(encoding="utf-8")
	start = source.index("def _clean_text")
	end = source.index("# ===== REST API Endpoints")
	namespace: dict[str, Any] = {
		"Any": Any,
		"BadRequest": BadRequest,
		"List": List,
		"Optional": Optional,
		"g": g,
		"os": __import__("os"),
		"request": request,
		"session": session,
	}
	exec(compile(source[start:end], str(API_PATH), "exec"), namespace)
	return namespace


def test_nlpc_api_no_longer_uses_fixed_or_placeholder_identity_context():
	source = API_PATH.read_text(encoding="utf-8")
	for stale_text in (
		"'default-tenant'",
		'"default-tenant"',
		"'default-user'",
		'"default-user"',
		"real implementation",
		"Ka" + "fka",
		"kaf" + "ka",
	):
		assert stale_text not in source

	assert 'request.headers.get("X-APG-Tenant-ID")' in source
	assert 'request.headers.get("X-APG-User-ID")' in source


def test_nlpc_api_context_resolves_request_context_before_session_headers_and_query(monkeypatch):
	helpers = _helpers()
	resolve_tenant = helpers["_get_tenant_id"]
	resolve_user = helpers["_get_user_id"]
	app = Flask(__name__)
	app.secret_key = "test"

	monkeypatch.setenv("APG_TENANT_ID", "env-tenant")
	monkeypatch.setenv("APG_USER_ID", "env-user")
	with app.test_request_context("/nlp"):
		assert resolve_tenant() == "env-tenant"
		assert resolve_user() == "env-user"

	with app.test_request_context(
		"/nlp?tenant_id=query-tenant&user_id=query-user",
		headers={"X-APG-Tenant-ID": "header-tenant", "X-APG-User-ID": "header-user"},
	):
		session["tenant_id"] = "session-tenant"
		session["user_id"] = "session-user"
		request.current_user = {"tenant_id": "request-tenant", "user_id": "request-user"}
		assert resolve_tenant() == "request-tenant"
		assert resolve_user() == "request-user"


def test_nlpc_api_context_resolves_g_and_header_fallbacks():
	helpers = _helpers()
	resolve_tenant = helpers["_get_tenant_id"]
	resolve_user = helpers["_get_user_id"]
	app = Flask(__name__)
	app.secret_key = "test"

	with app.test_request_context(
		"/nlp?tenant=query-tenant&user_id=query-user",
		headers={"X-Tenant-ID": "header-tenant", "X-User-ID": "header-user"},
	):
		g.current_user = {"tenant_id": "g-tenant", "user_id": "g-user"}
		assert resolve_tenant() == "g-tenant"
		assert resolve_user() == "g-user"

	with app.test_request_context(
		"/nlp?tenant=query-tenant&user_id=query-user",
		headers={"X-Tenant-ID": "header-tenant", "X-User-ID": "header-user"},
	):
		assert resolve_tenant() == "header-tenant"
		assert resolve_user() == "header-user"
