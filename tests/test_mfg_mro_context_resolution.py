"""Context resolution regressions for Predictive Maintenance views."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, g, has_request_context, request


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTEXT_PATH = REPO_ROOT / "capabilities" / "mfg" / "mro" / "context.py"
VIEWS_PATH = REPO_ROOT / "capabilities" / "mfg" / "mro" / "views.py"


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


def test_mro_views_delegate_context_resolution():
	source = VIEWS_PATH.read_text(encoding="utf-8")

	for stale_text in (
		'return "default_tenant"',
		"from flask_appbuilder.security import current_user",
		"return str(current_user.id) if current_user and current_user.is_authenticated else None",
	):
		assert stale_text not in source

	assert "from .context import get_current_user_id, get_tenant_id_from_request" in source
	assert source.count("get_tenant_id_from_request()") >= 3
	assert source.count("get_current_user_id()") >= 3


def test_mro_context_resolves_tenant_and_user_from_request(monkeypatch):
	helpers = _context_helpers()
	resolve_tenant = helpers["get_tenant_id_from_request"]
	resolve_user = helpers["get_current_user_id"]
	app = Flask(__name__)

	monkeypatch.setenv("APG_DEFAULT_TENANT_ID", "mro-env-tenant")
	assert resolve_tenant() == "mro-env-tenant"
	assert resolve_user() is None

	with app.test_request_context("/mro?tenant_id=query-tenant", headers={"X-Tenant-ID": "header-tenant"}):
		assert resolve_tenant({"tenant_id": "payload-tenant"}) == "payload-tenant"

	with app.test_request_context("/mro?tenant_id=query-tenant", headers={"X-Tenant-ID": "header-tenant"}):
		g.tenant_id = "context-tenant"
		assert resolve_tenant({}) == "context-tenant"

	with app.test_request_context("/mro?tenant=query-tenant"):
		assert resolve_tenant({}) == "query-tenant"

	with app.test_request_context("/mro", headers={"X-APG-User-ID": "header-user"}):
		assert resolve_user() == "header-user"

	with app.test_request_context("/mro", environ_base={"APG_USER_ID": "env-user"}):
		assert resolve_user() == "env-user"

	with app.test_request_context("/mro"):
		g.current_user = type("User", (), {"id": "context-user"})()
		assert resolve_user() == "context-user"
