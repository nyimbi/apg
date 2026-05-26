"""Context resolution regressions for Accounts Receivable views."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, g, has_request_context, request


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTEXT_PATH = REPO_ROOT / "capabilities" / "fin" / "arc" / "accounts_receivable" / "context.py"
VIEWS_PATH = REPO_ROOT / "capabilities" / "fin" / "arc" / "accounts_receivable" / "views.py"
BLUEPRINT_PATH = REPO_ROOT / "capabilities" / "fin" / "arc" / "accounts_receivable" / "blueprint.py"


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


def test_ar_views_delegate_context_resolution():
	context_source = CONTEXT_PATH.read_text(encoding="utf-8")
	source = VIEWS_PATH.read_text(encoding="utf-8")

	assert '"default_tenant"' not in context_source

	for stale_text in (
		'return "default_tenant"',
		'return "default_user"',
		"This would typically come from session",
	):
		assert stale_text not in source

	assert "from .context import get_current_user_id, get_tenant_id_from_request" in source
	assert source.count("get_tenant_id_from_request()") >= 10
	assert source.count("get_current_user_id()") >= 7


def test_ar_blueprint_delegates_context_resolution():
	source = BLUEPRINT_PATH.read_text(encoding="utf-8")

	for stale_text in (
		"request.headers.get('X-Tenant-ID', 'default_tenant')",
		"request.headers.get('X-User-ID', 'system_user')",
		"tenant_id='default_tenant'",
		"use a default tenant for now",
	):
		assert stale_text not in source

	assert "from .context import get_current_user_id as resolve_current_user_id, get_tenant_id_from_request" in source
	assert "return get_tenant_id_from_request()" in source
	assert 'return resolve_current_user_id() or "system"' in source


def test_ar_context_resolves_tenant_and_user_from_request(monkeypatch):
	helpers = _context_helpers()
	resolve_tenant = helpers["get_tenant_id_from_request"]
	resolve_user = helpers["get_current_user_id"]
	app = Flask(__name__)

	monkeypatch.setenv("APG_DEFAULT_TENANT_ID", "ar-env-tenant")
	assert resolve_tenant() == "ar-env-tenant"
	assert resolve_user() is None

	with app.test_request_context("/ar?tenant_id=query-tenant", headers={"X-Tenant-ID": "header-tenant"}):
		assert resolve_tenant({"tenant_id": "payload-tenant"}) == "payload-tenant"

	with app.test_request_context("/ar?tenant_id=query-tenant", headers={"X-Tenant-ID": "header-tenant"}):
		g.tenant_id = "context-tenant"
		assert resolve_tenant({}) == "context-tenant"

	with app.test_request_context("/ar?tenant=query-tenant"):
		assert resolve_tenant({}) == "query-tenant"

	with app.test_request_context("/ar", headers={"X-APG-User-ID": "header-user"}):
		assert resolve_user() == "header-user"

	with app.test_request_context("/ar", environ_base={"APG_USER_ID": "env-user"}):
		assert resolve_user() == "env-user"

	with app.test_request_context("/ar"):
		g.current_user = type("User", (), {"id": "context-user"})()
		assert resolve_user() == "context-user"
