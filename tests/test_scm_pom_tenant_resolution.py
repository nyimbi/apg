"""Context resolution regressions for Purchase Order Management surfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, g, has_request_context, request


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTEXT_PATH = REPO_ROOT / "capabilities" / "scm" / "pom" / "context.py"
API_PATH = REPO_ROOT / "capabilities" / "scm" / "pom" / "api.py"
VIEWS_PATH = REPO_ROOT / "capabilities" / "scm" / "pom" / "views.py"


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


def test_purchase_order_surfaces_delegate_tenant_resolution():
	api_source = API_PATH.read_text(encoding="utf-8")
	views_source = VIEWS_PATH.read_text(encoding="utf-8")

	for source in (api_source, views_source):
		assert 'return "default_tenant"' not in source
		assert "from .context import get_tenant_id_from_request" in source

	assert "get_tenant_id_from_request(payload)" in api_source
	assert "get_tenant_id_from_request()" in views_source


def test_purchase_order_context_resolves_tenant(monkeypatch):
	helpers = _context_helpers()
	resolve_tenant = helpers["get_tenant_id_from_request"]
	app = Flask(__name__)

	monkeypatch.setenv("APG_DEFAULT_TENANT_ID", "pom-env-tenant")
	assert resolve_tenant() == "pom-env-tenant"

	with app.test_request_context("/po?tenant_id=query-tenant", headers={"X-Tenant-ID": "header-tenant"}):
		assert resolve_tenant({"tenant_id": "payload-tenant"}) == "payload-tenant"

	with app.test_request_context("/po?tenant_id=query-tenant", headers={"X-Tenant-ID": "header-tenant"}):
		g.tenant_id = "context-tenant"
		assert resolve_tenant({}) == "context-tenant"

	with app.test_request_context("/po", headers={"X-Tenant-ID": "header-tenant"}):
		g.user = type("User", (), {"tenant_id": "fab-user-tenant"})()
		assert resolve_tenant({}) == "fab-user-tenant"

	with app.test_request_context("/po", environ_base={"APG_TENANT_ID": "request-env-tenant"}):
		assert resolve_tenant({}) == "request-env-tenant"
