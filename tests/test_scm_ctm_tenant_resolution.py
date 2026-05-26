"""Tenant context regressions for Contract Management surfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, g, has_request_context, request


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTEXT_PATH = REPO_ROOT / "capabilities" / "scm" / "ctm" / "contract_management" / "context.py"
API_PATH = REPO_ROOT / "capabilities" / "scm" / "ctm" / "contract_management" / "api.py"
VIEWS_PATH = REPO_ROOT / "capabilities" / "scm" / "ctm" / "contract_management" / "views.py"


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


def test_contract_management_surfaces_delegate_tenant_resolution():
	api_source = API_PATH.read_text(encoding="utf-8")
	views_source = VIEWS_PATH.read_text(encoding="utf-8")

	for source in (api_source, views_source):
		assert 'return "default_tenant"' not in source
		assert "from .context import get_tenant_id_from_request" in source
		assert "get_tenant_id_from_request()" in source


def test_contract_management_tenant_resolver_precedence(monkeypatch):
	resolver = _context_helpers()["get_tenant_id_from_request"]
	app = Flask(__name__)

	monkeypatch.setenv("APG_DEFAULT_TENANT_ID", "ctm-env-tenant")
	assert resolver() == "ctm-env-tenant"

	with app.test_request_context("/contracts?tenant_id=query-tenant", headers={"X-Tenant-ID": "header-tenant"}):
		assert resolver({"tenant_id": "payload-tenant"}) == "payload-tenant"

	with app.test_request_context("/contracts?tenant_id=query-tenant", headers={"X-Tenant-ID": "header-tenant"}):
		g.tenant_id = "context-tenant"
		assert resolver({}) == "context-tenant"

	with app.test_request_context("/contracts", headers={"X-Tenant-ID": "header-tenant"}):
		g.user = type("User", (), {"tenant_id": "fab-user-tenant"})()
		assert resolver({}) == "fab-user-tenant"

	with app.test_request_context("/contracts?tenant=query-tenant"):
		assert resolver({}) == "query-tenant"

	with app.test_request_context("/contracts", environ_base={"APG_TENANT_ID": "request-env-tenant"}):
		assert resolver({}) == "request-env-tenant"
