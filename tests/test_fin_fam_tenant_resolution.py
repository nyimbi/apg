"""Tenant resolution regressions for Fixed Asset Management surfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, g, has_request_context, request


REPO_ROOT = Path(__file__).resolve().parents[1]
API_PATH = REPO_ROOT / "capabilities" / "fin" / "fam" / "fixed_asset_management" / "api.py"
VIEWS_PATH = REPO_ROOT / "capabilities" / "fin" / "fam" / "fixed_asset_management" / "views.py"
TENANT_PATH = REPO_ROOT / "capabilities" / "fin" / "fam" / "fixed_asset_management" / "tenant.py"


def _tenant_helpers() -> dict[str, Any]:
	source = TENANT_PATH.read_text(encoding="utf-8")
	start = source.index("def _clean_tenant_id")
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
	exec(compile(source[start:], str(TENANT_PATH), "exec"), namespace)
	return namespace


def test_fam_surfaces_delegate_to_shared_tenant_resolver():
	api_source = API_PATH.read_text(encoding="utf-8")
	views_source = VIEWS_PATH.read_text(encoding="utf-8")

	for stale_text in (
		"return \"default_tenant\"",
		"Get current tenant ID - placeholder implementation",
		"TODO: Implement proper tenant context",
	):
		assert stale_text not in api_source
		assert stale_text not in views_source

	assert "from .tenant import get_tenant_id_from_request" in api_source
	assert "from .tenant import get_tenant_id_from_request" in views_source
	assert api_source.count("get_tenant_id_from_request()") >= 11
	assert views_source.count("get_tenant_id_from_request()") >= 1


def test_fam_tenant_resolver_prefers_payload_context_headers_and_query(monkeypatch):
	resolver = _tenant_helpers()["get_tenant_id_from_request"]
	app = Flask(__name__)

	monkeypatch.setenv("APG_DEFAULT_TENANT_ID", "fam-env-tenant")
	assert resolver() == "fam-env-tenant"

	with app.test_request_context("/assets?tenant_id=query-tenant", headers={"X-Tenant-ID": "header-tenant"}):
		assert resolver({"tenant_id": " payload-tenant "}) == "payload-tenant"

	with app.test_request_context("/assets?tenant_id=query-tenant", headers={"X-Tenant-ID": "header-tenant"}):
		g.tenant_id = "context-tenant"
		assert resolver({}) == "context-tenant"

	with app.test_request_context("/assets?tenant_id=query-tenant", headers={"X-Tenant-ID": "header-tenant"}):
		g.current_user = type("User", (), {"tenant_id": "user-tenant"})()
		assert resolver({}) == "user-tenant"

	with app.test_request_context("/assets?tenant_id=query-tenant", headers={"X-Tenant-ID": "header-tenant"}):
		assert resolver({}) == "header-tenant"

	with app.test_request_context("/assets?tenant=query-tenant"):
		assert resolver({}) == "query-tenant"
