"""Tenant resolution regression tests for Financial Cost Accounting surfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, g, has_request_context, request


REPO_ROOT = Path(__file__).resolve().parents[1]
API_PATH = REPO_ROOT / "capabilities" / "fin" / "cos" / "api.py"
VIEWS_PATH = REPO_ROOT / "capabilities" / "fin" / "cos" / "views.py"
TENANT_PATH = REPO_ROOT / "capabilities" / "fin" / "cos" / "tenant.py"
BLUEPRINT_PATH = REPO_ROOT / "capabilities" / "fin" / "cos" / "blueprint.py"


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


def test_fin_cos_surfaces_use_shared_tenant_resolver():
	api_source = API_PATH.read_text(encoding="utf-8")
	views_source = VIEWS_PATH.read_text(encoding="utf-8")
	tenant_source = TENANT_PATH.read_text(encoding="utf-8")

	for stale_lookup in (
		"request.args.get('tenant_id', 'default_tenant')",
		"request.json.get('tenant_id', 'default_tenant')",
		"data.get('tenant_id', 'default_tenant')",
		"CostAccountingService(tenant_id='default_tenant')",
		'"default_tenant"',
	):
		assert stale_lookup not in api_source
		assert stale_lookup not in views_source
		assert stale_lookup not in tenant_source

	assert "from .tenant import get_tenant_id_from_request" in api_source
	assert "from .tenant import get_tenant_id_from_request" in views_source
	assert api_source.count("get_tenant_id_from_request(") >= 9
	assert views_source.count("get_tenant_id_from_request(") >= 11


def test_fin_cos_default_seed_data_uses_current_tenant_context():
	source = BLUEPRINT_PATH.read_text(encoding="utf-8")

	assert "from .tenant import get_tenant_id_from_request" in source
	assert "tenant_id = get_tenant_id_from_request()" in source
	assert "tenant_id='default_tenant'" not in source
	assert '"default_tenant"' not in source
	assert "filter_by(tenant_id=tenant_id)" in source
	assert "tenant_id=tenant_id,\n\t\t\t\t\t\tcategory_code=cat_data['parent_category']" in source
	assert "tenant_id=tenant_id,\n\t\t\t\t\tdriver_code=activity_data['primary_driver']" in source


def test_tenant_resolver_prefers_payload_context_headers_and_query(monkeypatch):
	resolver = _tenant_helpers()["get_tenant_id_from_request"]
	app = Flask(__name__)

	monkeypatch.setenv("APG_DEFAULT_TENANT_ID", "env-tenant")
	assert resolver() == "env-tenant"

	with app.test_request_context("/cost_centers?tenant_id=query-tenant", headers={"X-Tenant-ID": "header-tenant"}):
		assert resolver({"tenant_id": " payload-tenant "}) == "payload-tenant"

	with app.test_request_context("/cost_centers?tenant_id=query-tenant", headers={"X-Tenant-ID": "header-tenant"}):
		g.tenant_id = "context-tenant"
		assert resolver({}) == "context-tenant"

	with app.test_request_context("/cost_centers?tenant_id=query-tenant", headers={"X-Tenant-ID": "header-tenant"}):
		g.current_user = type("User", (), {"tenant_id": "user-tenant"})()
		assert resolver({}) == "user-tenant"

	with app.test_request_context("/cost_centers?tenant_id=query-tenant", headers={"X-Tenant-ID": "header-tenant"}):
		assert resolver({}) == "header-tenant"

	with app.test_request_context("/cost_centers?tenant=query-tenant"):
		assert resolver({}) == "query-tenant"
