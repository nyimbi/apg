"""Tenant resolution regression tests for Financial Cost Accounting API."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, g, has_request_context, request


API_PATH = Path(__file__).resolve().parents[1] / "capabilities" / "fin" / "cos" / "api.py"


def _tenant_helpers() -> dict[str, Any]:
	source = API_PATH.read_text(encoding="utf-8")
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
	exec(compile(source[start:], str(API_PATH), "exec"), namespace)
	return namespace


def test_fin_cos_api_uses_shared_tenant_resolver():
	source = API_PATH.read_text(encoding="utf-8")

	for stale_lookup in (
		"request.args.get('tenant_id', 'default_tenant')",
		"request.json.get('tenant_id', 'default_tenant')",
		"data.get('tenant_id', 'default_tenant')",
	):
		assert stale_lookup not in source

	assert source.count("get_tenant_id_from_request(") >= 9


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
		assert resolver({}) == "header-tenant"

	with app.test_request_context("/cost_centers?tenant=query-tenant"):
		assert resolver({}) == "query-tenant"
