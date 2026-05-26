"""Tenant context regressions for top-level APG blueprints."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
REQUEST_CONTEXT_PATH = REPO_ROOT / "capabilities" / "common" / "request_context.py"
SCM_BLUEPRINT_PATH = REPO_ROOT / "capabilities" / "scm" / "blueprint.py"
HCM_BLUEPRINT_PATH = REPO_ROOT / "capabilities" / "hcm" / "blueprint.py"
CRAWLER_BLUEPRINT_PATH = REPO_ROOT / "capabilities" / "intel" / "crawler" / "blueprint.py"
COMPOSITION_BLUEPRINT_PATH = REPO_ROOT / "capabilities" / "composition" / "orchestration" / "blueprint.py"
SECURITY_ENGINE_PATH = REPO_ROOT / "capabilities" / "composition" / "config" / "security_engine.py"


def _request_context_helpers() -> dict[str, Any]:
	source = REQUEST_CONTEXT_PATH.read_text(encoding="utf-8")
	start = source.index("def _clean_text")
	namespace: dict[str, Any] = {
		"Any": Any,
		"Dict": Dict,
		"Iterable": Iterable,
		"Optional": Optional,
		"os": os,
	}
	exec(compile(source[start:], str(REQUEST_CONTEXT_PATH), "exec"), namespace)
	return namespace


def test_top_level_blueprints_delegate_tenant_context_resolution():
	scm_source = SCM_BLUEPRINT_PATH.read_text(encoding="utf-8")
	hcm_source = HCM_BLUEPRINT_PATH.read_text(encoding="utf-8")
	crawler_source = CRAWLER_BLUEPRINT_PATH.read_text(encoding="utf-8")
	composition_source = COMPOSITION_BLUEPRINT_PATH.read_text(encoding="utf-8")
	security_source = SECURITY_ENGINE_PATH.read_text(encoding="utf-8")

	assert "from ..common.request_context import get_tenant_id_from_context" in scm_source
	assert "from ..common.request_context import get_tenant_id_from_context" in hcm_source
	assert "from ...common.request_context import get_tenant_id_from_context" in crawler_source
	assert "from ...common.request_context import get_tenant_id_from_context" in composition_source
	assert "from ...common.request_context import get_tenant_id_from_context" in security_source

	assert 'return "default_tenant"' not in scm_source
	assert 'return "default_tenant"' not in hcm_source
	assert "request.args.get('tenant_id', 'default_tenant')" not in crawler_source
	assert '"default_tenant"' not in crawler_source
	assert '"default_tenant"' not in composition_source
	assert '"default_tenant"' not in security_source

	assert "return get_tenant_id_from_context()" in scm_source
	assert "return get_tenant_id_from_context()" in hcm_source
	assert "return get_tenant_id_from_context()" in crawler_source
	assert "return get_tenant_id_from_context()" in composition_source
	assert "tenant_id = get_tenant_id_from_context(credentials)" in security_source


def test_request_context_resolves_tenant_from_payload_flask_and_environment(monkeypatch):
	helpers = _request_context_helpers()
	resolve_tenant = helpers["get_tenant_id_from_context"]

	monkeypatch.setenv("APG_DEFAULT_TENANT_ID", "env-tenant")
	assert resolve_tenant() == "env-tenant"
	assert resolve_tenant({"tenant_id": "payload-tenant"}) == "payload-tenant"
	assert resolve_tenant({"tenant": "payload-tenant-alias"}) == "payload-tenant-alias"
	assert resolve_tenant({"organization_id": "payload-org"}) == "payload-org"

	from flask import Flask, g, session

	app = Flask(__name__)
	app.secret_key = "test-secret"

	with app.test_request_context(
		"/?tenant_id=query-tenant",
		headers={"X-Tenant-ID": "header-tenant", "X-APG-Tenant-ID": "apg-header-tenant"},
	):
		session["tenant_id"] = "session-tenant"
		g.tenant_id = "flask-g-tenant"
		assert resolve_tenant() == "session-tenant"

	with app.test_request_context(
		"/?tenant_id=query-tenant",
		headers={"X-Tenant-ID": "header-tenant", "X-APG-Tenant-ID": "apg-header-tenant"},
	):
		g.tenant_id = "flask-g-tenant"
		assert resolve_tenant() == "flask-g-tenant"

	with app.test_request_context("/", headers={"X-APG-Tenant-ID": "apg-header-tenant"}):
		assert resolve_tenant() == "apg-header-tenant"

	with app.test_request_context("/?tenant=query-tenant"):
		assert resolve_tenant() == "query-tenant"
