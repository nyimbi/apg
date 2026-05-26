"""Context resolution regressions for HCM Employee Data Management."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, g, has_request_context, request, session


REPO_ROOT = Path(__file__).resolve().parents[1]
CAPABILITY_PATH = REPO_ROOT / "capabilities" / "hcm" / "chr" / "employee_data_management"
CONTEXT_PATH = CAPABILITY_PATH / "context.py"
VIEWS_PATH = CAPABILITY_PATH / "views.py"
API_PATH = CAPABILITY_PATH / "api.py"
API_INTEGRATION_PATH = CAPABILITY_PATH / "api_integration.py"


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
		"session": session,
	}
	exec(compile(source[start:], str(CONTEXT_PATH), "exec"), namespace)
	return namespace


def test_employee_management_surfaces_delegate_context_resolution():
	views_source = VIEWS_PATH.read_text(encoding="utf-8")
	api_source = API_PATH.read_text(encoding="utf-8")
	api_integration_source = API_INTEGRATION_PATH.read_text(encoding="utf-8")
	combined = "\n".join([views_source, api_source, api_integration_source])

	assert "from .context import get_tenant_id_from_request" in views_source
	assert "from .context import get_tenant_id_from_request" in api_source
	assert "from .context import get_current_user_id, get_tenant_id_from_request" in api_integration_source
	assert "from flask_login import current_user" not in views_source
	assert "from flask import Blueprint, request, jsonify, g" not in api_integration_source
	assert "from .service import RevolutionaryEmployeeDataManagementService as EmployeeDataManagementService" in views_source
	assert "from .service import RevolutionaryEmployeeDataManagementService as EmployeeDataManagementService" in api_source
	assert 'return "default_tenant"' not in combined
	assert "TODO: Implement tenant resolution" not in combined
	assert "Would extract from user session" not in combined
	assert views_source.count("get_tenant_id_from_request()") >= 12
	assert api_source.count("get_tenant_id_from_request(") >= 1
	assert api_integration_source.count("get_tenant_id_from_request(") >= 2
	assert api_integration_source.count("get_current_user_id(") >= 1


def test_employee_management_context_resolves_tenant_and_user(monkeypatch):
	helpers = _context_helpers()
	resolve_tenant = helpers["get_tenant_id_from_request"]
	resolve_user = helpers["get_current_user_id"]
	app = Flask(__name__)
	app.secret_key = "test-secret"

	monkeypatch.setenv("APG_DEFAULT_TENANT_ID", "hcm-env-tenant")
	monkeypatch.setenv("APG_DEFAULT_USER_ID", "hcm-env-user")
	assert resolve_tenant() == "hcm-env-tenant"
	assert resolve_user() == "hcm-env-user"

	with app.test_request_context("/hcm?tenant_id=query-tenant", headers={"X-Tenant-ID": "header-tenant"}):
		session["tenant_id"] = "session-tenant"
		session["user_id"] = "session-user"
		assert resolve_tenant({"tenant_id": "payload-tenant"}) == "payload-tenant"
		assert resolve_user({"user_id": "payload-user"}) == "payload-user"

	with app.test_request_context("/hcm?tenant_id=query-tenant", headers={"X-Tenant-ID": "header-tenant"}):
		session["tenant_id"] = "session-tenant"
		g.tenant_id = "context-tenant"
		assert resolve_tenant({}) == "context-tenant"

	with app.test_request_context("/hcm", headers={"X-Tenant-ID": "header-tenant"}):
		session["tenant_id"] = "session-tenant"
		session["user_id"] = "session-user"
		g.user = type("User", (), {"tenant_id": "fab-user-tenant", "username": "fab-user"})()
		assert resolve_tenant({}) == "fab-user-tenant"
		assert resolve_user({}) == "fab-user"

	with app.test_request_context("/hcm", headers={"X-APG-User-ID": "header-user"}):
		assert resolve_user({}) == "header-user"

	with app.test_request_context("/hcm", environ_base={"APG_TENANT_ID": "request-env-tenant"}):
		assert resolve_tenant({}) == "request-env-tenant"
