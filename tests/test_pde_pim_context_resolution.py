"""Context resolution regressions for Product Information Management blueprint."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, g, has_request_context, request, session


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTEXT_PATH = REPO_ROOT / "capabilities" / "pde" / "pim" / "context.py"
BLUEPRINT_PATH = REPO_ROOT / "capabilities" / "pde" / "pim" / "blueprint.py"
APP_INTEGRATION_PATH = REPO_ROOT / "capabilities" / "pde" / "pim" / "app_integration.py"


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


def test_pim_blueprint_delegates_context_resolution():
	source = BLUEPRINT_PATH.read_text(encoding="utf-8")

	assert "session.get('tenant_id', 'default_tenant')" not in source
	assert "session.get('user_id', 'system')" not in source
	assert "from .context import get_current_user_id, get_tenant_id_from_request" in source
	assert source.count("get_tenant_id_from_request()") >= 10
	assert source.count("get_current_user_id()") >= 9


def test_pim_app_integration_delegates_context_resolution():
	source = APP_INTEGRATION_PATH.read_text(encoding="utf-8")

	assert "'tenant_default'" not in source
	assert "'system'" not in source
	assert "from .context import get_current_user_id, get_tenant_id_from_request" in source
	assert "tenant_id = get_tenant_id_from_request()" in source
	assert "user_id = get_current_user_id()" in source
	assert "service.get_capability_metrics(tenant_id, user_id)" in source


def test_pim_context_resolves_tenant_and_user(monkeypatch):
	helpers = _context_helpers()
	resolve_tenant = helpers["get_tenant_id_from_request"]
	resolve_user = helpers["get_current_user_id"]
	app = Flask(__name__)
	app.secret_key = "test-secret"

	monkeypatch.setenv("APG_DEFAULT_TENANT_ID", "pim-env-tenant")
	monkeypatch.setenv("APG_DEFAULT_USER_ID", "pim-env-user")
	assert resolve_tenant() == "pim-env-tenant"
	assert resolve_user() == "pim-env-user"

	with app.test_request_context("/pim?tenant_id=query-tenant", headers={"X-Tenant-ID": "header-tenant"}):
		session["tenant_id"] = "session-tenant"
		session["user_id"] = "session-user"
		assert resolve_tenant({"tenant_id": "payload-tenant"}) == "payload-tenant"
		assert resolve_user({"user_id": "payload-user"}) == "payload-user"

	with app.test_request_context("/pim?tenant_id=query-tenant", headers={"X-Tenant-ID": "header-tenant"}):
		session["tenant_id"] = "session-tenant"
		g.tenant_id = "context-tenant"
		assert resolve_tenant({}) == "context-tenant"

	with app.test_request_context("/pim", headers={"X-Tenant-ID": "header-tenant"}):
		session["tenant_id"] = "session-tenant"
		session["user_id"] = "session-user"
		g.user = type("User", (), {"tenant_id": "fab-user-tenant", "username": "fab-user"})()
		assert resolve_tenant({}) == "fab-user-tenant"
		assert resolve_user({}) == "fab-user"

	with app.test_request_context("/pim", headers={"X-APG-User-ID": "header-user"}):
		assert resolve_user({}) == "header-user"

	with app.test_request_context("/pim", environ_base={"APG_TENANT_ID": "request-env-tenant"}):
		assert resolve_tenant({}) == "request-env-tenant"


def test_pim_context_resolves_permissions_and_wildcards(monkeypatch):
	helpers = _context_helpers()
	resolve_permissions = helpers["get_current_permissions"]
	has_permission = helpers["has_current_permission"]
	permission_matches = helpers["permission_matches"]
	app = Flask(__name__)
	app.secret_key = "test-secret"

	monkeypatch.setenv("APG_DEFAULT_PERMISSIONS", "plm.products.read,plm.changes.*")
	assert resolve_permissions() == ["plm.products.read", "plm.changes.*"]
	assert permission_matches("plm.changes.*", "plm.changes.approve")
	assert not permission_matches("plm.products.read", "plm.products.delete")

	with app.test_request_context("/pim", headers={"X-APG-Permissions": "plm.products.create plm.ai.*"}):
		assert resolve_permissions({}) == ["plm.products.create", "plm.ai.*"]
		assert has_permission("plm.products.create")
		assert has_permission("plm.ai.insights")
		assert not has_permission("plm.products.delete")

	with app.test_request_context("/pim"):
		session["permissions"] = ["plm.products.update", "plm.collaboration.*"]
		assert has_permission("plm.products.update")
		assert has_permission("plm.collaboration.participate")


def test_pim_api_permission_check_no_longer_allows_every_authenticated_user():
	source = (REPO_ROOT / "capabilities" / "pde" / "pim" / "api.py").read_text(encoding="utf-8")

	assert "return True  # For now, allow all authenticated users" not in source
	assert "from .context import get_current_permissions, permission_matches" in source
	assert "auth_service = _get_auth_rbac_service()" in source
	assert "return any(permission_matches(granted, permission) for granted in get_current_permissions())" in source
