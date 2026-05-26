"""Executable context and async-call regressions for Payment Gateway webhooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, g, has_request_context, request, session


REPO_ROOT = Path(__file__).resolve().parents[1]
GATEWAY_PATH = REPO_ROOT / "capabilities" / "fintech" / "gateway"
CONTEXT_PATH = GATEWAY_PATH / "context.py"
WEBHOOK_API_PATH = GATEWAY_PATH / "webhook_api.py"


def _context_helpers(gateway_user: dict[str, Any] | None = None) -> dict[str, Any]:
	source = CONTEXT_PATH.read_text(encoding="utf-8")
	start = source.index("def _clean_text")
	namespace: dict[str, Any] = {
		"Any": Any,
		"Dict": Dict,
		"List": List,
		"Optional": Optional,
		"g": g,
		"get_current_user": lambda: gateway_user or {},
		"has_request_context": has_request_context,
		"os": __import__("os"),
		"request": request,
		"session": session,
	}
	exec(compile(source[start:], str(CONTEXT_PATH), "exec"), namespace)
	return namespace


def test_webhook_api_is_sync_executable_and_context_backed():
	source = WEBHOOK_API_PATH.read_text(encoding="utf-8")

	assert "from .context import get_current_user_id, get_tenant_id_from_request" in source
	assert "data['tenant_id'] = data.get('tenant_id', 'default_tenant')" not in source
	assert "request.args.get('tenant_id', 'default_tenant')" not in source
	assert "required_fields = ['tenant_id', 'event_type', 'payload']" not in source
	assert "required_fields = ['event_type', 'payload']" in source
	assert "tenant_id = get_tenant_id_from_request(data)" in source
	assert "tenant_id = get_tenant_id_from_request()" in source
	assert "data.setdefault('created_by', get_current_user_id(data))" in source
	assert "await self._ensure_initialized()" not in source
	assert "self._run_async(self._ensure_initialized())" in source
	assert source.count("self._run_async(self.webhook_service.") >= 8


def test_gateway_context_resolves_tenant_and_user(monkeypatch):
	helpers = _context_helpers({"tenant_id": "gateway-tenant", "id": "gateway-user"})
	resolve_tenant = helpers["get_tenant_id_from_request"]
	resolve_user = helpers["get_current_user_id"]
	app = Flask(__name__)
	app.secret_key = "test-secret"

	monkeypatch.setenv("APG_DEFAULT_TENANT_ID", "gateway-env-tenant")
	monkeypatch.setenv("APG_DEFAULT_USER_ID", "gateway-env-user")
	assert resolve_tenant() == "gateway-tenant"
	assert resolve_user() == "gateway-user"

	with app.test_request_context("/webhooks?tenant_id=query-tenant", headers={"X-Tenant-ID": "header-tenant"}):
		session["tenant_id"] = "session-tenant"
		session["user_id"] = "session-user"
		assert resolve_tenant({"tenant_id": "payload-tenant"}) == "payload-tenant"
		assert resolve_user({"user_id": "payload-user"}) == "payload-user"

	with app.test_request_context("/webhooks?tenant_id=query-tenant", headers={"X-Tenant-ID": "header-tenant"}):
		session["tenant_id"] = "session-tenant"
		g.tenant_id = "context-tenant"
		assert resolve_tenant({}) == "gateway-tenant"

	helpers_without_gateway_user = _context_helpers({})
	resolve_tenant_without_gateway_user = helpers_without_gateway_user["get_tenant_id_from_request"]
	resolve_user_without_gateway_user = helpers_without_gateway_user["get_current_user_id"]

	with app.test_request_context("/webhooks", headers={"X-Tenant-ID": "header-tenant"}):
		session["tenant_id"] = "session-tenant"
		session["user_id"] = "session-user"
		g.user = type("User", (), {"tenant_id": "fab-user-tenant", "username": "fab-user"})()
		assert resolve_tenant_without_gateway_user({}) == "fab-user-tenant"
		assert resolve_user_without_gateway_user({}) == "fab-user"

	with app.test_request_context("/webhooks", headers={"X-APG-User-ID": "header-user"}):
		assert resolve_user_without_gateway_user({}) == "header-user"

	with app.test_request_context("/webhooks", environ_base={"APG_TENANT_ID": "request-env-tenant"}):
		assert resolve_tenant_without_gateway_user({}) == "request-env-tenant"
