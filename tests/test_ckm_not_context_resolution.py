"""Context resolution regressions for CKM notification capability surfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, g, has_request_context, request


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTEXT_PATH = REPO_ROOT / "capabilities" / "ckm" / "not" / "context.py"
SURFACE_PATHS = [
	REPO_ROOT / "capabilities" / "ckm" / "not" / "views.py",
	REPO_ROOT / "capabilities" / "ckm" / "not" / "api.py",
	REPO_ROOT / "capabilities" / "ckm" / "not" / "blueprint.py",
	REPO_ROOT / "capabilities" / "ckm" / "not" / "websocket.py",
	REPO_ROOT / "capabilities" / "ckm" / "not" / "personalization" / "api.py",
]


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


def test_ckm_notification_surfaces_delegate_context_resolution():
	for path in SURFACE_PATHS:
		source = path.read_text(encoding="utf-8")
		assert "'default_tenant'" not in source
		assert '"default_tenant"' not in source
		assert "'user_123'" not in source
		assert '"user_123"' not in source

	assert "from .context import get_tenant_id_from_context" in SURFACE_PATHS[0].read_text(encoding="utf-8")
	assert "from .context import get_tenant_id_from_context" in SURFACE_PATHS[1].read_text(encoding="utf-8")
	assert "from .context import get_tenant_id_from_context" in SURFACE_PATHS[2].read_text(encoding="utf-8")
	assert "from .context import get_current_user_id, get_tenant_id_from_context" in SURFACE_PATHS[3].read_text(encoding="utf-8")
	assert "from ..context import get_current_user_id, get_tenant_id_from_context" in SURFACE_PATHS[4].read_text(encoding="utf-8")


def test_ckm_notification_context_resolves_tenant_and_user(monkeypatch):
	helpers = _context_helpers()
	resolve_tenant = helpers["get_tenant_id_from_context"]
	resolve_user = helpers["get_current_user_id"]
	app = Flask(__name__)

	monkeypatch.setenv("APG_DEFAULT_TENANT_ID", "ckm-not-env-tenant")
	monkeypatch.setenv("APG_DEFAULT_USER_ID", "ckm-not-env-user")
	assert resolve_tenant() == "ckm-not-env-tenant"
	assert resolve_user() == "ckm-not-env-user"

	assert resolve_tenant({"tenant_id": "payload-tenant"}) == "payload-tenant"
	assert resolve_user({"user_id": "payload-user"}) == "payload-user"

	with app.test_request_context("/ckm-not?tenant_id=query-tenant", headers={"X-Tenant-ID": "header-tenant"}):
		g.tenant_id = "context-tenant"
		assert resolve_tenant({}) == "context-tenant"

	with app.test_request_context("/ckm-not", headers={"X-Tenant-ID": "header-tenant"}):
		g.user = type("User", (), {"tenant_id": "fab-user-tenant", "username": "fab-user"})()
		assert resolve_tenant({}) == "fab-user-tenant"
		assert resolve_user({}) == "fab-user"

	with app.test_request_context("/ckm-not", headers={"X-APG-User-ID": "header-user"}):
		assert resolve_user({}) == "header-user"

	with app.test_request_context("/ckm-not", environ_base={"APG_TENANT_ID": "request-env-tenant"}):
		assert resolve_tenant({}) == "request-env-tenant"
