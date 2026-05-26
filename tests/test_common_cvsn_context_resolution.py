"""Context resolution regressions for Computer Vision services."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
CAPABILITY_PATH = REPO_ROOT / "capabilities" / "common" / "cvsn"
CONTEXT_PATH = CAPABILITY_PATH / "context.py"
API_PATH = CAPABILITY_PATH / "api.py"
VIEWS_PATH = CAPABILITY_PATH / "views.py"
BLUEPRINT_PATH = CAPABILITY_PATH / "blueprints" / "blueprint.py"


def _context_helpers() -> dict[str, Any]:
	source = CONTEXT_PATH.read_text(encoding="utf-8")
	start = source.index("DEFAULT_PERMISSIONS")
	namespace: dict[str, Any] = {
		"Any": Any,
		"Iterable": Iterable,
		"Optional": Optional,
		"os": __import__("os"),
	}
	exec(compile(source[start:], str(CONTEXT_PATH), "exec"), namespace)
	return namespace


def _request(headers: dict[str, str] | None = None, query: dict[str, str] | None = None, state: Any = None):
	return SimpleNamespace(
		headers=headers or {},
		query_params=query or {},
		state=state or SimpleNamespace(),
	)


def test_cvsn_runtime_surfaces_delegate_request_context_resolution():
	api_source = API_PATH.read_text(encoding="utf-8")
	views_source = VIEWS_PATH.read_text(encoding="utf-8")
	blueprint_source = BLUEPRINT_PATH.read_text(encoding="utf-8")

	assert "from .context import resolve_current_user_info" in api_source
	assert "request: Request" in api_source
	assert "return resolve_current_user_info(request=request, credentials=credentials)" in api_source
	assert '"user_id": "user_123"' not in api_source
	assert '"tenant_id": "tenant_456"' not in api_source
	assert "Placeholder implementation - would integrate with APG RBAC" not in api_source

	assert "from .context import resolve_current_user_info" in views_source
	assert "return resolve_current_user_info(request=request, session=session, g=g)" in views_source
	assert '"user_id": "user_123"' not in views_source
	assert '"tenant_id": "tenant_456"' not in views_source

	assert "from ..context import resolve_current_user_info" in blueprint_source
	assert 'g.user_id = user_info["user_id"]' in blueprint_source
	assert 'g.user_permissions = user_info["permissions"]' in blueprint_source


def test_cvsn_context_resolves_identity_tenant_and_permissions(monkeypatch):
	helpers = _context_helpers()
	resolve = helpers["resolve_current_user_info"]

	monkeypatch.setenv("APG_DEFAULT_USER_ID", "cvsn-env-user")
	monkeypatch.setenv("APG_DEFAULT_TENANT_ID", "cvsn-env-tenant")
	monkeypatch.setenv("APG_CVSN_PERMISSIONS", "cv:read,cv:inspect")
	assert resolve(_request()) == {
		"user_id": "cvsn-env-user",
		"tenant_id": "cvsn-env-tenant",
		"permissions": ["cv:read", "cv:inspect"],
	}

	state = SimpleNamespace(
		current_user={
			"user_id": "state-user",
			"tenant_id": "state-tenant",
			"permissions": ["cv:admin"],
		}
	)
	request = _request(
		headers={
			"X-User-ID": "header-user",
			"X-Tenant-ID": "header-tenant",
			"X-APG-Permissions": "cv:read, cv:write",
		},
		state=state,
	)
	assert resolve(request) == {
		"user_id": "state-user",
		"tenant_id": "state-tenant",
		"permissions": ["cv:admin"],
	}

	request = _request(
		headers={
			"X-APG-User-ID": "header-user",
			"X-APG-Tenant-ID": "header-tenant",
			"X-APG-Permissions": "cv:read, cv:write",
		}
	)
	assert resolve(request) == {
		"user_id": "header-user",
		"tenant_id": "header-tenant",
		"permissions": ["cv:read", "cv:write"],
	}

	request = _request(query={"user_id": "query-user", "tenant": "query-tenant"})
	assert resolve(request, session={"user_id": "session-user", "tenant_id": "session-tenant"}) == {
		"user_id": "query-user",
		"tenant_id": "query-tenant",
		"permissions": ["cv:read", "cv:inspect"],
	}

	assert resolve(_request(), g=SimpleNamespace(user_id="g-user", tenant_id="g-tenant")) == {
		"user_id": "g-user",
		"tenant_id": "g-tenant",
		"permissions": ["cv:read", "cv:inspect"],
	}
