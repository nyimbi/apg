"""Context resolution regressions for MTen and CRM API auth helpers."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
MTEN_API_PATH = REPO_ROOT / "capabilities" / "common" / "mten" / "api.py"
CRM_API_PATH = REPO_ROOT / "capabilities" / "crm" / "adv" / "api.py"
CRM_BLUEPRINT_PATH = REPO_ROOT / "capabilities" / "crm" / "adv" / "blueprint.py"


def _mten_helpers() -> dict[str, Any]:
	source = MTEN_API_PATH.read_text(encoding="utf-8")
	start = source.index("def _clean_text")
	end = source.index("class MultiTenantAPI")
	namespace: dict[str, Any] = {
		"Any": Any,
		"Dict": Dict,
		"List": List,
		"Optional": Optional,
		"Request": SimpleNamespace,
		"os": __import__("os"),
	}
	exec(compile(source[start:end], str(MTEN_API_PATH), "exec"), namespace)
	return namespace


def _crm_helpers() -> dict[str, Any]:
	source = CRM_API_PATH.read_text(encoding="utf-8")
	start = source.index("def _clean_text")
	end = source.index("def get_tenant_id")
	namespace: dict[str, Any] = {
		"Any": Any,
		"Dict": Dict,
		"List": List,
		"Optional": Optional,
		"Depends": lambda dependency=None: None,
		"HTTPAuthorizationCredentials": SimpleNamespace,
		"Request": SimpleNamespace,
		"os": __import__("os"),
		"security": object(),
	}
	exec(compile(source[start:end], str(CRM_API_PATH), "exec"), namespace)
	return namespace


def _crm_blueprint_helpers(has_context: bool = False) -> dict[str, Any]:
	source = CRM_BLUEPRINT_PATH.read_text(encoding="utf-8")
	start = source.index("def _clean_text")
	end = source.index("\n\n# CRM Configuration Class")
	namespace: dict[str, Any] = {
		"Any": Any,
		"AppBuilder": SimpleNamespace,
		"Dict": Dict,
		"List": List,
		"Optional": Optional,
		"g": SimpleNamespace(),
		"has_request_context": lambda: has_context,
		"os": __import__("os"),
		"request": SimpleNamespace(headers={}, args={}),
		"session": {},
	}
	exec(compile(source[start:end], str(CRM_BLUEPRINT_PATH), "exec"), namespace)
	return namespace


def test_mten_and_crm_no_longer_use_fixed_mock_auth_identity():
	for path in (MTEN_API_PATH, CRM_API_PATH):
		source = path.read_text(encoding="utf-8")
		for stale_text in (
			"return \"user-123\"",
			"For now, return mock user ID",
			"For now, return mock user",
			'"mock_user_001"',
			'"mock_tenant_001"',
			"TODO: Implement proper JWT token validation",
		):
			assert stale_text not in source

	assert "request: Request" in MTEN_API_PATH.read_text(encoding="utf-8")
	assert "request: Request" in CRM_API_PATH.read_text(encoding="utf-8")


def test_crm_blueprint_no_longer_sets_fixed_tenant_context():
	source = CRM_BLUEPRINT_PATH.read_text(encoding="utf-8")

	assert "getattr(g, 'user', {}).get('tenant_id', 'default_tenant')" not in source
	assert "'default_tenant'" not in source
	assert "g.tenant_id = context['tenant_id']" in source
	assert "g.user_id = context['user_id']" in source


def test_mten_user_resolution_prefers_state_then_headers_then_env(monkeypatch):
	resolve = _mten_helpers()["resolve_apg_user_id"]

	monkeypatch.setenv("APG_DEFAULT_USER_ID", "env-user")
	request = SimpleNamespace(state=SimpleNamespace(), headers={}, query_params={})
	assert resolve(request) == "env-user"

	request = SimpleNamespace(
		state=SimpleNamespace(current_user={"user_id": "state-user"}),
		headers={"X-APG-User-ID": "header-user"},
		query_params={"user_id": "query-user"},
	)
	assert resolve(request) == "state-user"

	request = SimpleNamespace(
		state=SimpleNamespace(),
		headers={"X-APG-User-ID": "header-user"},
		query_params={"user_id": "query-user"},
	)
	assert resolve(request) == "header-user"


def test_crm_user_resolution_prefers_state_headers_and_roles(monkeypatch):
	resolve_user = _crm_helpers()["get_current_user"]

	monkeypatch.setenv("APG_DEFAULT_USER_ID", "env-user")
	monkeypatch.setenv("APG_DEFAULT_TENANT_ID", "env-tenant")
	request = SimpleNamespace(state=SimpleNamespace(), headers={}, query_params={})
	context = asyncio.run(resolve_user(request, SimpleNamespace(scheme="Bearer")))
	assert context["user_id"] == "env-user"
	assert context["tenant_id"] == "env-tenant"
	assert context["roles"] == ["crm_user"]

	request = SimpleNamespace(
		state=SimpleNamespace(current_user={"user_id": "state-user", "tenant_id": "state-tenant", "roles": ["crm_admin"]}),
		headers={"X-APG-User-ID": "header-user", "X-APG-Tenant-ID": "header-tenant"},
		query_params={"tenant": "query-tenant"},
	)
	context = asyncio.run(resolve_user(request, SimpleNamespace(scheme="Bearer")))
	assert context["user_id"] == "state-user"
	assert context["tenant_id"] == "state-tenant"
	assert context["roles"] == ["crm_admin"]

	request = SimpleNamespace(
		state=SimpleNamespace(),
		headers={"X-APG-User-ID": "header-user", "X-APG-Tenant-ID": "header-tenant", "X-APG-Roles": "crm_user, crm_admin"},
		query_params={},
	)
	context = asyncio.run(resolve_user(request, SimpleNamespace(scheme="Bearer")))
	assert context["user_id"] == "header-user"
	assert context["tenant_id"] == "header-tenant"
	assert context["roles"] == ["crm_user", "crm_admin"]


def test_crm_blueprint_request_context_resolves_runtime_tenant_and_actor(monkeypatch):
	helpers = _crm_blueprint_helpers()
	resolve = helpers["_resolve_crm_request_context"]

	monkeypatch.setenv("APG_DEFAULT_USER_ID", "env-user")
	monkeypatch.setenv("APG_DEFAULT_TENANT_ID", "env-tenant")
	assert resolve() == {"tenant_id": "env-tenant", "user_id": "env-user"}

	helpers = _crm_blueprint_helpers(has_context=True)
	helpers["g"] = SimpleNamespace(current_user={"tenant_id": "g-tenant", "user_id": "g-user"})
	helpers["request"] = SimpleNamespace(
		headers={"X-APG-Tenant-ID": "header-tenant", "X-APG-User-ID": "header-user"},
		args={"tenant_id": "query-tenant", "user_id": "query-user"},
	)
	helpers["session"] = {"tenant_id": "session-tenant", "user_id": "session-user"}
	assert helpers["_resolve_crm_request_context"]() == {"tenant_id": "g-tenant", "user_id": "g-user"}

	helpers = _crm_blueprint_helpers(has_context=True)
	helpers["g"] = SimpleNamespace()
	helpers["request"] = SimpleNamespace(
		headers={"X-APG-Tenant-ID": "header-tenant", "X-APG-User-ID": "header-user"},
		args={"tenant_id": "query-tenant", "user_id": "query-user"},
	)
	helpers["session"] = {}
	assert helpers["_resolve_crm_request_context"]() == {"tenant_id": "header-tenant", "user_id": "header-user"}
