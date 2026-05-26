"""Context resolution regressions for MTen and CRM API auth helpers."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
MTEN_API_PATH = REPO_ROOT / "capabilities" / "common" / "mten" / "api.py"
CRM_API_PATH = REPO_ROOT / "capabilities" / "crm" / "adv" / "api.py"


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
