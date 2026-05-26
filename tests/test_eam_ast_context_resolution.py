"""Context resolution regressions for Enterprise Asset Management API auth."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
API_PATH = REPO_ROOT / "capabilities" / "eam" / "ast" / "api.py"


def _api_context_helpers() -> dict[str, Any]:
	source = API_PATH.read_text(encoding="utf-8")
	start = source.index("def _clean_text")
	end = source.index("async def get_database_session")
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
	exec(compile(source[start:end], str(API_PATH), "exec"), namespace)
	return namespace


def test_eam_api_auth_no_longer_uses_fixed_mock_identity():
	source = API_PATH.read_text(encoding="utf-8")

	for stale_text in (
		'"user-123"',
		'"tenant-456"',
		"mock user data",
		"For now, return mock user",
		'["eam.asset.view", "eam.asset.create", "eam.workorder.view"]',
	):
		assert stale_text not in source

	assert "request: Request" in source
	assert "headers.get(\"X-APG-User-ID\")" in source
	assert "headers.get(\"X-APG-Tenant-ID\")" in source
	assert "headers.get(\"X-APG-Permissions\")" in source
	assert 'or ["eam.asset.view"]' in source


def test_eam_api_auth_context_resolves_from_state_headers_query_and_env(monkeypatch):
	helpers = _api_context_helpers()
	resolve_user = helpers["get_current_user"]

	monkeypatch.setenv("APG_DEFAULT_USER_ID", "env-user")
	monkeypatch.setenv("APG_DEFAULT_TENANT_ID", "env-tenant")
	request = SimpleNamespace(state=SimpleNamespace(), headers={}, query_params={})
	context = asyncio.run(resolve_user(request, SimpleNamespace(scheme="Bearer")))
	assert context["user_id"] == "env-user"
	assert context["tenant_id"] == "env-tenant"
	assert context["permissions"] == ["eam.asset.view"]

	request = SimpleNamespace(
		state=SimpleNamespace(
			current_user={
				"user_id": "state-user",
				"tenant_id": "state-tenant",
				"permissions": ["eam.workorder.view"],
			}
		),
		headers={"X-APG-User-ID": "header-user", "X-APG-Tenant-ID": "header-tenant"},
		query_params={"tenant": "query-tenant"},
	)
	context = asyncio.run(resolve_user(request, SimpleNamespace(scheme="Bearer")))
	assert context["user_id"] == "state-user"
	assert context["tenant_id"] == "state-tenant"
	assert context["permissions"] == ["eam.workorder.view"]

	request = SimpleNamespace(
		state=SimpleNamespace(),
		headers={
			"X-APG-User-ID": "header-user",
			"X-APG-Tenant-ID": "header-tenant",
			"X-APG-Permissions": "eam.asset.view, eam.asset.create",
		},
		query_params={"tenant": "query-tenant"},
	)
	context = asyncio.run(resolve_user(request, SimpleNamespace(scheme="Bearer")))
	assert context["user_id"] == "header-user"
	assert context["tenant_id"] == "header-tenant"
	assert context["permissions"] == ["eam.asset.view", "eam.asset.create"]
