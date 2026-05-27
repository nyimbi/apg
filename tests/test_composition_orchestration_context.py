"""Context regressions for Composition Workflow Orchestration APIs."""

from __future__ import annotations

import asyncio
import base64
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.requests import Request


REPO_ROOT = Path(__file__).resolve().parents[1]
API_PATH = REPO_ROOT / "capabilities" / "composition" / "orchestration" / "api.py"
ADVANCED_API_PATH = REPO_ROOT / "capabilities" / "composition" / "orchestration" / "advanced_api.py"
COMPONENT_LIBRARY_PATH = REPO_ROOT / "capabilities" / "composition" / "orchestration" / "component_library.py"
USER_EXPERIENCE_PATH = REPO_ROOT / "capabilities" / "composition" / "orchestration" / "user_experience.py"


def _request(path: str = "/workflows", headers: dict[str, str] | None = None) -> Request:
	raw_headers = [
		(name.lower().encode("latin-1"), value.encode("latin-1"))
		for name, value in (headers or {}).items()
	]
	path_part, _, query = path.partition("?")
	return Request(
		{
			"type": "http",
			"method": "GET",
			"path": path_part,
			"headers": raw_headers,
			"query_string": query.encode("latin-1"),
		}
	)


def _jwt_token(claims: dict[str, Any]) -> str:
	header = {"alg": "none", "typ": "JWT"}
	parts = []
	for value in (header, claims):
		encoded = base64.urlsafe_b64encode(json.dumps(value).encode("utf-8")).decode("ascii")
		parts.append(encoded.rstrip("="))
	return f"{parts[0]}.{parts[1]}."


def _api_helpers() -> dict[str, Any]:
	source = API_PATH.read_text(encoding="utf-8")
	start = source.index("def _clean_text")
	end = source.index("async def get_tenant_id")
	namespace: dict[str, Any] = {
		"Any": Any,
		"Depends": Depends,
		"Dict": Dict,
		"HTTPAuthorizationCredentials": HTTPAuthorizationCredentials,
		"HTTPException": HTTPException,
		"List": List,
		"Optional": Optional,
		"Request": Request,
		"json": json,
		"logger": SimpleNamespace(error=lambda message: None),
		"os": os,
		"security": HTTPBearer(),
	}
	exec(compile(source[start:end], str(API_PATH), "exec"), namespace)
	return namespace


def _advanced_helpers() -> dict[str, Any]:
	source = ADVANCED_API_PATH.read_text(encoding="utf-8")
	start = source.index("def _clean_context_text")
	end = source.index("\n\nclass APIVersion")
	namespace: dict[str, Any] = {
		"Any": Any,
		"List": List,
		"Optional": Optional,
		"os": os,
	}
	exec(compile(source[start:end], str(ADVANCED_API_PATH), "exec"), namespace)
	return namespace


def _component_library_helpers() -> dict[str, Any]:
	source = COMPONENT_LIBRARY_PATH.read_text(encoding="utf-8")
	start = source.index("def _clean_component_context_text")
	end = source.index("\n\nclass ComponentType")
	namespace: dict[str, Any] = {
		"Any": Any,
		"Dict": Dict,
		"List": List,
		"Optional": Optional,
		"os": os,
	}
	exec(compile(source[start:end], str(COMPONENT_LIBRARY_PATH), "exec"), namespace)
	return namespace


def test_orchestration_context_sources_no_longer_use_fixed_placeholders():
	api_source = API_PATH.read_text(encoding="utf-8")
	advanced_source = ADVANCED_API_PATH.read_text(encoding="utf-8")
	component_library_source = COMPONENT_LIBRARY_PATH.read_text(encoding="utf-8")

	assert '"default_tenant"' not in api_source
	assert "'default_tenant'" not in advanced_source
	assert "'default_tenant'" not in component_library_source
	assert "payload.get(\"tenant_id\", \"default_tenant\")" not in api_source
	assert "getattr(info.context, 'tenant_id', 'default_tenant')" not in advanced_source
	assert "'created_by': 'current_user'" not in advanced_source
	assert "getattr(self, 'tenant_id', 'default_tenant')" not in component_library_source


def test_orchestration_user_experience_search_uses_request_tenant_context():
	source = USER_EXPERIENCE_PATH.read_text(encoding="utf-8")

	assert "from ...common.request_context import get_tenant_id_from_context" in source
	assert "params['tenant_id'] = 'default'" not in source
	assert "Get from context" not in source
	assert "params['tenant_id'] = get_tenant_id_from_context()" in source


def test_orchestration_rest_auth_resolves_claims_headers_query_and_env(monkeypatch):
	resolve_user = _api_helpers()["get_current_user"]
	credentials = HTTPAuthorizationCredentials(
		scheme="Bearer",
		credentials=_jwt_token({
			"sub": "claim-user",
			"tenant_id": "claim-tenant",
			"roles": ["workflow_admin"],
			"permissions": ["workflow.execute"],
		}),
	)

	user = asyncio.run(resolve_user(_request(), credentials))
	assert user == {
		"user_id": "claim-user",
		"tenant_id": "claim-tenant",
		"roles": ["workflow_admin"],
		"permissions": ["workflow.execute"],
	}

	monkeypatch.setenv("APG_DEFAULT_USER_ID", "env-user")
	monkeypatch.setenv("APG_DEFAULT_TENANT_ID", "env-tenant")
	credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="opaque")
	user = asyncio.run(
		resolve_user(
			_request(
				"/workflows?tenant_id=query-tenant",
				{"X-APG-User-ID": "header-user", "X-APG-Permissions": "workflow.read workflow.write"},
			),
			credentials,
		)
	)
	assert user["user_id"] == "header-user"
	assert user["tenant_id"] == "query-tenant"
	assert user["permissions"] == ["workflow.read", "workflow.write"]


def test_orchestration_graphql_context_helpers_resolve_tenant_and_actor(monkeypatch):
	helpers = _advanced_helpers()
	resolve_tenant = helpers["resolve_graphql_tenant_id"]
	resolve_user = helpers["resolve_graphql_user_id"]

	info = SimpleNamespace(context=SimpleNamespace(
		tenant_id="context-tenant",
		current_user={"user_id": "context-user", "tenant_id": "user-tenant"},
	))
	assert resolve_tenant(info) == "context-tenant"
	assert resolve_user(info) == "context-user"

	monkeypatch.setenv("APG_DEFAULT_TENANT_ID", "env-tenant")
	monkeypatch.setenv("APG_DEFAULT_USER_ID", "env-user")
	assert resolve_tenant(SimpleNamespace(context=SimpleNamespace())) == "env-tenant"
	assert resolve_user(SimpleNamespace(context=SimpleNamespace())) == "env-user"


def test_component_library_resolves_custom_component_tenant(monkeypatch):
	resolve_tenant = _component_library_helpers()["resolve_component_library_tenant_id"]

	monkeypatch.setenv("APG_DEFAULT_TENANT_ID", "env-tenant")
	assert resolve_tenant() == "env-tenant"

	assert resolve_tenant({"tenant_id": "definition-tenant"}) == "definition-tenant"
	assert resolve_tenant({"organization_id": "org-tenant"}) == "org-tenant"

	service = SimpleNamespace(tenant_id="service-tenant")
	assert resolve_tenant({"tenant_id": "definition-tenant"}, service) == "service-tenant"
