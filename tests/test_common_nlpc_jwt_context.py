"""JWT context regressions for the NLP API gateway."""

from __future__ import annotations

import asyncio
import base64
import json
import textwrap
from pathlib import Path
from typing import Any, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
API_GATEWAY_PATH = REPO_ROOT / "capabilities" / "common" / "nlpc" / "api_gateway.py"


def _helpers() -> dict[str, Any]:
	source = API_GATEWAY_PATH.read_text(encoding="utf-8")
	start = source.index("\tasync def _validate_jwt_token")
	end = source.index("\tasync def _check_rate_limit")
	class_body = textwrap.dedent(source[start:end])
	namespace: dict[str, Any] = {
		"Any": Any,
		"Optional": Optional,
		"base64": base64,
		"json": json,
		"logger": type("Logger", (), {"warning": lambda self, message: None})(),
		"os": __import__("os"),
	}
	exec(compile(f"class Gateway:\n{textwrap.indent(class_body, '    ')}", str(API_GATEWAY_PATH), "exec"), namespace)
	return namespace


def _token(payload: dict[str, Any]) -> str:
	def encode(part: dict[str, Any]) -> str:
		raw = json.dumps(part, separators=(",", ":")).encode("utf-8")
		return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

	return f"{encode({'alg': 'none'})}.{encode(payload)}.signature"


def test_nlpc_jwt_validation_no_longer_returns_demo_user():
	source = API_GATEWAY_PATH.read_text(encoding="utf-8")
	assert '"user_id": "demo_user"' not in source
	assert 'payload.get("user_id")' in source
	assert 'payload.get("sub")' in source
	assert 'return None' in source


def test_nlpc_jwt_validation_requires_user_claim_and_resolves_tenant(monkeypatch):
	gateway = _helpers()["Gateway"]()
	gateway.tenant_id = "gateway-tenant"

	assert asyncio.run(gateway._validate_jwt_token("not-a-jwt")) is None
	assert asyncio.run(gateway._validate_jwt_token(_token({"tenant_id": "tenant-only"}))) is None

	context = asyncio.run(gateway._validate_jwt_token(_token({
		"sub": "jwt-user",
		"tenant_id": "jwt-tenant",
		"scope": "nlp:read nlp:write",
	})))
	assert context == {
		"user_id": "jwt-user",
		"tenant_id": "jwt-tenant",
		"scopes": ["nlp:read", "nlp:write"],
	}

	monkeypatch.setenv("APG_TENANT_ID", "env-tenant")
	context = asyncio.run(gateway._validate_jwt_token(_token({"user_id": "env-user"})))
	assert context == {
		"user_id": "env-user",
		"tenant_id": "env-tenant",
		"scopes": ["nlp:read"],
	}
