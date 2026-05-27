"""Executable token-auth regressions for INT API gateway."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import jwt


REPO_ROOT = Path(__file__).resolve().parents[1]
GATEWAY_PATH = REPO_ROOT / "capabilities" / "int" / "api" / "gateway.py"


def _auth_middleware_class():
	source = GATEWAY_PATH.read_text(encoding="utf-8")
	start = source.index("class AuthenticationMiddleware:")
	end = source.index("class PolicyEnforcementMiddleware:")
	namespace: dict[str, Any] = {
		"Any": Any,
		"Callable": Callable,
		"ConsumerManagementService": object,
		"Dict": Dict,
		"GatewayRequest": object,
		"Optional": Optional,
		"asyncio": asyncio,
		"jwt": jwt,
		"os": os,
	}
	exec(compile(source[start:end], str(GATEWAY_PATH), "exec"), namespace)
	return namespace["AuthenticationMiddleware"]


def _jwt_token(secret: str, claims: dict[str, Any]) -> str:
	payload = {
		"exp": datetime.now(timezone.utc) + timedelta(minutes=5),
		**claims,
	}
	return jwt.encode(payload, secret, algorithm="HS256")


def test_gateway_jwt_validation_decodes_claims_and_tenant():
	AuthenticationMiddleware = _auth_middleware_class()
	secret = "test-secret-key-32-characters-long"
	middleware = AuthenticationMiddleware(object(), jwt_secret=secret, jwt_algorithm="HS256")
	token = _jwt_token(secret, {
		"sub": "consumer-1",
		"tenant_id": "tenant-a",
		"scope": "read write",
	})

	result = asyncio.run(middleware._validate_jwt_token(token))

	assert result["success"] is True
	assert result["consumer_id"] == "consumer-1"
	assert result["tenant_id"] == "tenant-a"
	assert result["auth_method"] == "jwt"


def test_gateway_bearer_validation_delegates_to_configured_validator():
	AuthenticationMiddleware = _auth_middleware_class()

	def validate_token(token: str) -> dict[str, Any]:
		assert token == "opaque-token"
		return {"consumer_id": "consumer-opaque", "tenant_id": "tenant-b"}

	middleware = AuthenticationMiddleware(object(), bearer_token_validator=validate_token)

	result = asyncio.run(middleware._validate_bearer_token("opaque-token"))

	assert result["success"] is True
	assert result["consumer_id"] == "consumer-opaque"
	assert result["tenant_id"] == "tenant-b"
	assert result["auth_method"] == "bearer"


def test_gateway_token_auth_no_longer_reports_not_implemented():
	source = GATEWAY_PATH.read_text(encoding="utf-8")

	assert "JWT validation not implemented" not in source
	assert "Bearer token validation not implemented" not in source
	assert "jwt.decode(" in source
	assert "def _normalize_token_validation_result(self, result: Any, auth_method: str) -> Dict[str, Any]:" in source
	assert "bearer_token_validator: Optional[Callable[[str], Any]] = None" in source
