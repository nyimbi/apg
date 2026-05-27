"""Context regressions for Composition Config security authentication."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
SECURITY_ENGINE_PATH = REPO_ROOT / "capabilities" / "composition" / "config" / "security_engine.py"


def _helpers() -> dict[str, Any]:
	source = SECURITY_ENGINE_PATH.read_text(encoding="utf-8")
	start = source.index("def _clean_security_text")
	end = source.index("\n\nclass SecurityLevel")
	namespace: dict[str, Any] = {
		"Any": Any,
		"Dict": Dict,
		"List": List,
		"Optional": Optional,
		"os": os,
	}
	exec(compile(source[start:end], str(SECURITY_ENGINE_PATH), "exec"), namespace)
	return namespace


def test_config_security_engine_no_longer_uses_fixed_api_key_actor():
	source = SECURITY_ENGINE_PATH.read_text(encoding="utf-8")

	assert 'user_id = "api_user"' not in source
	assert "For now, simple validation" not in source
	assert "_resolve_api_key_user_id(credentials)" in source
	assert "_resolve_api_key_permissions(credentials)" in source


def test_config_security_engine_api_key_identity_resolves_credentials_and_env(monkeypatch):
	helpers = _helpers()
	resolve_user = helpers["_resolve_api_key_user_id"]
	resolve_permissions = helpers["_resolve_api_key_permissions"]

	assert resolve_user({"user_id": "credential-user", "client_id": "client-user"}) == "credential-user"
	assert resolve_user({"client_id": "client-user"}) == "client-user"
	assert resolve_permissions({"permissions": ["read", "write"]}) == ["read", "write"]
	assert resolve_permissions({"scope": "read audit"}) == ["read", "audit"]

	monkeypatch.setenv("APG_API_KEY_USER_ID", "env-key-user")
	monkeypatch.setenv("APG_API_KEY_PERMISSIONS", "read,write,admin")
	assert resolve_user({}) == "env-key-user"
	assert resolve_permissions({}) == ["read", "write", "admin"]
