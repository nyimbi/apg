"""Secure IMEX API identity regressions."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
API_SECURE_PATH = REPO_ROOT / "capabilities" / "common" / "imex" / "api_secure.py"


def test_imex_secure_login_does_not_emit_fixed_demo_user_id():
	source = API_SECURE_PATH.read_text(encoding="utf-8")

	assert 'id="user_123"' not in source
	assert '"user_123"' not in source
	assert "username=auth_request.username" in source
	assert "tenant_id=auth_request.tenant_id" in source
