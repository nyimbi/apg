"""Context resolution regressions for HCM Time & Attendance APIs."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
CAPABILITY_PATH = REPO_ROOT / "capabilities" / "hcm" / "tat" / "time_attendance"
CONTEXT_PATH = CAPABILITY_PATH / "context.py"
API_PATH = CAPABILITY_PATH / "api.py"
MOBILE_API_PATH = CAPABILITY_PATH / "mobile_api.py"
MONITORING_PATH = CAPABILITY_PATH / "monitoring.py"


def _context_helpers() -> dict[str, Any]:
	source = CONTEXT_PATH.read_text(encoding="utf-8")
	start = source.index("def _clean_text")
	namespace: dict[str, Any] = {
		"Any": Any,
		"Dict": Dict,
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


def _jwt_token(claims: dict[str, Any]) -> str:
	header = {"alg": "none", "typ": "JWT"}
	parts = []
	for value in (header, claims):
		encoded = base64.urlsafe_b64encode(json.dumps(value).encode("utf-8")).decode("ascii")
		parts.append(encoded.rstrip("="))
	return f"{parts[0]}.{parts[1]}."


def test_time_attendance_api_delegates_auth_context_resolution():
	source = API_PATH.read_text(encoding="utf-8")

	assert "from .context import resolve_current_user_context" in source
	assert "Request" in source
	assert '"user_id": "user_123"' not in source
	assert '"tenant_id": "tenant_default"' not in source
	assert "TODO: Implement actual JWT token validation" not in source
	assert "return resolve_current_user_context(request)" in source


def test_time_attendance_mobile_api_uses_apg_auth_context():
	source = MOBILE_API_PATH.read_text(encoding="utf-8")
	monitoring_source = MONITORING_PATH.read_text(encoding="utf-8")

	assert "resolve_current_user_context(request, credentials=credentials, roles=[\"employee\"])" in source
	assert "decode_bearer_claims(credentials)" in source
	assert '"mobile_user_123"' not in source
	assert '"emp_123"' not in source
	assert '"device_mobile_123"' not in source
	assert '"tenant_id": "tenant_default"' not in source
	assert "TODO: Implement mobile-specific JWT validation" not in source
	assert '"tenant_default"' not in monitoring_source


def test_time_attendance_context_resolves_user_and_tenant(monkeypatch):
	helpers = _context_helpers()
	resolve = helpers["resolve_current_user_context"]

	monkeypatch.setenv("APG_DEFAULT_USER_ID", "tat-env-user")
	monkeypatch.setenv("APG_DEFAULT_TENANT_ID", "tat-env-tenant")
	assert resolve(_request()) == {
		"user_id": "tat-env-user",
		"tenant_id": "tat-env-tenant",
		"roles": ["employee", "manager"],
	}

	state = SimpleNamespace(current_user={"user_id": "state-user", "tenant_id": "state-tenant"})
	assert resolve(_request(headers={"X-User-ID": "header-user", "X-Tenant-ID": "header-tenant"}, state=state)) == {
		"user_id": "state-user",
		"tenant_id": "state-tenant",
		"roles": ["employee", "manager"],
	}

	assert resolve(_request(headers={"X-APG-User-ID": "header-user", "X-APG-Tenant-ID": "header-tenant"})) == {
		"user_id": "header-user",
		"tenant_id": "header-tenant",
		"roles": ["employee", "manager"],
	}

	assert resolve(_request(query={"user_id": "query-user", "tenant_id": "query-tenant"}), roles=["approver"]) == {
		"user_id": "query-user",
		"tenant_id": "query-tenant",
		"roles": ["approver"],
	}


def test_time_attendance_context_resolves_bearer_claims(monkeypatch):
	helpers = _context_helpers()
	resolve = helpers["resolve_current_user_context"]
	credentials = SimpleNamespace(credentials=_jwt_token({"sub": "claim-user", "tenant_id": "claim-tenant"}))

	assert resolve(_request(), credentials=credentials) == {
		"user_id": "claim-user",
		"tenant_id": "claim-tenant",
		"roles": ["employee", "manager"],
	}

	monkeypatch.setenv("APG_DEFAULT_USER_ID", "fallback-user")
	monkeypatch.setenv("APG_DEFAULT_TENANT_ID", "fallback-tenant")
	assert resolve(_request(), credentials=SimpleNamespace(credentials="opaque")) == {
		"user_id": "fallback-user",
		"tenant_id": "fallback-tenant",
		"roles": ["employee", "manager"],
	}
