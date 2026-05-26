"""Context resolution regressions for HCM Time & Attendance APIs."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
CAPABILITY_PATH = REPO_ROOT / "capabilities" / "hcm" / "tat" / "time_attendance"
CONTEXT_PATH = CAPABILITY_PATH / "context.py"
API_PATH = CAPABILITY_PATH / "api.py"


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


def test_time_attendance_api_delegates_auth_context_resolution():
	source = API_PATH.read_text(encoding="utf-8")

	assert "from .context import resolve_current_user_context" in source
	assert "Request" in source
	assert '"user_id": "user_123"' not in source
	assert '"tenant_id": "tenant_default"' not in source
	assert "TODO: Implement actual JWT token validation" not in source
	assert "return resolve_current_user_context(request)" in source


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
