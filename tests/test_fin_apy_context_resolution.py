"""Context resolution regressions for Accounts Payable APIs."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
CAPABILITY_PATH = REPO_ROOT / "capabilities" / "fin" / "apy" / "accounts_payable"
CONTEXT_PATH = CAPABILITY_PATH / "context.py"
API_PATH = CAPABILITY_PATH / "api.py"


def _context_helpers() -> dict[str, Any]:
	source = CONTEXT_PATH.read_text(encoding="utf-8")
	start = source.index("DEFAULT_PERMISSIONS")
	namespace: dict[str, Any] = {
		"Any": Any,
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


def test_accounts_payable_api_delegates_request_context_resolution():
	source = API_PATH.read_text(encoding="utf-8")

	assert "from .context import resolve_apg_user_context" in source
	assert "request: Request" in source
	assert "return APGUserContext(**resolve_apg_user_context(request=request))" in source
	assert 'user_id="user_123"' not in source
	assert 'tenant_id="tenant_456"' not in source
	assert "return a mock user context" not in source
	assert "validate the JWT token" not in source


def test_accounts_payable_context_resolves_identity_permissions_and_roles(monkeypatch):
	helpers = _context_helpers()
	resolve = helpers["resolve_apg_user_context"]

	monkeypatch.setenv("APG_DEFAULT_USER_ID", "apy-env-user")
	monkeypatch.setenv("APG_DEFAULT_TENANT_ID", "apy-env-tenant")
	monkeypatch.setenv("APG_APY_PERMISSIONS", "ap.read,ap.pay")
	monkeypatch.setenv("APG_APY_ROLES", "ap_clerk,ap_reviewer")
	assert resolve(_request()) == {
		"user_id": "apy-env-user",
		"tenant_id": "apy-env-tenant",
		"permissions": ["ap.read", "ap.pay"],
		"roles": ["ap_clerk", "ap_reviewer"],
	}

	state = SimpleNamespace(
		current_user={
			"user_id": "state-user",
			"tenant_id": "state-tenant",
			"permissions": ["ap.admin"],
			"roles": ["ap_manager"],
		}
	)
	request = _request(
		headers={
			"X-User-ID": "header-user",
			"X-Tenant-ID": "header-tenant",
			"X-APG-Permissions": "ap.read, ap.write",
			"X-APG-Roles": "ap_reviewer",
		},
		state=state,
	)
	assert resolve(request) == {
		"user_id": "state-user",
		"tenant_id": "state-tenant",
		"permissions": ["ap.admin"],
		"roles": ["ap_manager"],
	}

	request = _request(
		headers={
			"X-APG-User-ID": "header-user",
			"X-APG-Tenant-ID": "header-tenant",
			"X-APG-Permissions": "ap.read, ap.write",
			"X-APG-Roles": "ap_reviewer, ap_approver",
		}
	)
	assert resolve(request) == {
		"user_id": "header-user",
		"tenant_id": "header-tenant",
		"permissions": ["ap.read", "ap.write"],
		"roles": ["ap_reviewer", "ap_approver"],
	}

	request = _request(query={"user_id": "query-user", "tenant": "query-tenant"})
	assert resolve(request) == {
		"user_id": "query-user",
		"tenant_id": "query-tenant",
		"permissions": ["ap.read", "ap.pay"],
		"roles": ["ap_clerk", "ap_reviewer"],
	}
