"""Context resolution regressions for the ETLP Flask blueprint."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
BLUEPRINT_PATH = REPO_ROOT / "capabilities" / "common" / "etlp" / "blueprint.py"


def _helpers(has_context: bool = False) -> dict[str, Any]:
	source = BLUEPRINT_PATH.read_text(encoding="utf-8")
	start = source.index("def _clean_text")
	end = source.index("\n\nclass ETLPDashboardView")
	namespace: dict[str, Any] = {
		"Any": Any,
		"Dict": Dict,
		"List": List,
		"Optional": Optional,
		"g": SimpleNamespace(),
		"has_request_context": lambda: has_context,
		"os": os,
		"request": SimpleNamespace(headers={}, args={}),
		"session": {},
	}
	exec(compile(source[start:end], str(BLUEPRINT_PATH), "exec"), namespace)
	return namespace


def _view(appbuilder_user: Any = None) -> Any:
	return SimpleNamespace(
		appbuilder=SimpleNamespace(
			sm=SimpleNamespace(get_user=lambda: appbuilder_user)
		)
	)


def test_etlp_blueprint_no_longer_returns_fixed_user_context():
	source = BLUEPRINT_PATH.read_text(encoding="utf-8")

	assert "'default_tenant'" not in source
	assert "'current_user'" not in source
	assert "For now, return a default user context" not in source
	assert source.count("return _resolve_current_user(self)") == 5


def test_etlp_blueprint_context_resolves_appbuilder_and_environment(monkeypatch):
	helpers = _helpers()
	resolve = helpers["_resolve_current_user"]

	monkeypatch.setenv("APG_DEFAULT_USER_ID", "env-user")
	monkeypatch.setenv("APG_DEFAULT_TENANT_ID", "env-tenant")
	monkeypatch.setenv("APG_DEFAULT_ROLES", "etl:read etl:write")
	assert resolve() == {
		"tenant_id": "env-tenant",
		"user_id": "env-user",
		"username": "env-user",
		"roles": ["etl:read", "etl:write"],
	}

	appbuilder_user = SimpleNamespace(
		id=42,
		username="fab-user",
		tenant_id="fab-tenant",
		roles=[SimpleNamespace(name="Admin"), SimpleNamespace(name="Operator")],
	)
	assert resolve(_view(appbuilder_user)) == {
		"tenant_id": "fab-tenant",
		"user_id": "42",
		"username": "fab-user",
		"roles": ["Admin", "Operator"],
	}


def test_etlp_blueprint_context_resolves_flask_request_context():
	helpers = _helpers(has_context=True)
	helpers["g"] = SimpleNamespace(tenant_id="g-tenant", user_id="g-user", user={"username": "g-name"})
	helpers["request"] = SimpleNamespace(headers={"X-APG-Roles": "runner reviewer"}, args={})
	helpers["session"] = {"tenant_id": "session-tenant", "user_id": "session-user"}

	assert helpers["_resolve_current_user"]() == {
		"tenant_id": "g-tenant",
		"user_id": "g-user",
		"username": "g-name",
		"roles": ["runner", "reviewer"],
	}

	helpers = _helpers(has_context=True)
	helpers["g"] = SimpleNamespace()
	helpers["request"] = SimpleNamespace(
		headers={"X-APG-User-ID": "header-user", "X-APG-Tenant-ID": "header-tenant"},
		args={"tenant_id": "query-tenant"},
	)
	helpers["session"] = {}

	assert helpers["_resolve_current_user"]()["user_id"] == "header-user"
	assert helpers["_resolve_current_user"]()["tenant_id"] == "header-tenant"
