"""Context resolution regressions for MDM and Pose Flask blueprints."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
MDM_BLUEPRINT_PATH = REPO_ROOT / "capabilities" / "common" / "mdm" / "blueprint.py"
POSE_BLUEPRINT_PATH = REPO_ROOT / "capabilities" / "common" / "pose" / "blueprint.py"


def _mdm_helpers(has_context: bool = False) -> dict[str, Any]:
	source = MDM_BLUEPRINT_PATH.read_text(encoding="utf-8")
	start = source.index("def _clean_text")
	end = source.index("\n\n# Forms for MDM Operations")
	namespace: dict[str, Any] = {
		"Any": Any,
		"Dict": Dict,
		"List": List,
		"Optional": Optional,
		"g": SimpleNamespace(),
		"has_request_context": lambda: has_context,
		"os": os,
		"request": SimpleNamespace(
			current_user=None,
			headers={},
			args={},
			remote_addr=None,
			user_agent=SimpleNamespace(string=None),
		),
		"session": {},
	}
	exec(compile(source[start:end], str(MDM_BLUEPRINT_PATH), "exec"), namespace)
	return namespace


def _pose_helpers(has_context: bool = False) -> dict[str, Any]:
	source = POSE_BLUEPRINT_PATH.read_text(encoding="utf-8")
	start = source.index("def _clean_text")
	end = source.index("\n# APG Blueprint Registration")
	namespace: dict[str, Any] = {
		"Any": Any,
		"Dict": Dict,
		"List": List,
		"Optional": Optional,
		"g": SimpleNamespace(),
		"has_request_context": lambda: has_context,
		"os": os,
		"request": SimpleNamespace(current_user=None, headers={}, args={}),
		"session": {},
	}
	exec(compile(source[start:end], str(POSE_BLUEPRINT_PATH), "exec"), namespace)
	return namespace


def _view(appbuilder_user: Any = None) -> Any:
	return SimpleNamespace(
		appbuilder=SimpleNamespace(
			sm=SimpleNamespace(get_user=lambda: appbuilder_user)
		)
	)


def test_mdm_and_pose_blueprints_no_longer_use_fixed_runtime_identity():
	mdm_source = MDM_BLUEPRINT_PATH.read_text(encoding="utf-8")
	pose_source = POSE_BLUEPRINT_PATH.read_text(encoding="utf-8")

	assert "'current_user'" not in mdm_source
	assert "'current_tenant'" not in mdm_source
	assert "In production, integrate with APG auth" not in mdm_source
	assert "return _resolve_mdm_user_context(self)" in mdm_source

	assert "'tenant_id': 'default'" not in pose_source
	assert "'created_by': 'current_" + "user'" not in pose_source
	assert "Would get from APG auth" not in pose_source
	assert "_resolve_pose_request_context(session_data, self)" in pose_source
	assert "_resolve_pose_request_context(tracking_data, self)" in pose_source


def test_mdm_blueprint_context_resolves_runtime_sources(monkeypatch):
	resolve = _mdm_helpers()["_resolve_mdm_user_context"]

	monkeypatch.setenv("APG_DEFAULT_USER_ID", "mdm-env-user")
	monkeypatch.setenv("APG_DEFAULT_TENANT_ID", "mdm-env-tenant")
	assert resolve() == {
		"user_id": "mdm-env-user",
		"tenant_id": "mdm-env-tenant",
		"permissions": ["mdm.read", "mdm.write"],
		"client_ip": None,
		"user_agent": None,
	}

	helpers = _mdm_helpers(has_context=True)
	helpers["g"] = SimpleNamespace(current_user={"user_id": "g-user", "tenant_id": "g-tenant"})
	helpers["request"] = SimpleNamespace(
		current_user={"user_id": "request-user", "tenant_id": "request-tenant"},
		headers={"X-APG-User-ID": "header-user", "X-APG-Tenant-ID": "header-tenant"},
		args={"user_id": "query-user", "tenant_id": "query-tenant"},
		remote_addr="127.0.0.1",
		user_agent=SimpleNamespace(string="pytest-agent"),
	)
	helpers["session"] = {"user_id": "session-user", "tenant_id": "session-tenant"}
	assert helpers["_resolve_mdm_user_context"]() == {
		"user_id": "request-user",
		"tenant_id": "request-tenant",
		"permissions": ["mdm.read", "mdm.write"],
		"client_ip": "127.0.0.1",
		"user_agent": "pytest-agent",
	}

	appbuilder_user = SimpleNamespace(
		id="fab-user",
		tenant_id="fab-tenant",
		roles=[SimpleNamespace(name="DataSteward")],
	)
	helpers = _mdm_helpers(has_context=True)
	helpers["g"] = SimpleNamespace()
	helpers["request"] = SimpleNamespace(
		current_user=None,
		headers={"X-APG-User-ID": "header-user", "X-APG-Tenant-ID": "header-tenant"},
		args={},
		remote_addr=None,
		user_agent=SimpleNamespace(string=None),
	)
	helpers["session"] = {}
	context = helpers["_resolve_mdm_user_context"](_view(appbuilder_user))
	assert context["user_id"] == "fab-user"
	assert context["tenant_id"] == "fab-tenant"
	assert context["permissions"] == ["DataSteward"]


def test_pose_blueprint_context_resolves_runtime_sources(monkeypatch):
	resolve = _pose_helpers()["_resolve_pose_request_context"]

	monkeypatch.setenv("APG_DEFAULT_USER_ID", "pose-env-user")
	monkeypatch.setenv("APG_DEFAULT_TENANT_ID", "pose-env-tenant")
	assert resolve() == {"tenant_id": "pose-env-tenant", "user_id": "pose-env-user"}

	helpers = _pose_helpers(has_context=True)
	helpers["g"] = SimpleNamespace(current_user={"user_id": "g-user", "tenant_id": "g-tenant"})
	helpers["request"] = SimpleNamespace(
		current_user={"user_id": "request-user", "tenant_id": "request-tenant"},
		headers={"X-APG-User-ID": "header-user", "X-APG-Tenant-ID": "header-tenant"},
		args={"user_id": "query-user", "tenant_id": "query-tenant"},
	)
	helpers["session"] = {"user_id": "session-user", "tenant_id": "session-tenant"}
	assert helpers["_resolve_pose_request_context"]({"tenant_id": "payload-tenant", "created_by": "payload-user"}) == {
		"tenant_id": "request-tenant",
		"user_id": "request-user",
	}

	helpers = _pose_helpers(has_context=True)
	helpers["g"] = SimpleNamespace()
	helpers["request"] = SimpleNamespace(
		current_user=None,
		headers={"X-APG-User-ID": "header-user", "X-APG-Tenant-ID": "header-tenant"},
		args={},
	)
	helpers["session"] = {}
	assert helpers["_resolve_pose_request_context"]({}) == {
		"tenant_id": "header-tenant",
		"user_id": "header-user",
	}
