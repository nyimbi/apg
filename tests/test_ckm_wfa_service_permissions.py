"""Executable permission regressions for CKM WFA service integration."""

from __future__ import annotations

import asyncio
import inspect
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
SERVICE_PATH = REPO_ROOT / "capabilities" / "ckm" / "wfa" / "service.py"


def _integration_helpers() -> dict[str, Any]:
	source = SERVICE_PATH.read_text(encoding="utf-8")
	start = source.index("WORKFLOW_PERMISSION_ALIASES")
	end = source.index("# =============================================================================\n# Core Service Classes")
	namespace: dict[str, Any] = {
		"Any": Any,
		"Dict": Dict,
		"List": List,
		"Optional": Optional,
		"APGTenantContext": SimpleNamespace,
		"WBPMServiceConfig": SimpleNamespace,
		"asyncio": asyncio,
		"inspect": inspect,
		"logger": logging.getLogger("test-wfa-service"),
	}
	exec(compile(source[start:end], str(SERVICE_PATH), "exec"), namespace)
	return namespace


def test_wfa_permission_aliases_bridge_internal_and_public_permissions():
	helpers = _integration_helpers()
	config = SimpleNamespace(
		apg_auth_service_url="",
		apg_audit_service_url="",
		apg_collaboration_service_url="",
		apg_ai_service_url="",
	)
	context = SimpleNamespace(
		tenant_id="tenant-a",
		user_id="user-a",
		permissions=["wbpm:process:write", "wbpm:process:read", "workflow_execute"],
		metadata={},
	)
	integration = helpers["APGPlatformIntegration"](config)

	assert asyncio.run(integration.validate_user_permissions(context, ["create_process", "view_process"]))
	assert asyncio.run(integration.validate_user_permissions(context, ["start_process"]))
	assert not asyncio.run(integration.validate_user_permissions(context, ["manage_instance"]))


def test_wfa_permission_validation_uses_auth_service_boundary():
	helpers = _integration_helpers()

	class AuthService:
		async def has_permissions(self, *, context: Any, required_permissions: List[str]) -> dict[str, bool]:
			return {"allowed": context.tenant_id == "tenant-a" and "manage_instance" in required_permissions}

	config = SimpleNamespace(
		apg_auth_service_url="",
		apg_audit_service_url="",
		apg_collaboration_service_url="",
		apg_ai_service_url="",
	)
	context = SimpleNamespace(
		tenant_id="tenant-a",
		user_id="user-a",
		permissions=[],
		metadata={},
	)
	integration = helpers["APGPlatformIntegration"](config, auth_service=AuthService())

	assert asyncio.run(integration.validate_user_permissions(context, ["manage_instance"]))


def test_wfa_permission_validation_no_longer_simulates_auth_service():
	source = SERVICE_PATH.read_text(encoding="utf-8")

	assert "For now, simulate permission validation" not in source
	assert "In production, this would make actual HTTP calls to APG auth service" not in source
	assert "WORKFLOW_PERMISSION_ALIASES" in source
	assert "_validate_permissions_with_auth_service" in source
