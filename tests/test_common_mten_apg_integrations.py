"""Executable APG integration regressions for MTEN."""

from __future__ import annotations

import asyncio
from typing import Any

from capabilities.common.mten.service import (
	APGAuditComplianceIntegration,
	MultiTenantManager,
	TenantPermissionSet,
)


class RecordingAuthService:
	def __init__(self) -> None:
		self.calls: list[tuple[str, str]] = []

	async def get_tenant_permissions(self, tenant_id: str, user_id: str) -> dict[str, Any]:
		self.calls.append((tenant_id, user_id))
		return {
			"tenant_id": tenant_id,
			"user_id": user_id,
			"roles": ["tenant_operator"],
			"capabilities": ["tenant.read", "tenant.scale"],
			"resource_access": {"apis": ["tenant.scale"]},
			"source": "recording_auth",
		}


class RecordingAuditService:
	enabled = True

	def __init__(self) -> None:
		self.events: list[Any] = []

	async def log_event(self, audit_log: Any) -> dict[str, Any]:
		self.events.append(audit_log)
		return {"logged": True, "event_count": len(self.events)}


def test_mten_configured_auth_service_drives_tenant_permissions() -> None:
	auth_service = RecordingAuthService()
	manager = MultiTenantManager(tenant_id="system", apg_auth_endpoint="http://auth.local")

	async def scenario() -> TenantPermissionSet:
		await manager.initialize({
			"apg_integrations": {"auth_service": auth_service},
			"enable_multi_cloud": False,
			"enable_security_compliance": False,
			"enable_analytics": False,
		})
		return await manager.get_tenant_permissions("tenant-a", "user-a")

	permissions = asyncio.run(scenario())

	assert auth_service.calls == [("tenant-a", "user-a")]
	assert permissions.roles == ["tenant_operator"]
	assert permissions.capabilities == ["tenant.read", "tenant.scale"]
	assert permissions.resource_access == {"apis": ["tenant.scale"]}
	assert permissions.source == "recording_auth"
	assert permissions.model_dump()["tenant_id"] == "tenant-a"


def test_mten_audit_service_records_tenant_lifecycle_events() -> None:
	audit_service = RecordingAuditService()
	manager = MultiTenantManager(tenant_id="system", apg_auth_endpoint="http://auth.local")

	async def scenario() -> None:
		await manager.initialize({
			"apg_integrations": {"audit_service": audit_service},
			"enable_multi_cloud": False,
			"enable_security_compliance": False,
			"enable_analytics": False,
		})
		await manager.create_tenant(
			name="audit-integration",
			display_name="Audit Integration",
			organization_name="Audit Org",
			contact_email="audit@example.com",
			primary_domain="audit.example.com",
			created_by="auditor",
		)

	asyncio.run(scenario())

	assert len(audit_service.events) == 1
	assert audit_service.events[0].action == "tenant_created"
	assert audit_service.events[0].actor_id == "auditor"


def test_mten_default_audit_integration_is_executable() -> None:
	audit = APGAuditComplianceIntegration(enabled=True, framework="SOC2")
	manager = MultiTenantManager(tenant_id="system", apg_auth_endpoint="http://auth.local")

	async def scenario() -> APGAuditComplianceIntegration:
		await manager.initialize({
			"apg_integrations": {"audit_service": audit},
			"enable_multi_cloud": False,
			"enable_security_compliance": False,
			"enable_analytics": False,
		})
		await manager._log_audit_event(
			tenant_id="tenant-a",
			action="tenant_created",
			actor_id="system",
			resource_type="tenant",
			resource_id="tenant-a",
		)
		return audit

	result = asyncio.run(scenario())

	assert result.events
	assert result.events[0]["tenant_id"] == "tenant-a"
	assert result.events[0]["framework"] == "SOC2"
