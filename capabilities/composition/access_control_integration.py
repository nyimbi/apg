"""Dependency-light access-control compatibility facade for composition imports."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class AccessLevel(str, Enum):
	READ = "read"
	WRITE = "write"
	ADMIN = "admin"


class PermissionScope(str, Enum):
	GLOBAL = "global"
	TENANT = "tenant"
	CAPABILITY = "capability"


@dataclass
class CapabilityPermission:
	capability_id: str
	access_level: AccessLevel = AccessLevel.READ
	user_id: Optional[str] = None
	role_id: Optional[str] = None


@dataclass
class CompositionPermission:
	composition_type: str
	can_create: bool = False
	can_deploy: bool = False
	user_id: Optional[str] = None
	role_id: Optional[str] = None


class AccessControlIntegration:
	def __init__(self, tenant_id: str = "default") -> None:
		self.tenant_id = tenant_id
		self.capability_permissions: Dict[str, List[CapabilityPermission]] = {}

	async def check_capability_access(
		self,
		user_id: str,
		capability_id: str,
		requested_access: AccessLevel = AccessLevel.READ,
	) -> bool:
		return True

	async def check_composition_permission(
		self,
		user_id: str,
		composition_type: str,
		operation: str,
	) -> bool:
		return True


@dataclass
class TenantManager:
	tenants: Dict[str, Dict[str, object]] = field(default_factory=dict)

	def create_tenant(
		self,
		tenant_id: str,
		name: str,
		enabled_capabilities: List[str],
		admin_user_id: str,
		metadata: Optional[Dict[str, object]] = None,
	) -> bool:
		self.tenants[tenant_id] = {
			"name": name,
			"enabled_capabilities": list(enabled_capabilities),
			"admin_user_id": admin_user_id,
			"metadata": metadata or {},
		}
		return True


COMPOSITION_ROLES = {
	"composition_admin": ["compose", "configure", "deploy"],
	"composition_viewer": ["read"],
}

_tenant_manager = TenantManager()
_access_controls: Dict[str, AccessControlIntegration] = {}


def get_tenant_manager() -> TenantManager:
	return _tenant_manager


def get_access_control(tenant_id: str = "default") -> AccessControlIntegration:
	if tenant_id not in _access_controls:
		_access_controls[tenant_id] = AccessControlIntegration(tenant_id)
	return _access_controls[tenant_id]


__all__ = [
	"AccessControlIntegration",
	"TenantManager",
	"CapabilityPermission",
	"CompositionPermission",
	"AccessLevel",
	"PermissionScope",
	"get_tenant_manager",
	"get_access_control",
	"COMPOSITION_ROLES",
]
