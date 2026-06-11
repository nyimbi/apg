"""SpiceDB authorization provider (Authzed / Google Zanzibar-style ReBAC).

SpiceDB handles ONLY authorization — not authentication.
Pair it with Keycloak, Clerk, or BetterAuth for authentication.

SpiceDB excels at fine-grained resource-level permissions using relationship tuples:
    user:alice   viewer   document:readme
    user:bob     editor   document:readme
    group:eng    member   user:alice

Config:
    APG_SPICEDB_URL    grpc://spicedb:50051 or https://grpc.authzed.com
    APG_SPICEDB_TOKEN  <pre-shared-key>
    APG_SPICEDB_SCHEMA <schema-file-path or inline schema>
"""
from __future__ import annotations

import logging
import os
from typing import Any

from capabilities.common.reliability import BoundedCache
from capabilities.common.reliability.circuit_breaker import get_circuit_breaker
from ..protocols import AuthzProvider, ProviderNotImplementedError

_log = logging.getLogger(__name__)


class SpiceDBAuthzProvider:
    """SpiceDB relationship-based authorization via its REST API or gRPC.

    Uses the SpiceDB HTTP gateway (v1 API) for simplicity.
    For high-throughput deployments, replace with the grpcio client.

    Schema example (write to APG_SPICEDB_SCHEMA file):
        definition user {}
        definition tenant {}
        definition document {
            relation owner: user
            relation editor: user | group#member
            relation viewer: user | group#member
            permission edit = owner + editor
            permission view = editor + viewer
        }
    """

    provider_name = "spicedb"

    def __init__(
        self,
        url: str | None = None,
        token: str | None = None,
    ) -> None:
        self._url = (url or os.environ.get("APG_SPICEDB_URL", "http://localhost:8080")).rstrip("/")
        self._token = token or os.environ.get("APG_SPICEDB_TOKEN", "")
        self._perm_cache = BoundedCache(max_size=50000)
        self._cb = get_circuit_breaker("spicedb_authz", failure_threshold=5, reset_timeout=60.0)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }

    async def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        import httpx
        await self._cb._before_call()
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(f"{self._url}{path}", json=body, headers=self._headers())
                resp.raise_for_status()
                await self._cb._on_success()
                return resp.json()
        except Exception as exc:
            await self._cb._on_failure(exc)
            raise

    async def check_permission(
        self,
        user_id: str,
        permission: str,
        tenant_id: str = "default",
        resource_id: str | None = None,
        resource_type: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> bool:
        """Check if user has permission, optionally scoped to a resource."""
        rtype = resource_type or "resource"
        rid = resource_id or tenant_id
        return await self.check_resource_access(user_id, rtype, rid, permission, tenant_id)

    async def check_resource_access(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        action: str,
        tenant_id: str = "default",
    ) -> bool:
        """SpiceDB CheckPermission — the primary authorization call."""
        cache_key = f"spicedb:{tenant_id}:{user_id}:{resource_type}/{resource_id}:{action}"
        cached = self._perm_cache.get(cache_key)
        if cached is not None:
            return bool(cached)
        try:
            data = await self._post("/v1/permissions/check", {
                "resource": {"objectType": resource_type, "objectId": resource_id},
                "permission": action,
                "subject": {"object": {"objectType": "user", "objectId": user_id}},
                "consistency": {"minimizeLatency": True},
            })
            # SpiceDB returns permissionship: PERMISSIONSHIP_HAS_PERMISSION | PERMISSIONSHIP_NO_PERMISSION
            result = data.get("permissionship") == "PERMISSIONSHIP_HAS_PERMISSION"
            self._perm_cache.set(cache_key, result, ttl=30)
            return result
        except Exception as exc:
            _log.error("SpiceDB check_resource_access failed: %s", exc)
            return False

    async def get_user_roles(self, user_id: str, tenant_id: str = "default") -> list[str]:
        """Read all role relationships for a user in a tenant.

        Requires SpiceDB schema to have a 'role' object type with a 'member' relation.
        Falls back to empty list if schema doesn't have this pattern.
        """
        try:
            data = await self._post("/v1/permissions/resources", {
                "resourceObjectType": "role",
                "permission": "member",
                "subject": {"object": {"objectType": "user", "objectId": user_id}},
                "consistency": {"minimizeLatency": True},
            })
            return [r.get("objectId", "") for r in data.get("resources", [])]
        except Exception as exc:
            _log.debug("SpiceDB get_user_roles failed (may not have role schema): %s", exc)
            return []

    async def assign_role(
        self, user_id: str, role: str, tenant_id: str = "default", granted_by: str = "system"
    ) -> None:
        await self.write_relationship("role", role, "member", "user", user_id)
        self._perm_cache.clear()

    async def revoke_role(
        self, user_id: str, role: str, tenant_id: str = "default", revoked_by: str = "system"
    ) -> None:
        await self.delete_relationship("role", role, "member", "user", user_id)
        self._perm_cache.clear()

    async def get_role_permissions(self, role: str, tenant_id: str = "default") -> list[str]:
        return []

    async def create_role(
        self, role: str, permissions: list[str], tenant_id: str = "default", description: str = ""
    ) -> dict[str, Any]:
        # SpiceDB roles are defined in schema, not created at runtime
        # Write relationship tuples for each permission
        for perm in permissions:
            await self.write_relationship("permission", perm, "has_role", "role", role)
        return {"role": role, "permissions": permissions}

    async def delete_role(self, role: str, tenant_id: str = "default") -> None:
        pass  # Schema changes required for role deletion in SpiceDB

    async def list_roles(self, tenant_id: str = "default") -> list[dict[str, Any]]:
        return []  # SpiceDB doesn't enumerate roles via API — defined in schema

    async def write_relationship(
        self,
        resource_type: str,
        resource_id: str,
        relation: str,
        subject_type: str,
        subject_id: str,
    ) -> None:
        """Write a relationship tuple to SpiceDB."""
        await self._post("/v1/relationships/write", {
            "updates": [{
                "operation": "OPERATION_CREATE",
                "relationship": {
                    "resource": {"objectType": resource_type, "objectId": resource_id},
                    "relation": relation,
                    "subject": {"object": {"objectType": subject_type, "objectId": subject_id}},
                },
            }],
        })
        self._perm_cache.clear()

    async def delete_relationship(
        self,
        resource_type: str,
        resource_id: str,
        relation: str,
        subject_type: str,
        subject_id: str,
    ) -> None:
        """Delete a relationship tuple from SpiceDB."""
        await self._post("/v1/relationships/write", {
            "updates": [{
                "operation": "OPERATION_DELETE",
                "relationship": {
                    "resource": {"objectType": resource_type, "objectId": resource_id},
                    "relation": relation,
                    "subject": {"object": {"objectType": subject_type, "objectId": subject_id}},
                },
            }],
        })
        self._perm_cache.clear()

    async def list_accessible_resources(
        self,
        user_id: str,
        resource_type: str,
        action: str,
        tenant_id: str = "default",
    ) -> list[str]:
        """LookupResources — return IDs of all resources user can access."""
        try:
            data = await self._post("/v1/permissions/resources", {
                "resourceObjectType": resource_type,
                "permission": action,
                "subject": {"object": {"objectType": "user", "objectId": user_id}},
                "consistency": {"minimizeLatency": True},
            })
            return [r.get("objectId", "") for r in data.get("resources", [])]
        except Exception as exc:
            _log.error("SpiceDB list_accessible_resources failed: %s", exc)
            return []

    async def bulk_check_permissions(
        self,
        user_id: str,
        checks: list[dict[str, Any]],
        tenant_id: str = "default",
    ) -> dict[str, bool]:
        """Run multiple permission checks concurrently."""
        import asyncio
        results = {}
        async def _check(check: dict[str, Any]) -> tuple[str, bool]:
            perm = check.get("permission", "")
            result = await self.check_permission(
                user_id, perm, tenant_id,
                resource_id=check.get("resource_id"),
                resource_type=check.get("resource_type"),
            )
            return perm, result
        checks_results = await asyncio.gather(*[_check(c) for c in checks], return_exceptions=True)
        for item in checks_results:
            if isinstance(item, Exception):
                _log.debug("Suppressed %s: %s", type(item).__name__, item)
            else:
                perm, result = item
                results[perm] = result
        return results

    async def health_check(self) -> dict[str, Any]:
        import httpx
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self._url}/healthz", headers=self._headers())
                return {"status": "ok" if resp.status_code == 200 else "degraded",
                        "provider": "spicedb", "url": self._url}
        except Exception as exc:
            return {"status": "unhealthy", "provider": "spicedb", "error": str(exc)}
