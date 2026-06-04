"""Adapter protocols for Distribution Network.

Each required service (from the REQUIRES contract list) is represented as a
Protocol here.  When running standalone, the Null* adapters are used
automatically — no external capabilities are needed.  When running inside
the APG platform, real adapters wrapping the installed capability packages
are wired in via :func:`get_adapters`.

Usage (standalone)::

    svc = DistributionNetworkService()                   # null adapters, in-memory store
    svc = DistributionNetworkService(db_url="postgresql+asyncpg://...")

Usage (platform)::

    from apg_common_auth import AuthService
    svc = DistributionNetworkService(auth=AuthService.from_env())
"""
from __future__ import annotations

import json
import os
from typing import Any, Protocol, runtime_checkable


# ─────────────────────────────────────────────────────────────
# Auth adapter
# ─────────────────────────────────────────────────────────────
@runtime_checkable
class AuthAdapter(Protocol):
    """Minimal auth surface required by this capability."""
    async def verify_token(self, token: str) -> dict[str, Any]: ...
    async def check_permission(self, user_id: str, permission: str, resource: str | None = None) -> bool: ...
    async def get_current_user(self, token: str) -> dict[str, Any]: ...


class NullAuthAdapter:
    """Standalone fallback — all tokens accepted, all permissions granted."""
    async def verify_token(self, token: str) -> dict[str, Any]:
        return {"user_id": token or "anonymous", "tenant_id": "default", "roles": ["admin"]}

    async def check_permission(self, user_id: str, permission: str, resource: str | None = None) -> bool:
        return True

    async def get_current_user(self, token: str) -> dict[str, Any]:
        return {"id": token or "anonymous", "name": "Standalone User", "roles": ["admin"]}


class _InstalledAuthAdapter:
    """Wraps apg-common-auth when installed."""
    def __init__(self, svc: Any) -> None:
        self._svc = svc

    async def verify_token(self, token: str) -> dict[str, Any]:
        return await self._svc.verify_token(token)

    async def check_permission(self, user_id: str, permission: str, resource: str | None = None) -> bool:
        return await self._svc.check_permission(user_id, permission, resource)

    async def get_current_user(self, token: str) -> dict[str, Any]:
        return await self._svc.get_current_user(token)


# ─────────────────────────────────────────────────────────────
# Audit adapter
# ─────────────────────────────────────────────────────────────
@runtime_checkable
class AuditAdapter(Protocol):
    async def log_event(self, event_type: str, actor_id: str, tenant_id: str,
                        resource_id: str, details: dict[str, Any]) -> None: ...


class NullAuditAdapter:
    """Standalone fallback — logs to stdout as JSON."""
    async def log_event(self, event_type: str, actor_id: str, tenant_id: str,
                        resource_id: str, details: dict[str, Any]) -> None:
        print(json.dumps({
            "event_type": event_type, "actor_id": actor_id, "tenant_id": tenant_id,
            "resource_id": resource_id, "details": details,
        }, default=str))


# ─────────────────────────────────────────────────────────────
# Notify adapter
# ─────────────────────────────────────────────────────────────
@runtime_checkable
class NotifyAdapter(Protocol):
    async def send(self, recipient: str, channel: str, subject: str, body: str,
                   metadata: dict[str, Any] | None = None) -> None: ...


class NullNotifyAdapter:
    """Standalone fallback — prints notification."""
    async def send(self, recipient: str, channel: str, subject: str, body: str,
                   metadata: dict[str, Any] | None = None) -> None:
        print(f"[NOTIFY] {channel}→{recipient}: {subject}")


# ─────────────────────────────────────────────────────────────
# Workflow adapter
# ─────────────────────────────────────────────────────────────
@runtime_checkable
class WorkflowAdapter(Protocol):
    async def start_workflow(self, definition_id: str, payload: dict[str, Any]) -> dict[str, Any]: ...
    async def complete_task(self, task_id: str, outcome: str, variables: dict[str, Any]) -> None: ...


class NullWorkflowAdapter:
    """Standalone fallback — synchronous in-process workflow."""
    async def start_workflow(self, definition_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {"instance_id": f"local-{definition_id}", "status": "running", "payload": payload}

    async def complete_task(self, task_id: str, outcome: str, variables: dict[str, Any]) -> None:
        pass  # No-op in standalone mode


# ─────────────────────────────────────────────────────────────
# Adapter factory
# ─────────────────────────────────────────────────────────────
def get_auth_adapter(auth_service: Any | None = None) -> AuthAdapter:
    if auth_service is not None:
        return _InstalledAuthAdapter(auth_service)
    try:
        from apg_common_auth import AuthService  # type: ignore[import]
        return _InstalledAuthAdapter(AuthService.from_env())
    except ImportError:
        return NullAuthAdapter()


def get_audit_adapter(audit_service: Any | None = None) -> AuditAdapter:
    if audit_service is not None:
        return audit_service
    try:
        from apg_common_audl import AuditService  # type: ignore[import]
        return AuditService.from_env()
    except ImportError:
        return NullAuditAdapter()


def get_notify_adapter(notify_service: Any | None = None) -> NotifyAdapter:
    if notify_service is not None:
        return notify_service
    try:
        from apg_common_ntfy import NotifyService  # type: ignore[import]
        return NotifyService.from_env()
    except ImportError:
        return NullNotifyAdapter()


def get_workflow_adapter(workflow_service: Any | None = None) -> WorkflowAdapter:
    if workflow_service is not None:
        return workflow_service
    try:
        from apg_common_wflo import WorkflowService  # type: ignore[import]
        return WorkflowService.from_env()
    except ImportError:
        return NullWorkflowAdapter()
