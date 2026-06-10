"""Adapter protocols for ussd_eng capability.

Null adapters are used in standalone mode. Platform wiring injects real adapters.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Protocol, runtime_checkable

_log = logging.getLogger(__name__)


# ── Auth adapter ──────────────────────────────────────────────────────────────

@runtime_checkable
class AuthAdapter(Protocol):
	async def verify_token(self, token: str) -> dict[str, Any]: ...
	async def check_permission(self, user_id: str, permission: str, resource: str | None = None) -> bool: ...


class NullAuthAdapter:
	async def verify_token(self, token: str) -> dict[str, Any]:
		return {"user_id": token or "anonymous", "tenant_id": "default", "roles": ["admin"]}

	async def check_permission(self, user_id: str, permission: str, resource: str | None = None) -> bool:
		return True


# ── Audit adapter ─────────────────────────────────────────────────────────────

@runtime_checkable
class AuditAdapter(Protocol):
	async def log_event(self, event_type: str, actor_id: str, tenant_id: str,
	                    resource_id: str, details: dict[str, Any]) -> None: ...


class NullAuditAdapter:
	async def log_event(self, event_type: str, actor_id: str, tenant_id: str,
	                    resource_id: str, details: dict[str, Any]) -> None:
		print(json.dumps({
			"event_type": event_type, "actor_id": actor_id, "tenant_id": tenant_id,
			"resource_id": resource_id, "details": details,
		}, default=str))


# ── Notify adapter ────────────────────────────────────────────────────────────

@runtime_checkable
class NotifyAdapter(Protocol):
	async def send(self, recipient: str, channel: str, subject: str, body: str,
	               metadata: dict[str, Any] | None = None) -> None: ...


class NullNotifyAdapter:
	async def send(self, recipient: str, channel: str, subject: str, body: str,
	               metadata: dict[str, Any] | None = None) -> None:
		print(f"[NOTIFY] {channel}→{recipient}: {subject}")


# ── NATS audit adapter factory ────────────────────────────────────────────────

def get_audit_adapter(capability_id: str = "ussd_eng") -> AuditAdapter | None:
	nats_url = os.environ.get("NATS_URL")
	if nats_url:
		try:
			from capabilities.common.nats.nats_adapter import NATSEventAdapter
			return NATSEventAdapter(capability_id)
		except Exception as exc:
			_log.debug("NATS unavailable: %s", exc)
	return None


def get_auth_adapter(auth_service: Any | None = None) -> AuthAdapter:
	if auth_service is not None:
		return auth_service
	try:
		from apg_common_auth import AuthService  # type: ignore[import]
		return AuthService.from_env()
	except ImportError:
		return NullAuthAdapter()


def get_notify_adapter(notify_service: Any | None = None) -> NotifyAdapter:
	if notify_service is not None:
		return notify_service
	try:
		from apg_common_ntfy import NotifyService  # type: ignore[import]
		return NotifyService.from_env()
	except ImportError:
		return NullNotifyAdapter()
