"""Domain adapters for Deposit Products Engine.

Protocol-based adapters for auth, audit, notify, and GL posting.
Null implementations enable standalone/test operation without any
external APG capability installed.

Usage (standalone)::

    svc = DepositProductsService()  # null adapters

Usage (platform)::

    from apg_common_auth import AuthService
    svc = DepositProductsService(auth=AuthService.from_env())

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""
from __future__ import annotations

import json
from typing import Any, Protocol, runtime_checkable


# ─────────────────────────────────────────────────────────────
# Auth adapter
# ─────────────────────────────────────────────────────────────

@runtime_checkable
class AuthAdapter(Protocol):
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
	async def send(self, recipient: str, channel: str, subject: str, body: str,
		metadata: dict[str, Any] | None = None) -> None:
		print(f"[NOTIFY] {channel}→{recipient}: {subject}")


# ─────────────────────────────────────────────────────────────
# GL posting adapter  (dep-specific: posts journal entries)
# ─────────────────────────────────────────────────────────────

@runtime_checkable
class GLAdapter(Protocol):
	async def post_journal(
		self,
		tenant_id: str,
		debit_account: str,
		credit_account: str,
		amount: str,
		currency: str,
		narration: str,
		reference: str,
	) -> dict[str, Any]: ...


class NullGLAdapter:
	"""Standalone fallback — prints journal entry."""
	async def post_journal(
		self,
		tenant_id: str,
		debit_account: str,
		credit_account: str,
		amount: str,
		currency: str,
		narration: str,
		reference: str,
	) -> dict[str, Any]:
		entry = {
			"status":         "posted",
			"debit_account":  debit_account,
			"credit_account": credit_account,
			"amount":         amount,
			"currency":       currency,
			"narration":      narration,
			"reference":      reference,
		}
		print(f"[GL] {json.dumps(entry)}")
		return entry


# ─────────────────────────────────────────────────────────────
# Adapter factories
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
		pass
	try:
		from capabilities.common.nats.nats_adapter import get_nats_audit_adapter  # type: ignore[import]
		adapter = get_nats_audit_adapter("fin_dep")
		if adapter is not None:
			return adapter
	except ImportError:
		pass
	return NullAuditAdapter()


def get_notify_adapter(notify_service: Any | None = None) -> NotifyAdapter:
	if notify_service is not None:
		return notify_service
	try:
		from apg_common_ntfy import NotifyService  # type: ignore[import]
		return NotifyService.from_env()
	except ImportError:
		return NullNotifyAdapter()


def get_gl_adapter(gl_service: Any | None = None) -> GLAdapter:
	if gl_service is not None:
		return gl_service
	try:
		from capabilities.fin.glr import GLService  # type: ignore[import]
		return GLService.from_env()
	except ImportError:
		return NullGLAdapter()
