"""NATS and adapter protocols for Land Registry (gov_lnd)."""
from __future__ import annotations

import logging
import os
from typing import Any, Protocol, runtime_checkable

_log = logging.getLogger(__name__)


@runtime_checkable
class AuditAdapter(Protocol):
	async def log_event(self, event_type: str, actor_id: str, tenant_id: str,
		resource_id: str, details: dict[str, Any]) -> None: ...


class NullAuditAdapter:
	"""Standalone fallback — logs to stderr."""
	async def log_event(self, event_type: str, actor_id: str, tenant_id: str,
		resource_id: str, details: dict[str, Any]) -> None:
		_log.info("AUDIT [%s] actor=%s tenant=%s resource=%s", event_type, actor_id, tenant_id, resource_id)


def get_audit_adapter(capability_id: str = "gov_lnd") -> AuditAdapter | None:
	nats_url = os.environ.get("NATS_URL")
	if nats_url:
		try:
			from capabilities.common.nats.nats_adapter import NATSEventAdapter
			return NATSEventAdapter(capability_id)
		except Exception as exc:
			_log.debug("NATS unavailable: %s", exc)
	try:
		from apg_common_audl import AuditService  # type: ignore[import]
		return AuditService.from_env()
	except ImportError as _exc:
		_log.debug("Handled exception: %s", _exc)
	return NullAuditAdapter()


@runtime_checkable
class NotifyAdapter(Protocol):
	async def send(self, recipient: str, channel: str, subject: str, body: str,
		metadata: dict[str, Any] | None = None) -> None: ...


class NullNotifyAdapter:
	async def send(self, recipient: str, channel: str, subject: str, body: str,
		metadata: dict[str, Any] | None = None) -> None:
		_log.info("NOTIFY %s→%s: %s", channel, recipient, subject)


def get_notify_adapter() -> NotifyAdapter:
	try:
		from apg_common_ntfy import NotifyService  # type: ignore[import]
		return NotifyService.from_env()
	except ImportError:
		return NullNotifyAdapter()
